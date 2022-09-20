import numpy as np
import tensorflow as tf
from past.builtins import xrange
from Dataset import Dataset
from collections import defaultdict
import time
import sys
import math
import random
import os
from utils import ProgressBar


class MTL(object):
    def __init__(self, config, sess):
        t1 = time.time()
        # data corpus and load data: multi-task
        self.data_dir = config['data_dir']
        # 1) task: app rec
        self.data_name_app = config['data_name_app']
        dataset_app = Dataset(self.data_dir + self.data_name_app)
        self.train_app,self.testRatings_app,self.testNegatives_app = dataset_app.trainMatrix,dataset_app.testRatings,dataset_app.testNegatives
        self.nUsers_app, self.nItems_app = self.train_app.shape
        print("Load app data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d, #test_neg=%d"
          %(time.time()-t1, self.nUsers_app, self.nItems_app, self.train_app.nnz, len(self.testRatings_app), len(self.testNegatives_app)*99))
        self.user_gt_item_app = defaultdict(int)
        for user, gt_item in self.testRatings_app:
            self.user_gt_item_app[user] = gt_item
        self.user_input_app, self.item_input_app, self.labels_app = [],[],[]
        self.test_user_input_app, self.test_item_input_app, self.test_labels_app = [],[],[]

        # 2) task: news rec
        self.data_name_news = config['data_name_news']
        dataset_news = Dataset(self.data_dir + self.data_name_news)
        self.train_news,self.testRatings_news,self.testNegatives_news = dataset_news.trainMatrix,dataset_news.testRatings,dataset_news.testNegatives
        self.nUsers_news, self.nItems_news = self.train_news.shape
        print("Load news data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d, #test_neg=%d"
          %(time.time()-t1, self.nUsers_news, self.nItems_news, self.train_news.nnz, len(self.testRatings_news), len(self.testNegatives_news)*99))
        self.user_gt_item_news = defaultdict(int)
        for user, gt_item in self.testRatings_news:
            self.user_gt_item_news[user] = gt_item
        self.user_input_news, self.item_input_news, self.labels_news = [],[],[]
        self.test_user_input_news, self.test_item_input_news, self.test_labels_news = [],[],[]

        if self.nUsers_app != self.nUsers_news:
            print('nUsers_app != nUsers_news. However, they should be shared. exit...')
            sys.exit(0)
        self.nUsers = self.nUsers_news

        # hyper-parameters
        self.init_std = config['init_std']
        self.batch_size = config['batch_size']
        self.nepoch = config['nepoch']
        self.layers = config['layers']
        self.edim_u = config['edim_u']
        self.edim_v = config['edim_v']
        self.edim = self.edim_u + self.edim_v  # concat
        self.nhop = len(self.layers)
        self.max_grad_norm = config['max_grad_norm']
        self.negRatio = config['negRatio']
        self.activation = config['activation']
        self.learner = config['learner']
        self.objective = config['objective']
        self.class_size = 2  # (pos, neg)
        self.input_size = 2  # (user, item)
        # save and restore
        self.show = config['show']
        self.checkpoint_dir = config['checkpoint_dir']
        self.input_app = tf.placeholder(tf.int32, [self.batch_size, self.input_size], name="input")  # (user, app)
        self.target_app = tf.placeholder(tf.float32, [self.batch_size, self.class_size], name="target")
        self.input_news = tf.placeholder(tf.int32, [self.batch_size, self.input_size], name="input")  # (user, news)
        self.target_news = tf.placeholder(tf.float32, [self.batch_size, self.class_size], name="target")
        self.lr = None
        self.init_lr = config['init_lr']
        self.current_lr = config['init_lr']
        self.loss_joint = None
        self.loss_app_joint = None
        self.loss_news_joint = None
        self.loss_app_only = None
        self.loss_news_only = None
        self.optim_joint = None
        self.optim_app = None
        self.optim_news = None
        self.step = None
        self.sess = sess
        self.log_loss_app = []
        self.log_perp_app = []
        self.log_loss_news = []
        self.log_perp_news = []
        self.isDebug = config['isDebug']
        self.isOneBatch = config['isOneBatch']

        # multi-task
        self.weights_app_news = config['weights_app_news']
        self.cross_layers = config['cross_layers']
        assert self.cross_layers > 0 and self.cross_layers < self.nhop

        # evaluation
        self.topK = config['topK']
        self.bestHR_app = 0.
        self.bestHR_epoch_app = -1
        self.bestNDCG_app = 0.
        self.bestNDCG_epoch_app = -1
        self.bestMRR_app = 0.
        self.bestMRR_epoch_app = -1
        self.bestAUC_app = 0.
        self.bestAUC_epoch_app = -1
        self.HR_app, self.NDCG_app, self.MRR_app, self.AUC_app = 0, 0, 0, 0
        self.bestHR_news = 0.
        self.bestHR_epoch_news = -1
        self.bestNDCG_news = 0.
        self.bestNDCG_epoch_news = -1
        self.bestMRR_news = 0.
        self.bestMRR_epoch_news = -1
        self.bestAUC_news = 0.
        self.bestAUC_epoch_news = -1
        self.HR_news, self.NDCG_news, self.MRR_news, self.AUC_news = 0, 0, 0, 0

    def build_memory_shared(self):
        ## ------- parameters: shared-------  ##
        # 1. embedding matrices for input <user, app>, <user, news>: shared user embedding matrix
        self.U = tf.Variable(tf.random_normal([self.nUsers, self.edim_u], stddev=self.init_std))  # sharing user factors
        # 2. match the dimensions
        self.shared_Hs = defaultdict(object)
        for h in xrange(1, self.cross_layers+1):  # only cross between
            self.shared_Hs[h] = tf.Variable(tf.random_normal([self.layers[h-1], self.layers[h]], stddev=self.init_std))

    def build_memory_app_specific(self):
        ## ------- parameters: app specific -------  ##
        # 1. embedding matrices for input <user, app>, <user, news>: shared user embedding matrix
        self.V_app = tf.Variable(tf.random_normal([self.nItems_app, self.edim_v], stddev=self.init_std))

        # 2. weights & biases for hidden layers: the input to hidden layers are the merged embedding
        self.weights_app = defaultdict(object)
        self.biases_app = defaultdict(object)
        for h in xrange(1, self.nhop):  # (nhop-1) weights matrix in hidden layers
            self.weights_app[h] = tf.Variable(tf.random_normal([self.layers[h-1], self.layers[h]], stddev=self.init_std))
            self.biases_app[h] = tf.Variable(tf.random_normal([self.layers[h]], stddev=self.init_std))
        # 3. output layer: weight and bias
        self.h_app = tf.Variable(tf.random_normal([self.layers[-1], self.class_size], stddev=self.init_std))
        self.b_app = tf.Variable(tf.random_normal([self.class_size], stddev=self.init_std))

    def build_model_app_training(self):
        ## ------- computational graph: app training only ------- ##
        # 1. input & embedding layer
        USERin_app = tf.nn.embedding_lookup(self.U, self.input_app[:,0])  # 3D due to batch
        ITEMin_app = tf.nn.embedding_lookup(self.V_app, self.input_app[:,1])
        UIin_app = tf.concat(values=[USERin_app, ITEMin_app], axis=1)  # no info loss, and edim = edim_u + edim_v

        # 2. MLP: hidden layers, http://www.jessicayung.com/explaining-tensorflow-code-for-a-multilayer-perceptron/
        self.layer_h_apps = defaultdict(object)
        layer_h_app = tf.reshape(UIin_app, [-1, self.edim])  # init: merged embedding
        self.layer_h_apps[0] = layer_h_app  # tf.identity(layer_h_app)
        for h in xrange(1, self.nhop):  # (nhop-1) weights matrix in hidden layers
            layer_h_app = tf.add(tf.matmul(self.layer_h_apps[h-1], self.weights_app[h]), self.biases_app[h])
            if self.activation == 'relu':
                layer_h_app = tf.nn.relu(layer_h_app)
            elif self.activation == 'sigmoid':
                layer_h_app = tf.nn.sigmoid(layer_h_app)
            self.layer_h_apps[h] = layer_h_app  #tf.identity(layer_h_app)
            # layer_h = tf.nn.dropout(layer_h, keep_prob) https://www.tensorflow.org/get_started/mnist/pros
        # 'layer_h' is now the representations of last hidden layer

        # 3. output layer: dense and linear
        self.z_app_only = tf.matmul(layer_h_app, self.h_app) + self.b_app
        self.pred_app_only = tf.nn.softmax(self.z_app_only)

        ## ------- loss and optimization ------- ##
        if self.objective == 'cross':
            self.loss_app_only = tf.nn.softmax_cross_entropy_with_logits(logits=self.z_app_only, labels=self.target_app)
            #self.loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.z, labels=self.target)
        elif self.objective == 'log':
            self.loss_app_only = tf.losses.log_loss(predictions=self.pred_app_only, labels=self.target_app)
        else:
            self.loss_app_only = tf.losses.hinge_loss(logits=self.z_app_only, labels=self.target_app)

        self.lr = tf.Variable(self.current_lr)
        if self.learner == 'adam':
            self.opt_app = tf.train.AdamOptimizer(self.lr)
        elif self.learner == 'rmsprop':
            self.opt_app = tf.train.RMSPropOptimizer(learning_rate=self.lr,decay=0.9)
        elif self.learner == 'adagrad':
            self.opt_app = tf.train.AdagradOptimizer(learning_rate=self.lr)
        else:
            self.opt_app = tf.train.GradientDescentOptimizer(learning_rate=self.lr)

        params = [self.U, self.V_app, self.h_app, self.b_app]
        for h in range(1, self.nhop):  # weighs/biases in hidden layers
            params.append(self.weights_app[h])
            params.append(self.biases_app[h])
        grads_and_vars = self.opt_app.compute_gradients(self.loss_app_only, params)
        clipped_grads_and_vars = [(tf.clip_by_norm(gv[0], self.max_grad_norm), gv[1]) for gv in grads_and_vars]

        inc = self.global_step.assign_add(1)
        with tf.control_dependencies([inc]):
            #self.optim = self.opt.apply_gradients(grads_and_vars)
            self.optim_app = self.opt_app.apply_gradients(clipped_grads_and_vars)

    def build_memory_news_specific(self):
        ## ------- parameters: news specific -------  ##
        # 1. embedding matrices for input <user, app>, <user, news>: shared user embedding matrix
        self.V_news = tf.Variable(tf.random_normal([self.nItems_news, self.edim_v], stddev=self.init_std))

        # 2. weights & biases for hidden layers: the input to hidden layers are the merged embedding
        self.weights_news = defaultdict(object)
        self.biases_news = defaultdict(object)
        for h in xrange(1, self.nhop):  # (nhop-1) weights matrix in hidden layers
            self.weights_news[h] = tf.Variable(tf.random_normal([self.layers[h-1], self.layers[h]], stddev=self.init_std))
            self.biases_news[h] = tf.Variable(tf.random_normal([self.layers[h]], stddev=self.init_std))
        # 3. output layer: weight and bias
        self.h_news = tf.Variable(tf.random_normal([self.layers[-1], self.class_size], stddev=self.init_std))
        self.b_news = tf.Variable(tf.random_normal([self.class_size], stddev=self.init_std))

    def build_model_news_training(self):
        ## ------- computational graph: news training only ------- ##
        # 1. input & embedding layer
        USERin_news = tf.nn.embedding_lookup(self.U, self.input_news[:,0])
        ITEMin_news = tf.nn.embedding_lookup(self.V_news, self.input_news[:,1])
        UIin_news = tf.concat(values=[USERin_news, ITEMin_news], axis=1)  # no info loss, and edim = edim_u + edim_v

        # 2. MLP: hidden layers, http://www.jessicayung.com/explaining-tensorflow-code-for-a-multilayer-perceptron/
        self.layer_h_newss = defaultdict(object)
        layer_h_news = tf.reshape(UIin_news, [-1, self.edim])  # init: cmerged embedding
        self.layer_h_newss[0] = layer_h_news  # tf.identity(layer_h_news)
        for h in xrange(1, self.nhop):  # (nhop-1) weights matrix in hidden layers
            layer_h_news = tf.add(tf.matmul(self.layer_h_newss[h-1], self.weights_news[h]), self.biases_news[h])
            if self.activation == 'relu':
                layer_h_news = tf.nn.relu(layer_h_news)
            elif self.activation == 'sigmoid':
                layer_h_news = tf.nn.sigmoid(layer_h_news)
            self.layer_h_newss[h] = layer_h_news
            # layer_h = tf.nn.dropout(layer_h, keep_prob) https://www.tensorflow.org/get_started/mnist/pros
        # 'layer_h' is now the representations of last hidden layer

        # 3. output layer: dense and linear
        self.z_news_only = tf.matmul(layer_h_news, self.h_news) + self.b_news
        self.pred_news_only = tf.nn.softmax(self.z_news_only)

        ## ------- loss and optimization ------- ##
        if self.objective == 'cross':
            self.loss_news_only = tf.nn.softmax_cross_entropy_with_logits(logits=self.z_news_only, labels=self.target_news)
            #self.loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.z, labels=self.target)
        elif self.objective == 'log':
            self.loss_news_only = tf.losses.log_loss(predictions=self.pred_news_only, labels=self.target_news)
        else:
            self.loss_news_only = tf.losses.hinge_loss(logits=self.z_news_only, labels=self.target_news)

        self.lr = tf.Variable(self.current_lr)
        if self.learner == 'adam':
            self.opt_news = tf.train.AdamOptimizer(self.lr)
        elif self.learner == 'rmsprop':
            self.opt_news = tf.train.RMSPropOptimizer(learning_rate=self.lr,decay=0.9)
        elif self.learner == 'adagrad':
            self.opt_news = tf.train.AdagradOptimizer(learning_rate=self.lr)
        else:
            self.opt_news = tf.train.GradientDescentOptimizer(learning_rate=self.lr)

        params = [self.U, self.V_news, self.h_news, self.b_news]
        for h in range(1, self.nhop):  # weighs/biases in hidden layers
            params.append(self.weights_news[h])
            params.append(self.biases_news[h])
        grads_and_vars = self.opt_news.compute_gradients(self.loss_news_only, params)
        clipped_grads_and_vars = [(tf.clip_by_norm(gv[0], self.max_grad_norm), gv[1]) for gv in grads_and_vars]

        inc = self.global_step.assign_add(1)
        with tf.control_dependencies([inc]):
            #self.optim = self.opt.apply_gradients(grads_and_vars)
            self.optim_news = self.opt_news.apply_gradients(clipped_grads_and_vars)

    def build_model_joint_training(self):
        ## ------- computational graph: for joint training ------- ##
        # 1. input & embedding layer
        USERin_app = tf.nn.embedding_lookup(self.U, self.input_app[:,0])  # 3D due to batch
        USERin_news = tf.nn.embedding_lookup(self.U, self.input_news[:,0])
        ITEMin_app = tf.nn.embedding_lookup(self.V_app, self.input_app[:,1])
        ITEMin_news = tf.nn.embedding_lookup(self.V_news, self.input_news[:,1])
        UIin_app = tf.concat(values=[USERin_app, ITEMin_app], axis=1)  # no info loss, and edim = edim_u + edim_v
        UIin_news = tf.concat(values=[USERin_news, ITEMin_news], axis=1)  # no info loss, and edim = edim_u + edim_v

        # 2. MLP: hidden layers, http://www.jessicayung.com/explaining-tensorflow-code-for-a-multilayer-perceptron/
        # cross computation between app network and news network
        self.layer_h_apps = defaultdict(object)
        layer_h_app = tf.reshape(UIin_app, [-1, self.edim])  # init: merged embedding
        self.layer_h_apps[0] = layer_h_app
        self.layer_h_newss = defaultdict(object)
        layer_h_news = tf.reshape(UIin_news, [-1, self.edim])  # init: merged embedding
        self.layer_h_newss[0] = layer_h_news
        for h in xrange(1, self.nhop):  # (nhop-1) weights matrix in hidden layers
            # 1) app-specific: o_app^t+1 = (W_app^t a_app^t + b_app^t) + a_news^t
            layer_h_app = tf.add(tf.matmul(self.layer_h_apps[h-1], self.weights_app[h]), self.biases_app[h])
            if h <= self.cross_layers:
                layer_h_app = tf.add(layer_h_app, tf.matmul(self.layer_h_newss[h-1], self.shared_Hs[h]))
            if self.activation == 'relu':
                layer_h_app = tf.nn.relu(layer_h_app)
            elif self.activation == 'sigmoid':
                layer_h_app = tf.nn.sigmoid(layer_h_app)
            self.layer_h_apps[h] = layer_h_app
            # 2) news-specific:  o_news^t+1 = (W_news^t a_news^t + b_news^t) + a_app^t
            layer_h_news = tf.add(tf.matmul(self.layer_h_newss[h-1], self.weights_news[h]), self.biases_news[h])
            if h <= self.cross_layers:
                layer_h_news = tf.add(layer_h_news, tf.matmul(self.layer_h_apps[h-1], self.shared_Hs[h]))
            if self.activation == 'relu':
                layer_h_news = tf.nn.relu(layer_h_news)
            elif self.activation == 'sigmoid':
                layer_h_news = tf.nn.sigmoid(layer_h_news)
            self.layer_h_newss[h] = layer_h_news

        # 3. output layer: dense and linear
        self.z_app_joint = tf.matmul(layer_h_app, self.h_app) + self.b_app
        self.pred_app_joint = tf.nn.softmax(self.z_app_joint)
        self.z_news_joint = tf.matmul(layer_h_news, self.h_news) + self.b_news
        self.pred_news_joint = tf.nn.softmax(self.z_news_joint)

        ## ------- loss and optimization ------- ##
        if self.objective == 'cross':
            self.loss_app_joint = tf.nn.softmax_cross_entropy_with_logits(logits=self.z_app_joint, labels=self.target_app)
            #self.loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.z, labels=self.target)
        elif self.objective == 'log':
            self.loss_app_joint = tf.losses.log_loss(predictions=self.pred_app_joint, labels=self.target_app)
        else:
            self.loss_app_joint = tf.losses.hinge_loss(logits=self.z_app_joint, labels=self.target_app)
        if self.objective == 'cross':
            self.loss_news_joint = tf.nn.softmax_cross_entropy_with_logits(logits=self.z_news_joint, labels=self.target_news)
            #self.loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.z, labels=self.target)
        elif self.objective == 'log':
            self.loss_news_joint = tf.losses.log_loss(predictions=self.pred_news_joint, labels=self.target_news)
        else:
            self.loss_news_joint = tf.losses.hinge_loss(logits=self.z_news_joint, labels=self.target_news)

        self.lr = tf.Variable(self.current_lr)
        if self.learner == 'adam':
            self.opt_joint = tf.train.AdamOptimizer(self.lr)
        elif self.learner == 'rmsprop':
            self.opt_joint = tf.train.RMSPropOptimizer(learning_rate=self.lr,decay=0.9)
        elif self.learner == 'adagrad':
            self.opt_joint = tf.train.AdagradOptimizer(learning_rate=self.lr)
        else:
            self.opt_joint = tf.train.GradientDescentOptimizer(learning_rate=self.lr)

        params = [self.U, self.V_app, self.h_app, self.b_app, self.V_news, self.h_news, self.b_news]
        for h in range(1, self.nhop):  # weighs/biases in hidden layers
            params.append(self.weights_app[h])
            params.append(self.biases_app[h])
            params.append(self.weights_news[h])
            params.append(self.biases_news[h])
        for h in xrange(1, self.cross_layers+1):
            params.append(self.shared_Hs[h])  # only cross these layers
        self.loss_joint = self.weights_app_news[0] * self.loss_app_joint + self.weights_app_news[1] * self.loss_news_joint
        grads_and_vars = self.opt_joint.compute_gradients(self.loss_joint, params)
        clipped_grads_and_vars = [(tf.clip_by_norm(gv[0], self.max_grad_norm), gv[1]) for gv in grads_and_vars]

        inc = self.global_step.assign_add(1)
        with tf.control_dependencies([inc]):
            #self.optim = self.opt.apply_gradients(grads_and_vars)
            self.optim_joint = self.opt_joint.apply_gradients(clipped_grads_and_vars)

    def build_model(self):
        self.global_step = tf.Variable(0, name="global_step")

        self.build_memory_shared()

        self.build_memory_app_specific()
        self.build_model_app_training()

        self.build_memory_news_specific()
        self.build_model_news_training()

        self.build_model_joint_training()

        tf.global_variables_initializer().run()

    def get_train_instances_app(self):
        self.user_input_app, self.item_input_app, self.labels_app = [],[],[]
        for (u, i) in self.train_app.keys():
            # positive instance
            self.user_input_app.append(u)
            self.item_input_app.append(i)
            self.labels_app.append(1)
            # negative negRatio instances
            for _ in xrange(self.negRatio):
                j = np.random.randint(self.nItems_app)
                while (u, j) in self.train_app:
                    j = np.random.randint(self.nItems_app)
                self.user_input_app.append(u)
                self.item_input_app.append(j)
                self.labels_app.append(0)

    def get_train_instances_news(self):
        self.user_input_news, self.item_input_news, self.labels_news = [],[],[]
        for (u, i) in self.train_news.keys():
            # positive instance
            self.user_input_news.append(u)
            self.item_input_news.append(i)
            self.labels_news.append(1)
            # negative negRatio instances
            for _ in xrange(self.negRatio):
                j = np.random.randint(self.nItems_news)
                while (u, j) in self.train_news:
                    j = np.random.randint(self.nItems_news)
                self.user_input_news.append(u)
                self.item_input_news.append(j)
                self.labels_news.append(0)

    def get_test_instances_app(self):
        self.test_user_input_app, self.test_item_input_app, self.test_labels_app = [],[],[]
        for idx in range(len(self.testRatings_app)):
            # leave-one-out test_item
            rating = self.testRatings_app[idx]
            u = rating[0]
            gtItem = rating[1]
            self.test_user_input_app.append(u)
            self.test_item_input_app.append(gtItem)
            self.test_labels_app.append(1)
            # random 99 neg_items
            items = self.testNegatives_app[idx]
            for i in items:
                self.test_user_input_app.append(u)
                self.test_item_input_app.append(i)
                self.test_labels_app.append(0)

    def get_test_instances_news(self):
        self.test_user_input_news, self.test_item_input_news, self.test_labels_news = [],[],[]
        for idx in range(len(self.testRatings_news)):
            # leave-one-out test_item
            rating = self.testRatings_news[idx]
            u = rating[0]
            gtItem = rating[1]
            self.test_user_input_news.append(u)
            self.test_item_input_news.append(gtItem)
            self.test_labels_news.append(1)
            # random 99 neg_items
            items = self.testNegatives_news[idx]
            for i in items:
                self.test_user_input_news.append(u)
                self.test_item_input_news.append(i)
                self.test_labels_news.append(0)

    def train_model(self):
        self.get_train_instances_app()  # randomly sample negatives each time / per epoch
        self.get_train_instances_news()  # randomly sample negatives each time / per epoch

        num_examples_app = len(self.labels_app)
        num_batches_app = int(math.ceil(num_examples_app / self.batch_size))
        num_examples_news = len(self.labels_news)
        num_batches_news = int(math.ceil(num_examples_news / self.batch_size))
        num_batches_max = max(num_batches_app,num_batches_news)
        num_batches_min = min(num_batches_app,num_batches_news)
        print('#batch: app={}, news={}, max={}, min={}'.format(num_batches_app, num_batches_news, num_batches_max, num_batches_min))
        if self.show:
            from utils import ProgressBar
        x_app = np.ndarray([self.batch_size, self.input_size], dtype=np.int32)  # (user,item)
        target_app = np.zeros([self.batch_size, self.class_size])  # (pos, neg)
        x_news = np.ndarray([self.batch_size, self.input_size], dtype=np.int32)  # (user,item)
        target_news = np.zeros([self.batch_size, self.class_size])  # (pos, neg)
        sample_ids_app = [sid for sid in range(num_examples_app)]
        random.shuffle(sample_ids_app)
        sample_ids_news = [sid for sid in range(num_examples_news)]
        random.shuffle(sample_ids_news)
        cost_total = 0.0
        cost_joint = 0.0
        cost_app = 0.0
        cost_news = 0.0
        batches_app = [b for b in range(num_batches_app)]
        batches_news = [b for b in range(num_batches_news)]
        ## should first single training and then joint train (i.e. first pre-train and then fine-tune)
        if num_batches_app < num_batches_news:  # app smaller than news, and hence single training on news first
            batches_join = batches_app
            batches_single = batches_news[num_batches_app:]
            bar_news = ProgressBar('train_news', max=len(batches_single))
            for _ in batches_single:
                if self.show: bar_news.next()
                target_news.fill(0)
                for b in xrange(self.batch_size):
                    if not sample_ids_news:  # if news used up
                        sample_id_news = random.randrange(0, num_examples_news)
                        x_news[b][0] = self.user_input_news[sample_id_news]
                        x_news[b][1] = self.item_input_news[sample_id_news]
                        if self.labels_news[sample_id_news] == 1:
                            target_news[b][0] = 1  # one-hot encoding for two classes of positive & negative
                        else:
                            target_news[b][1] = 1  # negative
                    else:
                        sample_id_news = sample_ids_news.pop()
                        x_news[b][0] = self.user_input_news[sample_id_news]
                        x_news[b][1] = self.item_input_news[sample_id_news]
                        if self.labels_news[sample_id_news] == 1:
                            target_news[b][0] = 1  # one-hot encoding for two classes of positive & negative
                        else:
                            target_news[b][1] = 1  # negative
                keys = [self.input_news, self.target_news]
                values = [x_news, target_news]
                _, loss_news, pred_news, self.step = self.sess.run([self.optim_news,
                                                    self.loss_news_only,
                                                    self.pred_news_only,
                                                    self.global_step],
                                                    feed_dict={
                                                        k:v for k,v in zip(keys, values)
                                                    })
                cost_news += np.sum(loss_news)
                cost_total += np.sum(loss_news)
            if self.show: bar_news.finish()
        else:  # app is not smaller than news, and hence single training on app first
            batches_join = batches_news
            batches_single = batches_app[num_batches_news:]
            bar_app = ProgressBar('train_app', max=len(batches_single))
            for _ in batches_single:
                if self.show: bar_app.next()
                target_app.fill(0)
                for b in xrange(self.batch_size):
                    if not sample_ids_app:  # if app used up
                        sample_id_app = random.randrange(0, num_examples_app)
                        x_app[b][0] = self.user_input_app[sample_id_app]
                        x_app[b][1] = self.item_input_app[sample_id_app]
                        if self.labels_app[sample_id_app] == 1:
                            target_app[b][0] = 1  # one-hot encoding for two classes of positive & negative
                        else:
                            target_app[b][1] = 1  # negative
                    else:
                        sample_id_app = sample_ids_app.pop()
                        x_app[b][0] = self.user_input_app[sample_id_app]
                        x_app[b][1] = self.item_input_app[sample_id_app]
                        if self.labels_app[sample_id_app] == 1:
                            target_app[b][0] = 1  # one-hot encoding for two classes of positive & negative
                        else:
                            target_app[b][1] = 1  # negative
                keys = [self.input_app, self.target_app]
                values = [x_app, target_app]
                _, loss_app, pred_app, self.step = self.sess.run([self.optim_app,
                                                    self.loss_app_only,
                                                    self.pred_app_only,
                                                    self.global_step],
                                                    feed_dict={
                                                        k:v for k,v in zip(keys, values)
                                                    })
                cost_app += np.sum(loss_app)
                cost_total += np.sum(loss_app)
            if self.show: bar_app.finish()
        # joint training on both datasets after single training on the bigger one
        bar_join = ProgressBar('train_join', max=len(batches_join))
        for _ in batches_join:
            if self.show: bar_join.next()
            target_app.fill(0)
            target_news.fill(0)
            for b in xrange(self.batch_size):
                if not sample_ids_app:  # if app used up
                    sample_id_app = random.randrange(0, num_examples_app)
                    x_app[b][0] = self.user_input_app[sample_id_app]
                    x_app[b][1] = self.item_input_app[sample_id_app]
                    if self.labels_app[sample_id_app] == 1:
                        target_app[b][0] = 1  # one-hot encoding for two classes of positive & negative
                    else:
                        target_app[b][1] = 1  # negative
                else:
                    sample_id_app = sample_ids_app.pop()
                    x_app[b][0] = self.user_input_app[sample_id_app]
                    x_app[b][1] = self.item_input_app[sample_id_app]
                    if self.labels_app[sample_id_app] == 1:
                        target_app[b][0] = 1  # one-hot encoding for two classes of positive & negative
                    else:
                        target_app[b][1] = 1  # negative
                if not sample_ids_news:  # if news used up
                    sample_id_news = random.randrange(0, num_examples_news)
                    x_news[b][0] = self.user_input_news[sample_id_news]
                    x_news[b][1] = self.item_input_news[sample_id_news]
                    if self.labels_news[sample_id_news] == 1:
                        target_news[b][0] = 1  # one-hot encoding for two classes of positive & negative
                    else:
                        target_news[b][1] = 1  # negative
                else:
                    sample_id_news = sample_ids_news.pop()
                    x_news[b][0] = self.user_input_news[sample_id_news]
                    x_news[b][1] = self.item_input_news[sample_id_news]
                    if self.labels_news[sample_id_news] == 1:
                        target_news[b][0] = 1  # one-hot encoding for two classes of positive & negative
                    else:
                        target_news[b][1] = 1  # negative
            keys = [self.input_app, self.input_news, self.target_app, self.target_news]
            values = [x_app, x_news, target_app, target_news]
            _, loss, loss_app, loss_news, pred_app, pred_news, self.step = self.sess.run([self.optim_joint,
                                                self.loss_joint,
                                                self.loss_app_joint,
                                                self.loss_news_joint,
                                                self.pred_app_joint,
                                                self.pred_news_joint,
                                                self.global_step],
                                                feed_dict={
                                                    k:v for k,v in zip(keys, values)
                                                })
            cost_joint += np.sum(loss)
            cost_total += np.sum(loss)
            cost_app += np.sum(loss_app)
            cost_news += np.sum(loss_news)
        if self.show: bar_join.finish()
        return [cost_total/num_batches_max/self.batch_size,cost_joint/num_batches_min/self.batch_size,
                cost_app/num_batches_app/self.batch_size,cost_news/num_batches_news/self.batch_size]

    def run(self):
        self.get_test_instances_app()  # only need to get once
        self.get_test_instances_news()  # only need to get once

        self.para_str = 'time'+str(int(time.time())) + '_' + 'eu'+str(self.edim_u)+'ev'+str(self.edim_v) \
                        + str(self.layers) + 'batch'+str(self.batch_size) + 'wtask'+str(self.weights_app_news) \
                        + 'cross_layers'+str(self.cross_layers) + 'lr'+str(self.init_lr) + 'std'+str(self.init_std) + 'nr'+str(self.negRatio) \
                        + str(self.activation) + str(self.learner) + 'loss'+str(self.objective) 
        print(self.para_str)
        with open('results_' + self.para_str + '.log', 'w') as ofile:
            ofile.write(self.para_str + '\n')
            start_time = time.time()
            for idx in xrange(self.nepoch):
                start = time.time()
                #train_loss, train_loss_app, train_loss_news = np.sum(self.train_model())
                train_loss_total, train_loss_joint, train_loss_app, train_loss_news = self.train_model()
                train_time = time.time() - start

                start = time.time()
                valid_loss_app = np.sum(self.valid_model_app())
                valid_loss_news = np.sum(self.valid_model_news())
                valid_loss = (valid_loss_app + valid_loss_news) / 2
                valid_time = time.time() - start

                if self.HR_app > self.bestHR_app and self.HR_app < 0.99 and idx > 3:
                    self.bestHR_app = self.HR_app
                    self.bestHR_epoch_app = idx
                if self.NDCG_app > self.bestNDCG_app and self.NDCG_app < 0.99 and idx > 3:
                    self.bestNDCG_app = self.NDCG_app
                    self.bestNDCG_epoch_app = idx
                if self.MRR_app > self.bestMRR_app and self.MRR_app < 0.99 and idx > 3:
                    self.bestMRR_app = self.MRR_app
                    self.bestMRR_epoch_app = idx
                if self.AUC_app > self.bestAUC_app and self.AUC_app < 0.99 and idx > 3:
                    self.bestAUC_app = self.AUC_app
                    self.bestAUC_epoch_app = idx

                if self.HR_news > self.bestHR_news and self.HR_news < 0.99 and idx > 3:
                    self.bestHR_news = self.HR_news
                    self.bestHR_epoch_news = idx
                if self.NDCG_news > self.bestNDCG_news and self.NDCG_news < 0.99 and idx > 3:
                    self.bestNDCG_news = self.NDCG_news
                    self.bestNDCG_epoch_news = idx
                if self.MRR_news > self.bestMRR_news and self.MRR_news < 0.99 and idx > 3:
                    self.bestMRR_news = self.MRR_news
                    self.bestMRR_epoch_news = idx
                if self.AUC_news > self.bestAUC_news and self.AUC_news < 0.99 and idx > 3:
                    self.bestAUC_news = self.AUC_news
                    self.bestAUC_epoch_news = idx

                print('{:.1f}s. epoch={}, loss_total={:.6f}, loss_joint={:.6f}, val_l={:.6f}. {:.1f}s'.format(
                        train_time, idx, train_loss_total, train_loss_joint, valid_loss, valid_time))
                print('App: loss={:.6f}, val_l={:.6f}, HR={:.6f}, NDCG={:.6f}, MRR={:.6f}, AUC={:.6f}.'.format(
                        train_loss_app, valid_loss_app, self.HR_app, self.NDCG_app, self.MRR_app, self.AUC_app))
                print('News: loss={:.6f}, val_l={:.6f}, HR={:.6f}, NDCG={:.6f}, MRR={:.6f}, AUC={:.6f}.'.format(
                        train_loss_news, valid_loss_news, self.HR_news, self.NDCG_news, self.MRR_news, self.AUC_news))
                ofile.write('{:.1f}s. epoch={}, loss_total={:.6f}, loss_joint={:.6f}, val_l={:.6f}. {:.1f}s\n'.format(
                        train_time, idx, train_loss_total, train_loss_joint, valid_loss, valid_time))
                ofile.write('App: loss={:.6f}, val_l={:.6f}, HR={:.6f}, NDCG={:.6f}, MRR={:.6f}, AUC={:.6f}\n'.format(
                        train_loss_app, valid_loss_app, self.HR_app, self.NDCG_app, self.MRR_app, self.AUC_app))
                ofile.write('News: loss={:.6f}, val_l={:.6f}, HR={:.6f}, NDCG={:.6f}, MRR={:.6f}, AUC={:.6f}\n'.format(
                        train_loss_news, valid_loss_news, self.HR_news, self.NDCG_news, self.MRR_news, self.AUC_news))
                ofile.flush()
            ofile.write('bestHR_app = {:.6f} at epoch {}\n'.format(self.bestHR_app, self.bestHR_epoch_app))
            ofile.write('bestNDCG_app = {:.6f} at epoch {}\n'.format(self.bestNDCG_app, self.bestNDCG_epoch_app))
            ofile.write('bestMRR_app = {:.6f} at epoch {}\n'.format(self.bestMRR_app, self.bestMRR_epoch_app))
            ofile.write('bestAUC_app = {:.6f} at epoch {}\n'.format(self.bestAUC_app, self.bestAUC_epoch_app))
            ofile.write('bestHR_news = {:.6f} at epoch {}\n'.format(self.bestHR_news, self.bestHR_epoch_news))
            ofile.write('bestNDCG_news = {:.6f} at epoch {}\n'.format(self.bestNDCG_news, self.bestNDCG_epoch_news))
            ofile.write('bestMRR_news = {:.6f} at epoch {}\n'.format(self.bestMRR_news, self.bestMRR_epoch_news))
            ofile.write('bestAUC_news = {:.6f} at epoch {}\n'.format(self.bestAUC_news, self.bestAUC_epoch_news))
            print('bestHR_app = {:.6f} at epoch {}'.format(self.bestHR_app, self.bestHR_epoch_app))
            print('bestNDCG_app = {:.6f} at epoch {}'.format(self.bestNDCG_app, self.bestNDCG_epoch_app))
            print('bestMRR_app = {:.6f} at epoch {}'.format(self.bestMRR_app, self.bestMRR_epoch_app))
            print('bestAUC_app = {:.6f} at epoch {}'.format(self.bestAUC_app, self.bestAUC_epoch_app))
            print('bestHR_news = {:.6f} at epoch {}'.format(self.bestHR_news, self.bestHR_epoch_news))
            print('bestNDCG_news = {:.6f} at epoch {}'.format(self.bestNDCG_news, self.bestNDCG_epoch_news))
            print('bestMRR_news = {:.6f} at epoch {}'.format(self.bestMRR_news, self.bestMRR_epoch_news))
            print('bestAUC_news = {:.6f} at epoch {}'.format(self.bestAUC_news, self.bestAUC_epoch_news))
            print('total time = {:.1f}m'.format((time.time() - start_time)/60))
            ofile.write('total time = {:.1f}\n'.format((time.time() - start_time)/60))
        print(self.para_str)

    def valid_model_app(self):
        num_test_examples = len(self.test_labels_app)
        num_test_batches = math.ceil(num_test_examples / self.batch_size)
        if self.show:
            from utils import ProgressBar
            bar = ProgressBar('val_app', max=num_test_batches)
        cost = 0
        x = np.ndarray([self.batch_size, self.input_size], dtype=np.int32)  # user,item
        target = np.zeros([self.batch_size, self.class_size])  # one-hot encoding: (pos, neg)
        sample_id = 0
        test_preds = []
        for current_batch in xrange(num_test_batches):
            if self.show: bar.next()
            target.fill(0)
            for b in xrange(self.batch_size):
                if sample_id >= len(self.test_labels_app):  # fill this batch; not be used when compute test metrics
                    x[b][0] = self.test_user_input_app[0]
                    x[b][1] = self.test_item_input_app[0]
                    if self.test_labels_app[0] == 1:
                        target[b][0] = 1  # one-hot encoding for two classes of positive & negative
                    else:
                        target[b][1] = 1  # negative
                else:
                    x[b][0] = self.test_user_input_app[sample_id]
                    x[b][1] = self.test_item_input_app[sample_id]
                    if self.test_labels_app[sample_id] == 1:
                        target[b][0] = 1  # one-hot encoding for two classes of positive & negative
                    else:
                        target[b][1] = 1  # negative
                sample_id += 1

            keys = [self.input_app, self.target_app]
            values = [x, target]
            loss, pred = self.sess.run([self.loss_app_only, self.pred_app_only],
                                        feed_dict={
                                            k:v for k,v in zip(keys, values)
                                        })
            cost += np.sum(loss)
            test_preds.extend(pred)
            if self.isOneBatch: break
        if self.show: bar.finish()

        # evaluation
        user_item_preds = defaultdict(lambda: defaultdict(float))
        user_pred_gtItem = defaultdict(float)
        for sample_id in range(len(self.test_labels_app)):
            user = self.test_user_input_app[sample_id]
            item = self.test_item_input_app[sample_id]
            label = self.test_labels_app[sample_id]
            pred = test_preds[sample_id]  # [pos_prob, neg_prob]
            user_item_preds[user][item] = pred[0]
            if item == self.user_gt_item_app[user]:
                user_pred_gtItem[user] = pred[0]
        HR, NDCG, MRR, AUC = 0, 0, 0, 0
        for user, item_preds in user_item_preds.items():
            # compute AUC
            gt_pred = user_pred_gtItem[user]
            hit = 0
            for i, p in item_preds.items():
                if i != self.user_gt_item_app[user] and p < gt_pred:
                    hit += 1
            AUC += hit / 99.0
            # compute HR, NDCG, MRR
            item_preds = sorted(item_preds.items(), key=lambda x:-x[1])
            item_preds_topK = item_preds[:self.topK]
            for item, pred in item_preds_topK:
                if item == self.user_gt_item_app[user]:
                    HR += 1
                    break
            for position in range(len(item_preds_topK)):
                item, pred = item_preds_topK[position]
                if item == self.user_gt_item_app[user]:
                    NDCG += math.log(2) / math.log(position + 2)
                    break
            rank = 1
            for item, pred in item_preds_topK:
                if item == self.user_gt_item_app[user]:
                    break
                rank += 1
            MRR += 1 / rank
            if self.isDebug and user == 1:
                print('gt_pred = {:.6f}, topK_pred={:.6f}'.format(gt_pred, item_preds[self.topK][1]))
        self.HR_app = HR / len(user_item_preds)
        self.NDCG_app = NDCG / len(user_item_preds)
        self.MRR_app = MRR / len(user_item_preds)
        self.AUC_app = AUC / len(user_item_preds)
        return cost/num_test_batches/self.batch_size

    def valid_model_news(self):
        num_test_examples = len(self.test_labels_news)
        num_test_batches = math.ceil(num_test_examples / self.batch_size)
        if self.show:
            from utils import ProgressBar
            bar = ProgressBar('val_news', max=num_test_batches)
        cost = 0
        x = np.ndarray([self.batch_size, self.input_size], dtype=np.int32)  # user,item
        target = np.zeros([self.batch_size, self.class_size])  # one-hot encoding: (pos, neg)
        sample_id = 0
        test_preds = []
        for current_batch in xrange(num_test_batches):
            if self.show: bar.next()
            target.fill(0)
            for b in xrange(self.batch_size):
                if sample_id >= len(self.test_labels_news):  # fill this batch; not be used when compute test metrics
                    x[b][0] = self.test_user_input_news[0]
                    x[b][1] = self.test_item_input_news[0]
                    if self.test_labels_news[0] == 1:
                        target[b][0] = 1  # one-hot encoding for two classes of positive & negative
                    else:
                        target[b][1] = 1  # negative
                else:
                    x[b][0] = self.test_user_input_news[sample_id]
                    x[b][1] = self.test_item_input_news[sample_id]
                    if self.test_labels_news[sample_id] == 1:
                        target[b][0] = 1  # one-hot encoding for two classes of positive & negative
                    else:
                        target[b][1] = 1  # negative
                sample_id += 1

            keys = [self.input_news, self.target_news]
            values = [x, target]
            loss, pred = self.sess.run([self.loss_news_only, self.pred_news_only],
                                        feed_dict={
                                            k:v for k,v in zip(keys, values)
                                        })
            cost += np.sum(loss)
            test_preds.extend(pred)
            if current_batch == 0 and self.isDebug:
                print()
                print(target[0:3, :])
                print(pred[0:3, :])
            if self.isOneBatch: break
        if self.show: bar.finish()

        # evaluation
        user_item_preds = defaultdict(lambda: defaultdict(float))
        user_pred_gtItem = defaultdict(float)
        for sample_id in range(len(self.test_labels_news)):
            user = self.test_user_input_news[sample_id]
            item = self.test_item_input_news[sample_id]
            label = self.test_labels_news[sample_id]
            pred = test_preds[sample_id]  # [pos_prob, neg_prob]
            user_item_preds[user][item] = pred[0]
            if item == self.user_gt_item_news[user]:
                user_pred_gtItem[user] = pred[0]
        HR, NDCG, MRR, AUC = 0, 0, 0, 0
        for user, item_preds in user_item_preds.items():
            # compute AUC
            gt_pred = user_pred_gtItem[user]
            hit = 0
            for i, p in item_preds.items():
                if i != self.user_gt_item_news[user] and p < gt_pred:
                    hit += 1
            AUC += hit / 99.0
            # compute HR, NDCG, MRR
            item_preds = sorted(item_preds.items(), key=lambda x:-x[1])
            item_preds_topK = item_preds[:self.topK]
            for item, pred in item_preds_topK:
                if item == self.user_gt_item_news[user]:
                    HR += 1
                    break
            for position in range(len(item_preds_topK)):
                item, pred = item_preds_topK[position]
                if item == self.user_gt_item_news[user]:
                    NDCG += math.log(2) / math.log(position + 2)
                    break
            rank = 1
            for item, pred in item_preds_topK:
                if item == self.user_gt_item_news[user]:
                    break
                rank += 1
            MRR += 1 / rank
            if self.isDebug and user == 1:
                print('gt_pred = {:.6f}, topK_pred={:.6f}'.format(gt_pred, item_preds[self.topK][1]))
        self.HR_news = HR / len(user_item_preds)
        self.NDCG_news = NDCG / len(user_item_preds)
        self.MRR_news = MRR / len(user_item_preds)
        self.AUC_news = AUC / len(user_item_preds)
        return cost/num_test_batches/self.batch_size
