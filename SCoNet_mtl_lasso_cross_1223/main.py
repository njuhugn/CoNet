import os
import pprint
import tensorflow as tf
import time
from collections import defaultdict
from model import MTL
import json
import random
import math
import numpy as np

pp = pprint.PrettyPrinter()


def main(_):

    paras_setting = {
        'edim_u': 32,
        'edim_v': 32,
        'layers': [64,32,16,8],  # layers[0] must equal to edim_u + edim_v
        'batch_size': 128,  # "batch size to use during training [128,256,512,]"
        'nepoch': 50,  # "number of epoch to use during training [80]"
        'init_lr': 0.001,  # "initial learning rate [0.01]"
        'init_std': 0.01,  # "weight initialization std [0.05]"
        'max_grad_norm': 10,  # "clip gradients to this norm [50]"
        'negRatio': 1,  # "negative sampling ratio [5]"
        'cross_layers': 2,  # cross between 1st & 2nd, and 2nd & 3rd layers
        #'merge_ui': 0,  # "merge embeddings of user and item: 0-add, 1-mult [1], 2-concat"
        'activation': 'relu',  # "0:relu, 1:tanh, 2:softmax"
        'learner': 'adam',  # {adam, rmsprop, adagrad, sgd}
        'objective': 'cross',  # 0:cross, 1: hinge, 2:log
        #'carry_trans_alpha': [0.5, 0.5],  # weight of carry/copy gate
        'topK': 10,
        'data_dir': '../data/',  # "data directory [../data]"
        'data_name_app': 'user_apps_NY',  # "user-info", "data state [user-info]"
        'data_name_news': 'user_articles_NY',  # "user-info", "data state [user-info]"
        'weights_app_news': [1,1],  # weights of each task [0.8,0.2], [0.5,0.5], [1,1]
        'lasso_weight': 0.01,  # weights of lasso regu for linear combination matrices
        'checkpoint_dir': 'checkpoints',  # "checkpoints", "checkpoint directory [checkpoints]"
        'show': True,  # "print progress [True]"
        #'isDebug': True,  # "isDebug mode [True]"
        'isDebug': False,  # "isDebug mode [True]"
        'isOneBatch': False,  # "isOneBatch mode for quickly run through [True]"
    }
    # setenv CUDA_VISIBLE_DEVICES 1
    isRandomSearch = False

    if not isRandomSearch:
        start_time = time.time()
        pp.pprint(paras_setting)

        #train_file = paras_setting['data_dir'] + paras_setting['data_name'] + '.train'
        #valid_file = paras_setting['data_dir'] + paras_setting['data_name'] + '.valid'
        #valid_neg_file = paras_setting['data_dir'] + paras_setting['data_name'] + '.neg.valid'
        with tf.Session() as sess:
            model = MTL(paras_setting, sess)
            model.build_model()
            model.run()
            metrics = {
                'bestHR_app': model.bestHR_app,
                'bestNDCG_app': model.bestNDCG_app,
                'bestMRR_app': model.bestMRR_app,
                'bestAUC_app': model.bestAUC_app,
                'bestHR_epoch_app': model.bestHR_epoch_app,
                'bestNDCG_epoch_app': model.bestNDCG_epoch_app,
                'bestMRR_epoch_app': model.bestMRR_epoch_app,
                'bestAUC_epoch_app': model.bestAUC_epoch_app,
                'bestHR_news': model.bestHR_news,
                'bestNDCG_news': model.bestNDCG_news,
                'bestMRR_news': model.bestMRR_news,
                'bestAUC_news': model.bestAUC_news,
                'bestHR_epoch_news': model.bestHR_epoch_news,
                'bestNDCG_epoch_news': model.bestNDCG_epoch_news,
                'bestMRR_epoch_news': model.bestMRR_epoch_news,
                'bestAUC_epoch_news': model.bestAUC_epoch_news,
            }
            pp.pprint(metrics)
            print(model.para_str)
            pp.pprint(paras_setting)
        print('total time {:.2f}m'.format((time.time() - start_time)/60))
    else:
        para_ranges_map = {
            'edim_u': [5, 10, 20, 32, 50, 64],
            'edim_v': [5, 10, 20, 32, 50, 64],
            'mem_size': [5,10,20,32,50,64,100],
            'nhop': [1,2,3,4,5,6],
            'init_lr': [0.001, 0.005, 0.01, 0.05],
            'negRatio': [1,2,3,4,5],
            'batch_size': [128,256,512],
            'activation': [0,1,2],
            'learner': [0,1,2],
            'init_std': [0.01,0.05,0.1]
        }
        total_random_searches = 100
        g_idx_paras = defaultdict(lambda: defaultdict(object))
        for idx_rand_search in range(total_random_searches):
            for key, value in para_ranges_map.items():
                rint = np.random.randint(len(value))
                paras_setting[key] = value[rint]
                paras_setting['lindim'] = math.floor(0.5 * (paras_setting['edim_u'] + paras_setting['edim_v']))
            g_idx_paras[idx_rand_search] = paras_setting

        start_time = time.time()
        g_bestHR10 = -1
        g_bestHR10_paras = defaultdict(object)
        g_bestNDCG = -1
        g_bestNDCG_paras = defaultdict(object)
        g_MetricsParas = []
        for idx_rand_search, paras in g_idx_paras.items():
            paras_setting = paras
            pp.pprint(paras_setting)

            train_file = paras_setting['data_dir'] + paras_setting['data_name'] + '.train'
            valid_file = paras_setting['data_dir'] + paras_setting['data_name'] + '.valid'
            valid_neg_file = paras_setting['data_dir'] + paras_setting['data_name'] + '.neg.valid'
            with tf.Session() as sess:
                model = MemN2N(paras_setting, sess, train_file, valid_file, valid_neg_file)
                model.build_model()
                model.run()
                metrics = {
                    #'bestMRR': model.bestMRR,
                    'bestHR10': model.bestHR10,
                    'bestNDCG': model.bestNDCG,
                    #'bestMRR_epoch': model.bestMRR_epoch,
                    'bestHR10_epoch': model.bestHR10_epoch,
                    'bestNDCG_epoch': model.bestNDCG_epoch,
                }
                pp.pprint(metrics)
                print(model.para_str)
                pp.pprint(paras_setting)
            print('total time {:.2f}m'.format((time.time() - start_time)/60))

            if model.bestHR10 > g_bestHR10:
                g_bestHR10 = model.bestHR10
                g_bestHR10_paras = paras_setting
                print('current best HR = {}'.format(g_bestHR10))
            if model.bestNDCG > g_bestNDCG:
                g_bestNDCG = model.bestNDCG
                g_bestNDCG_paras = paras_setting
                print('current best NDCG = {}'.format(g_bestNDCG))
            g_MetricsParas.append([metrics, paras_setting])
        print('best metric (HR, NDCG) and corresponding meta-parameters')
        print(g_bestHR10)
        print(g_bestHR10_paras)
        print(g_bestNDCG)
        print(g_bestNDCG_paras)
        print('total time {:.2f}m'.format((time.time() - start_time)/60))

        with open('g_MetricsParas.txt', 'w', encoding='utf-8') as ofile:
            ofile.write('\n'.join([str(e) for e in g_MetricsParas]))

if __name__ == '__main__':
    tf.app.run()

