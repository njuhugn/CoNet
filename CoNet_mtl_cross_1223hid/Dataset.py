import scipy.sparse as sp
import numpy as np


class Dataset(object):

    def __init__(self, path):
        self.trainMatrix = self.load_rating_file_as_matrix(path + ".train.rating")
        self.testRatings = self.load_rating_file_as_list(path + ".valid.rating")
        self.testNegatives = self.load_negative_file(path + ".neg.valid.rating")
        assert len(self.testRatings) == len(self.testNegatives)
        
        self.num_users, self.num_items = self.trainMatrix.shape

    def load_rating_file_as_matrix(self, filename):
        print('read {}...'.format(filename))
        # Get number of users and items
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)

                line = f.readline()
        print('maxUserId = {}, maxItemId = {}'.format(num_users, num_items))

        # Construct matrix
        users_set = set()
        items_set = set()
        mat = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                if (rating > 0):
                    mat[user, item] = 1.0
                users_set.add(user)
                items_set.add(item)
                line = f.readline()
        print('#train(user,item,feed)={},{},{}'.format(len(users_set), len(items_set), mat.nnz))
        return mat

    def load_rating_file_as_list(self, filename):
        print('read {}...'.format(filename))
        users_set = set()
        items_set = set()
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item])
                users_set.add(user)
                items_set.add(item)
                line = f.readline()
        print('#test_pos(user,item,feed)={},{},{}'.format(len(users_set), len(items_set), len(ratingList)))
        return ratingList
    
    def load_negative_file(self, filename):
        print('read {}...'.format(filename))
        negativeList = []
        items_set = set()
        nNegFeed = 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                negatives = []
                for x in arr[1:]:  # arr[0] = (user, pos_item)
                    item = int(x)
                    negatives.append(item)
                    items_set.add(item)
                    nNegFeed += 1
                negativeList.append(negatives)
                line = f.readline()
        print('#test_neg(item,feed)={},{}'.format(len(items_set), nNegFeed))
        return negativeList
    

