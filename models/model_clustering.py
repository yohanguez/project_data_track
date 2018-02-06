import numpy as np
import glob
import pickle
from sklearn.metrics.pairwise import pairwise_distances
from matplotlib import pyplot as plt
from keras.models import Sequential
import pandas as pd
import os
import shutil
from sklearn import model_selection
from keras.preprocessing.image import ImageDataGenerator
from sklearn.cluster import DBSCAN
from shutil import copyfile


class model_clustering():
    def __init__(self, p_features_list, p_pic_list):
        print('TODO')
        pass


    def pickle_load(self, path1, path2): #path1= path/features_list_VGG16.pkl'
        with open(path1, 'rb') as f:
            self.features_list = pickle.load(f)

        with open(path2, 'rb') as f:
            self.pic_list = pickle.load(f)


    def fit_predict(self, eps, minPts):

        self.model_DBSCAN = DBSCAN(eps=eps, min_samples=minPts)
        self.labels = model_DBSCAN.fit_predict(dist_eucl)

    def compute_statistics(self):
        #statistics
        self.nb_image = nb_image = len(pic_list)
        self.nb_clusters = len(set(self.labels)) - (1 if -1 in self.labels
                                                  else 0)
        self.nb_element_clustered = self.labels[self.labels != -1].shape[0]
        self.number_element_per_cluster = int(self.nb_element_clustered / self.nb_clusters)
        self.number_element_no_clusterized = (self.labels==-1).sum()/self.nb_image

    def print_statistics(self):
        compute_statistics()
        print('Value for eps', eps)
        print('% Number of element no clustered: ',(labels == -1).sum() / nb_image)
        print('# clusters:', nb_clusters)
        print('Average of number in cluster:', nb_element_clustered)
        print('Average number of element per cluster:', number_element_per_cluster)

    def plot_statitics(selfself):
        labels_only_clustered = self.labels[self.labels != -1]
        f, axe = plt.subplots(1, 2, figsize=(30, 20))
        axe[0].hist(self.labels, bins=range(self.nb_clusters))


    def create_folder_one(self, no_cluster, path):
        path_name = path + '/cluster_' + str(no_cluster)
        if not os.path.exists(path_name):
            os.makedirs(path_name)
        indice = np.argwhere(self.labels == no_cluster)
        indice = indice.reshape(indice.shape[0]).tolist()
        nb_image = len(indice)
        for image_ind in indice:
            name_image = self.pic_list[image_ind].split('/')[-1]
            src = self.pic_list[image_ind]
            des = path_name + '/' + name_image
            copyfile(src, des)

    def create_folder_all(self, path):
        for no_cluster in range(self.nb_clusters):
            create_folder_one(no_cluster, path)

