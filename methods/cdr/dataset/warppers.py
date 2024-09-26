#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import math

from torch.utils.data import DataLoader

from .samplers import CustomSampler
from .transforms import SimCLRDataTransform
from .datasets import *
from ..utils.nn_utils import compute_knn_graph


class DataSetWrapper(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.batch_num = 0
        self.test_batch_num = 0
        self.knn_indices = None
        self.knn_distances = None
        self.symmetric_nn_indices = None
        self.symmetric_nn_weights = None
        self.symmetric_nn_dists = None
        self.sym_no_norm_weights = None
        self.n_neighbor = 0
        self.shifted_data = None

    def get_data_loaders(self, epoch_num, data_matrix, n_neighbors, is_image=True, symmetric="UMAP"):
        self.n_neighbor = n_neighbors
        train_dataset = UMAPCDRTextDataset(data_matrix)

        self.knn_indices, self.knn_distances = compute_knn_graph(train_dataset.data, n_neighbors, accelerate=True)

        self.distance2prob(train_dataset, symmetric)

        train_indices, train_num = self.update_transform(epoch_num, is_image, train_dataset)

        train_loader = self._get_train_validation_data_loaders(train_dataset, train_indices)

        return train_loader, train_num

    def update_transform(self, epoch_num, is_image, train_dataset):
        train_dataset.update_transform(SimCLRDataTransform(epoch_num, train_dataset, is_image,
                                                           self.n_neighbor, self.symmetric_nn_indices,
                                                           self.symmetric_nn_weights))
        train_num = train_dataset.data_num

        train_indices = list(range(train_num))
        self.batch_num = math.floor(train_num / self.batch_size)

        return train_indices, train_num

    def distance2prob(self, train_dataset, symmetric):
        train_dataset.umap_process(self.knn_indices, self.knn_distances, self.n_neighbor, symmetric)
        self.symmetric_nn_indices = train_dataset.symmetry_knn_indices
        self.symmetric_nn_weights = train_dataset.symmetry_knn_weights
        self.symmetric_nn_dists = train_dataset.symmetry_knn_dists
        self.sym_no_norm_weights = train_dataset.sym_no_norm_weights

    def _get_train_validation_data_loaders(self, train_dataset, train_indices):
        np.random.shuffle(train_indices)
        train_sampler = CustomSampler(train_indices)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  drop_last=True, shuffle=False)

        return train_loader
