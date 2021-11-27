"""
This file contains the definition of different heterogeneous datasets used for training
"""
import torch
import numpy as np
import math

from .base_dataset import BaseDataset

class MixedDataset(torch.utils.data.Dataset):
    # def __init__(self, options, **kwargs):
    #     self.dataset_list = ['h36m', 'lsp-orig', 'mpii', 'lspet', 'coco', options.data_train, 'mpi-inf-3dhp']
    #     self.dataset_dict = {'h36m': 0, 'lsp-orig': 1, 'mpii': 2, 'lspet': 3, 'coco': 4, options.data_train: 5, 'mpi-inf-3dhp': 6}
    #     self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
    #     total_length = sum([len(ds) for ds in self.datasets])
    #     length_itw = sum([len(ds) for ds in self.datasets[1:-2]])
    #     # a=[print(len(ds)) for ds in self.datasets]
    #     self.length = math.floor(len(self.datasets[-2]) / 0.3)
    #     """
    #     Data distribution inside each batch:
    #     30% H36M - 30% ITW - 30% SLP - 10% MPI-INF
    #     """
    #     self.partition = [.3, .6*.5*len(self.datasets[1])/length_itw,
    #                       .6*.5*len(self.datasets[2])/length_itw,
    #                       .6*.5*len(self.datasets[3])/length_itw, 
    #                       .6*.5*len(self.datasets[4])/length_itw,
    #                       .6*.5,
    #                       0.1]
    #     self.partition = np.array(self.partition).cumsum()

    def __init__(self, options, **kwargs):
        self.dataset_list = [options.data_train]
        self.dataset_dict = {options.data_train: 0}
        self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
        total_length = sum([len(ds) for ds in self.datasets])
        length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
        self.length = max([len(ds) for ds in self.datasets])
        """
        Data distribution inside each batch:
        100% slp
        """
        self.partition = [1]
        self.partition = np.array(self.partition).cumsum()

    def __getitem__(self, index):
        return self.datasets[0][index % len(self.datasets[0])]

    # def __init__(self, options, **kwargs):
    #     self.dataset_list = ['h36m', 'lsp-orig', 'mpii', 'lspet', 'coco', 'mpi-inf-3dhp']
    #     self.dataset_dict = {'h36m': 0, 'lsp-orig': 1, 'mpii': 2, 'lspet': 3, 'coco': 4, 'mpi-inf-3dhp': 5}
    #     self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
    #     total_length = sum([len(ds) for ds in self.datasets])
    #     length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
    #     self.length = max([len(ds) for ds in self.datasets])
    #     """
    #     Data distribution inside each batch:
    #     30% H36M - 60% ITW - 10% MPI-INF
    #     """
    #     self.partition = [.3, .6*len(self.datasets[1])/length_itw,
    #                       .6*len(self.datasets[2])/length_itw,
    #                       .6*len(self.datasets[3])/length_itw, 
    #                       .6*len(self.datasets[4])/length_itw,
    #                       0.1]
    #     self.partition = np.array(self.partition).cumsum()

    # def __getitem__(self, index):
    #     p = np.random.rand()
    #     for i in range(7):
    #         if p <= self.partition[i]:
    #             return self.datasets[i][index % len(self.datasets[i])]

    def __len__(self):
        return self.length
