# I/O
import os
import pickle
import torch
import numpy as np

# Pytorch
from torch.utils.data import Dataset

class LSMDataset(Dataset):
    '''
    root: 输入数据的根目录
    '''
    def __init__(self, config):
        self.config = config
        self.root = os.path.join(config['root'], 'TEMP')
        self.data_paths = os.listdir(self.root)
        
    def __getitem__(self, idx):
        # 获取数据路径
        data_path = self.data_paths[idx]
        
        # 加载数据
        with open(os.path.join(self.root, data_path), 'rb') as file:
            data_pkg = pickle.load(file)
        input_pkg = data_pkg[0]
        label_pkg = data_pkg[1]

        label_pkg = label_pkg[:, self.config['output']*8: (self.config['output']+1)*8]
        if self.config['output']==0:
            label_pkg = label_pkg - 273.15

        label_pkg = label_pkg[:, self.config['soil_layer_num']]
        return torch.tensor(input_pkg, dtype=torch.double), torch.tensor(label_pkg, dtype=torch.double)

    def __len__(self):
        return len(self.data_paths)
            