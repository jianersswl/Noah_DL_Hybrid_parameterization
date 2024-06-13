# I/O
import os
import pickle
import torch
import numpy as np
import pandas as pd

# Pytorch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class PGDataset(Dataset):
    '''
    root: 输入数据的根目录
    '''
    def __init__(self, config):
        self.config = config
        self.root = os.path.join(config['root'], 'PG_DATASET')
        
        self.input_csv = pd.read_csv(os.path.join(self.root, 'static_propertise.csv'))
        self.input_csv = self.input_csv.drop(['ORIG_FID', 'Soil_htype'], axis=1)
        self.input_csv['lat'] = self.input_csv['CMFD_lat']
        self.input_csv['lon'] = self.input_csv['CMFD_lon']
        self.input_csv = self.input_csv.drop(self.input_csv.columns[4:22], axis=1)
        
        if config['standardization'] == True:
            # 初始化 StandardScaler
            scaler = StandardScaler()
            # 对每一列进行标准化
            self.input_csv.iloc[:, 2:] = scaler.fit_transform(self.input_csv.iloc[:, 2:])
            
        self.label_paths = os.listdir(os.path.join(self.root, 'GRID_NPY_QQ'))
#         print(self.input_csv.info())
#         print(len(self.label_paths))
        
    def __getitem__(self, idx):
        # 获取数据路径
        label_path = self.label_paths[idx]
        
        # 获取经纬度
        lon = float(label_path.split('_')[-3])
        lat = float(label_path.split('_')[-2])
#         print(lon, lat)
        
        # 加载label数据
        with open(os.path.join(os.path.join(self.root, 'GRID_NPY_QQ'), label_path), 'rb') as file:
            label_pkg = np.load(file)
        
        # load input data for generator
        gen_pkg = self.input_csv[self.input_csv['lon']==lon]
        gen_pkg = gen_pkg[gen_pkg['lat']==lat]
#         print(input_pkg[['CMFD_lon', 'CMFD_lat']])
        gen_pkg = np.asarray(gen_pkg.iloc[:, 2:])[0]
#         print(f'gen_pkg.shape: {gen_pkg.shape}')
        
        # load input data for surrogate
        cmfd_filename = f'{lon}_{lat}_.npy'
        cmfd_path = os.path.join(os.path.join(self.root, 'CMFD2FLOAT'), cmfd_filename)
        # 加载数据
        with open(os.path.join(cmfd_path), 'rb') as file:
            surr_pkg = np.load(file)
        params = np.zeros((365,4))
        surr_pkg = np.concatenate((params, surr_pkg), axis=1)
        
        if self.config['output']==0:
            label_pkg = label_pkg - 273.15
        
        meta_pkg = [lon, lat]
        return torch.tensor(gen_pkg, dtype=torch.double), torch.tensor(surr_pkg, dtype=torch.double), torch.tensor(label_pkg, dtype=torch.double), meta_pkg

    def __len__(self):
        return len(self.label_paths)

class PGDataset_TEST(Dataset):
    '''
    root: 输入数据的根目录
    '''
    def __init__(self, config):
        self.config = config
        self.root = os.path.join(config['root'], 'PG_DATASET')
        
        self.input_csv = pd.read_csv(os.path.join(self.root, 'static_propertise.csv'))
        self.input_csv = self.input_csv.drop(['ORIG_FID', 'Soil_htype'], axis=1)
        self.input_csv['lat'] = self.input_csv['CMFD_lat']
        self.input_csv['lon'] = self.input_csv['CMFD_lon']
        self.input_csv = self.input_csv.drop(self.input_csv.columns[4:22], axis=1)
        
        if config['standardization'] == True:
            # 初始化 StandardScaler
            scaler = StandardScaler()
            # 对每一列进行标准化
            self.input_csv.iloc[:, 2:] = scaler.fit_transform(self.input_csv.iloc[:, 2:])
            
        self.label_paths = os.listdir(os.path.join(self.root, 'GRID_NPY_QQ'))
#         print(self.input_csv.info())
#         print(len(self.label_paths))
        
    def __getitem__(self, idx):
        # 获取数据路径
        label_path = self.label_paths[idx]
        
        # 获取经纬度
        lon = float(label_path.split('_')[-3])
        lat = float(label_path.split('_')[-2])
#         print(lon, lat)
        
        # 加载label数据
        with open(os.path.join(os.path.join(self.root, 'GRID_NPY_QQ'), label_path), 'rb') as file:
            label_pkg = np.load(file)
        
        # load input data for generator
        gen_pkg = self.input_csv[self.input_csv['lon']==lon]
        gen_pkg = gen_pkg[gen_pkg['lat']==lat]
#         print(input_pkg[['CMFD_lon', 'CMFD_lat']])
        gen_pkg = np.asarray(gen_pkg.iloc[:, 2:])[0]
#         print(f'gen_pkg.shape: {gen_pkg.shape}')
        
        # load input data for surrogate
        test_year = self.config['test_year']
        cmfd_filename = f'{lon}_{lat}_{test_year}_.npy'
        cmfd_path = os.path.join(os.path.join(self.root, 'CMFD2FLOAT_year'), cmfd_filename)
        # 加载数据
        with open(os.path.join(cmfd_path), 'rb') as file:
            surr_pkg = np.load(file)
        params = np.zeros((365,4))
        surr_pkg = np.concatenate((params, surr_pkg), axis=1)
        
        meta_pkg = [lon, lat]
        return torch.tensor(gen_pkg, dtype=torch.double), torch.tensor(surr_pkg, dtype=torch.double), torch.tensor(label_pkg, dtype=torch.double), meta_pkg

    def __len__(self):
        return len(self.label_paths)
            