import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
import h5py
from scipy.signal import spectrogram
from utils.data_utils import frft_wrapper
from tqdm import tqdm
import pickle
tqdm.disable = True


class AMRDatasets(Dataset):
    def __init__(self, data_path, phase):
        self.load_2016_file(phase)
        # if os.path.exists(f'{data_path}/{phase}_data.npy'):
        #     self.data = np.load(f'{data_path}/{phase}_data.npy')
        #     self.label = np.load(f'{data_path}/{phase}_label.npy')
        # else:
        #     self.load_file(data_path, phase)
        # # (416000, 2, 1024) (416000,) 信号I/Q波形域特征
        # self.data, self.label = np.squeeze(self.data), np.squeeze(self.label)
        # # (416000, 1, 1024)提取frft特征
        # self.frft_data = self.extract_frft_feature()
        # self.frft_data = self.frft_data[:, np.newaxis, :]
        
        self.data = (self.data - np.mean(self.data)) / np.std(self.data)
        self.frft_data = (self.frft_data - np.mean(self.frft_data)) / np.std(self.frft_data)
        print(self.data.shape, self.label.shape, self.frft_data.shape)
        # unique_elements, unique_counts = np.unique(self.label, return_counts=True)
        # # 输出每个元素及其对应的频次
        # for element, count in zip(unique_elements, unique_counts):
        #     print(f"{element}: {count}")

    def extract_frft_feature(self, data):
        frft_data_list = []
        for data_ in tqdm(data):
            frft_data = frft_wrapper(data_)
            frft_data_list.append(frft_data)
        return np.array(frft_data_list)
    
    def load_2016_file(self, phase):
        Xd =pickle.load(open('./data/RML2016.10a_dict.pkl','rb'),encoding='iso-8859-1')#Xd2(22W,2,128)
        mods,snrs = [sorted(list(set([k[j] for k in Xd.keys()]))) for j in [0,1] ] 
        X = []
        # X2=[]
        lbl = []
        # lbl2=[]
        train_idx=[]
        val_idx=[]
        np.random.seed(2016)
        a=0

        for mod in mods:
            for snr in snrs:
                X.append(Xd[(mod,snr)])     #ndarray(1000,2,128)
                for i in range(Xd[(mod,snr)].shape[0]):
                    lbl.append((mod,snr))
                train_idx+=list(np.random.choice(range(a*1000,(a+1)*1000), size=600, replace=False))
                val_idx+=list(np.random.choice(list(set(range(a*1000,(a+1)*1000))-set(train_idx)), size=200, replace=False))
                a+=1
        X = np.vstack(X)                    #(220000,2,128)  mods * snr * 1000,total 220000 samples            #(162060,2,128)
        X_frft = self.extract_frft_feature(X)
        X_frft = X_frft[:, np.newaxis, :]
        n_examples=X.shape[0]
        # n_test=X2.shape[0] 
        test_idx = list(set(range(0,n_examples))-set(train_idx)-set(val_idx))
        np.random.shuffle(train_idx)
        np.random.shuffle(val_idx)
        np.random.shuffle(test_idx)
        # test_idx=np.random.choice(range(0,n_test),size=n_test,replace=False)
        X_train = X[train_idx]
        X_frft_train = X_frft[train_idx]
        X_val=X[val_idx]
        X_frft_val=X_frft[val_idx]
        X_test =  X[test_idx]
        X_frft_test =  X_frft[test_idx]
        
        # yy = list(map(lambda x: mods.index(lbl[x][0]), train_idx))

        Y_train = np.array(list(map(lambda x: mods.index(lbl[x][0]), train_idx)))
        Y_val=np.array(list(map(lambda x: mods.index(lbl[x][0]), val_idx)))
        Y_test = np.array(list(map(lambda x: mods.index(lbl[x][0]),test_idx)))

        self.mods = mods
        self.snrs = snrs
        self.lbl = lbl
        if phase == 'train':
            self.data = X_train
            self.frft_data = X_frft_train
            self.label = Y_train
            self.idx = train_idx
        elif phase == 'val':
            self.data = X_val
            self.frft_data = X_frft_val
            self.label = Y_val
            self.idx = val_idx
        elif phase == "test":
            self.data = X_test
            self.frft_data = X_frft_test
            self.label = Y_test
            self.idx = test_idx
        


    def load_2019_file(self, data_path, signal):
        self.data = h5py.File(f"{data_path}/{signal}.mat",'r')['data_save'][:].swapaxes(0,2)
        self.data = np.expand_dims(self.data, axis=3)

        self.label = pd.read_csv(f"{data_path}/{signal}_labels1.csv", header=None)
        self.label = np.array(self.label)

        if signal == "test":
            np.save(f"{data_path}/{signal}_data.npy", self.data)
            np.save(f"{data_path}/{signal}_label.npy", self.label)
            return

        # Count samples for each class
        class_counts = np.bincount(self.label.flatten())

        # Initialize lists to store indices for each class
        class_indices = [[] for _ in range(len(class_counts))]

        # Iterate through each class and store indices
        for class_idx in range(len(class_counts)):
            indices = np.where(self.label == class_idx)[0]
            np.random.shuffle(indices)
            class_indices[class_idx] = indices

        # Split data and labels into train and val using balanced class indices
        train_indices = []
        val_indices = []
        for indices in class_indices:
            split_index = int(len(indices) * 0.8)
            train_indices.extend(indices[:split_index])
            val_indices.extend(indices[split_index:])

        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)

        np.save(f"{data_path}/train_data.npy", self.data[train_indices])
        np.save(f"{data_path}/train_label.npy", self.label[train_indices])
        np.save(f"{data_path}/val_data.npy", self.data[val_indices])
        np.save(f"{data_path}/val_label.npy", self.label[val_indices])

        if signal == 'train':
            self.data = self.data[train_indices]
            self.label = self.label[train_indices]

        elif signal == 'val':
            self.data = self.data[val_indices]
            self.label = self.label[val_indices]


    def __getitem__(self, index):
        sample = torch.tensor(self.data[index], dtype=torch.float32)
        frft_sample = torch.tensor(self.frft_data[index], dtype=torch.float32)
        label = torch.tensor(self.label[index], dtype=torch.long)

        return sample, frft_sample, label
    
    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    train_data = AMRDatasets("HisarModdataset/train/", 'train')
    # val_data = AMRDatasets("HisarModdataset/train/", 'val')
    # test_data = AMRDatasets("HisarModdataset/test/", 'test')