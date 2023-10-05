import torch
import torch.nn as nn
import numpy as np
import time
import os, glob, sys
from torch.utils.data import Dataset


class RHO_dataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.density_noisy, self.density_true  = self.get_data(self.data_dir)  

    def get_data(self, data_path):
        data1 = []
        data2 = []
        for den_path in glob.glob(data_path + os.sep + 'run_*'):
            #print(den_path)
            sden_path = den_path + "/sDFT/density-out"
            dden_path = den_path + "/dDFT/density-out"
            data1.append(sden_path)
            data2.append(dden_path)
        return data1, data2 

    def __getitem__(self, index):
        # read the density data and return them
        noisy  = np.loadtxt(self.density_noisy[index], dtype='float32')
        true   = np.loadtxt(self.density_true[index], dtype='float32')
        #no normalization for density
        noisy = np.reshape(noisy, (1,32,32,32))
        true = np.reshape(true, (1,32,32,32))
        noisy_tensor = torch.tensor(noisy, dtype=torch.float)
        true_tensor  = torch.tensor(true, dtype=torch.float)

        return noisy_tensor, true_tensor

    def __len__(self):
        return len(self.density_noisy)


def load_data(input_args):

    if input_args.train_size == None:
        train_dataset = RHO_dataset(input_args.train_data_dir)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = input_args.batch_size, shuffle = True)

    else:
        full_train_dataset = RHO_dataset(input_args.train_data_dir)
        train_size = input_args.train_size
        rest_train = len(full_train_dataset) - train_size

        train_dataset,  _ = torch.utils.data.random_split(full_train_dataset, [train_size, rest_train])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = input_args.batch_size, shuffle = True)

    if input_args.val_size == None:    
        val_dataset = RHO_dataset(input_args.val_data_dir)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = input_args.batch_size, shuffle = not True)
    else:
        full_val_dataset = RHO_dataset(input_args.val_data_dir)
        val_size = input_args.val_size
        rest_val = len(full_val_dataset) - val_size

        val_dataset,  _ = torch.utils.data.random_split(full_val_dataset, [val_size, rest_val])
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = input_args.batch_size, shuffle = not True)


    print("Train_size = ", len(train_dataset))
    print("Val_train = ", len(val_dataset))

    return train_loader, val_loader


def load_test_data(input_args):

    if input_args.test_size == None:
        test_dataset = RHO_dataset(input_args.test_data_dir)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 1, shuffle = not True)

    else:
        full_test_dataset = RHO_dataset(input_args.test_data_dir)
        test_size = input_args.test_size
        rest_test = len(full_test_dataset) - test_size

        test_dataset,  _ = torch.utils.data.random_split(full_test_dataset, [test_size, rest_test])
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 1, shuffle = not True)

    print("Test_size = ", len(test_dataset))

    return test_loader


