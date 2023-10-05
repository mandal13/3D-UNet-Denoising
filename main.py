import torch
import torch.nn as nn
import numpy as np
import time
import os, glob, sys
from torch.utils.data import Dataset
import argparse

import fit


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print('device: ', device)


parser = argparse.ArgumentParser(description='De-noise Electron Density with U-Net')
parser.add_argument('--epochs', default=10000, type=int, help='number of epochs/steps')
parser.add_argument('--lr', default=1.0e-5, type=float, help='learning rate')
parser.add_argument('--batch_size', default=8, type=int, help='Batch size')
parser.add_argument('--print_freq', default=1, type=int, help='Print frequency of the logfiles for restarting')

parser.add_argument("--restart", action="store_true", help="Do you want to restart from a checkpoint file ?")
parser.add_argument('--log_dir',default='',type=str, help='logfile/checkpoint file directory')
parser.add_argument('--log_file',default='',type=str, help='log/checkpoint file name')

parser.add_argument("--train", action="store_true", help="Do you want to Train the model ?")
parser.add_argument('--train_data_dir',default='',type=str, help='Directory of training data')
parser.add_argument('--train_size', default=None, type=int, help='Training data size (if not specified takes all data)')
parser.add_argument('--val_data_dir',default='',type=str, help='Directory of validation data')
parser.add_argument('--val_size', default=None, type=int, help='Validation data size (if not specified takes all data)')

parser.add_argument("--test", action="store_true", help="Do you want to Test the model ?")
parser.add_argument('--test_data_dir',default='',type=str, help='Directory of test data')
parser.add_argument('--test_size', default=None, type=int, help='Test data size (if not specified takes all data)')
parser.add_argument("--print_density", action="store_true", help="Do you want to print the density predictions ? ")
parser.add_argument('--pred_dir',default='',type=str, help='Directory for printing density predictions with ground truth data')




if __name__=='__main__':

    input_args = parser.parse_args()

    start_time = time.time()

    if input_args.train:
        print("\nStarting training the model\n")
        #
        fit.fit(input_args, device)
        #
        print("\nFinished training the model")
    elif input_args.test:
        print("\nStarting test the model\n")  
        #
        fit.predict(input_args, device)
        #
        print("\nFinished test the model")  
    else:
        print("\nPlease decide either Training / Testing !!") 
 
    end_time = time.time()

    tot_time = end_time - start_time 
    m,s = divmod(tot_time, 60)
    h,m = divmod(m, 60)
    print('\nTotal time: {} hrs {} mins {} secs'.format(int(h), int(m), int(s)))


