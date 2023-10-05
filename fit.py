import torch
import torch.nn as nn
import numpy as np
import time
import os, glob, sys
from torch.utils.data import Dataset

import models
import utils


def fit(input_args, device):

    epochs = input_args.epochs
    lr = input_args.lr
    batch_size = input_args.batch_size
    print_freq = input_args.print_freq
    
    model = models.UNet().to(device)
    loss_fn = nn.MSELoss()
    opt_func=torch.optim.Adam
   
    train_loader, val_loader = utils.load_data(input_args)

    if not input_args.restart:
         print('\nfrom scratch')
         train_epoch_loss = []
         val_epoch_loss = []
         running_train_loss = []
         running_val_loss = []
         epochs_till_now = 0
    else:
         ckpt_path = os.path.join(input_args.log_dir, input_args.log_file)
         ckpt = torch.load(ckpt_path)
         print(f'\nckpt loaded: {ckpt_path}')
         model_state_dict = ckpt['model_state_dict']
         model.load_state_dict(model_state_dict)
         model.to(device)
         losses = ckpt['losses']
         running_train_loss = losses['running_train_loss']
         running_val_loss = losses['running_val_loss']
         train_epoch_loss = losses['train_epoch_loss']
         val_epoch_loss = losses['val_epoch_loss']
         epochs_till_now = ckpt['epochs_till_now']         

    print('\nmodel has {} M parameters'.format(count_parameters(model)))
    print(f'\nloss_fn        : {loss_fn}')
    print(f'lr             : {lr}')
    print(f'epochs_till_now: {epochs_till_now}')
    print(f'epochs from now: {epochs}')

    
    optimizer = opt_func(filter(lambda p: p.requires_grad, model.parameters()), lr = lr)
    
    for epoch in range(epochs_till_now, epochs_till_now + epochs):
        print('\nTRAINING...')
        epoch_train_start_time = time.time()
        # Training Phase 
        model.train()
       
        epoch_loss_train = [] 
        for idx, (density, density_target) in enumerate(train_loader):
            batch_start_time = time.time()
            density_target = density_target.to(device)
            density = density.to(device)
            density_predict = model(density)
            loss = loss_fn(density_predict, density_target)
            running_train_loss.append(loss.item())
            epoch_loss_train.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
           
            batch_time = time.time() - batch_start_time 
            print('train loss for epoch: {} batch id: {}, loss: {}, time: {}'.format(epoch, idx, loss.item(), batch_time), flush=True)
        
        print('\nAverage TRAIN LOSS for epoch: {}, loss: {}'.format(epoch, np.array(epoch_loss_train).mean()), flush=True)

        train_epoch_loss.append(np.array(running_train_loss).mean())      
 
        epoch_train_time = time.time() - epoch_train_start_time
        m,s = divmod(epoch_train_time, 60)
        h,m = divmod(m, 60)
        print('\nepoch train time: {} hrs {} mins {} secs'.format(int(h), int(m), int(s)))
 
        print('\nVALIDATION...')
        epoch_val_start_time = time.time()
        model.eval()
        
        epoch_loss_val = [] 
        with torch.no_grad():
            for idx, (density, density_target) in enumerate(val_loader):
                density_target = density_target.to(device)
                density = density.to(device)
                density_predict = model(density)
                loss = loss_fn(density_predict, density_target)

                running_val_loss.append(loss.item())
                epoch_loss_val.append(loss.item())

                print('val loss for epoch: {} batch id: {}, loss: {}'.format(epoch, idx, loss.item()), flush=True)
       
        print('\nAverage VAL LOSS for epoch: {}, loss: {}'.format(epoch, np.array(epoch_loss_val).mean()), flush=True)

        val_epoch_loss.append(np.array(running_val_loss).mean()) 
       
        epoch_val_time = time.time() - epoch_val_start_time
        m,s = divmod(epoch_val_time, 60)
        h,m = divmod(m, 60)
        print('\nepoch val   time: {} hrs {} mins {} secs'.format(int(h), int(m), int(s))) 

        if (epoch % print_freq ) == 0:
            torch.save({'model_state_dict': model.state_dict(), 
                'losses': {'running_train_loss': running_train_loss, 
                           'running_val_loss': running_val_loss, 
                           'train_epoch_loss': train_epoch_loss, 
                           'val_epoch_loss': val_epoch_loss}, 
                'epochs_till_now': epoch+1}, 
                os.path.join(input_args.log_dir, 'model{}.pth'.format(str(epoch + 1).zfill(2))))
        
    return 



def predict(input_args, device):
   
    model = models.UNet().to(device)
    loss_fn = nn.MSELoss()
    
    test_loader = utils.load_test_data(input_args)
    predict_dir = input_args.pred_dir
      
    ckpt_path = os.path.join(input_args.log_dir, input_args.log_file)
    ckpt = torch.load(ckpt_path)
    print(f'\nckpt loaded: {ckpt_path}')
    model_state_dict = ckpt['model_state_dict']
    model.load_state_dict(model_state_dict)
    model.to(device)
 
    print('\nTESTING...')
    model.eval()
    
    with torch.no_grad():
        for idx, (density, density_target) in enumerate(test_loader):
            density_target = density_target.to(device)
            density = density.to(device)
            density_predict = model(density)
            loss = loss_fn(density_predict, density_target)

            if input_args.print_density:
                #save density_predict to file with index number in filename
                name = 'density-target-{}'.format(str(idx).zfill(2))
                path = os.path.join(predict_dir, name)
                #print(path)
                density_target = density_target.cpu().detach().numpy()
                np.savetxt(path, np.reshape(density_target, (-1,1)))

                name = 'density-predict-{}'.format(str(idx).zfill(2))
                path = os.path.join(predict_dir, name)
                #print(path)
                density_predict = density_predict.cpu().detach().numpy()
                np.savetxt(path, np.reshape(density_predict, (-1,1)))

                name = 'density-actual-{}'.format(str(idx).zfill(2))
                path = os.path.join(predict_dir, name)
                #print(path)
                density = density.cpu().detach().numpy()
                np.savetxt(path, np.reshape(density, (-1,1)))

            print('test loss for batch id: {}, loss: {}'.format(idx, loss.item()), flush=True)
        
    return 


def count_parameters(model):
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_parameters/1e6 # in terms of millions

