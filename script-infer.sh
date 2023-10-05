#!/bin/bash
# for running on gilbreth


module load learning/conda-2020.11-py38-gpu
module load ml-toolkit-gpu/all 

host=`hostname -s`
echo $CUDA_VISIBLE_DEVICES

#testing the model
python main.py --log_dir "./logdir" --restart --log_file "model05.pth" --test --test_data_dir "./data/test_data/" --test_size 4

#testing the model and printing the predicted densities with actual and noisy densities
python main.py --log_dir "./logdir" --restart --log_file "model05.pth" --test --test_data_dir "./data/test_data/" --test_size 2  --pred_dir "./predict/" --print_density
