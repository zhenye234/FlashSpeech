#!/bin/bash

#SBATCH --job-name=speech
#SBATCH --time=72:00:00 
#SBATCH --nodes=2         
#SBATCH --partition=buildlam
#SBATCH --gpus-per-node=8  
#SBATCH --ntasks-per-node=8
#SBATCH --account buildlam  
#SBATCH --cpus-per-task=28 
#SBATCH --mem=1024   
#SBATCH --exclusive
export LOGLEVEL=INFO

 
exp_dir='egs/tts/NaturalSpeech2'
work_dir='/scratch/buildlam/speech_yz/Amphion2'

export WORK_DIR=$work_dir
export PYTHONPATH=$work_dir
export PYTHONIOENCODING=UTF-8
 
######## Set Experiment Configuration ###########
exp_config="$exp_dir/exp_config_s1.json"
exp_name="/project/buildlam/zhenye/flashspeech_log/ns2_ict_normal_lignt_666_12node_smaller_lr_old_phone_s1_crop_mid_s1_new"

######## Train Model ###########
srun python \
    bins/tts/train_new.py \
    --config=$exp_config \
    --exp_name=$exp_name \
    --log_level debug \
    --resume