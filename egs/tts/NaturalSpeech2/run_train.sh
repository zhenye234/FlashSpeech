# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


######## Build Experiment Environment ###########
exp_dir='egs/tts/NaturalSpeech2'
work_dir='/scratch/buildlam/speech_yz/FlashSpeech'

export WORK_DIR=$work_dir
export PYTHONPATH=$work_dir
export PYTHONIOENCODING=UTF-8
 
######## Set Experiment Configuration ###########
exp_config="$exp_dir/exp_config_s1.json" #s1 or s2
exp_name="/project/buildlam/zhenye/flashspeech_log/"

######## Train Model ###########
python \
    bins/tts/train_new.py \
    --config=$exp_config \
    --exp_name=$exp_name \
    --log_level debug 