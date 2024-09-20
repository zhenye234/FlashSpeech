# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


######## Build Experiment Environment ###########
exp_dir=$(cd `dirname $0`; pwd)
work_dir=$(dirname $(dirname $(dirname $exp_dir)))

export WORK_DIR=$work_dir
export PYTHONPATH=$work_dir
export PYTHONIOENCODING=UTF-8

######## Set Experiment Configuration ###########
exp_config="$exp_dir/exp_config.json"
exp_name="ns2_ict_normal"
ref_audio="/scratch/buildlam/speech_yz/Amphion2/egs/tts/NaturalSpeech2/7.wav"
checkpoint_path="/project/buildlam/zhenye/flashspeech_log/ns2_ict_normal_lignt_666_12node_smaller_lr_old_phone_s1_crop_mid_s2/epochepoch=81-stepstep=69454.ckpt"
output_dir="$work_dir/output-fs"
mode="single"

export CUDA_VISIBLE_DEVICES="0"



######## Train Model ###########
python "${work_dir}"/bins/tts/inference.py \
    --config=$exp_config \
    --text='you must look at him in the face, fight him, conquer him, with what scathe you may. you need not think to keep out of the way of him.' \
    --mode=$mode \
    --checkpoint_path=$checkpoint_path \
    --ref_audio=$ref_audio \
    --output_dir=$output_dir \

