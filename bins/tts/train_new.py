# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import torch
from utils.util import load_config
import glob

# 导入 PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
# 导入新的 FlashSpeech 的 LightningModule 和 DataModule
# from models.tts.naturalspeech2.flashspeech_trainer import FlashSpeechLightningModule, NS2DataModule
# from models.tts.naturalspeech2.flashspeech_trainer_stage2 import FlashSpeechLightningModule, NS2DataModule

def find_latest_checkpoint(dirpath):
    """
    在指定目录中查找最新的检查点文件。
    """
    checkpoint_files = glob.glob(os.path.join(dirpath, '*.ckpt'))
    if not checkpoint_files:
        return None
    latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
    return latest_checkpoint

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config.json",
        help="json files for configurations.",
        required=True,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="random seed",
        required=False,
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="exp_name",
        help="A specific name to note the experiment",
        required=True,
    )
    parser.add_argument(
        "--resume", action="store_true", help="Whether to resume training from the latest checkpoint"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Checkpoint path for resuming training or testing. If not provided and --resume is set, the latest checkpoint will be used.",
    )
    parser.add_argument(
        "--test", action="store_true", default=False, help="Test the model"
    )
    parser.add_argument(
        "--log_level", default="warning", help="logging level (debug, info, warning)"
    )
    # 如果需要添加其他参数，请在此处添加
    args = parser.parse_args()
    cfg = load_config(args.config)

    # 设置随机种子
    pl.seed_everything(args.seed)

    # Data Augmentation
    if hasattr(cfg, "preprocess"):
        if hasattr(cfg.preprocess, "data_augment"):
            if (
                isinstance(cfg.preprocess.data_augment, list)
                and len(cfg.preprocess.data_augment) > 0
            ):
                new_datasets_list = []
                for dataset in cfg.preprocess.data_augment:
                    new_datasets = [
                        (
                            f"{dataset}_pitch_shift"
                            if getattr(cfg.preprocess, 'use_pitch_shift', False)
                            else None
                        ),
                        (
                            f"{dataset}_formant_shift"
                            if getattr(cfg.preprocess, 'use_formant_shift', False)
                            else None
                        ),
                        (
                            f"{dataset}_equalizer"
                            if getattr(cfg.preprocess, 'use_equalizer', False)
                            else None
                        ),
                        (
                            f"{dataset}_time_stretch"
                            if getattr(cfg.preprocess, 'use_time_stretch', False)
                            else None
                        ),
                    ]
                    new_datasets_list.extend(filter(None, new_datasets))
                cfg.dataset.extend(new_datasets_list)

    print("Experiment name: ", args.exp_name)

 
    if args.config.endswith("s1"):
        from models.tts.naturalspeech2.flashspeech_trainer import FlashSpeechLightningModule,NS2DataModule
    elif args.config.endswith("s2"):
        from models.tts.naturalspeech2.flashspeech_trainer_stage2 import FlashSpeechLightningModule,NS2DataModule

    # 实例化 FlashSpeech 的 LightningModule 和 DataModule
    model = FlashSpeechLightningModule(cfg)
    data_module = NS2DataModule(cfg)  # 如果有特定的数据模块，请替换为正确的类

    # 配置 accelerator 和 devices
    if torch.cuda.is_available():
        accelerator = 'gpu'
        devices = -1  # 使用所有可用的 GPU
    else:
        accelerator = 'cpu'
        devices = 1   # 使用 CPU

    # 配置 ModelCheckpoint 回调
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.exp_name,
        filename='epoch{epoch}-step{step}',
        save_top_k=-1,
        every_n_epochs=1,
        save_last=True,
    )

    # 配置 WandbLogger
    logger = WandbLogger(project='flashspeech_log', name=args.exp_name)
    num_nodes = int(os.getenv('SLURM_NNODES', 1))
    # 配置 PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epoch,
        accelerator=accelerator,
        num_nodes=2, #!num_nodes, 1 2
        devices=devices,
        precision=16,
        gradient_clip_val=0.5,
        accumulate_grad_batches=cfg.train.gradient_accumulation_step,
        default_root_dir=args.exp_name,  # 设置默认的保存路径
        logger=logger,
        use_distributed_sampler=False,
        # 如果需要分布式训练，可以设置 strategy 参数，例如 'ddp'
        strategy='ddp_find_unused_parameters_true',
        callbacks=[checkpoint_callback],  # 添加回调
    )

    ckpt_path = None
    if args.resume:
        if args.checkpoint_path:
            if os.path.isfile(args.checkpoint_path):
                ckpt_path = args.checkpoint_path
                print(f"Resuming training from specified checkpoint: {ckpt_path}")
            else:
                print(f"Specified checkpoint path does not exist: {args.checkpoint_path}")
                return
        else:
            # 自动查找最新的检查点
            latest_ckpt = find_latest_checkpoint(args.exp_name)
            if latest_ckpt:
                ckpt_path = latest_ckpt
                print(f"Resuming training from latest checkpoint: {ckpt_path}")
            else:
                print(f"No checkpoint found in {args.exp_name}. Starting training from scratch.")
    elif args.test and args.checkpoint_path:
        # 如果是测试模式且指定了检查点路径
        if os.path.isfile(args.checkpoint_path):
            ckpt_path = args.checkpoint_path
            print(f"Testing model from checkpoint: {ckpt_path}")
        else:
            print(f"Specified checkpoint path does not exist: {args.checkpoint_path}")
            return

    if args.test:
        if ckpt_path:
            trainer.test(model, datamodule=data_module, ckpt_path=ckpt_path)
        else:
            print("Checkpoint path is required for testing.")
    else:
        if args.resume and ckpt_path:
            # 从检查点恢复训练
            trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path)
        else:
            # 否则，开始训练
            trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    main()
