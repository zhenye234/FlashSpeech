# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import time
import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import ConcatDataset, DataLoader
from models.tts.naturalspeech2.ns2_dataset import NS2Dataset, NS2Collator, batch_by_size,NS2Dataset_New
from models.tts.naturalspeech2.ns2_loss import (
    log_pitch_loss,
    log_dur_loss,
)
from models.tts.naturalspeech2.flashspeech import FlashSpeech
from torch.optim import AdamW
from diffusers import get_scheduler
import json5
from models.base.base_sampler import VariableSampler
import torch.distributed as dist
from transformers import WavLMModel 
#WavLMLosscond
from models.tts.naturalspeech2.wavlm_loss import WavLMLosscond


def calculate_adaptive_weight(loss1, loss2, last_layer=None):
    loss1_grad = torch.autograd.grad(loss1, last_layer, retain_graph=True)[0]
    loss2_grad = torch.autograd.grad(loss2, last_layer, retain_graph=True)[0]
    d_weight = torch.norm(loss1_grad) / (torch.norm(loss2_grad) + 1e-4)
    d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
    return d_weight


class FlashSpeechLightningModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = FlashSpeech(cfg=self.cfg.model)

        # 
 
        ckpt = torch.load('/project/buildlam/zhenye/flashspeech_log/ns2_ict_normal_lignt_666_2node_smaller_lr/epochepoch=499-stepstep=160000.ckpt', map_location="cpu")
        state_dict = ckpt['state_dict']
        # 调整键名
        new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}

        # 加载模型参数
        self.model.load_state_dict(new_state_dict,strict=False)
         




        self.criterion = torch.nn.L1Loss(reduction="mean")
        self.save_hyperparameters()

        self.wavlm = WavLMModel.from_pretrained('microsoft/wavlm-large')
        # i delete hidden_states.requires_grad = True RuntimeError: you can only change requires_grad flags of leaf variables.
        self.wavlm.train() 
        self.wavlm.requires_grad_(False)
        self.discriminator = WavLMLosscond()


    def forward(self, batch):
        diff_out, prior_out = self.model(
            code=batch["code"],
            pitch=batch["pitch"],
            duration=batch["duration"],
            phone_id=batch["phone_id"],
            ref_code=batch["ref_code"],
            phone_mask=batch["phone_mask"],
            mask=batch["mask"],
            ref_mask=batch["ref_mask"],
            trainstep=self.global_step,
        )
        return diff_out, prior_out

    def training_step(self, batch, batch_idx):
        if (batch_idx ) % 2 == 0:
            # 生成器步骤
            # 冻结判别器参数
            for param in self.discriminator.parameters():
                param.requires_grad = False

            diff_out, prior_out = self.forward(batch)
            pitch_loss = log_pitch_loss(prior_out["pitch_pred_log"], batch["pitch"], mask=batch["mask"])
            dur_loss = log_dur_loss(prior_out["dur_pred_log"], batch["duration"], mask=batch["phone_mask"])
            diff_loss_x0 = diff_out["ict_loss"].mean()

 
            latent_gen = diff_out['x0_pred']
            latent_gt = diff_out['x0_gt']
 
            wav_gen = self.model.soundstream.decoder_2(latent_gen*self.model.latent_norm) #  
            wav_gt = self.model.soundstream.decoder_2(latent_gt*self.model.latent_norm)
            wav_cond =  self.model.soundstream.decode(batch['ref_code'].long().transpose(0,1))    

            # 提取生成音频的特征
            y_rec = self.wavlm(wav_gen.squeeze(1), output_hidden_states=True).hidden_states
            y_rec = torch.stack(y_rec, dim=1).transpose(-1, -2).flatten(start_dim=1, end_dim=2)

            # 提取条件音频的特征（不计算梯度）
            with torch.no_grad():
                y_cond = self.wavlm(wav_cond.squeeze(1), output_hidden_states=True).hidden_states
                y_cond = torch.stack(y_cond, dim=1).transpose(-1, -2).flatten(start_dim=1, end_dim=2)
                y_cond = y_cond.mean(2)

            # 计算生成器的对抗损失
            loss_g = self.discriminator.generator(y_rec, y_cond.detach())

            # 计算平衡权重
            last_layer = self.model.diffusion.diff_estimator.layers[-1].out_proj.weight
            balance_weight =  calculate_adaptive_weight(diff_loss_x0, loss_g, last_layer=last_layer)

            # 总损失
            gen_loss = (loss_g) * balance_weight * 0.5
            total_loss = pitch_loss + dur_loss + diff_loss_x0 + gen_loss

            # 记录损失
            self.log('train_total_loss', total_loss, prog_bar=True)
            self.log('train_pitch_loss', pitch_loss, prog_bar=True)
            self.log('train_dur_loss', dur_loss, prog_bar=True)
            self.log('train_diff_loss_x0', diff_loss_x0, prog_bar=True)
            self.log('train_gen_loss', gen_loss, prog_bar=True)
            self.log('balance_weight', balance_weight, prog_bar=True)
            return total_loss

        if (batch_idx ) % 2 == 1:
            # 判别器步骤
            # 允许判别器参数更新
            for param in self.discriminator.parameters():
                param.requires_grad = True

            # 不计算生成器的梯度
            with torch.no_grad():
                diff_out, prior_out = self.forward(batch)
                latent_gen = diff_out['x0_pred']
                latent_gt = diff_out['x0_gt']
    
                wav_gen = self.model.soundstream.decoder_2(latent_gen*self.model.latent_norm)  
                wav_gt = self.model.soundstream.decoder_2(latent_gt*self.model.latent_norm)
                wav_cond =   self.model.soundstream.decode(batch['ref_code'].long().transpose(0,1))    


            # 提取生成音频的特征
            y_rec = self.wavlm(wav_gen.squeeze(1), output_hidden_states=True).hidden_states
            y_rec = torch.stack(y_rec, dim=1).transpose(-1, -2).flatten(start_dim=1, end_dim=2)

            # 提取条件音频的特征
            y_cond = self.wavlm(wav_cond.squeeze(1), output_hidden_states=True).hidden_states
            y_cond = torch.stack(y_cond, dim=1).transpose(-1, -2).flatten(start_dim=1, end_dim=2)
            y_cond = y_cond.mean(2)

            # 提取真实音频的特征
            y_gt = self.wavlm(wav_gt.squeeze(1), output_hidden_states=True).hidden_states
            y_gt = torch.stack(y_gt, dim=1).transpose(-1, -2).flatten(start_dim=1, end_dim=2)

            # 计算判别器的损失
            loss_d = self.discriminator.discriminator(y_gt.detach(), y_rec.detach(), y_cond.detach())

            # 记录损失
            self.log('train_dis_loss', loss_d, prog_bar=True)

            return loss_d

    def validation_step(self, batch, batch_idx):
        diff_out, prior_out = self.forward(batch)
        # 计算损失
        pitch_loss = log_pitch_loss(prior_out["pitch_pred_log"], batch["pitch"], mask=batch["mask"])
        dur_loss = log_dur_loss(prior_out["dur_pred_log"], batch["duration"], mask=batch["phone_mask"])
        diff_loss_x0 = diff_out["ict_loss"].mean()
        total_loss = pitch_loss + dur_loss + diff_loss_x0

        # 记录损失
        self.log('val_total_loss', total_loss, prog_bar=True)
        self.log('val_pitch_loss', pitch_loss, prog_bar=True)
        self.log('val_dur_loss', dur_loss, prog_bar=True)
        self.log('val_diff_loss_x0', diff_loss_x0, prog_bar=True)

        return total_loss

    def configure_optimizers(self):
        # optimizer = AdamW(
        #     filter(lambda p: p.requires_grad, self.model.parameters()),
        #     **self.cfg.train.adam
        # )
        trainable_params = (
            list(self.model.parameters()) +
            list(self.discriminator.parameters())
        )

        # 过滤掉 requires_grad 为 False 的参数（如果有）
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, trainable_params),
            **self.cfg.train.adam
        )

        # scheduler = get_scheduler(
        #     self.cfg.train.lr_scheduler,
        #     optimizer=optimizer,
        #     num_warmup_steps=self.cfg.train.lr_warmup_steps,
        #     num_training_steps=self.cfg.train.num_train_steps,
        # )
        # scheduler_config = {
        #     'scheduler': scheduler,
        #     'interval': 'step',  # 或 'epoch'，取决于您的需求
        #     'frequency': 1,
        # }
        return [optimizer] #, [scheduler_config]

 
class NS2DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.Dataset = NS2Dataset
        # self.Dataset = NS2Dataset_New
        self.Collator = NS2Collator
        self.train_dataset = None
        self.val_dataset = None
        self.collate_fn = None

    def _build_dataset(self):
        return self.Dataset, self.Collator

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.Dataset, self.Collator = self._build_dataset()
            self.train_dataset = self.Dataset(self.cfg, self.cfg.dataset[0], is_valid=False)
            self.val_dataset = self.Dataset(self.cfg, self.cfg.dataset[0], is_valid=True)
            self.collate_fn = self.Collator(self.cfg)

    def train_dataloader(self):
        if self.cfg.train.use_dynamic_batchsize:
            print("Use Dynamic Batchsize for training...")
            batch_sampler = batch_by_size(
                self.train_dataset.num_frame_indices,
                self.train_dataset.get_num_frames,
                max_tokens=self.cfg.train.max_tokens*torch.cuda.device_count(),
                max_sentences=self.cfg.train.max_sentences*torch.cuda.device_count(),
                required_batch_size_multiple=torch.cuda.device_count(),
            )
            np.random.seed(980205)
            batches = np.random.shuffle(batch_sampler)
 

            # num_replicas = dist.get_world_size()
            
            # rank = dist.get_rank()
            # print("DDP, .....", num_replicas, rank, flush=True)
            # batches = [x[rank::num_replicas] for x in batches if len(x) % num_replicas == 0]
            num_replicas = dist.get_world_size()
            rank = dist.get_rank()
            batches = [
                x[
                    rank :: num_replicas
                ]
                for x in batch_sampler
                if len(x) % num_replicas == 0
            ]
            train_loader = DataLoader(
                self.train_dataset,
                collate_fn=self.collate_fn,
                num_workers=self.cfg.train.dataloader.num_worker,
                batch_sampler=VariableSampler(
                    batches, drop_last=False 
                ),
                pin_memory=True,
            )
        else:
            print("Use Fixed Batchsize for training...")
            train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.cfg.train.batch_size,
                shuffle=True,
                num_workers=self.cfg.train.dataloader.num_worker,
                collate_fn=self.collate_fn,
                pin_memory=False,
            )
        return train_loader

    def val_dataloader(self):
        if self.cfg.train.use_dynamic_batchsize:
            print("Use Dynamic Batchsize for validation...")
            batch_sampler = batch_by_size(
                self.val_dataset.num_frame_indices,
                self.val_dataset.get_num_frames,
                max_tokens=self.cfg.train.max_tokens*torch.cuda.device_count(),
                max_sentences=self.cfg.train.max_sentences*torch.cuda.device_count(),
                required_batch_size_multiple=torch.cuda.device_count(),
            )
            num_replicas = dist.get_world_size()
            batches = batch_sampler
            rank = dist.get_rank()
            print("DDP, .....", num_replicas, rank, flush=True)
            batches = [x[rank::num_replicas] for x in batches if len(x) % num_replicas == 0]

            val_loader = DataLoader(
                self.val_dataset,
                collate_fn=self.collate_fn,
                num_workers=self.cfg.train.dataloader.num_worker,
                batch_sampler=VariableSampler(
                    batches, drop_last=False 
                ),
                pin_memory=False,
            )
        else:
            print("Use Fixed Batchsize for validation...")
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.cfg.train.batch_size,
                shuffle=False,
                num_workers=self.cfg.train.dataloader.num_worker,
                collate_fn=self.collate_fn,
                pin_memory=self.cfg.train.dataloader.pin_memory,
            )
        # for i in val_loader:
        #     print(i)
        return val_loader


    # def train_dataloader(self):
    #     if self.cfg.train.use_dynamic_batchsize:
    #         print("Use Dynamic Batchsize for training...")
    #         batch_sampler = batch_by_size(
    #             self.train_dataset.num_frame_indices,
    #             self.train_dataset.get_num_frames,
    #             max_tokens=self.cfg.train.max_tokens,
    #             max_sentences=self.cfg.train.max_sentences,
    #             required_batch_size_multiple=1,
    #         )
    #         # np.random.seed(980205)
    #         np.random.shuffle(batch_sampler)

    #         variable_sampler = VariableSampler(
    #             batch_sampler, drop_last=False, use_random_sampler=True
    #         )

    #         # 如果使用分布式训练，需要确保每个进程获取不同的样本
    #         if self.trainer.world_size > 1:
    #             variable_sampler = DistributedSampler(
    #                 variable_sampler,
    #                 num_replicas=self.trainer.world_size,
    #                 rank=self.trainer.global_rank,
    #                 shuffle=True,
    #                 drop_last=self.cfg.train.dataloader.drop_last,
    #             )

    #         train_loader = DataLoader(
    #             self.train_dataset,
    #             collate_fn=self.collate_fn,
    #             num_workers=self.cfg.train.dataloader.num_worker,
    #             batch_sampler=variable_sampler,
    #             pin_memory=True,
    #         )
    #     else:
    #         print("Use Fixed Batchsize for training...")
    #         train_loader = DataLoader(
    #             self.train_dataset,
    #             batch_size=self.cfg.train.batch_size,
    #             shuffle=True,
    #             num_workers=self.cfg.train.dataloader.num_worker,
    #             collate_fn=self.collate_fn,
    #             pin_memory=self.cfg.train.dataloader.pin_memory,
    #         )
    #     return train_loader

    # def val_dataloader(self):
    #     if self.cfg.train.use_dynamic_batchsize:
    #         print("Use Dynamic Batchsize for validation...")
    #         batch_sampler = batch_by_size(
    #             self.val_dataset.num_frame_indices,
    #             self.val_dataset.get_num_frames,
    #             max_tokens=self.cfg.train.max_tokens,
    #             max_sentences=self.cfg.train.max_sentences,
    #             required_batch_size_multiple=1,
    #         )

    #         variable_sampler = VariableSampler(
    #             batch_sampler, drop_last=False, use_random_sampler=False
    #         )

    #         if self.trainer.world_size > 1:
    #             variable_sampler = DistributedSampler(
    #                 variable_sampler,
    #                 num_replicas=self.trainer.world_size,
    #                 rank=self.trainer.global_rank,
    #                 shuffle=False,
    #                 drop_last=self.cfg.train.dataloader.drop_last,
    #             )

    #         val_loader = DataLoader(
    #             self.val_dataset,
    #             collate_fn=self.collate_fn,
    #             num_workers=self.cfg.train.dataloader.num_worker,
    #             batch_sampler=variable_sampler,
    #             pin_memory=self.cfg.train.dataloader.pin_memory,
    #         )
    #     else:
    #         print("Use Fixed Batchsize for validation...")
    #         val_loader = DataLoader(
    #             self.val_dataset,
    #             batch_size=self.cfg.train.batch_size,
    #             shuffle=False,
    #             num_workers=self.cfg.train.dataloader.num_worker,
    #             collate_fn=self.collate_fn,
    #             pin_memory=self.cfg.train.dataloader.pin_memory,
    #         )
    #     return val_loader
        


  