import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
from transformers import AutoModel
from torch.nn import Conv1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, spectral_norm
LRELU_SLOPE = 0.1
import numpy as np
 
    
 
class WavLMDiscriminatorcond(nn.Module):
    """docstring for Discriminator."""

    def __init__(self, slm_hidden=1024, 
                 slm_layers=25, 
                 initial_channel=128, #64 128
                 use_spectral_norm=False):
        super(WavLMDiscriminatorcond, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
 
        initial_channel = 256
        self.pre = norm_f(Conv1d(slm_hidden * slm_layers, initial_channel , 1, 1, padding=0))
        self.convs = nn.ModuleList([
            norm_f(nn.Conv1d(initial_channel , initial_channel * 2, kernel_size=5, padding=2)),
            norm_f(nn.Conv1d(initial_channel * 2, initial_channel * 4, kernel_size=5, padding=2)),
            norm_f(nn.Conv1d(initial_channel * 4, initial_channel * 4, 5, 1, padding=2)),
        ])
 

        self.conv_post = norm_f(Conv1d(initial_channel * 4, 1, 3, 1, padding=1))
        self.cmap_dim =25600
    def forward(self, x,cond):
 
        x = (x * cond.unsqueeze(-1))   * (1 / np.sqrt(self.cmap_dim))
        # x=x
        x = self.pre(x)
 
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        x = torch.flatten(x, 1, -1)

        return x

class WavLMLosscond(torch.nn.Module):

    def __init__(self,  ):
        super(WavLMLosscond, self).__init__()
 
        self.wd = WavLMDiscriminatorcond()
     
    def forward(self, wav, y_rec):
 
        floss = torch.mean(torch.abs(wav - y_rec))
        
        return floss.mean()
    
    def generator(self, y_rec,cond):
 
        y_df_hat_g = self.wd(y_rec,cond)
        loss_gen = torch.mean((1-y_df_hat_g)**2)
        
        return loss_gen
    
    def discriminator(self, wav, y_rec,cond):
 
        y_d_rs = self.wd(wav,cond)
        y_d_gs = self.wd(y_rec,cond)
        
        y_df_hat_r, y_df_hat_g = y_d_rs, y_d_gs
        
        r_loss = torch.mean((1-y_df_hat_r)**2)
        g_loss = torch.mean((y_df_hat_g)**2)
        
        loss_disc_f = r_loss + g_loss
                        
        return loss_disc_f.mean()

    def discriminator_forward(self, wav):
 
        y_d_rs = self.wd(wav)
        
        return y_d_rs

    def feature_matching(self,wav, y_rec,cond):
        y_d_rs = self.wd(wav,cond)
        y_d_gs = self.wd(y_rec,cond)

        d_loss = F.l1_loss(y_d_rs, y_d_gs)
        return d_loss