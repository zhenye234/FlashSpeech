 

from typing import Sequence, Optional, Union
import sys
sys.path.append('/scratch/buildlam/speech_yz/codec_final')
import math
import random

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

 
from quantization  import ResidualVectorQuantizer#,VectorQuantize
 
 
import descriptaudiocodec.dac.model.dac  as dac2
 
 
class SoundStream(nn.Module):
    """ SoundStream model or EnCodec model.
    
    Args:
        n_filters (int): n_filters (int): Base width for the model.
        D (int): Intermediate representation dimension.
        target_bandwidths (Sequence[int]): Target bandwidths in K-bits/second.
        ratios (Sequence[int]): downsampling factors, whose multiplication is the hop size.
        sample_rate (int): wave sampling rate.
        bins (int): number of code words in a codebook.
        normalize (bool): audio normalization.

    """
    def __init__(
        self,
        n_filters: int = 32,
        D: int = 128,
        # target_bandwidths: Sequence[Union[int, float]] = [0.5, 1, 1.5, 2, 4, 6],
        target_bandwidths: Sequence[Union[int, float]] = [1, 1.5, 2, 4, 6],
        ratios: Sequence[int] = [8, 5, 4, 2], #  downsampling by 320
        sample_rate: int = 16000,
        bins: int = 1024,
        normalize: bool = False,
        causal: bool = False,
    ):
        super().__init__()
        self.hop_length = np.prod(ratios)
        # total nb of codebooks, e.g., 6Kb/s, sr=16000 and hop_length=320 => nq = 12
        n_q = int(1000 * target_bandwidths[-1] // (math.ceil(sample_rate / self.hop_length) * 10))
        self.frame_rate = math.ceil(sample_rate / np.prod(ratios)) # 50 Hz
        self.bits_per_codebook = int(math.log2(bins)) # 1024 => 10
        self.target_bandwidths = target_bandwidths
        self.n_q = n_q
        self.sample_rate = sample_rate

 
        self.encoder = dac2.Encoder(            64,ratios,D)
 
        self.quantizer = ResidualVectorQuantizer(dimension=D, n_q=n_q, bins=bins)
 
        self.decoder_2 = dac2.Decoder(            D,1024,ratios,)

     

    def get_last_layer(self):
        return self.decoder.layers[-1].weight
    
 

 
 
    def forward(self, x: torch.Tensor, feat: torch.Tensor,bw: int):

 
        e = self.encoder(x)
   
        quantized, codes, bandwidth, commit_loss  = self.quantizer(e, self.frame_rate, bw)
      
        o = self.decoder_2(quantized)
        
        return o, commit_loss, commit_loss,None
 
    def encode(self, x: torch.Tensor,feat: torch.Tensor, target_bw: Optional[int] = None) -> torch.Tensor:
 
        bw = target_bw
 
        e = self.encoder(x)
   
 
 
        quantized, codes, bandwidth, commit_loss  = self.quantizer(e, self.frame_rate, bw)
        return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        quantized = self.quantizer.decode(codes)
 
        o = self.decoder_2(quantized)
        return o

# test
if __name__ == '__main__':
    soundstream = SoundStream(n_filters=32, D=256).cuda(0)
 
    for i in range(10):
        print(f"Iter {i}: ")
        x = torch.rand(1, 1, 16000).cuda(0)
        feat= torch.rand(1, 1, 50, 160).cuda(0)
        o, commit_loss, distill_loss,_= soundstream(x,feat,soundstream.target_bandwidths[-1])
        print('output', o.shape)
