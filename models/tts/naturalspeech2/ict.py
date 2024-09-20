import sys
sys.path.append('/scratch/buildlam/speech_yz/Amphion2')
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.tts.naturalspeech2.wavenet import WaveNet

import math
from torch import Tensor


def improved_timesteps_schedule(
    current_training_step: int,
    total_training_steps: int,
    initial_timesteps: int = 10,
    final_timesteps: int = 1280,
) -> int:
    """Implements the improved timestep discretization schedule.

    Parameters
    ----------
    current_training_step : int
        Current step in the training loop.
    total_training_steps : int
        Total number of steps the model will be trained for.
    initial_timesteps : int, default=2
        Timesteps at the start of training.
    final_timesteps : int, default=150
        Timesteps at the end of training.

    Returns
    -------
    int
        Number of timesteps at the current point in training.

    References
    ----------
    [1] [Improved Techniques For Consistency Training](https://arxiv.org/pdf/2310.14189.pdf)
    """
    total_training_steps_prime = math.floor(
        total_training_steps
        / (math.log2(math.floor(final_timesteps / initial_timesteps)) + 1)
    )
    num_timesteps = initial_timesteps * math.pow(
        2, math.floor(current_training_step / total_training_steps_prime)
    )
    # total_training_steps_prime = math.floor(total_training_steps / (final_timesteps / initial_timesteps ))
    # beishu = final_timesteps / initial_timesteps
    # num_timesteps = (beishu * (current_training_step / total_training_steps)+1)*initial_timesteps

    num_timesteps = min(num_timesteps, final_timesteps) + 1
    # num_timesteps = initial_timesteps * math.floor(math.pow(
    #     2, current_training_step / total_training_steps_prime)
    # )
    # num_timesteps = min(num_timesteps, final_timesteps) + 1
    # num_timesteps =100
    return int(num_timesteps)

def karras_schedule(
    num_timesteps: int,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    device: torch.device = 'cpu',
) -> Tensor:
    """Implements the karras schedule that controls the standard deviation of
    noise added.

    Parameters
    ----------
    num_timesteps : int
        Number of timesteps at the current point in training.
    sigma_min : float, default=0.002
        Minimum standard deviation.
    sigma_max : float, default=80.0
        Maximum standard deviation
    rho : float, default=7.0
        Schedule hyper-parameter.
    device : torch.device, default=None
        Device to generate the schedule/sigmas/boundaries/ts on.

    Returns
    -------
    Tensor
        Generated schedule/sigmas/boundaries/ts.
    """
    rho_inv = 1.0 / rho
    # Clamp steps to 1 so that we don't get nans
    steps = torch.arange(num_timesteps, device=device) / max(num_timesteps - 1, 1)
    sigmas = sigma_min**rho_inv + steps * (
        sigma_max**rho_inv - sigma_min**rho_inv
    )
    sigmas = sigmas**rho

    return sigmas


def lognormal_timestep_distribution(
    num_samples: int,
    sigmas: Tensor,
    mean: float = -1.1,
    std: float = 2.0,
) -> Tensor:
    """Draws timesteps from a lognormal distribution.

    Parameters
    ----------
    num_samples : int
        Number of samples to draw.
    sigmas : Tensor
        Standard deviations of the noise.
    mean : float, default=-1.1
        Mean of the lognormal distribution.
    std : float, default=2.0
        Standard deviation of the lognormal distribution.

    Returns
    -------
    Tensor
        Timesteps drawn from the lognormal distribution.

    References
    ----------
    [1] [Improved Techniques For Consistency Training](https://arxiv.org/pdf/2310.14189.pdf)
    """
    pdf = torch.erf((torch.log(sigmas[1:]) - mean) / (std * math.sqrt(2))) - torch.erf(
        (torch.log(sigmas[:-1]) - mean) / (std * math.sqrt(2))
    )
    pdf = pdf / pdf.sum()

    timesteps = torch.multinomial(pdf, num_samples, replacement=True)

    return timesteps

def get_current_next_sigma(current_training_step,total_training_steps,batch_size,is_fixed=False):

    if current_training_step ==None:
        current_training_step=0

    # if hparams['is_fixed']:
    if is_fixed:
        num_timesteps = 1281
        # num_timesteps = 81
 
   
    else:
        num_timesteps = improved_timesteps_schedule(
            current_training_step,
            total_training_steps,

        )
 
    sigmas = karras_schedule(
        num_timesteps 
    )


    timesteps = lognormal_timestep_distribution(
        batch_size, sigmas 
    )

    current_sigmas = sigmas[timesteps]
    next_sigmas = sigmas[timesteps + 1]
 
    return current_sigmas,next_sigmas,num_timesteps


class Ict(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.diff_estimator = WaveNet(cfg.wavenet)
 
 
        self.sigma_data =0.5 
 
        self.sigma_min= 0.002
        self.sigma_max= 80
        self.rho=7
        self.all_ict_training_steps = cfg.all_steps
        self.is_fixed = cfg.is_fixed
    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)
    def EDMPrecond(self, x, sigma ,x_mask,cond,   spk ):
 
 
        sigma = sigma.reshape(-1, 1, 1 )
 
        c_skip = self.sigma_data ** 2 / ((sigma-self.sigma_min) ** 2 + self.sigma_data ** 2)
        c_out = (sigma-self.sigma_min) * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise =  sigma.log() / 4
 
 
        # F_x = self.estimator(spec=(c_in * x), x_mask=x_mask, cond=mu, diffusion_step=c_noise.flatten(), spk=spk, prompt=prompt, prompt_mask=prompt_mask)
        F_x = self.diff_estimator(c_in * x, x_mask, cond, c_noise.flatten() , spk)

        D_x = c_skip * x + c_out * (F_x  )
        return D_x

    def CTLoss_T(self,x_start, x_mask,cond, spk,  global_steps=None): 
        z = torch.randn_like(x_start) 
 
        tn ,tn_1 ,num_timesteps= get_current_next_sigma(global_steps,self.all_ict_training_steps ,x_start.shape[0],self.is_fixed)
 
        tn_1 = tn_1.reshape(-1, 1,   1).to(x_start.device)
        tn = tn.reshape(-1, 1,   1).to(x_start.device)
 

        y_tn_1 = x_start + tn_1 * z  
        # f_theta = self.EDMPrecond(y_tn_1, tn_1 ,x_mask,cond,   spk=spk, pos_ids=pos_ids, prompt=prompt, prompt_mask=prompt_mask)
        f_theta = self.EDMPrecond(y_tn_1, tn_1 ,x_mask,cond, spk)
        with torch.no_grad():

            y_tn = x_start  +  tn  * z 
 
            f_theta_ema = self.EDMPrecond(y_tn, tn ,x_mask,cond,   spk)

        lamda_n = 1./(tn_1-tn )
 
        loss = torch.sqrt ((f_theta - f_theta_ema.detach())** 2 +9e-4)-3e-2
 
        loss = lamda_n * (  loss)
 
        # diff_out = {"x0_pred": f_theta, "loss": loss, "noise": z}
        # return f_theta,loss,z
        diff_out = {"x0_pred": f_theta, "ict_loss": loss, "noise": z,"x0_gt":x_start}
        return diff_out 

    def get_t_steps(self,N):
        N=N+1
        step_indices = torch.arange( N )  
        t_steps = (self.sigma_min ** (1 / self.rho) + step_indices / (N- 1) * (self.sigma_max ** (1 / self.rho) - self.sigma_min ** (1 / self.rho))) ** self.rho

        return  t_steps.flip(0)
    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

    def ct_sampler(self, z, x_mask,cond, n_timesteps, spk ): 
        t_steps = n_timesteps
        latents=z 
        if t_steps ==1:
            t_steps=[80  ]
        else:
            t_steps=self.get_t_steps(t_steps)
        # t_steps=[80,hparams['temp'],0]
        t_steps = torch.as_tensor(t_steps).to(latents.device)
        latents = latents * t_steps[0]
        x = self.EDMPrecond(latents, t_steps[0] ,x_mask,cond,   spk)        
        for t in t_steps[1:-1]:
            z = torch.randn_like(x)  
 
            x_tn = x +  t*z
            x = self.EDMPrecond(x_tn, t ,x_mask,cond,   spk=spk)
 
        return x
 
    @torch.no_grad()
    def reverse_diffusion(self, z, x_mask,cond, n_timesteps, spk):

        return self.ct_sampler(z=z, x_mask=x_mask,cond=cond, n_timesteps=n_timesteps, spk=spk)
   
    def forward(self, x0, x_mask,cond, spk=None,global_steps=None):
 
        loss = self.CTLoss_T(x0, x_mask,cond,   spk=spk, global_steps=global_steps)
        return loss  


def main():
 
    from utils.util import load_config
    cfg = load_config('egs/tts/NaturalSpeech2/exp_config.json')
    # 初始化 Diffusion 对象
    diffusion = Ict(cfg.model.diffusion)

    # 创建模拟输入数据
    batch_size = 1
    feature_dim = 256
    seq_len = 100
    cond_dim = 512
    query_emb_dim = 512
    query_emb_num = 32

    x = torch.randn(batch_size, feature_dim, seq_len)  # 模拟音频特征数据
    x_mask = torch.ones(batch_size, seq_len).bool()  # 假设所有时间步都有效
    cond = torch.randn(batch_size, seq_len, cond_dim)  # 条件向量
    spk_query_emb = torch.randn(batch_size, query_emb_num, query_emb_dim)  # 说话人查询嵌入

    # 执行前向传播
    output = diffusion.forward(x, x_mask, cond, spk_query_emb,global_steps=10000)

    # 打印输出结果
    print("Output:", output)

if __name__ == "__main__":
    main()
