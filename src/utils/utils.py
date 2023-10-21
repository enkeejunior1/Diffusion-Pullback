import json
import os
import types
import time
import gc
from tqdm import tqdm

import math
import numpy as np
import torch
import torch.nn as nn

from einops import rearrange, einsum

########
# path #
########
from configs.paths import (
    DATASET_PATHS,
    # PROMPT_PATHS,
    MODEL_PATHS,
)

####################
# uncond diffusion #
####################
from diffusers import DDIMScheduler, DDIMPipeline

# from models.guided_diffusion.script_util import g_DDPM

def get_custom_diffusion_scheduler(args):
    '''
    DDIM scheduler
    '''
    if args.use_yh_custom_scheduler:
        scheduler = YHCustomScheduler(args)

    elif 'HF' in args.model_name:
        scheduler = None

    else:
        scheduler = DDIMScheduler(
            num_train_timesteps     = args.config.diffusion.num_diffusion_timesteps,
            beta_start              = args.config.diffusion.beta_start,
            beta_end                = args.config.diffusion.beta_end,
            beta_schedule           = args.config.diffusion.beta_schedule,
            trained_betas           = None,
            clip_sample             = False, 
            set_alpha_to_one        = False,    # follow Stable-Diffusion setting. NOTE : need to verify this setting 
            steps_offset            = 1,        # follow Stable-Diffusion setting. NOTE : need to verify this setting
            prediction_type         = "epsilon",
        )

    return scheduler
    
def get_custom_diffusion_model(args):
    '''
    CelebA_HQ, LSUN, AFHQ

    TODO : IMAGENET
    '''
    # pretrained weight url
    if args.model_name == "CelebA_HQ":
        url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt"
        init_ckpt = torch.hub.load_state_dict_from_url(
            url, map_location=args.device
        )
    elif args.model_name in ['LSUN_bedroom', 'LSUN_cat', 'LSUN_horse']:
        init_ckpt = torch.load(MODEL_PATHS[args.model_name], map_location=args.device)
    elif args.model_name in ["ImageNet256Uncond", "ImageNet64Uncond", "ImageNet256Cond", "ImageNet128Cond", "ImageNet64Cond", "CIFAR10Uncond"]:
        init_ckpt = torch.load(MODEL_PATHS[args.model_name], map_location=args.device)
    elif args.model_name == "CelebA_HQ_HF":
        model_id = "google/ddpm-ema-celebahq-256"
    elif args.model_name == "LSUN_church_HF":
        model_id = "google/ddpm-ema-church-256"
    elif args.model_name == "LSUN_bedroom_HF":
        model_id = "google/ddpm-ema-bedroom-256"
    elif args.model_name == "FFHQ_HF":
        model_id = "google/ncsnpp-ffhq-256"
    elif args.model_name in ["FFHQ_P2", "AFHQ_P2", "Flower_P2"]:
        init_ckpt = torch.load(MODEL_PATHS[args.model_name])
    else:
        raise ValueError()

    # load model and weight
    if args.model_name in ["CelebA_HQ"]:
        model = PullBackDDPM(args)
        model.learn_sigma = False
        model.load_state_dict(init_ckpt)

    elif args.model_name in ["FFHQ_P2", "AFHQ_P2", "Flower_P2"]:
        model = g_DDPM(args)
        model.learn_sigma = True
        model.load_state_dict(init_ckpt)

    elif args.model_name in ["ImageNet256Uncond", "ImageNet64Uncond", "CIFAR10Uncond", "ImageNet256Cond", "ImageNet128Cond", "ImageNet64Cond", "LSUN_bedroom", "LSUN_cat", "LSUN_horse"]:
        model = g_DDPM(args)
        model.learn_sigma = True
        model.load_state_dict(init_ckpt)

    elif args.model_name in ["CelebA_HQ_HF", "LSUN_bedroom_HF", "LSUN_church_HF", "FFHQ_HF"]:
        model = DDIMPipeline.from_pretrained(model_id) 
        model.unet.get_h                        = types.MethodType(get_h_uncond, model.unet)
        model.unet.local_encoder_pullback_xt    = types.MethodType(local_encoder_pullback_xt, model.unet)

    else:
        raise ValueError('Not implemented dataset')

    # load weight to model
    model = model.to(args.device)
    return model

# monkey patch (local method - uncond)
def get_h_uncond(
        self, x=None, t=None, op=None, block_idx=None, verbose=False, **kwargs,
    ):
    '''
    Args
        - x : zt
        - op : ['down', 'mid', 'up']
        - block_idx : op == down, up : [0,1,2,3], op == mid : [0]
    Returns
        - h : hidden feature
    '''
    # assert (after_res_or_sa) == ((op == 'down') & (block_idx == 1)), 'after_res_or_sa is only implemented for 16x16 resolution yet'
    # assert (op == 'mid') & (block_idx == 0), 'mid block is only implemented yet'

    # 1. time
    timesteps = t
    if not torch.is_tensor(timesteps):
        timesteps = torch.tensor([timesteps], dtype=torch.long, device=x.device)
    elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
        timesteps = timesteps[None].to(x.device)

    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
    timesteps = timesteps * torch.ones(x.shape[0], dtype=timesteps.dtype, device=timesteps.device)
    t_emb = self.time_proj(timesteps)
    t_emb = t_emb.to(dtype=self.dtype)
    emb = self.time_embedding(t_emb)

    # 2. pre-process
    skip_sample = x
    x = self.conv_in(x)

    # 3. down
    down_block_res_samples = (x,)
    for downsample_block in self.down_blocks:
        if hasattr(downsample_block, "skip_conv"):
            x, res_samples, skip_sample = downsample_block(
                hidden_states=x, temb=emb, skip_sample=skip_sample
            )
        else:
            x, res_samples = downsample_block(hidden_states=x, temb=emb)
        down_block_res_samples += res_samples

    # 4. mid
    x = self.mid_block(x, emb)
    if (op == 'mid') & (block_idx == 0):
        if verbose:
            print(f'op : {op}, block_idx : {block_idx}, return h.shape : {x.shape}')
        return x

    raise ValueError(f'(op, block_idx) = ({op, block_idx}) is not valid')

def local_encoder_pullback_xt(
        self, x, t, op=None, block_idx=None,
        pca_rank=50, chunk_size=25, min_iter=10, max_iter=100, convergence_threshold=1e-3,
    ):
    '''
    Args
        - x : zt
        - op : ['down', 'mid', 'up']
        - block_idx : op == down, up : [0,1,2,3], op == mid : [0]
        - pooling : ['pixel-sum', 'channel-sum', 'single-channel', 'multiple-channel']
    Returns
        - h : hidden feature
    '''
    num_chunk = pca_rank // chunk_size if pca_rank % chunk_size == 0 else pca_rank // chunk_size + 1

    # get h samples
    time_s = time.time()

    # necessary variables
    h_shape = self.get_h(
        x, t=t, op=op, block_idx=block_idx, 
    ).shape

    c_i, w_i, h_i = x.size(1), x.size(2), x.size(3)
    c_o, w_o, h_o = h_shape[1], h_shape[2], h_shape[3]

    a = torch.tensor(0., device=x.device, dtype=x.dtype)

    # Algorithm 1
    vT = torch.randn(c_i*w_i*h_i, pca_rank, device=x.device, dtype=torch.float)
    vT, _ = torch.linalg.qr(vT)
    v = vT.T
    v = v.view(-1, c_i, w_i, h_i)

    time_s = time.time()
    for i in range(max_iter):
        v = v.to(device=x.device, dtype=x.dtype)
        v_prev = v.detach().cpu().clone()
        
        
        u = []
        v_buffer = list(v.chunk(num_chunk))
        for vi in v_buffer:
            # g = lambda a : get_h(x + a*vi)

            g = lambda a : self.get_h(
                x + a*vi, t=t, op=op, block_idx=block_idx, 
            )

            ui = torch.func.jacfwd(g, argnums=0, has_aux=False, randomness='error')(a)
            u.append(ui.detach().cpu().clone())
        u = torch.cat(u, dim=0)
        u = u.to(x.device, x.dtype)

        # time_s = time.time()
        # g = lambda a : get_h(x + a*v)
        # u = torch.func.jacfwd(g, argnums=0, has_aux=False, randomness='error')(a)
        # time_e = time.time()
        # print('single vi jacfwd t ==', time_e - time_s)

        g = lambda x : einsum(
            u, self.get_h(
                x, t=t, op=op, block_idx=block_idx, 
            ), 'b c w h, i c w h -> b'
        )
        v_ = torch.autograd.functional.jacobian(g, x)
        v_ = v_.view(-1, c_i*w_i*h_i)

        _, s, v = torch.linalg.svd(v_, full_matrices=False)
        v = v.view(-1, c_i, w_i, h_i)
        u = u.view(-1, c_o, w_o, h_o)
        
        convergence = torch.dist(v_prev, v.detach().cpu()).item()
        print(f'power method : {i}-th step convergence : ', convergence)
        
        if torch.allclose(v_prev, v.detach().cpu(), atol=convergence_threshold) and (i > min_iter):
            print('reach convergence threshold : ', convergence)
            break

    time_e = time.time()
    print('power method runtime ==', time_e - time_s)

    # u, s, vT = u.view(-1, c_o*w_o*h_o).T.detach(), s.sqrt().detach(), v.view(-1, c_i*w_i*h_i).detach()
    u, s, vT = u.reshape(-1, c_o*w_o*h_o).T.detach(), s.sqrt().detach(), v.reshape(-1, c_i*w_i*h_i).detach()
    return u, s, vT

####################
# stable-diffusion #
####################
from diffusers import (
    StableDiffusionPipeline, 
    DDIMScheduler,
)

# from models.ddpm.diffusion import PullBackDDPM

def get_stable_diffusion_scheduler(args, scheduler):
    # monkey patch (replace scheduler for better inversion quality)
    if args.use_yh_custom_scheduler:
        scheduler.t_max = 999
        scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device=args.device, dtype=args.dtype)
        scheduler.betas = scheduler.betas.to(device=args.device, dtype=args.dtype)
        scheduler.set_timesteps = types.MethodType(set_timesteps, scheduler)
        scheduler.step = types.MethodType(step, scheduler)
    else:
        pass
    return scheduler

def set_timesteps(self, num_inferences, device=None, is_inversion=False):
    device = 'cpu' if device is None else device
    if is_inversion:
        seq = torch.linspace(0, 1, num_inferences, device=device) * self.t_max
        seq = seq + 1e-6
        seq_prev = torch.cat([torch.tensor([-1], device=device), seq[:-1]], dim = 0)
        self.timesteps = seq_prev[1:]
        self.timesteps_next = seq[1:]
    
    else:
        seq = torch.linspace(0, 1, num_inferences, device=device) * self.t_max
        seq_prev = torch.cat([torch.tensor([-1], device=device), seq[:-1]], dim = 0)
        self.timesteps = reversed(seq[1:])
        self.timesteps_next = reversed(seq_prev[1:])
    
def step(self, et, t, xt, eta=0.0, **kwargs):
    '''
    Notation
        - a : alpha / b : beta / e : epsilon
    '''
    t_idx   = self.timesteps.tolist().index(t)
    t_next  = self.timesteps_next[t_idx]

    # extract need parameters : at, at_next
    at = extract(self.alphas_cumprod, t, xt.shape)
    at_next = extract(self.alphas_cumprod, t_next, xt.shape)

    # DDIM step ; xt-1 = sqrt(at-1 / at) (xt - sqrt(1-at)*e(xt, t)) + sqrt(1-at-1)*e(xt, t)
    P_xt = (xt - et * (1 - at).sqrt()) / at.sqrt()

    # Deterministic. (ODE)
    if eta == 0:
        D_xt = (1 - at_next).sqrt() * et
        xt_next = at_next.sqrt() * P_xt + D_xt

    # Add noise. (SDE)
    else:
        sigma_t = ((1 - at / (at_next)) * (1 - at_next) / (1 - at)).sqrt()

        D_xt = (1 - at_next - eta * sigma_t ** 2).sqrt() * et
        xt_next = at_next.sqrt() * P_xt + D_xt + eta * sigma_t * torch.randn_like(xt)

    return SchedulerOutput(xt_next, P_xt)

def get_stable_diffusion_model(args):
    # load from hf
    model = StableDiffusionPipeline.from_pretrained(args.model_name, torch_dtype=args.dtype)
    model = model.to(torch_device = args.device)

    # turn-off xformers memory efficient attention for using forward AD
    model.disable_xformers_memory_efficient_attention()

    # monkey patch (basic method)
    model.unet.get_h                = types.MethodType(get_h, model.unet)
    model.unet.get_h_to_e           = types.MethodType(get_h_to_e, model.unet)
    model.unet.forward_dh           = types.MethodType(forward_dh, model.unet)
    model.unet.inv_jac_zt           = types.MethodType(inv_jac_zt, model.unet)
    
    # monkey patch (local method)
    # model.unet.local_pca_zt                 = types.MethodType(local_pca_zt, model.unet)
    model.unet.local_encoder_pullback_zt    = types.MethodType(local_encoder_pullback_zt, model.unet)
    # model.unet.local_decoder_pullback_zt    = types.MethodType(local_decoder_pullback_zt, model.unet)

    # monkey patch (global method)
    model.unet.global_pca_zt    = types.MethodType(global_pca_zt, model.unet)
    
    # change scheduler
    model.scheduler = DDIMScheduler.from_config(model.scheduler.config)

    # change config (TODO need strict verification)
    model.vae.config.sample_size = 512

    # assert config
    assert model.scheduler.config.prediction_type == 'epsilon'
    return model

# monkey patch (basic method)
def forward_dh(
        self, sample=None, timestep=None, encoder_hidden_states=None, cross_attention_kwargs=None,
        op=None, block_idx=None, uk=None, verbose=False, 
    ):
    '''
    Args
        - sample : zt
        - op : ['down', 'mid', 'up']
        - block_idx : op == down, up : [0,1,2,3], op == mid : [0]
    Returns
        - h : hidden feature
    '''
    # time
    timesteps = timestep
    if not torch.is_tensor(timesteps):
        # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
        # This would be a good case for the `match` statement (Python 3.10+)
        is_mps = sample.device.type == "mps"
        if isinstance(timestep, float):
            dtype = torch.float32 if is_mps else torch.float64
        else:
            dtype = torch.int32 if is_mps else torch.int64
        timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
    elif len(timesteps.shape) == 0:
        timesteps = timesteps[None].to(sample.device)

    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
    timesteps = timesteps.expand(sample.shape[0])
    t_emb = self.time_proj(timesteps)
    t_emb = t_emb.to(dtype=self.dtype)
    emb = self.time_embedding(t_emb)

    # pre-process
    sample = self.conv_in(sample)

    # down
    down_block_res_samples = (sample,)
    for down_block_idx, downsample_block in enumerate(self.down_blocks):
        if (op == 'down') & (block_idx == down_block_idx) & after_res_or_sa:
            sample, res_samples, _ = down_block_forward(
                downsample_block, hidden_states=sample, temb=emb, encoder_hidden_states=encoder_hidden_states, 
                uk=uk, after_res=self.after_res, after_sa=self.after_sa,
            )
        elif hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
            sample, res_samples = downsample_block(
                hidden_states=sample, temb=emb, encoder_hidden_states=encoder_hidden_states,
            )
        else:
            sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
        
        if (op == 'down') & (block_idx == down_block_idx) & (not after_res_or_sa):
            sample += uk.view(-1, *sample.shape[1:])

        down_block_res_samples += res_samples

    # mid
    sample = self.mid_block(sample, emb, encoder_hidden_states=encoder_hidden_states)
    if (op == 'mid') & (block_idx == 0):
        sample += uk.view(-1, *sample.shape[1:])
    
    # up
    for up_block_idx, upsample_block in enumerate(self.up_blocks):
        res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
        down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

        if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
            sample = upsample_block(
                hidden_states=sample,
                temb=emb,
                res_hidden_states_tuple=res_samples,
                encoder_hidden_states=encoder_hidden_states,
            )
        else:
            sample = upsample_block(
                hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=None,
            )

        # return h
        if (op == 'up') & (block_idx == up_block_idx):
            sample += uk.view(-1, *sample.shape[1:])

    # 6. post-process
    if self.conv_norm_out:
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    return sample

def get_h(
        self, sample=None, timestep=None, encoder_hidden_states=None, 
        op=None, block_idx=None, verbose=False,
    ):
    '''
    Args
        - sample : zt
        - op : ['down', 'mid', 'up']
        - block_idx : op == down, up : [0,1,2,3], op == mid : [0]
    Returns
        - h : hidden feature
    '''
    # time
    timesteps = timestep
    if not torch.is_tensor(timesteps):
        # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
        # This would be a good case for the `match` statement (Python 3.10+)
        is_mps = sample.device.type == "mps"
        if isinstance(timestep, float):
            dtype = torch.float32 if is_mps else torch.float64
        else:
            dtype = torch.int32 if is_mps else torch.int64
        timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
    elif len(timesteps.shape) == 0:
        timesteps = timesteps[None].to(sample.device)

    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
    timesteps = timesteps.expand(sample.shape[0])
    t_emb = self.time_proj(timesteps)
    t_emb = t_emb.to(dtype=self.dtype)
    emb = self.time_embedding(t_emb)

    # pre-process
    sample = self.conv_in(sample)

    # down
    down_block_res_samples = (sample,)
    for down_block_idx, downsample_block in enumerate(self.down_blocks):
        if (op == 'down') & (block_idx == down_block_idx):
            sample, res_samples, h_space = down_block_forward(
                downsample_block, hidden_states=sample, temb=emb, encoder_hidden_states=encoder_hidden_states, timesteps=timestep, uk=None, 
            )
            return h_space
        
        elif hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
            sample, res_samples = downsample_block(
                hidden_states=sample, temb=emb, encoder_hidden_states=encoder_hidden_states,
            )
        else:
            sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

        if (op == 'down') & (block_idx == down_block_idx):
            return sample

        down_block_res_samples += res_samples

    # mid
    sample = self.mid_block(sample, emb, encoder_hidden_states=encoder_hidden_states)
    if (op == 'mid') & (block_idx == 0):
        if verbose:
            print(f'op : {op}, block_idx : {block_idx}, return h.shape : {sample.shape}')
        return sample
    
    # up
    for up_block_idx, upsample_block in enumerate(self.up_blocks):
        res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
        down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

        if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
            sample = upsample_block(
                hidden_states=sample,
                temb=emb,
                res_hidden_states_tuple=res_samples,
                encoder_hidden_states=encoder_hidden_states,
            )
        else:
            sample = upsample_block(
                hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=None,
            )

        if verbose:
            print(f'down_block_idx : {down_block_idx}, sample.shape : {sample.shape}')

        # return h
        if (op == 'up') & (block_idx == up_block_idx):
            if verbose:
                print(f'op : {op}, block_idx : {block_idx}, return h.shape : {sample.shape}')
            return sample

    raise ValueError(f'(op, block_idx) = ({op, block_idx}) is not valid')

def get_h_to_e(
        self, sample=None, timestep=None, encoder_hidden_states=None, 
        input_h=None, op=None, block_idx=None, verbose=False,
    ):
    '''
    Args
        - sample : zt
        - op : ['down', 'mid', 'up']
        - block_idx : op == down, up : [0,1,2,3], op == mid : [0]
    Returns
        - h : hidden feature
    '''
    after_res_or_sa = self.after_res or self.after_sa
    # assert (after_res_or_sa) == ((op == 'down') & (block_idx == 1)), 'after_res_or_sa is only implemented for 16x16 resolution yet'
    # assert (op == 'mid') & (block_idx == 0), 'mid block is only implemented yet'
    assert after_res_or_sa != True, 'after_res_or_sa not implemented yet'
    assert op in ['mid', 'down'], 'up block is not implemented yet'

    pca_rank = input_h.size(0)

    # time
    timesteps = timestep
    if not torch.is_tensor(timesteps):
        # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
        # This would be a good case for the `match` statement (Python 3.10+)
        is_mps = sample.device.type == "mps"
        if isinstance(timestep, float):
            dtype = torch.float32 if is_mps else torch.float64
        else:
            dtype = torch.int32 if is_mps else torch.int64
        timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
    elif len(timesteps.shape) == 0:
        timesteps = timesteps[None].to(sample.device)

    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
    timesteps = timesteps.expand(sample.shape[0])
    t_emb = self.time_proj(timesteps)
    t_emb = t_emb.to(dtype=self.dtype)
    emb = self.time_embedding(t_emb)

    # pre-process
    sample = self.conv_in(sample)

    # down
    down_block_res_samples = (sample,)
    for down_block_idx, downsample_block in enumerate(self.down_blocks):
        if (op == 'down') & (block_idx == down_block_idx) & after_res_or_sa:
            sample, res_samples, h_space = down_block_forward(
                downsample_block, hidden_states=sample, temb=emb, encoder_hidden_states=encoder_hidden_states, timesteps=timestep, uk=None, after_res=self.after_res, after_sa=self.after_sa,
            )
            assert (sample == res_samples[-1]).all()
            sample = input_h.view(pca_rank, *sample.shape[1:])
            res_samples[-1] = input_h.view(pca_rank, *res_samples[-1].shape[1:])
            res_samples = [res_sample.repeat(pca_rank, 1, 1, 1) for res_sample in res_samples]

        elif hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
            sample, res_samples = downsample_block(
                hidden_states=sample, temb=emb, encoder_hidden_states=encoder_hidden_states,
            )
        else:
            sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

        if (op == 'down') & (block_idx == down_block_idx) & (not after_res_or_sa):
            assert (sample == res_samples[-1]).all()
            sample = input_h.view(pca_rank, *sample.shape[1:])
            res_samples[-1] = input_h.view(pca_rank, *res_samples[-1].shape[1:])
            res_samples = [res_sample.repeat(pca_rank, 1, 1, 1) for res_sample in res_samples]
            encoder_hidden_states = encoder_hidden_states.repeat(pca_rank, 1, 1)

        down_block_res_samples += res_samples

    # mid
    sample = self.mid_block(sample, emb, encoder_hidden_states=encoder_hidden_states)
    
    if (op == 'mid') & (block_idx == 0):
        sample = input_h.view(pca_rank, *sample.shape[1:])
        down_block_res_samples = [res_sample.repeat(pca_rank, 1, 1, 1) for res_sample in down_block_res_samples]
        encoder_hidden_states = encoder_hidden_states.repeat(pca_rank, 1, 1)
    
    # up
    for up_block_idx, upsample_block in enumerate(self.up_blocks):
        res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
        down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

        if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
            sample = upsample_block(
                hidden_states=sample,
                temb=emb,
                res_hidden_states_tuple=res_samples,
                encoder_hidden_states=encoder_hidden_states,
            )
        else:
            sample = upsample_block(
                hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=None,
            )

        # return h
        if (op == 'up') & (block_idx == up_block_idx):
            sample = input_h.view(pca_rank, *sample.shape[1:])
            res_samples = [res_sample.repeat(pca_rank, 1, 1, 1) for res_sample in res_samples]

    # 6. post-process
    if self.conv_norm_out:
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    return sample

# non-monkey patch (basic method)
def down_block_forward(down_block, hidden_states, temb, encoder_hidden_states, timestep, uk=None, after_res=False, after_sa=False):
    assert after_res != after_sa
    
    # TODO(Patrick, William) - attention mask is not used
    output_states = ()

    for resnet, attn in zip(down_block.resnets, down_block.attentions):
        hidden_states = resnet(hidden_states, temb)

        if after_res:
            h_space = hidden_states.clone()
            
            if uk is not None:
                hidden_states = hidden_states + uk.view(-1, *hidden_states.shape[1:])
            
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=None,
            ).sample

        elif after_sa:
            # 1. Input
            if attn.is_input_continuous:
                batch, _, height, width = hidden_states.shape
                residual = hidden_states

                hidden_states = attn.norm(hidden_states)
                if not attn.use_linear_projection:
                    hidden_states = attn.proj_in(hidden_states)
                    inner_dim = hidden_states.shape[1]
                    hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
                else:
                    inner_dim = hidden_states.shape[1]
                    hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
                    hidden_states = attn.proj_in(hidden_states)
            elif attn.is_input_vectorized:
                hidden_states = attn.latent_image_embedding(hidden_states)
            elif attn.is_input_patches:
                hidden_states = attn.pos_embed(hidden_states)

            # 2. Blocks
            for block_idx, block in enumerate(attn.transformer_blocks):
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timestep,
                    cross_attention_kwargs=None,
                    class_labels=None,
                )

                if block_idx == 1:
                    h_space = hidden_states.clone()
                    
                    if uk is not None:
                        hidden_states = hidden_states + uk.view(-1, *hidden_states.shape[1:])

            # 3. Output
            if attn.is_input_continuous:
                if not attn.use_linear_projection:
                    hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
                    hidden_states = attn.proj_out(hidden_states)
                else:
                    hidden_states = attn.proj_out(hidden_states)
                    hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()

                output = hidden_states + residual
            elif attn.is_input_vectorized:
                raise ValueError()
            elif attn.is_input_patches:
                raise ValueError()
            hidden_states = output
            
        output_states += (hidden_states,)

    if down_block.downsamplers is not None:
        for downsampler in down_block.downsamplers:
            hidden_states = downsampler(hidden_states)

        output_states += (hidden_states,)
    
    return hidden_states, output_states, h_space

# monkey patch (local method)
def local_encoder_pullback_zt(
        self, sample, timestep, encoder_hidden_states=None, op=None, block_idx=None,
        pca_rank=50, chunk_size=25, min_iter=10, max_iter=100, convergence_threshold=1e-3,
    ):
    '''
    Args
        - sample : zt
        - op : ['down', 'mid', 'up']
        - block_idx : op == down, up : [0,1,2,3], op == mid : [0]
        - pooling : ['pixel-sum', 'channel-sum', 'single-channel', 'multiple-channel']
    Returns
        - h : hidden feature
    '''
    # get h samples
    time_s = time.time()

    # necessary variables
    h_shape = self.get_h(
        sample, timestep=timestep, encoder_hidden_states=encoder_hidden_states,
        op=op, block_idx=block_idx, 
    ).shape

    c_i, w_i, h_i = sample.size(1), sample.size(2), sample.size(3)
    c_o, w_o, h_o = h_shape[1], h_shape[2], h_shape[3]

    a = torch.tensor(0., device=sample.device, dtype=sample.dtype)

    # Algorithm 1
    vT = torch.randn(c_i*w_i*h_i, pca_rank, device=sample.device, dtype=torch.float)
    vT, _ = torch.linalg.qr(vT)
    v = vT.T
    v = v.view(-1, c_i, w_i, h_i)

    time_s = time.time()
    for i in range(max_iter):
        v = v.to(device=sample.device, dtype=sample.dtype)
        v_prev = v.detach().cpu().clone()
        
        u = []
        if v.size(0) // chunk_size != 0:
            v_buffer = list(v.chunk(v.size(0) // chunk_size))
        else:
            v_buffer = [v]

        for vi in v_buffer:
            g = lambda a : self.get_h(
                sample + a*vi, timestep=timestep, encoder_hidden_states=encoder_hidden_states.repeat(vi.size(0), 1, 1),
                op=op, block_idx=block_idx, 
            )

            ui = torch.func.jacfwd(g, argnums=0, has_aux=False, randomness='error')(a)
            u.append(ui.detach().cpu().clone())
        u = torch.cat(u, dim=0)
        u = u.to(sample.device, sample.dtype)
        
        # if v.size(0) == 1:
        #     g = lambda a : self.get_h(
        #         sample + a*v, timestep=timestep, encoder_hidden_states=encoder_hidden_states.repeat(v.size(0), 1, 1),
        #         op=op, block_idx=block_idx, 
        #     )
        #     u = torch.func.jacfwd(g, argnums=0, has_aux=False, randomness='error')(a)
            
        # time_s = time.time()
        # g = lambda a : get_h(x + a*v)
        # u = torch.func.jacfwd(g, argnums=0, has_aux=False, randomness='error')(a)
        # time_e = time.time()
        # print('single vi jacfwd t ==', time_e - time_s)

        g = lambda sample : einsum(
            u, self.get_h(
                sample, timestep=timestep, encoder_hidden_states=encoder_hidden_states,
                op=op, block_idx=block_idx, 
            ), 'b c w h, i c w h -> b'
        )
        v_ = torch.autograd.functional.jacobian(g, sample)
        v_ = v_.view(-1, c_i*w_i*h_i)

        _, s, v = torch.linalg.svd(v_, full_matrices=False)
        v = v.view(-1, c_i, w_i, h_i)
        u = u.view(-1, c_o, w_o, h_o)
        
        convergence = torch.dist(v_prev, v.detach().cpu()).item()
        print(f'power method : {i}-th step convergence : ', convergence)
        
        if torch.allclose(v_prev, v.detach().cpu(), atol=convergence_threshold) and (i > min_iter):
            print('reach convergence threshold : ', convergence)
            break

    u, s, vT = u.view(-1, c_o*w_o*h_o).T.detach(), s.sqrt().detach(), v.view(-1, c_i*w_i*h_i).detach()
    # u, s, vT = u.reshape(-1, c_o*w_o*h_o).T.detach(), s.sqrt().detach(), v.reshape(-1, c_i*w_i*h_i).detach()

    time_e = time.time()
    print('power method runtime ==', time_e - time_s)

    return u, s, vT

def local_decoder_pullback_zt(
        self, sample, timestep, encoder_hidden_states=None, op=None, block_idx=None,
        pca_rank=50, chunk_size=25, min_iter=10, max_iter=100, convergence_threshold=None,
    ):
    '''
    Args
        - sample : zt
        - op : ['down', 'mid', 'up']
        - block_idx : op == down, up : [0,1,2,3], op == mid : [0]
        - pooling : ['pixel-sum', 'channel-sum', 'single-channel', 'multiple-channel']
    Returns
        - h : hidden feature
    '''
    # get h samples
    time_s = time.time()

    # necessary variables
    h = self.get_h(
        sample, timestep=timestep, encoder_hidden_states=encoder_hidden_states,
        op=op, block_idx=block_idx, 
    )

    get_h_to_e = lambda h : self.get_h_to_e(
        sample=sample, timestep=timestep, encoder_hidden_states=encoder_hidden_states,
        input_h=h, op=op, block_idx=block_idx, verbose=False,
    )

    c_i, w_i, h_i = h.size(1), h.size(2), h.size(3)
    c_o, w_o, h_o = sample.size(1), sample.size(2), sample.size(3)

    a = torch.tensor(0., device=sample.device, dtype=sample.dtype)

    # Algorithm 1
    vT = torch.randn(c_i*w_i*h_i, pca_rank, device=sample.device, dtype=torch.float)
    vT, _ = torch.linalg.qr(vT)
    v = vT.T
    v = v.view(-1, c_i, w_i, h_i)

    for i in range(max_iter):
        v_prev = v.detach().cpu().clone()
        u = []
        
        time_s = time.time()
        v = v.to(device=sample.device, dtype=sample.dtype)
        v_buffer = list(v.chunk(v.size(0) // chunk_size))
        for vi in v_buffer:
            g = lambda a : get_h_to_e(h + a*vi)
            ui = torch.func.jacfwd(g, argnums=0, has_aux=False, randomness='error')(a)
            u.append(ui.detach().cpu().clone())
        u = torch.cat(u, dim=0)
        u = u.to(sample.device, sample.dtype)
        time_e = time.time()
        print('single v jacfwd t ==', time_e - time_s)

        # time_s = time.time()
        # g = lambda a : get_h(x + a*v)
        # u = torch.func.jacfwd(g, argnums=0, has_aux=False, randomness='error')(a)
        # time_e = time.time()
        # print('single vi jacfwd t ==', time_e - time_s)

        g = lambda h : einsum(u, get_h_to_e(h), 'b c w h, i c w h -> b')
        v_ = torch.autograd.functional.jacobian(g, h)
        v_ = v_.view(-1, c_i*w_i*h_i)

        _, s, v = torch.linalg.svd(v_, full_matrices=False)
        v = v.view(-1, c_i, w_i, h_i)
        u = u.view(-1, c_o, w_o, h_o)
        
        convergence = torch.dist(v_prev, v.detach().cpu()).item()
        print(f'power method : {i}-th step convergence : ', convergence)

        if torch.allclose(v_prev, v.detach().cpu(), atol=convergence_threshold) and (i > min_iter):
            print('reach convergence threshold : ', convergence)
            break

        if i == max_iter - 1:
            print('last convergence : ', convergence)

    # decoder jac do not return vT
    u, s, vT = v.view(-1, c_i*w_i*h_i).T.detach(), s.sqrt().detach(), u.view(-1, c_o*w_o*h_o).detach()
    return u, s, vT

def local_pca_zt(
        self, sample=None, timestep=None, encoder_hidden_states=None, op = None, block_idx=None, 
        memory_bound=5, num_pca_samples=50000, pca_rank=2000, pca_device='cpu',
        return_x_direction=True, perturb_h=1e-1,
    ):
    '''
    Args
        - sample : zt
        - op : ['down', 'mid', 'up']
        - block_idx : op == down, up : [0,1,2,3], op == mid : [0]
    Returns
        - h : hidden feature
    '''
    # get h samples
    time_s = time.time()
    
    h_list = []
    for _ in tqdm(range(num_pca_samples // memory_bound)):
        x = sample.repeat(memory_bound, 1, 1, 1)

        get_h = lambda x : self.get_h(
            x, timestep=timestep, encoder_hidden_states=encoder_hidden_states.repeat(memory_bound, 1, 1),
            op=op, block_idx=block_idx,
        )

        x += normalize_wrt_batch(torch.randn_like(x, device=x.device, dtype=x.dtype))
        h = get_h(x)

        if pca_device == 'cpu':
            h_list.append(h.detach().cpu().type(torch.float32).clone())
        else:
            h_list.append(h.detach().to(device=pca_device, dtype=torch.float32).clone())
            
    h = torch.cat(h_list, dim=0)
    time_e = time.time()
    print('h sampling t ==', time_e - time_s)

    # local pca -> h direction
    time_s = time.time()
    _, s, u = torch.pca_lowrank(h.view(num_pca_samples, -1), q=pca_rank, center=True, niter=2)
    s = s.to(device=sample.device, dtype=sample.dtype)
    u = u.to(device=sample.device, dtype=sample.dtype)
    time_e = time.time()
    print('torch.pca_lowrank t ==', time_e - time_s)
    print(f'eigenvalue spectrum : {s}')

    del h, h_list, _
    torch.cuda.empty_cache()
    gc.collect()

    # get corresponding x direction
    if return_x_direction:
        time_s = time.time()
        original_h = self.get_h(
            sample, timestep=timestep, encoder_hidden_states=encoder_hidden_states,
            op=op, block_idx=block_idx,
        )

        print(f'original_h norm : {original_h.view(1, -1).norm(dim=-1)}, original_h std : {original_h.view(1, -1).std(dim=-1)}')
        original_h = original_h.repeat(pca_rank, 1, 1, 1).detach()
        perturbed_h = original_h + perturb_h * u.view(*original_h.shape)
        
        jacx = lambda x : (perturbed_h - self.get_h(
            x, timestep=timestep, encoder_hidden_states=encoder_hidden_states,
            op=op, block_idx=block_idx,
        )).view(pca_rank, -1).norm(dim=-1)
        jac  = torch.autograd.functional.jacobian(jacx, sample)

        vT = normalize_wrt_batch(jac).view(pca_rank, -1)
        time_e = time.time()
        print('torch.jac t ==', time_e - time_s)
        
    else:
        vT = None

    return u.detach(), s.detach(), vT.detach()

# monkey patch (global method)
def global_pca_zt(
        self, sample=None, timestep=None, encoder_hidden_states=None, op=None, block_idx=None, 
        memory_bound=5, pca_rank=100, pca_device='cpu',
    ):
    '''
    Args
        - sample : zt w/ zT ~ N(0, I)
        - op : ['down', 'mid', 'up']
        - block_idx : op == down, up : [0,1,2,3], op == mid : [0]
    Returns
        - h : hidden feature
    '''
    # get h samples
    time_s = time.time()

    h_list = list(sample.chunk(sample.size(0) // memory_bound + 1))
    for buffer_idx, x in tqdm(enumerate(h_list)):
        x = x.to(dtype=self.dtype, device=self.device)

        h = self.get_h(
            sample=x, timestep=timestep, encoder_hidden_states=encoder_hidden_states.repeat(x.size(0), 1, 1),
            op=op, block_idx=block_idx,
        )

        h_list[buffer_idx] = h.detach().to(device=pca_device, dtype=torch.float).clone()
            
    h = torch.cat(h_list, dim=0)
    time_e = time.time()
    num_pca_samples = h.size(0)
    print('num_pca_samples ==', num_pca_samples)
    print('h sampling t ==', time_e - time_s)
    print('h shape : ', h.shape)

    del h_list
    gc.collect()

    # global pca -> h direction
    time_s = time.time()
    _, s, u = torch.pca_lowrank(h.view(num_pca_samples, -1), q=pca_rank, center=True, niter=5)
    s = s.to(device=pca_device, dtype=sample.dtype)
    u = u.to(device=pca_device, dtype=sample.dtype)
    time_e = time.time()
    print('torch.pca_lowrank t ==', time_e - time_s)
    print(f'eigenvalue spectrum : {s}')

    del h, _
    torch.cuda.empty_cache()

    return u.detach(), s.detach()

# deprecated
def local_pca_text(
        self, sample=None, timestep=None, encoder_hidden_states=None, op = None, block_idx=None, 
        memory_bound=250, num_pca_samples=50000, pca_rank=100, pca_device='cpu',
        return_x_direction=True, perturb_h=1e-1,
    ):
    '''
    Args
        - sample : zt
        - op : ['down', 'mid', 'up']
        - block_idx : op == down, up : [0,1,2,3], op == mid : [0]
    Returns
        - h : hidden feature
    '''
    # get h samples
    time_s = time.time()
    
    h_list = []
    for _ in tqdm(range(num_pca_samples // memory_bound)):
        t_emb = encoder_hidden_states.repeat(memory_bound, 1, 1)

        get_h = lambda encoder_hidden_states : self.get_h(
            sample=sample.repeat(memory_bound, 1, 1, 1), timestep=timestep, encoder_hidden_states=encoder_hidden_states,
            op=op, block_idx=block_idx,
        )

        t_emb += normalize_wrt_batch(torch.randn_like(t_emb, device=t_emb.device, dtype=t_emb.dtype))
        h = get_h(t_emb)

        h_list.append(h.detach().to(device=pca_device, dtype=torch.float).clone())
            
    h = torch.cat(h_list, dim=0)
    time_e = time.time()
    print('h sampling t ==', time_e - time_s)

    # local pca -> h direction
    time_s = time.time()
    _, s, u = torch.pca_lowrank(h.view(num_pca_samples, -1), q=pca_rank, center=True, niter=2)
    s = s.to(device=sample.device, dtype=sample.dtype)
    u = u.to(device=sample.device, dtype=sample.dtype)
    time_e = time.time()
    print('torch.pca_lowrank t ==', time_e - time_s)
    print(f'eigenvalue spectrum : {s}')

    del h, h_list, _
    torch.cuda.empty_cache()
    gc.collect()

    # get corresponding x direction
    if return_x_direction:
        time_s = time.time()
        original_h = self.get_h(
            sample=sample, timestep=timestep, encoder_hidden_states=encoder_hidden_states,
            op=op, block_idx=block_idx,
        )

        print(f'original_h norm : {original_h.view(1, -1).norm(dim=-1)}, original_h std : {original_h.view(1, -1).std(dim=-1)}')
        original_h = original_h.repeat(pca_rank, 1, 1).detach()
        perturbed_h = original_h + perturb_h * u.view(*original_h.shape)
        
        jacx = lambda t_emb : (perturbed_h - self.get_h(
            sample=sample, timestep=timestep, encoder_hidden_states=t_emb,
            op=op, block_idx=block_idx,
        )).view(pca_rank, -1).norm(dim=-1)
        jac  = torch.autograd.functional.jacobian(jacx, encoder_hidden_states)

        vT = normalize_wrt_batch(jac).view(pca_rank, -1)
        time_e = time.time()
        print('torch.jac t ==', time_e - time_s)
        
    else:
        vT = None

    return u.detach(), s.detach(), vT.detach()

def global_pca_text(
        self, sample=None, timestep=None, encoder_hidden_states=None, op = None, block_idx=None, 
        memory_bound=5, pca_rank=100, pca_device='cpu',
    ):
    '''
    Args
        - sample : zt w/ zT ~ N(0, I)
        - op : ['down', 'mid', 'up']
        - block_idx : op == down, up : [0,1,2,3], op == mid : [0]
    Returns
        - h : hidden feature
    '''
    pass

def inv_jac_zt(
        self, sample=None, timestep=None, encoder_hidden_states=None, op=None, block_idx=None,
        u=None, perturb_h=1e-1, 
    ):
    '''
    I think it's... kinda like classifier guidance.

    Note
        - get v = Jac^-1 u, where Jac = dh / dx
    Args
        - sample : zt. sample for editing
        - u : direction in hidden space
    '''
    # original h
    h = self.get_h(
        sample=sample, timestep=timestep, encoder_hidden_states=encoder_hidden_states,
        op=op, block_idx=block_idx,
    )

    # get number of h space directions 
    if len(u.shape) > 1:
        raise NotImplementedError('')
        pca_rank = u.size(1)
        h = h.repeat(pca_rank, 1, 1, 1).detach()
        u = rearrange(u, '(c w h) k -> k c w h', c=h.size(1), w=h.size(2), h=h.size(3))
    else:
        assert sample.size(0) == 1, 'sample size should be 1'
        pca_rank = 1
        u = u.view(-1, *h.shape[1:])

    # perturb h
    perturbed_h = h + perturb_h * u
    
    # get corresponding x direction
    jacx = lambda sample : (perturbed_h - self.get_h(
        sample, timestep=timestep, encoder_hidden_states=encoder_hidden_states,
        op=op, block_idx=block_idx, 
    )).view(pca_rank, -1).norm(dim=-1)
    jac = torch.autograd.functional.jacobian(jacx, sample)

    # normalize direction
    vT = jac.view(pca_rank, -1)
    vT = vT / vT.norm(dim=1, keepdim=True)
    return vT


###################
# diffusion utils #
###################
class SchedulerOutput(object):
    def __init__(self, xt_next, P_xt):
        self.prev_sample = xt_next
        self.x0 = P_xt
    
class YHCustomScheduler(object):
    def __init__(self, args):
        # NOTE : verify this
        self.t_max = 999
        self.noise_schedule = 'linear' if args.noise_schedule is None else args.noise_schedule
        self.timesteps = None
        self.learn_sigma = False

        # get SNR schedule
        self.get_alphas_cumprod(args)

    def set_timesteps(self, num_inferences, device=None, is_inversion=False):
        device = 'cpu' if device is None else device
        if is_inversion:
            seq = torch.linspace(0, 1, num_inferences, device=device) * self.t_max
            seq = seq + 1e-6
            seq_prev = torch.cat([torch.tensor([-1], device=device), seq[:-1]], dim = 0)
            self.timesteps = seq_prev[1:]
            self.timesteps_next = seq[1:]
        
        else:
            seq = torch.linspace(0, 1, num_inferences, device=device) * self.t_max
            seq_prev = torch.cat([torch.tensor([-1], device=device), seq[:-1]], dim = 0)
            self.timesteps = reversed(seq[1:])
            self.timesteps_next = reversed(seq_prev[1:])

    def step(self, et, t, xt, eta=0.0, **kwargs):
        '''
        Notation
            - a : alpha / b : beta / e : epsilon
        '''
        if self.learn_sigma:
            et, logvar = torch.split(et, et.shape[1] // 2, dim=1)
        else:
            logvar = None
        assert et.shape == xt.shape, 'et, xt shape should be same'

        t_idx   = self.timesteps.tolist().index(t)
        t_next  = self.timesteps_next[t_idx]

        # print('t, t_next : ', t, t_next)
        
        # extract need parameters : at, at_next
        at = extract(self.alphas_cumprod, t, xt.shape)
        at_next = extract(self.alphas_cumprod, t_next, xt.shape)

        # print('at, at_next : ', at[0].squeeze(), at_next[0].squeeze())

        # DDIM step ; xt-1 = sqrt(at-1 / at) (xt - sqrt(1-at)*e(xt, t)) + sqrt(1-at-1)*e(xt, t)
        P_xt = (xt - et * (1 - at).sqrt()) / at.sqrt()

        # Deterministic.
        if eta == 0:
            D_xt = (1 - at_next).sqrt() * et
            xt_next = at_next.sqrt() * P_xt + D_xt

        # Add noise. When eta is 1 and time step is 1000, it is equal to ddpm.
        elif logvar is None:
            sigma_t = ((1 - at / (at_next)) * (1 - at_next) / (1 - at)).sqrt()

            D_xt = (1 - at_next - eta * sigma_t ** 2).sqrt() * et
            xt_next = at_next.sqrt() * P_xt + D_xt + eta * sigma_t * torch.randn_like(xt)

        elif logvar is not None:
            bt = extract(self.betas, t, xt.shape)
            
            mean = 1 / torch.sqrt(1.0 - bt) * (xt - bt / torch.sqrt(1 - at) * et)
            xt_next = mean + torch.exp(0.5 * logvar) * torch.randn_like(xt, device=xt.device, dtype=xt.dtype)
            P_xt = None

        return SchedulerOutput(xt_next, P_xt)

    def get_alphas_cumprod(self, args):
        # betas
        if self.noise_schedule == 'linear':
            betas = self.linear_beta_schedule(
                beta_start = 0.0001, # args.config.diffusion.beta_start,
                beta_end   = 0.02,   # args.config.diffusion.beta_end,
                timesteps  = 1000,   # args.config.diffusion.num_diffusion_timesteps,
            )

        elif self.noise_schedule == 'cosine':
            betas = self.cosine_beta_schedule(
                timesteps = self.t_max + 1
            )
        self.betas = betas.to(device=args.device, dtype=args.dtype)

        # alphas
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod = alphas_cumprod.to(device=args.device, dtype=args.dtype)

    def linear_beta_schedule(self, beta_start, beta_end, timesteps):
        return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)
    
    # def cosine_beta_schedule(self, timesteps):
    #     return betas_for_alpha_bar(
    #         timesteps, lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
    #     )
    
    def cosine_beta_schedule(self, timesteps, s = 0.008):
        """
        cosine schedule (improved DDPM)
        proposed in https://arxiv.org/abs/2102.09672
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                    produces the cumulative product of (1-beta) up to that
                    part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                    prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def extract(a, t, x_shape):
    """Extract coefficients from a based on t and reshape to make it
    broadcastable with x_shape."""
    if isinstance(t, int):
        t = torch.tensor([t])
        t = t.repeat(x_shape[0])
    elif isinstance(t, torch.Tensor):
        t = t.repeat(x_shape[0])
    else:
        raise ValueError(f"t must be int or torch.Tensor, got {type(t)}")
    bs, = t.shape
    assert x_shape[0] == bs, f"{x_shape[0]}, {t.shape}"
    out = torch.gather(a, 0, t.long())
    assert out.shape == (bs,)
    out = out.reshape((bs,) + (1,) * (len(x_shape) - 1))
    return out

###########
# dataset #
###########
from PIL import Image
from glob import glob
from torch.utils.data import Dataset
import torchvision.transforms as tfs

def get_dataset(args):
    '''
    Args
        - image_name : [Astronaut, Cyberpunk, VanGogh]
    Returns
        - dataset[idx] = img
    '''
    if args.dataset_name == 'Examples':
        dataset = ImgDataset(
            image_root = DATASET_PATHS['Examples'], 
            device = args.device, 
            dtype = args.dtype, 
            image_size = 512, # Stable-Diffusion
            dataset_name = 'Examples',
        )
    elif args.dataset_name == 'CelebA_HQ':
        # dataset = BenchmarkDataset(
        #     image_root = DATASET_PATHS['CelebA_HQ'],
        #     img_size = 256 # High resolution DM
        # )
        dataset = ImgDataset(
            image_root = DATASET_PATHS['CelebA_HQ'], 
            device = args.device, 
            dtype = args.dtype, 
            image_size = 256, # High resolution DM
            dataset_name = 'CelebA_HQ',
        )
    elif args.dataset_name == 'Flower':
        dataset = HFDataset(
            device = args.device, 
            dtype = args.dtype, 
            image_size = 256, # High resolution DM
            dataset_name = 'Flower',
        )
    elif args.dataset_name == 'LSUN_church':
        dataset = get_lsun_dataset(
            data_root = DATASET_PATHS['LSUN_church'], 
            image_size = 256, # High resolution DM
            category='church_outdoor',
        )
    else:
        raise ValueError('Invalid dataset name')
    return dataset

class HFDataset(Dataset):
    def __init__(self, device, dtype, image_size, dataset_name):
        super().__init__()
        from datasets import load_dataset
        self.dataset_name = dataset_name
        self.dataset = load_dataset("huggan/flowers-102-categories", split='train')
        self.transform = tfs.Compose([
            tfs.ToTensor(), 
            tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        ])
        self.img_size = image_size
        self.device = device
        self.dtype = dtype

    def __getitem__(self, index):
        x = self.dataset['image'][index]
        
        # Crop the center of the image
        w, h = x.size
        crop_size = min(w, h)

        left    = (w - crop_size)/2
        top     = (h - crop_size)/2
        right   = (w + crop_size)/2
        bottom  = (h + crop_size)/2

        # Crop the center of the image
        x = x.crop((left, top, right, bottom))

        # resize the image
        x = x.resize((self.img_size, self.img_size))
        if self.transform is not None:
            x = self.transform(x) 
        return x.unsqueeze(0).to(device=self.device, dtype=self.dtype)

    def __len__(self):
        return len(self.image_paths)

class BenchmarkDataset(Dataset):
    def __init__(self, image_root, img_size=256, is_train=True):
        super().__init__()
        if is_train:
            self.image_dir = os.path.join(image_root, 'raw_images', 'train', 'images')
        else:
            self.image_dir = os.path.join(image_root, 'raw_images', 'test', 'images')
        imgs_path_list = os.listdir(self.image_dir)
        imgs_path_list = [path for path in imgs_path_list if path.split('.')[1] in ['jpg', 'jpeg', 'png']]
        self.image_paths = sorted(imgs_path_list, key=lambda path: int(path.split('.')[0]))

        self.transform = tfs.Compose([
            tfs.ToTensor(), 
            tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
            inplace=True)
        ])
        self.img_size = img_size

        print(f'number of dataset : {len(self.image_paths)}')

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.image_paths[index])
        x = Image.open(image_path)
        x = x.resize((self.img_size, self.img_size))
        if self.transform is not None:
            x = self.transform(x) 
        return x.unsqueeze(dim=0)

    def __len__(self):
        return len(self.image_paths)

class ImgDataset(Dataset):
    def __init__(self, image_root, device, dtype, image_size, dataset_name):
        super().__init__()
        self.dataset_name = dataset_name
        self.image_dir = os.path.join(image_root)
        imgs_path_list = os.listdir(self.image_dir)
        imgs_path_list = [path for path in imgs_path_list if path.split('.')[1] in ['jpg', 'jpeg', 'png']]

        self.image_paths = sorted(imgs_path_list, key=lambda path: int(path.split('.')[0]))
        self.transform = tfs.Compose([
            tfs.ToTensor(), 
            tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        ])
        self.img_size = image_size
        self.device = device
        self.dtype = dtype

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.image_paths[index])
        x = Image.open(image_path)
        
        # Crop the center of the image
        w, h = x.size
        crop_size = min(w, h)

        left    = (w - crop_size)/2
        top     = (h - crop_size)/2
        right   = (w + crop_size)/2
        bottom  = (h + crop_size)/2

        # Crop the center of the image
        x = x.crop((left, top, right, bottom))

        # resize the image
        x = x.resize((self.img_size, self.img_size))
        if self.transform is not None:
            x = self.transform(x) 
        return x.unsqueeze(0).to(device=self.device, dtype=self.dtype)

    def __len__(self):
        return len(self.image_paths)

################
# LSUN dataset #
################
import os.path
from collections.abc import Iterable
from torchvision.datasets.utils import verify_str_arg, iterable_to_str

from PIL import Image
import io
import pickle
import os
import torch
import torch.utils.data as data
import torchvision.transforms as tfs

class VisionDataset(data.Dataset):
    _repr_indent = 4

    def __init__(self, root, transforms=None, transform=None, target_transform=None):
        # if isinstance(root, torch._six.string_classes):
        #     root = os.path.expanduser(root)
        self.root = root

        has_transforms = transforms is not None
        has_separate_transform = transform is not None or target_transform is not None
        if has_transforms and has_separate_transform:
            raise ValueError("Only transforms or transform/target_transform can "
                             "be passed as argument")

        # for backwards-compatibility
        self.transform = transform
        self.target_transform = target_transform

        if has_separate_transform:
            transforms = StandardTransform(transform, target_transform)
        self.transforms = transforms

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __repr__(self):
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        body += self.extra_repr().splitlines()
        if hasattr(self, 'transform') and self.transform is not None:
            body += self._format_transform_repr(self.transform,
                                                "Transforms: ")
        if hasattr(self, 'target_transform') and self.target_transform is not None:
            body += self._format_transform_repr(self.target_transform,
                                                "Target transforms: ")
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    def _format_transform_repr(self, transform, head):
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])

    def extra_repr(self):
        return ""

class StandardTransform(object):
    def __init__(self, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

    def __call__(self, input, target):
        if self.transform is not None:
            input = self.transform(input)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return input, target

    def _format_transform_repr(self, transform, head):
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])

    def __repr__(self):
        body = [self.__class__.__name__]
        if self.transform is not None:
            body += self._format_transform_repr(self.transform,
                                                "Transform: ")
        if self.target_transform is not None:
            body += self._format_transform_repr(self.target_transform,
                                                "Target transform: ")

        return '\n'.join(body)

class LSUNClass(VisionDataset):
    def __init__(self, root, transform=None, target_transform=None):
        import lmdb

        super(LSUNClass, self).__init__(
            root, transform=transform, target_transform=target_transform
        )

        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()["entries"]
        root_split = root.split("/")
        cache_file = os.path.join("/".join(root_split[:-1]), f"_cache_{root_split[-1]}")
        if os.path.isfile(cache_file):
            self.keys = pickle.load(open(cache_file, "rb"))
        else:
            with self.env.begin(write=False) as txn:
                self.keys = [key for key, _ in txn.cursor()]
            pickle.dump(self.keys, open(cache_file, "wb"))

    def __getitem__(self, index):
        img, target = None, None
        env = self.env
        with env.begin(write=False) as txn:
            imgbuf = txn.get(self.keys[index])

        buf = io.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.length

class LSUN(VisionDataset):
    """
    `LSUN <https://www.yf.io/p/lsun>`_ dataset.
    Args:
        root (string): Root directory for the database files.
        classes (string or list): One of {'train', 'val', 'test'} or a list of
            categories to load. e,g. ['bedroom_train', 'church_outdoor_train'].
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, classes="train", transform=None, target_transform=None):
        super(LSUN, self).__init__(
            root, transform=transform, target_transform=target_transform
        )
        self.classes = self._verify_classes(classes)

        # for each class, create an LSUNClassDataset
        self.dbs = []
        for c in self.classes:
            self.dbs.append(
                LSUNClass(root=root + "/" + c + "_lmdb", transform=transform)
            )

        self.indices = []
        count = 0
        for db in self.dbs:
            count += len(db)
            self.indices.append(count)

        self.length = count

    def _verify_classes(self, classes):
        categories = [
            "bedroom",
            "bridge",
            "church_outdoor",
            "classroom",
            "conference_room",
            "dining_room",
            "kitchen",
            "living_room",
            "restaurant",
            "tower",
        ]
        dset_opts = ["train", "val", "test"]

        try:
            verify_str_arg(classes, "classes", dset_opts)
            if classes == "test":
                classes = [classes]
            else:
                classes = [c + "_" + classes for c in categories]
        except ValueError:
            if not isinstance(classes, Iterable):
                msg = (
                    "Expected type str or Iterable for argument classes, "
                    "but got type {}."
                )
                raise ValueError(msg.format(type(classes)))

            classes = list(classes)
            msg_fmtstr = (
                "Expected type str for elements in argument classes, "
                "but got type {}."
            )
            for c in classes:
                verify_str_arg(c, custom_msg=msg_fmtstr.format(type(c)))
                c_short = c.split("_")
                category, dset_opt = "_".join(c_short[:-1]), c_short[-1]

                msg_fmtstr = "Unknown value '{}' for {}. Valid values are {{{}}}."
                msg = msg_fmtstr.format(
                    category, "LSUN class", iterable_to_str(categories)
                )
                verify_str_arg(category, valid_values=categories, custom_msg=msg)

                msg = msg_fmtstr.format(dset_opt, "postfix", iterable_to_str(dset_opts))
                verify_str_arg(dset_opt, valid_values=dset_opts, custom_msg=msg)

        return classes

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target) where target is the index of the target category.
        """
        target = 0
        sub = 0
        for ind in self.indices:
            if index < ind:
                break
            target += 1
            sub = ind

        db = self.dbs[target]
        index = index - sub

        if self.target_transform is not None:
            target = self.target_transform(target)

        img, _ = db[index]
        return img.unsqueeze(0)

    def __len__(self):
        return self.length

    def extra_repr(self):
        return "Classes: {classes}".format(**self.__dict__)

def get_lsun_dataset(data_root, category='church_outdoor', image_size=256):
    assert category in ["church_outdoor"]

    train_folder = "{}_train".format(category)
    # val_folder = "{}_val".format(category)

    train_dataset = LSUN(
        root=os.path.join(data_root),
        classes=[train_folder],
        transform=tfs.Compose(
            [
                tfs.Resize(image_size),
                tfs.CenterCrop(image_size),
                tfs.ToTensor(),
                tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
                                     inplace=True)
            ]
        ),
    )

    # test_dataset = LSUN(
    #     root=os.path.join(data_root),
    #     classes=[val_folder],
    #     transform=tfs.Compose(
    #         [
    #             tfs.Resize(image_size),
    #             tfs.CenterCrop(image_size),
    #             tfs.ToTensor(),
    #             tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
    #                                  inplace=True)
    #             ,
    #         ]
    #     ),
    # )

    return train_dataset

###########
# caption #
###########
def get_laion_coco_prompt_list(num_captions=50):
    # load captions
    caption_path = os.path.join(PROMPT_PATHS['Laion-COCO'], 'caption.txt')
    caption_file = open(caption_path, "r")
    caption_json = json.load(caption_file)

    # extract captions
    caption_list = []
    for row in caption_json['rows']:
        caption_list.append(row['row']['top_caption'])

        if len(caption_list) == num_captions:
            break
    return caption_list

def get_ms_coco_prompt_list(num_captions=50):
    # sample random idxs
    # import random
    # indice = random.sample(range(202654), 50)
    
    # load captions
    caption_path = os.path.join(PROMPT_PATHS['MS-COCO'], 'captions_val2014.json')
    caption_file = open(caption_path, "r")
    caption_json = json.load(caption_file)

    # extract captions
    total_caption_list = []
    for annotation in caption_json['annotations']:
        total_caption_list.append(annotation['caption'])
    
    # random sample from captions
    caption_list = []
    caption_idxs = [114694, 179612, 185955, 21611, 20508, 119396, 186100, 102659, 117671, 9735, 71010, 68133, 182172, 36994, 127395, 116134, 165049, 145223, 20862, 139334, 24278, 102032, 140819, 72597, 202511, 156837, 13009, 8760, 195923, 28947, 14512, 141519, 176863, 169228, 126755, 90161, 113614, 161434, 131845, 146514, 42761, 185535, 180435, 7769, 34601, 122562, 109762, 103587, 186105, 128251]
    if len(caption_idxs) > num_captions:
        caption_idxs = caption_idxs[:num_captions]
        assert len(caption_idxs) == num_captions
    elif len(caption_idxs) < num_captions:
        caption_idxs = caption_idxs + [58356, 137734, 92355, 22745, 194352, 188512, 11869, 107059, 130335, 45468, 105371, 11340, 81133, 6925, 7260, 57598, 96075, 193805, 90701, 100074, 131850, 170891, 69908, 95499, 35393, 133518, 161788, 114858, 29263, 157174, 145698, 45155, 92954, 109720, 130806, 3338, 170187, 121130, 25776, 59724, 105466, 79769, 115491, 137827, 187348, 108271, 76080, 110157, 154149, 50551]
        caption_idxs = caption_idxs[:num_captions]
        assert len(caption_idxs) == num_captions
    for idx in caption_idxs:
        caption_list.append(total_caption_list[idx])
    assert len(caption_list) == num_captions
    return caption_list

#########
# utils #
#########
import argparse

def dict2namespace(args_dict, namespace = None):
    '''
    convert dict to namespace
    notice ; {k : {k : v}} -> args.k.k.v
    '''
    if namespace is None:
        namespace = argparse.Namespace()

    for k, v in args_dict.items():
        if isinstance(v, dict):
            try:
                v = dict2namespace(v, getattr(namespace, k))
            except:
                setattr(namespace, k, None)
                v = dict2namespace(v, getattr(namespace, k))
        setattr(namespace, k, v)
    return namespace

def normalize_wrt_batch(x, norm=1):
    # figure out the shape of x
    x_shape = [1 for _ in x.shape]
    x_shape[0] = x.size(0)

    # normalize x w.r.t. batch
    x = norm * x / x.view(x.size(0), -1).norm(dim=-1).view(x_shape)

    try:
        assert ((x.view(x.size(0), -1).norm(dim=-1) - norm) < 1e-3 * norm).all(), f'x is not normalized : {(x.view(x.size(0), -1).norm(dim=-1) - norm).abs()}'
    except:
        print('!!!!!!!!!!!!!!!!!')
        print(f'input is not normalized : {(x.view(x.size(0), -1).norm(dim=-1) - norm).abs()}')
        print('!!!!!!!!!!!!!!!!!')
    return x