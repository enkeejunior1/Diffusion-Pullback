import os
import gc
import time
from copy import deepcopy
import matplotlib.pyplot as plt

import math
import torch
import torch.nn as nn

from einops import rearrange, reduce, einsum
import torchvision.utils as tvu
import numpy as np

import skimage

from tqdm import tqdm

#########
# model #
#########
class DDPM(nn.Module):
    '''Pretrained DDPM ; SDEdit : https://github.com/ermongroup/SDEdit'''
    def __init__(self, config):
        super().__init__()
        self.config = config
        ch, out_ch, ch_mult = config.model.ch, config.model.out_ch, tuple(config.model.ch_mult)
        num_res_blocks = config.model.num_res_blocks
        attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        in_channels = config.model.in_channels
        resolution = config.data.image_size
        resamp_with_conv = config.model.resamp_with_conv

        self.ch = ch
        self.temb_ch = self.ch * 4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch,
                            self.temb_ch),
            torch.nn.Linear(self.temb_ch,
                            self.temb_ch),
        ])

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + ch_mult
        self.down = nn.ModuleList()
        block_in = None
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            skip_in = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                if i_block == self.num_res_blocks:
                    skip_in = ch * in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in + skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x, t):
        raise NotImplementedError
    
class PullBackDDPM(DDPM):
    '''
    Methods
        - get_h
        - inv_jac_zt
        - local_pca_zt
        - global_pca_zt
        - local_pullback_zt
    '''
    def __init__(self, args):
        super().__init__(args.config)
        self.device = args.device
        self.dtype = args.dtype

    def forward(
            self, x, t, u=None, op=None, block_idx=None, 
        ):
        if not (x.shape[2] == x.shape[3] == self.resolution):
            raise ValueError(f'Input image resolution {x.shape[2]}x{x.shape[3]} must be {self.resolution}x{self.resolution}')
        t = t.unsqueeze(0) if len(t.shape) == 0 else t
        t = t.to(device=self.device, dtype=self.dtype)

        # timestep embedding
        temb = get_timestep_embedding(t, self.ch, device=self.device)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

            if (op == 'down') & (block_idx == i_level):
                hs[-1] = hs[-1] + u.view(-1, *hs[-1].shape)

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        if (op == 'mid') & (block_idx == 0):
            h = h + u.view(-1, *h[-1].shape)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

            # return h
            if (op == 'up') & (block_idx == i_level):
                h = h + u.view(-1, *h[-1].shape)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

    def get_h(
            self, x=None, t=None, 
            op=None, block_idx=None, verbose=False,
        ):
        '''
        Args
            - x : xt
            - op : ['down', 'mid', 'up']
            - block_idx : down, up ; [0,1,2,3], mid ; [0]
        Returns
            - h : hidden feature
        '''
        assert x.shape[2] == x.shape[3] == self.resolution
        if isinstance(t, int):
            t = torch.tensor([t]).to(self.device)
        elif isinstance(t, torch.Tensor):
            t = t.unsqueeze(0) if len(t.shape) == 0 else t
        else:
            raise ValueError('t must be int or torch.Tensor')
        
        # timestep embedding
        temb = get_timestep_embedding(t, self.ch, device=self.device)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

            if (op == 'down') & (block_idx == i_level):
                if verbose:
                    print(f'op : {op}, block_idx : {block_idx}, return h.shape : {hs[-1].shape}')
                return hs[-1]

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        if (op == 'mid') & (block_idx == 0):
            if verbose:
                print(f'op : {op}, block_idx : {block_idx}, return h.shape : {h.shape}')
            return h

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

            # return h
            if (op == 'up') & (block_idx == i_level):
                if verbose:
                    print(f'op : {op}, block_idx : {block_idx}, return h.shape : {h.shape}')
                return h

        raise ValueError(f'(op, block_idx) = ({op, block_idx}) is not valid')

    def get_h_to_e(
            self, x=None, t=None, input_h=None,
            op=None, block_idx=None, verbose=False,
        ):
        '''
        Args
            - x : xt
            - op : ['down', 'mid', 'up']
            - block_idx : down, up ; [0,1,2,3], mid ; [0]
        Returns
            - h : hidden feature
        '''
        assert x.shape[2] == x.shape[3] == self.resolution
        if isinstance(t, int):
            t = torch.tensor([t]).to(self.device)
        elif isinstance(t, torch.Tensor):
            t = t.unsqueeze(0) if len(t.shape) == 0 else t
        else:
            raise ValueError('t must be int or torch.Tensor')
        assert (op == 'mid') & (block_idx == 0)
        
        pca_rank = input_h.size(0)
        
        # timestep embedding
        temb = get_timestep_embedding(t, self.ch, device=self.device)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

            if (op == 'down') & (block_idx == i_level):
                hs[-1] = input_h.view(-1, *hs[-1].shape[:1])
                raise NotImplementedError('not implemented yet for op : down')

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        if (op == 'mid') & (block_idx == 0):
            h = input_h.view(-1, *h.shape[1:])
            hs = [h.repeat(pca_rank, 1, 1, 1) for h in hs]

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

            # return h
            if (op == 'up') & (block_idx == i_level):
                h = input_h.view(-1, *h.shape[:1])
                raise NotImplementedError('not implemented yet for op : up')

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

    def inv_jac_xt(
            self, x=None, t=None, op=None, block_idx=None,
            u=None, perturb_h=1e-1, 
        ):
        # original h
        h = self.get_h(
            x=x, t=t, op=op, block_idx=block_idx,
        )

        # get number of h space directions 
        if len(u.shape) > 1:
            pca_rank = u.size(1)
            h = h.repeat(pca_rank, 1, 1, 1).detach()
            u = rearrange(u, '(c w h) k -> k c w h', c=h.size(1), w=h.size(2), h=h.size(3))

        else:
            pca_rank = 1
            u = u.view(*h.shape)

        # perturb h
        perturbed_h = h + perturb_h * u
        
        # get corresponding x direction (argmin_v perturbed_h - f(xt + v))
        jacx = lambda x : (perturbed_h - self.get_h(
            x=x, t=t, op=op, block_idx=block_idx, 
        )).view(pca_rank, -1).norm(dim=-1)
        jac = torch.autograd.functional.jacobian(jacx, x)

        # normalize direction
        vT = normalize_wrt_batch(jac).view(pca_rank, -1)
        return vT

    def local_pca_xt(
            self, x=None, t=None, op=None, block_idx=None, memory_bound=5, num_pca_samples=50000, pca_rank=512, pca_device='cpu',
            return_x_direction=True, perturb_h=1e-1,
        ):
        '''
        Args
            - x : xt
            - op : ['down', 'mid', 'up']
            - block_idx : op == down, up : [0,1,2,3], op == mid : [0]
        Returns
            - h : hidden feature
        '''
        assert num_pca_samples % memory_bound == 0
        original_x = x.clone()

        # get h samples
        time_s = time.time()
        h_list = []
        for _ in tqdm(range(num_pca_samples // memory_bound), desc='h sampling for local pca'):
            # local perturbation of x
            x = original_x.repeat(memory_bound, 1, 1, 1)
            normalized_noise = normalize_wrt_batch(torch.randn_like(x, device=x.device, dtype=x.dtype))
            x = x + normalized_noise
            
            # local perturbation of h = f(x + ep)
            h = self.get_h(x=x, t=t, op=op, block_idx=block_idx)

            # save in list
            h_list.append(h.detach().to(device=pca_device, dtype=torch.float32).clone())
                
        h = torch.cat(h_list, dim=0)
        time_e = time.time()
        print('h sampling t ==', time_e - time_s)

        # local pca -> h direction
        _, s, u = torch.pca_lowrank(h.view(num_pca_samples, -1), q=pca_rank, center=True, niter=2)
        s = s.to(device=x.device, dtype=x.dtype)
        u = u.to(device=x.device, dtype=x.dtype)
        print('torch.pca_lowrank t ==', time_e - time_s)
        print(f'eigenvalue spectrum : {s}')

        del h, h_list, _
        torch.cuda.empty_cache()
        gc.collect()

        # get corresponding x direction
        if return_x_direction:
            time_s = time.time()
            vT = self.inv_jac_xt(
                x=original_x, t=t, op=op, block_idx=block_idx, u=u, perturb_h=perturb_h,
            )
            time_e = time.time()
            print('inv_jac_xt t ==', time_e - time_s)
            
        else:
            vT = None

        return u, s, vT

    def global_pca_xt(
            self, x=None, t=None, op=None, block_idx=None, 
            memory_bound=5, pca_rank=512, pca_device='cpu',
        ):
        '''
        Args
            - sample : zt w/ zT ~ N(0, I)
            - op : ['down', 'mid', 'up']
            - block_idx : op == down, up : [0,1,2,3], op == mid : [0]
        Returns
            - h : hidden feature
        '''
        num_pca_samples = x.size(0)
        print('num_pca_samples ==', num_pca_samples)

        # get h samples
        time_s = time.time()
        h_list = list(x.chunk(x.size(0) // memory_bound + 1))
        for buffer_idx, x in tqdm(enumerate(h_list), desc='h sampling for global pca'):
            x = x.to(dtype=self.dtype, device=self.device)
            h = self.get_h(
                x=x, t=t, op=op, block_idx=block_idx,
            )
            h_list[buffer_idx] = h.detach().to(device=pca_device, dtype=torch.float).clone()
        h = torch.cat(h_list, dim=0)
        time_e = time.time()
        print('h sampling t ==', time_e - time_s)
        print('h shape : ', h.shape)

        del h_list
        gc.collect()

        # global pca -> h direction
        time_s = time.time()
        _, s, u = torch.pca_lowrank(h.view(num_pca_samples, -1), q=pca_rank, center=True, niter=5)
        s = s.to(device=pca_device, dtype=self.dtype)
        u = u.to(device=pca_device, dtype=self.dtype)
        time_e = time.time()
        print('torch.pca_lowrank t ==', time_e - time_s)
        print(f'eigenvalue spectrum : {s}')

        del h, _
        torch.cuda.empty_cache()

        return u.detach(), s.detach()

    def local_encoder_pullback_xt(
            self, x=None, t=None, op=None, block_idx=None, pca_rank=16, chunk_size=25,
            min_iter=10, max_iter=100, convergence_threshold=1e-3,
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
        # necessary variables
        num_chunk = pca_rank // chunk_size if pca_rank % chunk_size == 0 else pca_rank // chunk_size + 1
        get_h = lambda x : self.get_h(
            x, t=t, op=op, block_idx=block_idx,
        )
        h_shape = get_h(x).shape

        c_i, w_i, h_i = x.size(1), x.size(2), x.size(3)
        c_o, w_o, h_o = h_shape[1], h_shape[2], h_shape[3]

        a = torch.tensor(0., device=x.device)

        # Algorithm 1
        vT = torch.randn(c_i*w_i*h_i, pca_rank, device=x.device)
        vT, _ = torch.linalg.qr(vT)
        v = vT.T
        v = v.view(-1, c_i, w_i, h_i)

        for i in range(max_iter):
            v_prev = v.detach().cpu().clone()

            u = []
            time_s = time.time()

            v_buffer = list(v.chunk(num_chunk))
            for vi in v_buffer:
                # g = lambda a : get_h(x + a*vi.unsqueeze(0) if vi.size(0) == v.size(-1) else x + a*vi)
                g = lambda a : get_h(x + a*vi)
                ui = torch.func.jacfwd(g, argnums=0, has_aux=False, randomness='error')(a)
                u.append(ui.detach().cpu().clone())
            time_e = time.time()
            print('single v jacfwd t ==', time_e - time_s)
            u = torch.cat(u, dim=0)
            u = u.to(x.device)

            # time_s = time.time()
            # g = lambda a : get_h(x + a*v)
            # u = torch.func.jacfwd(g, argnums=0, has_aux=False, randomness='error')(a)
            # time_e = time.time()
            # print('single vi jacfwd t ==', time_e - time_s)

            g = lambda x : einsum(u, get_h(x), 'b c w h, i c w h -> b')
            v_ = torch.autograd.functional.jacobian(g, x)
            v_ = v_.view(-1, c_i*w_i*h_i)

            _, s, v = torch.linalg.svd(v_, full_matrices=False)
            v = v.view(-1, c_i, w_i, h_i)
            u = u.view(-1, c_o, w_o, h_o)
            
            convergence = torch.dist(v_prev, v.detach().cpu())
            print(f'power method : {i}-th step convergence : ', convergence)
            if torch.allclose(v_prev, v.detach().cpu(), atol=convergence_threshold) and (i > min_iter):
                print('reach convergence threshold : ', convergence)
                break

            if i == max_iter - 1:
                print('last convergence : ', convergence)

        u, s, vT = u.view(-1, c_o*w_o*h_o).T.detach(), s.sqrt().detach(), v.view(-1, c_i*w_i*h_i).detach()
        return u, s, vT
    
    def local_decoder_pullback_xt(
            self, x=None, t=None, op=None, block_idx=None, pca_rank=50, chunk_size=10,
            min_iter=10, max_iter=100, convergence_threshold=1e-3,
        ):
        '''
        Returns
            - u     : h space direction (dim = |dh, pc|)
            - s     : eigenvalue spectrum
            - vT    : e space direction (dim = |pc, de|)
        '''
        # necessary variables
        num_chunk = pca_rank // chunk_size if pca_rank % chunk_size == 0 else pca_rank // chunk_size + 1
        assert op == 'mid'
        assert block_idx == 0
        h = self.get_h(
            x=x, t=t, op=op, block_idx=block_idx,
        )

        get_h_to_e = lambda h : self.get_h_to_e(
            x=x, t=t, input_h=h,
            op=op, block_idx=block_idx, verbose=False,
        )

        c_i, w_i, h_i = h.size(1), h.size(2), h.size(3)
        c_o, w_o, h_o = x.size(1), x.size(2), x.size(3)

        a = torch.tensor(0., device=x.device)

        # Algorithm 1
        vT = torch.randn(c_i*w_i*h_i, pca_rank, device=x.device)
        vT, _ = torch.linalg.qr(vT)
        v = vT.T
        v = v.view(-1, c_i, w_i, h_i)

        for i in range(max_iter):
            # iterate over pca directions to avoid OOM
            v_prev = v.detach().cpu().clone()
            u = []
            time_s = time.time()

            v_buffer = list(v.chunk(num_chunk))
            for vi in v_buffer:
                g = lambda a : get_h_to_e(h + a*vi)
                ui = torch.func.jacfwd(g, argnums=0, has_aux=False, randomness='error')(a)
                u.append(ui.detach().cpu().clone())
            time_e = time.time()
            print('single v jacrev t ==', time_e - time_s)
            u = torch.cat(u, dim=0)
            u = u.to(x.device)
            
            # time_s = time.time()
            # g = lambda a : get_h_to_e(h + a*v)
            # u = torch.func.jacfwd(g, argnums=0, has_aux=False, randomness='error')(a)
            # time_e = time.time()
            # print('single vi jacrev t ==', time_e - time_s)

            g = lambda h : einsum(u, get_h_to_e(h), 'b c w h, i c w h -> b')
            v_ = torch.autograd.functional.jacobian(g, h)
            v_ = v_.view(-1, c_i*w_i*h_i)

            _, s, v = torch.linalg.svd(v_, full_matrices=False)
            v = v.view(-1, c_i, w_i, h_i)
            u = u.view(-1, c_o, w_o, h_o)
            
            convergence = torch.dist(v_prev, v.detach().cpu())
            print(f'power method : {i}-th step convergence : ', convergence)
            if torch.allclose(v_prev, v.detach().cpu(), atol=convergence_threshold) and (i > min_iter):
                print('reach convergence threshold : ', convergence)
                break
            else:
                print('convergence ==', torch.dist(v_prev, v.detach().cpu()))
        
        # decoder jac do not return vT
        u, s, vT = v.view(-1, c_i*w_i*h_i).T.detach(), s.sqrt().detach(), u.view(-1, c_o*w_o*h_o).detach()
        return u, s, vT
    
    def local_x0_decoder_pullback_xt(
            self, x=None, t=None, at=None, op=None, block_idx=None, pca_rank=50, chunk_size=10,
            num_chunk=None, min_iter=10, max_iter=100, convergence_threshold=1e-3,
        ):
        '''
        Returns
            - u     : h space direction (dim = |dh, pc|)
            - s     : eigenvalue spectrum
            - vT    : e space direction (dim = |pc, de|)
        '''
        # necessary variables
        assert op == 'mid'
        assert block_idx == 0
        h = self.get_h(
            x=x, t=t, op=op, block_idx=block_idx,
        )

        get_h_to_x0 = lambda h : (x - (1-at).sqrt() * self.get_h_to_e(
            x=x, t=t, input_h=h,
            op=op, block_idx=block_idx, verbose=False,
        )) / at.sqrt()

        c_i, w_i, h_i = h.size(1), h.size(2), h.size(3)
        c_o, w_o, h_o = x.size(1), x.size(2), x.size(3)

        a = torch.tensor(0., device=x.device)

        # Algorithm 1
        vT = torch.randn(c_i*w_i*h_i, pca_rank, device=x.device)
        vT, _ = torch.linalg.qr(vT)
        v = vT.T
        v = v.view(-1, c_i, w_i, h_i)

        for i in range(max_iter):
            # iterate over pca directions to avoid OOM
            v_prev = v.detach().cpu().clone()
            u = []
            time_s = time.time()

            v_buffer = list(v.chunk(v.size(0) // chunk_size))
            for vi in v_buffer:
                g = lambda a : get_h_to_x0(h + a*vi)
                ui = torch.func.jacfwd(g, argnums=0, has_aux=False, randomness='error')(a)
                u.append(ui.detach().cpu().clone())
            time_e = time.time()
            print('single v jacrev t ==', time_e - time_s)
            u = torch.cat(u, dim=0)
            u = u.to(x.device)
            
            # time_s = time.time()
            # g = lambda a : get_h_to_x0(h + a*v)
            # u = torch.func.jacfwd(g, argnums=0, has_aux=False, randomness='error')(a)
            # time_e = time.time()
            # print('single vi jacrev t ==', time_e - time_s)

            g = lambda h : einsum(u, get_h_to_x0(h), 'b c w h, i c w h -> b')
            v_ = torch.autograd.functional.jacobian(g, h)
            v_ = v_.view(-1, c_i*w_i*h_i)

            _, s, v = torch.linalg.svd(v_, full_matrices=False)
            v = v.view(-1, c_i, w_i, h_i)
            u = u.view(-1, c_o, w_o, h_o)
            
            convergence = torch.dist(v_prev, v.detach().cpu())
            print(f'power method : {i}-th step convergence : ', convergence)
            if torch.allclose(v_prev, v.detach().cpu(), atol=convergence_threshold) and (i > min_iter):
                print('reach convergence threshold : ', convergence)
                break
            else:
                print('convergence ==', torch.dist(v_prev, v.detach().cpu()))
        
        # decoder jac do not return vT
        u, s, vT = v.view(-1, c_i*w_i*h_i).T.detach(), s.sqrt().detach(), u.view(-1, c_o*w_o*h_o).detach()
        return u, s, vT

##################
# pullback-utils #  
##################
import skimage
def fourier_regularization(src, perturbed_src, noise_p, noise_q, fft_smoothing=False, histogram_matching=False):
    # ref : https://github.com/parlance-zz/g-diffuser-bot/blob/old-diffuserslib-beta/diffuser_server.py
    # src -> F(src)
    src_fft = _fft2(src)
    src_dist = np.absolute(src_fft)
    src_phase = src_fft / src_dist

    # perturbed_src -> F(src)
    perturbed_src_fft = _fft2(perturbed_src)
    perturbed_src_rgb = np.real(_ifft2(perturbed_src_fft))

    # perform the actual shaping
    shaped_perturbed_src_fft = _fft2(perturbed_src_rgb)
    shaped_perturbed_src_fft_dist = np.absolute(shaped_perturbed_src_fft)**noise_p * src_dist**noise_q

    print('shaped_perturbed_src_fft_dist.max(), shaped_perturbed_src_fft_dist.min(), shaped_perturbed_src_fft_dist.mean(), shaped_perturbed_src_fft_dist.std()')
    print( shaped_perturbed_src_fft_dist.max(), shaped_perturbed_src_fft_dist.min(), shaped_perturbed_src_fft_dist.mean(), shaped_perturbed_src_fft_dist.std())

    # signal 크기는 유지하되, low freq 가 지나치게 dominant 하지 않도록 regularize
    if fft_smoothing:
        var_src_dist = src_dist - src_dist.mean()
        var_shaped_perturbed_src_fft_dist = shaped_perturbed_src_fft_dist - shaped_perturbed_src_fft_dist.mean()
        shaped_perturbed_src_fft_dist     = shaped_perturbed_src_fft_dist.mean() + var_shaped_perturbed_src_fft_dist * \
                                            (var_src_dist.std() / var_shaped_perturbed_src_fft_dist.std())

        print('ratio')
        print( var_src_dist.std() / var_shaped_perturbed_src_fft_dist.std())

    shaped_perturbed_src_fft = shaped_perturbed_src_fft_dist * src_phase
    shaped_noise = np.real(_ifft2(shaped_perturbed_src_fft))

    # scikit-image is used for histogram matching, very convenient!
    if histogram_matching:
        shaped_noise -= np.min(shaped_noise)
        shaped_noise /= np.max(shaped_noise)
        shaped_noise = skimage.exposure.match_histograms(shaped_noise, src, channel_axis=2)
    
    return shaped_noise

def _fft2(data):
    if data.ndim > 2: # has channels
        out_fft = np.zeros((data.shape[0], data.shape[1], data.shape[2]), dtype=np.complex128)
        for c in range(data.shape[2]):
            c_data = data[:,:,c]
            out_fft[:,:,c] = np.fft.fft2(np.fft.fftshift(c_data),norm="ortho")
            out_fft[:,:,c] = np.fft.ifftshift(out_fft[:,:,c])
    else: # one channel
        out_fft = np.zeros((data.shape[0], data.shape[1]), dtype=np.complex128)
        out_fft[:,:] = np.fft.fft2(np.fft.fftshift(data),norm="ortho")
        out_fft[:,:] = np.fft.ifftshift(out_fft[:,:])
    
    return out_fft

def _ifft2(data):
    if data.ndim > 2: # has channels
        out_ifft = np.zeros((data.shape[0], data.shape[1], data.shape[2]), dtype=np.complex128)
        for c in range(data.shape[2]):
            c_data = data[:,:,c]
            out_ifft[:,:,c] = np.fft.ifft2(np.fft.fftshift(c_data),norm="ortho")
            out_ifft[:,:,c] = np.fft.ifftshift(out_ifft[:,:,c])
    else: # one channel
        out_ifft = np.zeros((data.shape[0], data.shape[1]), dtype=np.complex128)
        out_ifft[:,:] = np.fft.ifft2(np.fft.fftshift(data),norm="ortho")
        out_ifft[:,:] = np.fft.ifftshift(out_ifft[:,:])
        
    return out_ifft

#########
# utils #
#########
def get_timestep_embedding(timesteps, embedding_dim, device=None):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    device = timesteps.device if device is None else device
    timesteps = timesteps.to(device)

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    emb = emb.to(device)
    return emb

def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)

def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

##############
# sub-module #
##############
class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(
            x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.temb_proj = torch.nn.Linear(temb_channels,
                                         out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        
        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_

#########
# utils #
#########
def normalize_wrt_batch(x, norm=1):
    # figure out the shape of x
    x_shape = [1 for _ in x.shape]
    x_shape[0] = x.size(0)

    # normalize x w.r.t. batch
    x = norm * x / (x.view(x.size(0), -1).norm(dim=-1).view(x_shape) + 1e-6)

    try:
        assert ((x.view(x.size(0), -1).norm(dim=-1) - norm) < 1e-3 * norm).all(), f'x is not normalized : {(x.view(x.size(0), -1).norm(dim=-1) - norm).abs()}'
    except:
        print(f'!!!!!!!!!!!!!!!!!input is not normalized : {(x.view(x.size(0), -1).norm(dim=-1) - norm).abs()}!!!!!!!!!!!!!!!!!')
    return x