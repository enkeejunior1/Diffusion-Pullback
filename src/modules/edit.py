import gc
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.utils as tvu

import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange, einsum

from utils.utils import (
    get_dataset,
    get_stable_diffusion_model,
    get_custom_diffusion_model,
    get_custom_diffusion_scheduler,
    get_stable_diffusion_scheduler,
)

######################
# LDM ; use diffuser #
######################
from diffusers import (
    # DDIMInverseScheduler,
    DDIMScheduler, 
)

class EditStableDiffusion(object):
    def __init__(self, args):
        # default setting
        self.seed = args.seed
        self.pca_device     = args.pca_device
        self.buffer_device  = args.buffer_device
        self.memory_bound   = args.memory_bound

        # get model
        self.pipe = get_stable_diffusion_model(args)
        self.vae  = self.pipe.vae
        self.unet = self.pipe.unet

        self.dtype  = args.dtype
        self.device = self.pipe._execution_device

        # args (diffusion schedule)
        self.scheduler = get_stable_diffusion_scheduler(args, self.pipe.scheduler)
        self.for_steps = args.for_steps
        self.inv_steps = args.inv_steps
        self.use_yh_custom_scheduler = args.use_yh_custom_scheduler

        # get image
        self.c_in = args.c_in
        self.image_size = args.image_size
        self.dataset = get_dataset(args)
        self.dataset_name = args.dataset_name

        # args (prompt)
        self.for_prompt         = args.for_prompt if len(args.for_prompt.split(',')[0]) <= 3 else ','.join([args.for_prompt.split(',')[0]])
        self.neg_prompt         = args.neg_prompt if len(args.neg_prompt.split(',')[0]) <= 3 else ','.join([args.neg_prompt.split(',')[0]])
        self.null_prompt        = ""
        self.inv_prompt         = args.inv_prompt if len(args.inv_prompt.split(',')[0]) <= 3 else ','.join([args.inv_prompt.split(',')[0]])

        self.for_prompt_emb     = self._get_prompt_emb(args.for_prompt)
        self.neg_prompt_emb     = self._get_prompt_emb(args.neg_prompt)
        self.null_prompt_emb    = self._get_prompt_emb("")
        self.inv_prompt_emb     = self._get_prompt_emb(args.inv_prompt)
        
        # args (guidance)
        self.guidance_scale     = args.guidance_scale
        
        # args (h space edit)        
        self.edit_prompt        = args.edit_prompt 
        self.edit_prompt_emb    = self._get_prompt_emb(args.edit_prompt)
        # self.h_space_guidance_scale = args.h_space_guidance_scale
        # self.use_match_prompt = args.use_match_prompt

        # self.edit_xt = args.edit_xt
        # self.edit_ht = args.edit_ht
        
        # self.unet.after_res  = args.after_res
        # self.unet.after_sa   = args.after_sa

        # x-space guidance
        self.x_edit_step_size = args.x_edit_step_size
        self.x_space_guidance_edit_step         = args.x_space_guidance_edit_step
        self.x_space_guidance_scale             = args.x_space_guidance_scale
        self.x_space_guidance_num_step          = args.x_space_guidance_num_step
        self.x_space_guidance_use_edit_prompt   = args.x_space_guidance_use_edit_prompt

        # args (h space edit + currently not using. please refer main.py)
        self.scheduler.set_timesteps(self.for_steps, device=self.device)
        self.edit_t         = args.edit_t
        self.edit_t_idx     = (self.scheduler.timesteps - self.edit_t * 1000).abs().argmin()

        # path
        self.result_folder = args.result_folder
        self.obs_folder    = args.obs_folder

    @torch.no_grad()
    def run_DDIMforward(self, num_samples=5):
        print('start DDIMforward')
        self.EXP_NAME = f'DDIMforward-for_{self.for_prompt}'

        # get latent code
        zT = torch.randn(num_samples, 4, 64, 64).to(device=self.device, dtype=self.dtype)

        # simple DDIMforward
        self.DDIMforwardsteps(zT, t_start_idx=0, t_end_idx=-1)

    @torch.no_grad()
    def run_DDIMinversion(self, idx, guidance=None, vis_traj=False):
        '''
        Prompt
            (CFG)       pos : inv_prompt, neg : null_prompt
            (no CFG)    pos : inv_prompt
        '''
        print('start DDIMinversion')
        self.EXP_NAME = f'DDIMinversion-{self.dataset_name}-{idx}-for_{self.for_prompt}-inv_{self.inv_prompt}'

        # inversion scheduler
        # self.pipe.scheduler = self.scheduler

        # before start
        num_inference_steps = self.inv_steps
        do_classifier_free_guidance = (self.guidance_scale > 1.0) & (guidance is not None)
        # cross_attention_kwargs      = None

        # set timestep (we do not use default scheduler set timestep method)
        if self.use_yh_custom_scheduler:
            self.scheduler.set_timesteps(num_inference_steps, device=self.device, is_inversion=True)
        else:
            raise ValueError('recommend to use yh custom scheduler')
            self.scheduler = DDIMInverseScheduler.from_config(self.scheduler.config)
            self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        # get image
        x0 = self.dataset[idx]
        tvu.save_image((x0 / 2 + 0.5).clamp(0, 1), os.path.join(self.result_folder, f'original_x0-{self.EXP_NAME}.png'))

        # get latent
        z0 = self.vae.encode(x0).latent_dist
        z0 = z0.sample()
        z0 = z0 * 0.18215

        ##################
        # denoising loop #
        ##################
        latents = z0
        for i, t in enumerate(timesteps):
            if i == len(timesteps) - 1:
                break

            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            if do_classifier_free_guidance:
                prompt_emb = torch.cat([self.null_prompt_emb.repeat(latents.size(0), 1, 1), self.inv_prompt_emb.repeat(latents.size(0), 1, 1)], dim=0)
            else:
                prompt_emb = self.inv_prompt_emb.repeat(latents.size(0), 1, 1)

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input, t,
                encoder_hidden_states=prompt_emb,
                # cross_attention_kwargs=cross_attention_kwargs,
            ).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, eta=0).prev_sample

        # check latent code with forward
        # self.DDIMforwardsteps(latents, t_start_idx=0, t_end_idx=-1, return_latents=False, guidance=None, save_as='image')

        return latents

    @torch.no_grad()
    def run_edit_local_encoder_pullback_zt(
            self, idx, op, block_idx, vis_num, vis_num_pc=1, vis_vT=False, pca_rank=50, edit_prompt=None, edit_t=None, 
        ):
        print(f'current experiment : idx : {idx}, op : {op}, block_idx : {block_idx}, vis_num : {vis_num}, vis_num_pc : {vis_num_pc}, pca_rank : {pca_rank}, edit_prompt : {edit_prompt}')
        '''
        edit latent variable w/ local direction derived from **PULLBACK METRIC**

        1. z0 -> zT -> zt -> z0 ; we edit latent variable zt
        2. get local basis of h-space (u) and x-space (v) by using the power method
        3. edit sample with x-space guidance
        '''
        # set edit prompt
        if edit_prompt is not None:
            self.edit_prompt = edit_prompt
            self.edit_prompt_emb = self._get_prompt_emb(self.edit_prompt)

        # set edit_t
        self.scheduler.set_timesteps(self.for_steps)
        # self.h_t = h_t
        # self.h_t_idx = (self.scheduler.timesteps - self.h_t * 1000).abs().argmin()
        # self.edit_t = edit_t
        # self.edit_t_idx = (self.scheduler.timesteps - self.edit_t * 1000).abs().argmin()

        # if self.use_match_prompt:
        #     self._set_match_prompt()

        # get latent code (zT -> zt)
        zT = self.run_DDIMinversion(idx=idx)
        zt, t, t_idx = self.DDIMforwardsteps(zT, t_start_idx=0, t_end_idx=self.edit_t_idx)
        assert t_idx == self.edit_t_idx

        # get local basis
        local_basis_name = f'local_basis-{self.dataset_name}_{idx}-{self.edit_t}T-"{self.edit_prompt}"-{op}-block_{block_idx}-seed_{self.seed}'

        save_dir = f'./inputs/local_encoder_pullback_stable_diffusion-dataset_{self.dataset_name}-num_steps_{self.for_steps}-pca_rank_{pca_rank}'
        os.makedirs(save_dir, exist_ok=True)

        u_path = os.path.join(save_dir, 'u-' + local_basis_name + '.pt')
        s_path = os.path.join(save_dir, 's-' + local_basis_name + '.pt')
        vT_path = os.path.join(save_dir, 'vT-' + local_basis_name + '.pt')
        
        # load pre-computed local basis
        if os.path.exists(u_path) and os.path.exists(vT_path):
            u = torch.load(u_path, map_location=self.device).type(self.dtype)
            vT = torch.load(vT_path, map_location=self.device).type(self.dtype)
            
        # computed local basis
        else:
            print('!!!RUN LOCAL PULLBACK!!!')
            # zt, t, t_idx = self.DDIMforwardsteps(zT, t_start_idx=0, t_end_idx=self.edit_t_idx)
            u, s, vT = self.unet.local_encoder_pullback_zt(
                sample=zt, timestep=t, encoder_hidden_states=self.edit_prompt_emb, op=op, block_idx=block_idx, 
                pca_rank=pca_rank, chunk_size=5, min_iter=10, max_iter=50, convergence_threshold=1e-4,
            )

            vT = vT.to(device=self.device, dtype=self.dtype)

            # save semantic direction in h-space
            torch.save(u, u_path)
            torch.save(s, s_path)
            torch.save(vT, vT_path)

            # plot eigenvalue spectrum
            plt.scatter(range(s.size(0)), s.cpu().tolist(), s=1)
            plt.savefig(os.path.join(save_dir, f'eigenvalue_spectrum-{local_basis_name}.png'))
            plt.close()

            # visualize vT (using pca basis)
            pca_vT = vT.view(-1, *zT.shape[1:]).permute(0, 2, 3, 1)
            pca_vT = pca_vT.reshape(-1, 4)
            _, _, pca_basis = torch.pca_lowrank(pca_vT, q=3, center=True, niter=2)
            vis_vT = einsum(vT.view(-1, *zT.shape[1:]), pca_basis, 'b c w h, c p -> b p w h')

            vis_vT = vis_vT - vis_vT.min()
            vis_vT = vis_vT / vis_vT.max()
            tvu.save_image(
                vis_vT, os.path.join(self.obs_folder, f'vT-{local_basis_name}.png')
            )
            del pca_vT, vis_vT, pca_basis
            torch.cuda.empty_cache()

        u = u / u.norm(dim=0, keepdim=True)
        vT = vT / vT.norm(dim=1, keepdim=True)

        # original_zt, _, _ = self.DDIMforwardsteps(
        #     zT, t_start_idx=0, t_end_idx=self.edit_t_idx, guidance=None,
        #     vis_psd=False, return_latents=True, save_image='image',
        # )

        original_zt = zt.clone()
        for pc_idx in range(vis_num_pc):
            for direction in [1, -1]: # +v, -v
                # declare experiment name
                if direction == 1:
                    self.EXP_NAME = f'Edit_zt-{self.dataset_name}_{idx}-edit_{self.edit_t}T-{op}-block_{block_idx}-pc_{pc_idx:0=3d}_pos-edit_prompt_{self.edit_prompt}'
                elif direction == -1:
                    self.EXP_NAME = f'Edit_zt-{self.dataset_name}_{idx}-edit_{self.edit_t}T-{op}-block_{block_idx}-pc_{pc_idx:0=3d}_neg-edit_prompt_{self.edit_prompt}'

                # skip experiment if already done
                if os.path.exists(os.path.join(self.result_folder, self.EXP_NAME + '.png')):
                    print(f'!!!ALREADY DONE EXP!!!')
                    print(f'EXP_NAME: {self.EXP_NAME}')
                    continue

                vk = direction*vT[pc_idx, :].view(-1, *zT.shape[1:])

                # edit zt along vk direction with **x-space guidance**
                zt_list = [original_zt.clone()]
                for _ in tqdm(range(self.x_space_guidance_num_step), desc='x_space_guidance edit'):
                    zt_edit = self.x_space_guidance(
                        zt_list[-1], t_idx=self.edit_t_idx, vk=vk, 
                        single_edit_step=self.x_space_guidance_edit_step,
                        use_edit_prompt=self.x_space_guidance_use_edit_prompt,
                    )
                    zt_list.append(zt_edit)
                zt = torch.cat(zt_list, dim=0)
                zt = zt[::(zt.size(0) // vis_num)]

                # zt -> z0
                self.DDIMforwardsteps(
                    zt, t_start_idx=self.edit_t_idx, t_end_idx=-1,
                )

    @torch.no_grad()
    def run_sample_encoder_local_tangent_space_zt(
            self, h_t, op, block_idx, pca_rank=50, num_local_basis=10, use_edit_prompt=None, edit_prompt=None, vis_vT=True,
        ):
        '''sample T_x for analysis'''
        # experiment parametersk
        self.scheduler.set_timesteps(self.for_steps)
        h_t_idx = (self.scheduler.timesteps - h_t * 1000).abs().argmin()

        # path to save local tangent space
        save_dir = f'./inputs/local_encoder_pullback_stable_diffusion_{self.sd_ver}-dataset_{self.dataset_name}-scheduler_{self.scheduler_name}-num_steps_{self.for_steps}-pca_rank_{pca_rank}'
        os.makedirs(save_dir, exist_ok=True)
        
        for local_basis_idx in range(num_local_basis):
            assert (use_edit_prompt is None) != (edit_prompt is None)
            self._set_edit_prompt(sample_idx=local_basis_idx, use_edit_prompt=use_edit_prompt, edit_prompt=edit_prompt)
            self.EXP_NAME = f'zt-{self.dataset_name}_{local_basis_idx}-{h_t}T-"{self.edit_prompt}"-{op}-block_{block_idx}-seed_{self.seed}-ver_{self.sd_ver}'

            if self.unet.after_res:
                self.EXP_NAME = self.EXP_NAME + '-after_res'
            elif self.unet.after_sa:
                self.EXP_NAME = self.EXP_NAME + '-after_sa'

            u_path      = os.path.join(save_dir, 'u-' + self.EXP_NAME + '.pt')
            s_path      = os.path.join(save_dir, 's-' + self.EXP_NAME + '.pt')
            vT_path     = os.path.join(save_dir, 'vT-' + self.EXP_NAME + '.pt')
            s_fig_path  = os.path.join(save_dir, 'eigenvalue_spectrum-' + self.EXP_NAME + '.png')

            if os.path.exists(u_path) & os.path.exists(s_path) & os.path.exists(vT_path):
                print(f'!!!ALREADY SAMPLED LOCAL BASIS IDX : {local_basis_idx}!!!')
                continue

            else:
                print(f'!!!SAMPLE LOCAL BASIS IDX : {local_basis_idx}!!!')

            # get latent code
            if self.dataset_name == 'Random':
                zT = self.dataset[local_basis_idx]
            else:
                zT = self.run_DDIMinversion(idx=local_basis_idx)

            # zT -> zt
            zt, t, _ = self.DDIMforwardsteps(zT, t_start_idx=0, t_end_idx=h_t_idx)
            zt = zt.to(self.device)

            u, s, vT = self.unet.local_encoder_pullback_zt(
                sample=zt, timestep=t, encoder_hidden_states=self.edit_prompt_emb, op=op, block_idx=block_idx, 
                pca_rank=pca_rank, chunk_size=5, min_iter=10, max_iter=50, convergence_threshold=1e-3,
            )

            if vis_vT:
                pca_vT = vT.view(-1, *zT.shape[1:]).permute(0, 2, 3, 1)
                pca_vT = pca_vT.reshape(-1, 4)
                _, _, pca_basis = torch.pca_lowrank(pca_vT, q=3, center=True, niter=2)
                vis_pca_vT = einsum(vT.view(-1, *zT.shape[1:]), pca_basis, 'b c w h, c p -> b p w h')

                vis_pca_vT = vis_pca_vT - vis_pca_vT.min()
                vis_pca_vT = vis_pca_vT / vis_pca_vT.max()
                tvu.save_image(
                    vis_pca_vT, os.path.join(save_dir, f'vT-{self.EXP_NAME}.png')
                )
                del pca_vT, vis_pca_vT, pca_basis
                torch.cuda.empty_cache()


            # save eigenvalues spectrum
            plt.scatter(range(s.size(0)), s.cpu().tolist(), s=1)
            plt.savefig(s_fig_path, dpi = 80)
            plt.close()

            # save semantic direction in h-space
            torch.save(u, u_path)
            torch.save(s, s_path)
            torch.save(vT, vT_path)
        return

    @torch.no_grad()
    def DDIMforwardsteps(
            self, zt, t_start_idx, t_end_idx, **kwargs
        ):
        '''
        Prompt
            (CFG)       pos : for_prompt, neg : neg_prompt
            (no CFG)    pos : for_prompt
        '''
        print('start DDIMforward')
        # before start
        num_inference_steps = self.for_steps
        do_classifier_free_guidance = self.guidance_scale > 1.0
        # cross_attention_kwargs      = None
        memory_bound = memory_bound // 2 if do_classifier_free_guidance else self.memory_bound
        print('do_classifier_free_guidance : ', do_classifier_free_guidance)

        # set timestep (we do not use default scheduler set timestep method)
        if self.use_yh_custom_scheduler:
            self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        else:
            self.scheduler = DDIMScheduler.from_config(self.scheduler.config)
            self.scheduler.set_timesteps(num_inference_steps, device=self.device)

        # print(f'timesteps : {timesteps}')
        print(f'self.neg_prompt : {self.neg_prompt}')
        print(f'self.for_prompt : {self.for_prompt}')

        # save traj
        latents = zt

        #############################################
        # denoising loop (t_start_idx -> t_end_idx) #
        #############################################
        for t_idx, t in enumerate(self.scheduler.timesteps):
            # skip
            if (t_idx < t_start_idx): 
                continue
                
            # start sampling
            elif t_start_idx == t_idx:
                print('t_start_idx : ', t_idx)

            # end sampling
            elif t_idx == t_end_idx:
                print('t_end_idx : ', t_idx)
                return latents, t, t_idx

            # split zt to avoid OOM
            latents = latents.to(device=self.buffer_device)
            if latents.size(0) == 1:
                latents_buffer = [latents]
            else:
                latents_buffer = list(latents.chunk(latents.size(0) // memory_bound))

            # loop over buffer
            for buffer_idx, latents in enumerate(latents_buffer):
                # overload to device
                latents = latents.to(device=self.device)

                # expand the latents if we are doing classifier free guidance
                if do_classifier_free_guidance:
                    latent_model_input = torch.cat([latents] * 2, dim=0)
                    prompt_emb = torch.cat([self.neg_prompt_emb.repeat(latents.size(0), 1, 1), self.for_prompt_emb.repeat(latents.size(0), 1, 1)], dim=0)
                else:
                    latent_model_input = latents
                    prompt_emb = self.for_prompt_emb.repeat(latents.size(0), 1, 1)
                
                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input, t,
                    encoder_hidden_states=prompt_emb,
                    # cross_attention_kwargs=cross_attention_kwargs,
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, eta=0).prev_sample

                # save latents in buffer
                latents_buffer[buffer_idx] = latents.to(self.buffer_device)

            latents = torch.cat(latents_buffer, dim=0)
            latents = latents.to(device=self.device)
            del latents_buffer
            torch.cuda.empty_cache()

        # decode with vae
        latents = 1 / 0.18215 * latents
        x0 = self.vae.decode(latents).sample
        x0 = (x0 / 2 + 0.5).clamp(0, 1)
        tvu.save_image(x0, os.path.join(self.result_folder, f'x0_gen-{self.EXP_NAME}.png'), nrow = x0.size(0))

        return latents

    @torch.no_grad()
    def x_space_guidance(self, zt, t_idx, vk, single_edit_step, use_edit_prompt=False):
        # necesary parameters
        t = self.scheduler.timesteps[t_idx]

        # edit zt with vk
        zt_edit = zt + single_edit_step * vk

        # predict the noise residual
        et = self.unet(
            torch.cat([zt, zt_edit], dim=0), t, 
            encoder_hidden_states=self.edit_prompt_emb.repeat(2, 1, 1)
            # cross_attention_kwargs=None,
        ).sample

        # DDS regularization
        et_null, et_edit = et.chunk(2)
        zt_edit = zt + self.x_space_guidance_scale * (et_edit - et_null)
        return zt_edit

    # utils
    def _get_prompt_emb(self, prompt):
        # prompt_embeds = self.pipe._encode_prompt(
        #     prompt,
        #     device = self.device,
        #     num_images_per_prompt = 1,
        #     do_classifier_free_guidance = False,
        # )
        prompt_embeds = self.pipe._encode_prompt(
            prompt,
            device = self.device,
            num_images_per_prompt = 1,
            do_classifier_free_guidance = False,
            # prompt_embeds = None,
            negative_prompt = None,
            # negative_prompt_embeds= None,
        )

        return prompt_embeds
    
    def _classifier_free_guidance(self, noise_pred_uncond, noise_pred_cond):
        return 
    
    def _set_edit_prompt(self,edit_prompt=None):
        self.edit_prompt = edit_prompt
        self.edit_prompt_emb = self._get_prompt_emb(self.edit_prompt)

    def _set_match_prompt(self):
        self.for_prompt = self.edit_prompt
        self.for_prompt_emb = self.edit_prompt_emb
        self.inv_prompt = self.edit_prompt
        self.inv_prompt_emb = self.edit_prompt_emb

################
# Uncond model #
################
class EditUncondDiffusion(object):
    def __init__(self, args):
        # default setting
        self.pca_device     = args.pca_device
        self.buffer_device  = args.buffer_device
        self.memory_bound   = args.memory_bound
        self.device = args.device
        self.dtype = args.dtype
        self.seed = args.seed
        self.save_result_as = args.save_result_as

        # get model
        self.unet = get_custom_diffusion_model(args)
        self.scheduler = get_custom_diffusion_scheduler(args)
        self.model_name = args.model_name

        if 'HF' in self.model_name:
            self.scheduler = self.unet.scheduler if self.scheduler is None else self.scheduler
            self.unet = self.unet.unet

        # args (model)        
        self.image_size = args.image_size
        self.c_in = 3

        # get image
        self.dataset = get_dataset(args)
        self.dataset_name = args.dataset_name

        # args (diffusion schedule)
        self.for_steps = args.for_steps
        self.inv_steps = args.inv_steps
        self.use_yh_custom_scheduler = args.use_yh_custom_scheduler

        # args (edit)
        # self.h_space_guidance_scale = args.h_space_guidance_scale
        # self.use_match_prompt = args.use_match_prompt

        # self.edit_xt = args.edit_xt
        # self.edit_ht = args.edit_ht
        
        self.edit_t     = args.edit_t

        self.scheduler.set_timesteps(self.for_steps, device=self.device)
        self.edit_t_idx     = (self.scheduler.timesteps - self.edit_t * 1000).abs().argmin()
        self.performance_boosting_t_idx = (self.scheduler.timesteps - args.performance_boosting_t * 1000).abs().argmin() if args.performance_boosting_t > 0 else 1000
        print(f'performance_boosting_t_idx: {self.performance_boosting_t_idx}')
        
        # self.x_edit_step_size = args.x_edit_step_size
        # self.h_edit_step_size = args.h_edit_step_size

        # args (x-space guidance)
        self.use_x_space_guidance           = args.use_x_space_guidance
        self.x_space_guidance_edit_step     = args.x_space_guidance_edit_step
        self.x_space_guidance_scale         = args.x_space_guidance_scale
        self.x_space_guidance_num_step      = args.x_space_guidance_num_step

        # path
        self.result_folder = args.result_folder
        self.obs_folder    = args.obs_folder

    @torch.no_grad()
    def run_DDIMforward(self, num_samples=5):
        print('start DDIMforward')
        self.EXP_NAME = f'DDIMforward'

        # get latent code
        xT = torch.randn(num_samples, self.c_in, self.image_size, self.image_size, device=self.device, dtype=self.dtype)
        print('shape of xT : ', xT.shape)
        print('norm of xT : ', xT.view(num_samples, -1).norm(dim=-1))

        # simple DDIMforward
        self.DDIMforwardsteps(xT, t_start_idx=0, t_end_idx=-1, vis_psd=False)

    @torch.no_grad()
    def run_DDIMinversion(self, idx):
        '''
        Args
            - img idx
        Returns
            - zT
            - zt_traj
            - et_traj
        '''
        print('start DDIMinversion')
        EXP_NAME = f'DDIMinversion-{self.dataset_name}_{idx}'

        # before start
        num_inference_steps = self.inv_steps        
        if self.use_yh_custom_scheduler:
            self.scheduler.set_timesteps(num_inference_steps, device=self.device, is_inversion=True)
        else:
            # self.scheduler = DDIMInverseScheduler.from_config(self.scheduler.config)
            # self.scheduler.set_timesteps(num_inference_steps, device=self.device)
            raise ValueError('please set use_yh_custom_scheduler = True')
        timesteps = self.scheduler.timesteps
            
        # print(f'timesteps : {timesteps}')

        # get image
        x0 = self.dataset[idx]
        tvu.save_image((x0 / 2 + 0.5).clamp(0, 1), os.path.join(self.result_folder, f'original_x0-{EXP_NAME}.png'))

        ##################
        # denoising loop #
        ##################
        xt = x0.to(self.device, dtype=self.dtype)

        for i, t in enumerate(timesteps):
            if i == len(timesteps) - 1:
                break

            # 1. predict noise model_output
            et = self.unet(xt, t)
            if not isinstance(et, torch.Tensor):
                et = et.sample

            # 2. predict previous mean of image x_t-1 and add variance depending on eta
            # eta corresponds to η in paper and should be between [0, 1]
            # do x_t -> x_t-1
            xt = self.scheduler.step(
                et, t, xt, eta=0, use_clipped_model_output=None, 
            ).prev_sample

            # save traj
            # xt_traj.append(xt.detach().cpu().type(torch.float32).clone())
            # et_traj.append(et.detach().cpu().type(torch.float32).clone())

        # visualize latents, zt_traj, et_traj
        tvu.save_image(
            (xt / 2 + 0.5).clamp(0, 1), os.path.join(self.result_folder, f'xT-{EXP_NAME}.png'),
        )

        # check latent code with forward
        # self.DDIMforwardsteps(
        #     xt, t_start_idx=0, t_end_idx=-1,  
        #     vis_psd=False, save_image=True, return_xt=False, 
        # )

        return xt

    @torch.no_grad()
    def run_edit_local_encoder_pullback_zt(
            self, idx, vis_num, vis_num_pc=5, pca_rank=50, op='mid', block_idx=0, **kwargs,
        ):
        '''
        edit latent variable w/ local direction derived from **PULLBACK METRIC**

        1. xT -> xt -> x0 ; we edit latent variable xt
        2. get local basis of h-space (u) and x-space (v) by approximating the SVD of the jacobian of f : x -> h
        3. edit sample with predefined step size (sample += step_size * v)

        Args
            - idx               : sample idx
            - vis_num           : number of visualization editing steps
            - vis_num_pc        : number of visualization pc indexs
            - pca_rank          : pca rank
            - op                : h space operation idx = (down, mid, up)
            - block_idx         : h space block idx
        '''
        # get latent code
        if self.dataset_name == 'Random':
            xT = torch.randn(1, 3, 256, 256, dtype=self.dtype, device=self.device)
        else:
            xT = self.run_DDIMinversion(idx=idx)

        # xT -> xt
        xt, t, t_idx = self.DDIMforwardsteps(xT, t_start_idx=0, t_end_idx=self.edit_t_idx)
        assert t_idx == self.edit_t_idx
    
        # get local basis
        local_basis_name = f'local_basis-{self.dataset_name}_{idx}-{self.edit_t}T-{op}-block_{block_idx}-seed_{self.seed}'
        save_dir = f'./inputs/local_encoder_pullback_uncond-model_{self.model_name}-dataset_{self.dataset_name}-num_steps_{self.for_steps}-pca_rank_{pca_rank}'
        os.makedirs(save_dir, exist_ok=True)
        u_path = os.path.join(save_dir, 'u-' + local_basis_name + '.pt')
        vT_path = os.path.join(save_dir, 'vT-' + local_basis_name + '.pt')
        
        # load pre-computed local basis
        if os.path.exists(u_path) and os.path.exists(vT_path):
            u = torch.load(u_path, map_location=self.device).type(self.dtype)
            vT = torch.load(vT_path, map_location=self.device).type(self.dtype)

        # computed local basis using power method approximation
        else:
            print('!!!RUN LOCAL PULLBACK!!!')
            xt = xt.to(device=self.device, dtype=self.dtype)
            u, s, vT = self.unet.local_encoder_pullback_xt(
                x=xt, t=t, op=op, block_idx=block_idx, pca_rank=pca_rank, 
                min_iter=10, max_iter=50, convergence_threshold=1e-4,
            )

            # save semantic direction in h-space
            torch.save(u, u_path)
            torch.save(vT, vT_path)

            # visualize v, s
            plt.scatter(range(s.size(0)), s.cpu().tolist(), s=1)
            plt.savefig(os.path.join(save_dir, f'eigenvalue_spectrum-{local_basis_name}.png'), dpi = 80)
            plt.close()

            # visualize vT (min, max normalize)
            vis_vT = vT
            vis_vT = vis_vT - vis_vT.min()
            vis_vT = vis_vT / vis_vT.max()
            tvu.save_image(vT, os.path.join(self.obs_folder, f'vT-{local_basis_name}.png'))

        # normalize u, vT
        u = u / u.norm(dim=0, keepdim=True)
        vT = vT / vT.norm(dim=1, keepdim=True)

        # get latent code
        original_xt = xt.detach()
        for pc_idx in range(vis_num_pc):
            for direction in [1, -1]:
                if direction == 1:
                    self.EXP_NAME = f'Edit_xt-{self.dataset_name}_{idx}-edit_{self.edit_t}T-{op}-block_{block_idx}-pc_{pc_idx:0=3d}_pos'
                elif direction == -1:
                    self.EXP_NAME = f'Edit_xt-{self.dataset_name}_{idx}-edit_{self.edit_t}T-{op}-block_{block_idx}-pc_{pc_idx:0=3d}_neg'

                if os.path.exists(os.path.join(self.result_folder, f'x0_gen-{self.EXP_NAME}.png')):
                    print('!!!ALREADY DONE!!!')
                    print(f'!!!{self.EXP_NAME}!!!')
                    continue

                # edit latent variable
                vk = direction*vT[pc_idx, :].view(-1, *xt.shape[1:])

                xt_list = [original_xt.clone()]
                for _ in tqdm(range(self.x_space_guidance_num_step), desc='x_space_guidance edit'):
                    xt_edit = self.x_space_guidance(
                        xt_list[-1], t_idx=self.edit_t_idx, vk=vk, 
                        single_edit_step=self.x_space_guidance_edit_step,
                    )
                    xt_list.append(xt_edit)
                xt = torch.cat(xt_list, dim=0)
                xt = xt[::(xt.size(0) // vis_num)]

                # xt -> x0
                self.DDIMforwardsteps(xt, t_start_idx=self.edit_t_idx, t_end_idx=-1, performance_boosting=True)

        return xt
    
    @torch.no_grad()
    def run_edit_parallel_transport(
            self, sample_idx_0=None, sample_idx_1=None, op='mid', block_idx=0, vis_num=None, vis_vT=True, vis_num_pc=None, vis_pc_list=None, pca_rank=None, **kwargs,
        ):
        '''
        edit sample_idx_1 with uk from sample_idx_0
        '''
        # check if we have done this before
        vis_num_pc_list = list(range(vis_num_pc)) if vis_pc_list is None else vis_pc_list
        if vis_num_pc_list == []:
            pass
        elif os.path.exists(os.path.join(
                self.result_folder, f'x0_gen-' + f'xt-{self.dataset_name}-sample_idx_0_{sample_idx_0}-sample_idx_1_{sample_idx_1}-h_{self.h_t}T-edit_{self.edit_t}T-{op}-block_{block_idx}-seed_{self.seed}-pc_{vis_num_pc_list[-1]:0=3d}_neg' + '.png'
            )):
            print('!!!ALREADY DONE EXPERIMENT!!!')
            print(f'xt-{self.dataset_name}-sample_idx_0_{sample_idx_0}-sample_idx_1_{sample_idx_1}-h_{self.h_t}T-edit_{self.edit_t}T-{op}-block_{block_idx}-seed_{self.seed}')
            return

        # get latent code
        if self.dataset_name == 'Random':
            xT_0 = self.dataset[sample_idx_0]
            xT_1 = self.dataset[sample_idx_1]
        else:
            xT_0 = self.run_DDIMinversion(idx=sample_idx_0)
            xT_1 = self.run_DDIMinversion(idx=sample_idx_1)

        # xT -> xt
        xt_0, t, i = self.DDIMforwardsteps(xT_0, t_start_idx=0, t_end_idx=self.h_t_idx)
        xt_1, t, i = self.DDIMforwardsteps(xT_1, t_start_idx=0, t_end_idx=self.h_t_idx)
        
        # get local basis
        assert pca_rank == 50
        save_dir = f'./inputs/local_encoder_pullback_uncond-model_{self.model_name}-dataset_{self.dataset_name}-scheduler_{self.scheduler_name}-num_steps_{self.for_steps}-pca_rank_{pca_rank}'
        os.makedirs(save_dir, exist_ok=True)
        u_0_path = os.path.join(save_dir, 'u-' + f'xt-{self.dataset_name}_{sample_idx_0}-{self.h_t}T-{op}-block_{block_idx}-seed_{self.seed}' + '.pt')
        u_1_path = os.path.join(save_dir, 'u-' + f'xt-{self.dataset_name}_{sample_idx_1}-{self.h_t}T-{op}-block_{block_idx}-seed_{self.seed}' + '.pt')
        vT_0_path = os.path.join(save_dir, 'vT-' + f'xt-{self.dataset_name}_{sample_idx_0}-{self.h_t}T-{op}-block_{block_idx}-seed_{self.seed}' + '.pt')
        vT_1_path = os.path.join(save_dir, 'vT-' + f'xt-{self.dataset_name}_{sample_idx_1}-{self.h_t}T-{op}-block_{block_idx}-seed_{self.seed}' + '.pt')
        
        try:
            # if os.path.exists(u_0_path) and os.path.exists(u_1_path) and os.path.exists(vT_0_path) and os.path.exists(vT_1_path):
            u_0 = torch.load(u_0_path, map_location='cpu')
            u_0 = u_0.type(self.dtype)
            u_0 = u_0 / u_0.norm(dim=0, keepdim=True)
            
            vT_0 = torch.load(vT_0_path, map_location='cpu')
            vT_0 = vT_0.type(self.dtype)
            vT_0 = vT_0 / vT_0.norm(dim=1, keepdim=True)
        
        except:
            print('!!!RUN LOCAL PULLBACK!!!')
            xt_0 = xt_0.to(device=self.device, dtype=self.dtype)
            u_0, s, vT_0 = self.unet.local_encoder_pullback_xt(
                x=xt_0, t=t, op=op, block_idx=block_idx, pca_rank=pca_rank, 
                min_iter=10, max_iter=50, convergence_threshold=1e-4,
            )

            # save semantic direction in h-space
            torch.save(u_0, u_0_path)
            torch.save(vT_0, vT_0_path)
            u_0 = u_0.to('cpu')
            vT_0 = vT_0.to('cpu')

        try:
            u_1 = torch.load(u_1_path, map_location='cpu')
            u_1 = u_1.type(self.dtype)
            u_1 = u_1 / u_1.norm(dim=0, keepdim=True)

            vT_1 = torch.load(vT_1_path, map_location='cpu')
            vT_1 = vT_1.type(self.dtype)
            vT_1 = vT_1 / vT_1.norm(dim=1, keepdim=True)

        except:
            xt_1 = xt_1.to(device=self.device, dtype=self.dtype)
            u_1, s, vT_1 = self.unet.local_encoder_pullback_xt(
                x=xt_1, t=t, op=op, block_idx=block_idx, pca_rank=pca_rank, 
                min_iter=10, max_iter=50, convergence_threshold=1e-4,
            )

            # save semantic direction in h-space
            torch.save(u_1, u_1_path)
            torch.save(vT_1, vT_1_path)
            u_1 = u_1.to('cpu')
            vT_1 = vT_1.to('cpu')

        # get latent code
        original_xt_0, _, _ = self.DDIMforwardsteps(xT_0, t_start_idx=0, t_end_idx=self.edit_t_idx)
        original_xt_0 = original_xt_0.detach().to(device=self.device, dtype=self.dtype)

        original_xt_1, _, _ = self.DDIMforwardsteps(xT_1, t_start_idx=0, t_end_idx=self.edit_t_idx)
        original_xt_1 = original_xt_1.detach().to(device=self.device, dtype=self.dtype)
        
        vis_num_pc_list = list(range(vis_num_pc)) if vis_pc_list is None else vis_pc_list
        for pc_idx in vis_num_pc_list:
            # parallel transport sample (sample_idx_0 -> sample_idx_1)
            vk_list = []
            for direction in [1, -1]:
                if direction == 1:
                    self.EXP_NAME = f'xt-{self.dataset_name}-sample_idx_0_{sample_idx_0}-sample_idx_1_{sample_idx_1}-h_{self.h_t}T-edit_{self.edit_t}T-{op}-block_{block_idx}-seed_{self.seed}-pc_{pc_idx:0=3d}_pos'
                elif direction == -1:
                    self.EXP_NAME = f'xt-{self.dataset_name}-sample_idx_0_{sample_idx_0}-sample_idx_1_{sample_idx_1}-h_{self.h_t}T-edit_{self.edit_t}T-{op}-block_{block_idx}-seed_{self.seed}-pc_{pc_idx:0=3d}_neg'

                if os.path.exists(os.path.join(self.result_folder, f'x0_gen-{self.EXP_NAME}.png')):
                    print('!!!ALREADY DONE!!!')
                    print(f'!!!{self.EXP_NAME}!!!')
                    continue

                # edit latent variable
                if (self.edit_xt == 'parallel-x') & (self.edit_ht == 'default'):
                    vk = direction*vT_1.T @ (u_1.T @ u_0[:, pc_idx])
                    vk = vk / vk.norm()
                    vk = vk.view(-1, *original_xt_0.shape[1:]).to(self.device)
                    vk_list.append(vk.clone())

                    if self.use_x_space_guidance:
                        xt_list = [original_xt_1.clone()]
                        for _ in tqdm(range(self.x_space_guidance_num_step), desc='x_space_guidance edit'):
                            xt_edit = self.x_space_guidance(
                                xt_list[-1], t_idx=self.edit_t_idx, vk=vk, 
                                single_edit_step=self.x_space_guidance_edit_step,
                            )
                            xt_list.append(xt_edit)
                        xt = torch.cat(xt_list, dim=0)
                        xt = xt[::int(xt.size(0) / vis_num)]
                    
                    # xt -> x0
                    self.DDIMforwardsteps(xt, t_start_idx=self.edit_t_idx, t_end_idx=-1, performance_boosting=True)
            
            # original sample
            for direction in [1, -1]:
                if direction == 1:
                    self.EXP_NAME = f'xt-{self.dataset_name}-sample_idx_0_{sample_idx_0}-sample_idx_1_{sample_idx_0}-h_{self.h_t}T-edit_{self.edit_t}T-{op}-block_{block_idx}-seed_{self.seed}-pc_{pc_idx:0=3d}_pos'
                elif direction == -1:
                    self.EXP_NAME = f'xt-{self.dataset_name}-sample_idx_0_{sample_idx_0}-sample_idx_1_{sample_idx_0}-h_{self.h_t}T-edit_{self.edit_t}T-{op}-block_{block_idx}-seed_{self.seed}-pc_{pc_idx:0=3d}_neg'

                if os.path.exists(os.path.join(self.result_folder, f'x0_gen-{self.EXP_NAME}.png')):
                    print('!!!ALREADY DONE!!!')
                    print(f'!!!{self.EXP_NAME}!!!')
                    continue

                # edit latent variable
                if (self.edit_xt == 'parallel-x') & (self.edit_ht == 'default'):
                    vk = direction*vT_0[pc_idx, :].view(-1, *original_xt_0.shape[1:]).to(self.device)
                    vk = vk / vk.norm()
                    vk_list.append(vk.clone())

                    if self.use_x_space_guidance:
                        xt_list = [original_xt_0.clone()]
                        for _ in tqdm(range(self.x_space_guidance_num_step), desc='x_space_guidance edit'):
                            xt_edit = self.x_space_guidance(
                                xt_list[-1], t_idx=self.edit_t_idx, vk=vk, 
                                single_edit_step=self.x_space_guidance_edit_step,
                            )
                            xt_list.append(xt_edit)
                        xt = torch.cat(xt_list, dim=0)
                        xt = xt[::int(xt.size(0) / vis_num)]
                    
                    # xt -> x0
                    self.DDIMforwardsteps(xt, t_start_idx=self.edit_t_idx, t_end_idx=-1, performance_boosting=True)

            # save vk
            try:
                tvu.save_image(
                    torch.cat(vk_list, dim=0), os.path.join(self.obs_folder, f'vk-sample_idx_0_{sample_idx_0}-sample_idx_1_{sample_idx_1}-pc_{pc_idx:0=3d}.png')
                )
            except:
                pass
        return

    @torch.no_grad()
    def run_edit_global_frechet_mean_zt(
            self, idx, op, block_idx, vis_num, vis_num_pc=None, vis_pc_list=None, pca_rank=50, num_local_basis=100, local_projection=True, **kwargs,
        ):
        '''
        Find Frechet mean over SO(n) by py

        1. sample local basis
        2. Construct Frechet basis with pymanopt
        3. Edit samples with constructed frechet basis

        Args
            - idx               : sample idx
            - op                : h space operation idx = (down, mid, up)
            - block_idx         : h space block idx
            - vis_num           : number of visualization editing steps
            - vis_num_pc        : number of visualization pc indexs
            - vis_vT            : visualize vT or not
            - pca_rank          : pca rank
            - num_local_basis : number of local basis for construct global frechet mean
        '''
        vis_num_pc_list = list(range(vis_num_pc)) if vis_pc_list is None else vis_pc_list
        if vis_num_pc_list == []:
            pass
        elif os.path.exists(os.path.join(
                self.result_folder, f'xt-{self.dataset_name}_{idx}-h_{self.h_t}T-edit_{self.edit_t}T-{op}-block_{block_idx}-seed_{self.seed}-pc_{vis_num_pc_list[-1]:0=3d}_neg' + '.png'
            )):
            print(f'!!!ALREADY DONE EXP :', f'xt-{self.dataset_name}_{idx}-h_{self.h_t}T-edit_{self.edit_t}T-{op}-block_{block_idx}-seed_{self.seed}-pc_{vis_num_pc_list[-1]:0=3d}_neg', '!!!')
            return
        else:
            print(f'!!!DO EXP :', f'xt-{self.dataset_name}_{idx}-h_{self.h_t}T-edit_{self.edit_t}T-{op}-block_{block_idx}-seed_{self.seed}-pc_{vis_num_pc_list[-1]:0=3d}_neg', '!!!')

        # experiment paths
        save_dir = f'./inputs/global_frechet_mean_uncond-model_{self.model_name}-dataset_{self.dataset_name}-scheduler_{self.scheduler_name}-num_steps_{self.for_steps}-pca_rank_{pca_rank}'
        os.makedirs(save_dir, exist_ok=True)
        frechet_basis_name = f'{self.h_t}T-{op}-block_{block_idx}-seed_{self.seed}-num_local_basis_{num_local_basis}'
        frechet_basis_path_u = os.path.join(save_dir, 'u-' + frechet_basis_name + '.pt')
        frechet_basis_path_v = os.path.join(save_dir, 'v-' + frechet_basis_name + '.pt')

        if self.frechet_mean_space == 'h':
            if os.path.exists(frechet_basis_path_u):
                frechet_basis_u = torch.load(frechet_basis_path_u, map_location=self.device)
                frechet_basis_u = frechet_basis_u.type(self.dtype)
                frechet_basis_u = frechet_basis_u / torch.norm(frechet_basis_u, dim=0, keepdim=True)
                run_frechet_basis_opt = False
            else:
                run_frechet_basis_opt = True
                
        if self.frechet_mean_space == 'x':
            if os.path.exists(frechet_basis_path_v):
                frechet_basis_v = torch.load(frechet_basis_path_v, map_location=self.device)
                frechet_basis_v = frechet_basis_v.type(self.dtype)
                frechet_basis_v = frechet_basis_v / torch.norm(frechet_basis_v, dim=0, keepdim=True)
                run_frechet_basis_opt = False
            else:
                run_frechet_basis_opt = True

        if run_frechet_basis_opt:
            # 1. sample local basis
            local_basis_list_u = []
            local_basis_list_v = []
            local_basis_save_dir = f'./inputs/local_encoder_pullback_uncond-model_{self.model_name}-dataset_Random-scheduler_{self.scheduler_name}-num_steps_{self.for_steps}-pca_rank_{pca_rank}'
            os.makedirs(local_basis_save_dir, exist_ok=True)

            for local_basis_idx in range(num_local_basis):
                self.EXP_NAME = f'xt-Random_{local_basis_idx}-{self.h_t}T-{op}-block_{block_idx}-seed_{self.seed}'
                u_path = os.path.join(local_basis_save_dir, 'u-' + self.EXP_NAME + '.pt')
                s_path = os.path.join(local_basis_save_dir, 's-' + self.EXP_NAME + '.pt')
                vT_path = os.path.join(local_basis_save_dir, 'vT-' + self.EXP_NAME + '.pt')

                if os.path.exists(u_path) and os.path.exists(vT_path):
                    u = torch.load(u_path, map_location='cpu')
                    u = u.type(self.dtype)
                    vT = torch.load(vT_path, map_location='cpu')
                    vT = vT.type(self.dtype)

                else:
                    print(f'!!!SAMPLE LOCAL BASIS IDX : {local_basis_idx}!!!')
                    from utils.utils import RandomLatentDataset
                    from configs.paths import DATASET_PATHS

                    # get latent code
                    random_dataset = RandomLatentDataset(
                        image_root = DATASET_PATHS['Random'],
                        c_in = self.c_in,
                        image_size = self.image_size,
                        dataset_name = 'Random',
                        device = self.device, 
                        dtype = self.dtype, 
                    )
                    xT = random_dataset[local_basis_idx]

                    # xT -> xt
                    xt, t, _ = self.DDIMforwardsteps(xT, t_start_idx=0, t_end_idx=self.h_t_idx)
                    xt = xt.to(dtype=self.dtype, device=self.device)

                    u, s, vT = self.unet.local_encoder_pullback_xt(
                        x=xt, t=t, op=op, block_idx=block_idx, pca_rank=pca_rank,
                        min_iter=10, max_iter=50, convergence_threshold=1e-4,
                    )
                    u, s, vT = u.to('cpu'), s.to('cpu'), vT.to('cpu')

                    # save semantic direction in h-space
                    torch.save(u, u_path)
                    torch.save(s, s_path)
                    torch.save(vT, vT_path)
                
                local_basis_list_u.append(u.detach().clone())
                local_basis_list_v.append(vT.T.detach().clone())

            # 2. Construct Frechet basis with pymanopt
            if self.frechet_mean_space == 'h':
                if not os.path.exists(frechet_basis_path_u):
                    frechet_basis_u = compute_frechet_basis(local_basis_list_u, max_iterations=10000, max_time=10000)
                    frechet_basis_u = torch.from_numpy(frechet_basis_u).to(dtype=self.dtype, device=self.device)
                    torch.save(frechet_basis_u, frechet_basis_path_u)
                else:
                    frechet_basis_u = torch.load(frechet_basis_path_u, map_location=self.device)
                    frechet_basis_u = frechet_basis_u.type(self.dtype)
                    frechet_basis_u = frechet_basis_u / torch.norm(frechet_basis_u, dim=0, keepdim=True)
            
            if self.frechet_mean_space == 'x':
                if not os.path.exists(frechet_basis_path_v):
                    frechet_basis_v = compute_frechet_basis(local_basis_list_v, max_iterations=10000, max_time=50000)
                    frechet_basis_v = torch.from_numpy(frechet_basis_v).to(dtype=self.dtype, device=self.device)
                    torch.save(frechet_basis_v, frechet_basis_path_v)
                else:
                    frechet_basis_v = torch.load(frechet_basis_path_v, map_location=self.device)
                    frechet_basis_v = frechet_basis_v.type(self.dtype)
                    frechet_basis_v = frechet_basis_v / torch.norm(frechet_basis_v, dim=0, keepdim=True)

        # 3. Edit samples with constructed frechet basis
        if self.dataset_name == 'Random':
            xT = self.dataset[idx]
        else:
            xT = self.run_DDIMinversion(idx=idx)

        # get local tangent space
        u_path = os.path.join(
            f'./inputs/local_encoder_pullback_uncond-model_{self.model_name}-dataset_{self.dataset_name}-scheduler_{self.scheduler_name}-num_steps_{self.for_steps}-pca_rank_{pca_rank}',
            'u-' + f'xt-{self.dataset_name}_{idx}-{self.h_t}T-{op}-block_{block_idx}-seed_{self.seed}' + '.pt'
        )

        vT_path = os.path.join(
            f'./inputs/local_encoder_pullback_uncond-model_{self.model_name}-dataset_{self.dataset_name}-scheduler_{self.scheduler_name}-num_steps_{self.for_steps}-pca_rank_{pca_rank}',
            'vT-' + f'xt-{self.dataset_name}_{idx}-{self.h_t}T-{op}-block_{block_idx}-seed_{self.seed}' + '.pt'
        )

        if os.path.exists(u_path) and os.path.exists(vT_path):
            u = torch.load(u_path, map_location=self.device)
            u = u.type(self.dtype)
            u = u / u.norm(dim=0, keepdim=True)
            print('u norm : ', u.norm(dim=0))
            vT = torch.load(vT_path, map_location=self.device)
            vT = vT.type(self.dtype)
            vT = vT / vT.norm(dim=1, keepdim=True)
            print('vT norm : ', vT.norm(dim=1))

        elif local_projection:
            print('!!!SAMPLE LOCAL TANGENT SPACE!!!')
            # xT -> xt
            xt, t, i = self.DDIMforwardsteps(xT, t_start_idx=0, t_end_idx=self.h_t_idx)
            
            # get local basis
            SAMPLE_EXP_NAME = f'xt-{self.dataset_name}_{idx}-{self.h_t}T-{op}-block_{block_idx}-seed_{self.seed}'
            save_dir = f'./inputs/local_encoder_pullback_uncond-model_{self.model_name}-dataset_{self.dataset_name}-scheduler_{self.scheduler_name}-num_steps_{self.for_steps}-pca_rank_{pca_rank}'
            os.makedirs(save_dir, exist_ok=True)
            u_path = os.path.join(save_dir, 'u-' + SAMPLE_EXP_NAME + '.pt')
            vT_path = os.path.join(save_dir, 'vT-' + SAMPLE_EXP_NAME + '.pt')
            
            xt = xt.to(device=self.device, dtype=self.dtype)
            u, s, vT = self.unet.local_encoder_pullback_xt(
                x=xt, t=t, op=op, block_idx=block_idx, pca_rank=pca_rank, 
                min_iter=10, max_iter=50, convergence_threshold=1e-4,
            )

            # save semantic direction in h-space
            torch.save(u, u_path)
            torch.save(vT, vT_path)
            
            u = u.type(self.dtype)
            u = u / u.norm(dim=0, keepdim=True)
            vT = vT.type(self.dtype)
            vT = vT / vT.norm(dim=1, keepdim=True)

        # xT -> xt
        original_xt, t, i = self.DDIMforwardsteps(xT, t_start_idx=0, t_end_idx=self.edit_t_idx)
        original_xt = original_xt.to(device=self.device, dtype=self.dtype)

        vis_num_pc_list = list(range(vis_num_pc)) if vis_pc_list is None else vis_pc_list
        for pc_idx in vis_num_pc_list:
            for direction in [1, -1]:
                if direction == 1:
                    self.EXP_NAME = f'xt-{self.dataset_name}_{idx}-h_{self.h_t}T-edit_{self.edit_t}T-{op}-block_{block_idx}-seed_{self.seed}-pc_{pc_idx:0=3d}_pos'
                elif direction == -1:
                    self.EXP_NAME = f'xt-{self.dataset_name}_{idx}-h_{self.h_t}T-edit_{self.edit_t}T-{op}-block_{block_idx}-seed_{self.seed}-pc_{pc_idx:0=3d}_neg'

                if os.path.exists(os.path.join(self.result_folder, f'x0_gen-{self.EXP_NAME}.png')):
                    continue

                # select global direction
                if self.frechet_mean_space == 'h':
                    uk = direction*frechet_basis_u[:, pc_idx]
                    uk = uk / uk.norm()
                elif self.frechet_mean_space == 'x':
                    vk = direction*frechet_basis_v[:, pc_idx]
                    vk = vk / vk.norm()

                # project uk to local tangent space
                if local_projection:
                    if self.frechet_mean_space == 'x':
                        vk = vT.T @ (vT @ vk)
                        vk = vk / vk.norm()
                    elif self.frechet_mean_space == 'h':
                        vk = vT.T @ (u.T @ uk)
                        vk = vk / vk.norm()
                    
                    # projection coeffecient
                    plt.scatter(range(pca_rank), sorted((u.T @ uk).abs().tolist()), label='u')
                    plt.scatter(range(pca_rank), sorted((vT  @ vk).abs().tolist()), label='v')
                    plt.legend()
                    plt.savefig(os.path.join(self.obs_folder, 'projection_coeff-' + self.EXP_NAME + '.png'))
                    plt.close()

                if (self.edit_xt == 'parallel-x') & (self.edit_ht == 'default'):
                    vk = vk.view(-1, *xT.shape[1:])
                    if self.use_x_space_guidance:
                        xt_list = [original_xt.clone()]
                        for _ in tqdm(range(self.x_space_guidance_num_step), desc='x_space_guidance edit'):
                            xt_edit = self.x_space_guidance(
                                xt_list[-1], t_idx=self.edit_t_idx, vk=vk, 
                                single_edit_step=self.x_space_guidance_edit_step,
                            )
                            xt_list.append(xt_edit)
                        xt = torch.cat(xt_list, dim=0)
                        xt = xt[::int(xt.size(0) / vis_num)]
                    
                    else:
                        vis_grid = torch.linspace(0, self.x_edit_step_size, vis_num, dtype=vk.dtype, device=vk.device)
                        xt = original_xt + vis_grid[:, None, None, None] * vk

                    # regularize xt
                    if self.use_dynamic_thresholding:
                        xt = self.dynamic_thresholding(xt, t_idx=self.edit_t_idx, p=self.dynamic_thresholding_q)
                    if self.use_preserve_contrast:
                        xt = self.preserve_contrast(xt, t_idx=self.edit_t_idx)
                    if self.use_preserve_norm:
                        xt = self.preserve_norm(xt)

                    # xt -> x0
                    self.DDIMforwardsteps(xt, t_start_idx=self.edit_t_idx, t_end_idx=-1, performance_boosting=True)

                elif (self.edit_xt == 'parallel-h') & (self.edit_ht == 'default'):
                    uk = uk.to(device=self.device, dtype=self.dtype)
                    xt = [original_xt.clone()]
                    
                    for _ in tqdm(range(vis_num), desc='inv_jac_xt'):
                        vk = self.unet.inv_jac_xt(
                            x=xt[-1], t=self.edit_t_idx, op=op, block_idx=block_idx,
                            u=uk, perturb_h=1e-1, 
                        )
                        
                        if self.use_sega_reg:
                            vk_std = vk.std()
                            vk = torch.where(vk.abs() < self.sega_reg_sigma * vk_std, torch.zeros_like(vk), vk)

                        xt.append(xt[-1] + self.x_edit_step_size / vis_num * vk.view(*xt[-1].shape))
                    xt = torch.cat(xt, dim=0)
                    xt = xt[::int(xt.size(0) / vis_num)]

                    # regularize xt
                    if self.use_dynamic_thresholding:
                        xt = self.dynamic_thresholding(xt, t_idx=self.edit_t_idx, p=self.dynamic_thresholding_q)
                    if self.use_preserve_norm:
                        xt = self.preserve_norm(xt)
                    if self.use_preserve_contrast:
                        xt = self.preserve_contrast(xt, t_idx=self.edit_t_idx)

                    # xt -> x0
                    self.DDIMforwardsteps(xt, t_start_idx=self.edit_t_idx, t_end_idx=-1,  performance_boosting=True)
                
                # h space guidance
                elif (self.edit_xt == 'default')  & (self.edit_ht == 'h_space_guidance'):
                    uk = self.h_edit_step_size * uk.unsqueeze(0)
                    xt = original_xt.repeat(vis_num, 1, 1, 1)

                    xt, t, _ = self.h_space_guidance(
                        xt, t_start_idx=self.edit_t_idx, t_end_idx=self.no_edit_t_idx, 
                        uk=uk, op=op, block_idx=block_idx,  
                    )

                    if self.no_edit_t_idx != -1:
                        self.DDIMforwardsteps(
                            xt, t_start_idx=self.no_edit_t_idx, t_end_idx=-1, 
                             performance_boosting=True
                        )
        return
    
    @torch.no_grad()
    def run_edit_global_hungarian_mean_zt(
            self, idx, op, block_idx, vis_num, vis_num_pc=None, vis_pc_list=None, pca_rank=50, num_local_basis=100, local_projection=True, **kwargs,
        ):
        '''
        Find hungarian mean over SO(n) by py

        1. sample local basis
        2. Construct hungarian basis with pymanopt
        3. Edit samples with constructed hungarian basis

        Args
            - idx               : sample idx
            - op                : h space operation idx = (down, mid, up)
            - block_idx         : h space block idx
            - vis_num           : number of visualization editing steps
            - vis_num_pc        : number of visualization pc indexs
            - vis_vT            : visualize vT or not
            - pca_rank          : pca rank
            - num_local_basis : number of local basis for construct global hungarian mean
        '''
        # experiment paths
        save_dir = f'./inputs/global_hungarian_mean_uncond-model_{self.model_name}-dataset_{self.dataset_name}-scheduler_{self.scheduler_name}-num_steps_{self.for_steps}-pca_rank_{pca_rank}'
        os.makedirs(save_dir, exist_ok=True)
        hungarian_basis_name = f'{self.h_t}T-{op}-block_{block_idx}-seed_{self.seed}'
        hungarian_basis_path_u = os.path.join(save_dir, 'u-' + hungarian_basis_name + '.pt')
        hungarian_basis_path_v = os.path.join(save_dir, 'v-' + hungarian_basis_name + '.pt')

        if os.path.exists(hungarian_basis_path_u) and (os.path.exists(hungarian_basis_path_v)):
            hungarian_basis_u = torch.load(hungarian_basis_path_u, map_location=self.device)
            hungarian_basis_u = hungarian_basis_u.type(self.dtype)
            hungarian_basis_v = torch.load(hungarian_basis_path_v, map_location=self.device)
            hungarian_basis_v = hungarian_basis_v.type(self.dtype)

            hungarian_basis_u = hungarian_basis_u / torch.norm(hungarian_basis_u, dim=0, keepdim=True)
            hungarian_basis_v = hungarian_basis_v / torch.norm(hungarian_basis_v, dim=1, keepdim=True)

        else:
            # 1. sample local basis
            local_basis_list_u = []
            local_basis_list_v = []
            local_basis_save_dir = f'./inputs/local_encoder_pullback_uncond-model_{self.model_name}-dataset_Random-scheduler_{self.scheduler_name}-num_steps_{self.for_steps}-pca_rank_{pca_rank}'
            os.makedirs(local_basis_save_dir, exist_ok=True)

            for local_basis_idx in range(num_local_basis):
                self.EXP_NAME = f'xt-Random_{local_basis_idx}-{self.h_t}T-{op}-block_{block_idx}-seed_{self.seed}'
                u_path = os.path.join(local_basis_save_dir, 'u-' + self.EXP_NAME + '.pt')
                s_path = os.path.join(local_basis_save_dir, 's-' + self.EXP_NAME + '.pt')
                vT_path = os.path.join(local_basis_save_dir, 'vT-' + self.EXP_NAME + '.pt')

                if os.path.exists(u_path) and os.path.exists(vT_path):
                    u = torch.load(u_path, map_location=self.device)
                    u = u.type(self.dtype)
                    vT = torch.load(vT_path, map_location=self.device)
                    vT = vT.type(self.dtype)

                else:
                    print(f'!!!SAMPLE LOCAL BASIS IDX : {local_basis_idx}!!!')
                    from utils.utils import RandomLatentDataset
                    from configs.paths import DATASET_PATHS

                    # get latent code
                    random_dataset = RandomLatentDataset(
                        image_root = DATASET_PATHS['Random'],
                        c_in = self.c_in,
                        image_size = self.image_size,
                        dataset_name = 'Random',
                        device = self.device, 
                        dtype = self.dtype, 
                    )
                    xT = random_dataset[local_basis_idx]

                    # xT -> xt
                    xt, t, _ = self.DDIMforwardsteps(xT, t_start_idx=0, t_end_idx=self.h_t_idx)
                    xt = xt.to(dtype=self.dtype, device=self.device)

                    u, s, vT = self.unet.local_encoder_pullback_xt(
                        x=xt, t=t, op=op, block_idx=block_idx, pca_rank=pca_rank,
                        min_iter=10, max_iter=50, convergence_threshold=1e-4,
                    )

                    # save semantic direction in h-space
                    torch.save(u, u_path)
                    torch.save(s, s_path)
                    torch.save(vT, vT_path)
                
                local_basis_list_u.append(u.detach().clone())
                local_basis_list_v.append(vT.T.detach().clone())

            # 2. Construct hungarian basis with pymanopt
            if not os.path.exists(hungarian_basis_path_u):
                hungarian_basis_u = compute_hungarian_basis(local_basis_list_u, initial_local_basis_idx=0, distance='1/cosine')
                hungarian_basis_u = hungarian_basis_u.to(dtype=self.dtype, device=self.device)
                torch.save(hungarian_basis_u, hungarian_basis_path_u)
            else:
                hungarian_basis_u = torch.load(hungarian_basis_path_u, map_location=self.device)
                hungarian_basis_u = hungarian_basis_u.type(self.dtype)

            if not os.path.exists(hungarian_basis_path_v):
                hungarian_basis_v = compute_hungarian_basis(local_basis_list_v, initial_local_basis_idx=0, distance='1/cosine')
                hungarian_basis_v = hungarian_basis_v.to(dtype=self.dtype, device=self.device)
                torch.save(hungarian_basis_v, hungarian_basis_path_v)
            else:
                hungarian_basis_u = torch.load(hungarian_basis_path_u, map_location=self.device)
                hungarian_basis_u = hungarian_basis_u.type(self.dtype)

        # 3. Edit samples with constructed hungarian basis
        if self.dataset_name == 'Random':
            xT = self.dataset[idx]
        else:
            xT = self.run_DDIMinversion(idx=idx)

        # get local tangent space
        u_path = os.path.join(
            f'./inputs/local_encoder_pullback_uncond-model_{self.model_name}-dataset_{self.dataset_name}-scheduler_{self.scheduler_name}-num_steps_{self.for_steps}-pca_rank_{pca_rank}',
            'u-' + f'xt-{self.dataset_name}_{idx}-{self.h_t}T-{op}-block_{block_idx}-seed_{self.seed}' + '.pt'
        )

        vT_path = os.path.join(
            f'./inputs/local_encoder_pullback_uncond-model_{self.model_name}-dataset_{self.dataset_name}-scheduler_{self.scheduler_name}-num_steps_{self.for_steps}-pca_rank_{pca_rank}',
            'vT-' + f'xt-{self.dataset_name}_{idx}-{self.h_t}T-{op}-block_{block_idx}-seed_{self.seed}' + '.pt'
        )

        if os.path.exists(u_path) and os.path.exists(vT_path):
            u = torch.load(u_path, map_location=self.device)
            u = u.type(self.dtype)
            u = u / u.norm(dim=0, keepdim=True)
            print('u norm : ', u.norm(dim=0))
            vT = torch.load(vT_path, map_location=self.device)
            vT = vT.type(self.dtype)
            vT = vT / vT.norm(dim=1, keepdim=True)
            print('vT norm : ', vT.norm(dim=1))
        else:
            print('!!!SAMPLE LOCAL TANGENT SPACE!!!')
            # xT -> xt
            xt, t, i = self.DDIMforwardsteps(xT, t_start_idx=0, t_end_idx=self.h_t_idx)
            
            # get local basis
            SAMPLE_EXP_NAME = f'xt-{self.dataset_name}_{idx}-{self.h_t}T-{op}-block_{block_idx}-seed_{self.seed}'
            save_dir = f'./inputs/local_encoder_pullback_uncond-model_{self.model_name}-dataset_{self.dataset_name}-scheduler_{self.scheduler_name}-num_steps_{self.for_steps}-pca_rank_{pca_rank}'
            os.makedirs(save_dir, exist_ok=True)
            u_path = os.path.join(save_dir, 'u-' + SAMPLE_EXP_NAME + '.pt')
            vT_path = os.path.join(save_dir, 'vT-' + SAMPLE_EXP_NAME + '.pt')
            
            xt = xt.to(device=self.device, dtype=self.dtype)
            u, s, vT = self.unet.local_encoder_pullback_xt(
                x=xt, t=t, op=op, block_idx=block_idx, pca_rank=pca_rank, 
                min_iter=10, max_iter=50, convergence_threshold=1e-4,
            )

            # save semantic direction in h-space
            torch.save(u, u_path)
            torch.save(vT, vT_path)
            
            u = u.type(self.dtype)
            u = u / u.norm(dim=0, keepdim=True)
            vT = vT.type(self.dtype)
            vT = vT / vT.norm(dim=1, keepdim=True)

        # xT -> xt
        original_xt, t, i = self.DDIMforwardsteps(xT, t_start_idx=0, t_end_idx=self.edit_t_idx)
        original_xt = original_xt.to(device=self.device, dtype=self.dtype)

        vis_num_pc_list = list(range(vis_num_pc)) if vis_pc_list is None else vis_pc_list
        for pc_idx in vis_num_pc_list:
            for direction in [1, -1]:
                if direction == 1:
                    self.EXP_NAME = f'xt-{self.dataset_name}_{idx}-h_{self.h_t}T-edit_{self.edit_t}T-{op}-block_{block_idx}-seed_{self.seed}-pc_{pc_idx:0=3d}_pos'
                elif direction == -1:
                    self.EXP_NAME = f'xt-{self.dataset_name}_{idx}-h_{self.h_t}T-edit_{self.edit_t}T-{op}-block_{block_idx}-seed_{self.seed}-pc_{pc_idx:0=3d}_neg'

                if os.path.exists(os.path.join(self.result_folder, f'x0_gen-{self.EXP_NAME}.png')):
                    continue

                # select global direction
                uk = direction*hungarian_basis_u[:, pc_idx]
                uk = uk / uk.norm()
                vk = direction*hungarian_basis_v[:, pc_idx]
                vk = vk / vk.norm()

                # project uk to local tangent space
                if local_projection:
                    uk = u @ (u.T @ uk)
                    uk = uk / uk.norm()

                    vk = vT.T @ (vT @ vk)
                    vk = vk / vk.norm()
                    
                    # projection coeffecient
                    plt.scatter(range(pca_rank), sorted((u.T @ uk).abs().tolist()), label='u')
                    plt.scatter(range(pca_rank), sorted((vT  @ vk).abs().tolist()), label='v')
                    plt.savefig(os.path.join(self.obs_folder, 'projection_coeff-' + self.EXP_NAME + '.png'))
                    plt.close()

                if (self.edit_xt == 'parallel-x') & (self.edit_ht == 'default'):
                    vk = vk.view(-1, *xT.shape[1:])
                    if self.use_x_space_guidance:
                        xt_list = [original_xt.clone()]
                        for _ in tqdm(range(self.x_space_guidance_num_step), desc='x_space_guidance edit'):
                            xt_edit = self.x_space_guidance(
                                xt_list[-1], t_idx=self.edit_t_idx, vk=vk, 
                                single_edit_step=self.x_space_guidance_edit_step,
                            )
                            xt_list.append(xt_edit)
                        xt = torch.cat(xt_list, dim=0)
                        xt = xt[::int(xt.size(0) / vis_num)]
                    
                    else:
                        vis_grid = torch.linspace(0, self.x_edit_step_size, vis_num, dtype=vk.dtype, device=vk.device)
                        xt = original_xt + vis_grid[:, None, None, None] * vk

                    # regularize xt
                    if self.use_dynamic_thresholding:
                        xt = self.dynamic_thresholding(xt, t_idx=self.edit_t_idx, p=self.dynamic_thresholding_q)
                    if self.use_preserve_contrast:
                        xt = self.preserve_contrast(xt, t_idx=self.edit_t_idx)
                    if self.use_preserve_norm:
                        xt = self.preserve_norm(xt)

                    # xt -> x0
                    self.DDIMforwardsteps(xt, t_start_idx=self.edit_t_idx, t_end_idx=-1, performance_boosting=True)

                elif (self.edit_xt == 'parallel-h') & (self.edit_ht == 'default'):
                    uk = uk.to(device=self.device, dtype=self.dtype)
                    xt = [original_xt.clone()]
                    
                    for _ in tqdm(range(vis_num), desc='inv_jac_xt'):
                        vk = self.unet.inv_jac_xt(
                            x=xt[-1], t=self.edit_t_idx, op=op, block_idx=block_idx,
                            u=uk, perturb_h=1e-1, 
                        )
                        
                        if self.use_sega_reg:
                            vk_std = vk.std()
                            vk = torch.where(vk.abs() < self.sega_reg_sigma * vk_std, torch.zeros_like(vk), vk)

                        xt.append(xt[-1] + self.x_edit_step_size / vis_num * vk.view(*xt[-1].shape))
                    xt = torch.cat(xt, dim=0)
                    xt = xt[::int(xt.size(0) / vis_num)]

                    # regularize xt
                    if self.use_dynamic_thresholding:
                        xt = self.dynamic_thresholding(xt, t_idx=self.edit_t_idx, p=self.dynamic_thresholding_q)
                    if self.use_preserve_norm:
                        xt = self.preserve_norm(xt)
                    if self.use_preserve_contrast:
                        xt = self.preserve_contrast(xt, t_idx=self.edit_t_idx)

                    # xt -> x0
                    self.DDIMforwardsteps(xt, t_start_idx=self.edit_t_idx, t_end_idx=-1,  performance_boosting=True)
                
                # h space guidance
                elif (self.edit_xt == 'default')  & (self.edit_ht == 'h_space_guidance'):
                    uk = self.h_edit_step_size * uk.unsqueeze(0)
                    xt = original_xt.repeat(vis_num, 1, 1, 1)

                    xt, t, _ = self.h_space_guidance(
                        xt, t_start_idx=self.edit_t_idx, t_end_idx=self.no_edit_t_idx, 
                        uk=uk, op=op, block_idx=block_idx,  
                    )

                    if self.no_edit_t_idx != -1:
                        self.DDIMforwardsteps(
                            xt, t_start_idx=self.no_edit_t_idx, t_end_idx=-1, 
                             performance_boosting=True
                        )
        return

    @torch.no_grad()
    def run_sample_encoder_local_tangent_space_zt(
            self, h_t, op, block_idx, pca_rank=50, num_local_basis=100, fix_xt=False, fix_t=False,
        ):
        '''
        Find Frechet mean over SO(n) by py

        1. sample local basis
        2. Construct Frechet basis with pymanopt
        3. Edit samples with constructed frechet basis

        Args
            - idx           : sample idx
            - t_edit_idx    : edit time step
            - op            : h space operation idx = (down, mid, up)
            - block_idx     : h space block idx
            - step_size     : step size of edit (x + step_size * v or h + step_size * u)
            - vis_num       : number of visualization editing steps
            - edit_xt       : how to edit xt ('parallel-x', 'parallel-h')
            - edit_ht       : how to edit ht ('multiple-t', 'single-t')
            - pca_rank      : low rank approximation of local tangent space
            - num_local_basis : number of local basis for construct global frechet mean
        '''
        assert (not fix_xt) or (not fix_t)

        # experiment parameters
        self.scheduler.set_timesteps(self.for_steps)
        h_t_idx = (self.scheduler.timesteps - h_t * 1000).abs().argmin()

        # path
        save_dir = f'./inputs/local_encoder_pullback_uncond-model_{self.model_name}-dataset_{self.dataset_name}-scheduler_{self.scheduler_name}-num_steps_{self.for_steps}-pca_rank_{pca_rank}'
        if fix_xt:
            save_dir = save_dir + '-fix_xt'
        elif fix_t:
            save_dir = save_dir + '-fix_t'
        os.makedirs(save_dir, exist_ok=True)
        
        for local_basis_idx in range(num_local_basis):
            self.EXP_NAME = f'xt-{self.dataset_name}_{local_basis_idx}-{h_t}T-{op}-block_{block_idx}-seed_{self.seed}'
            u_path = os.path.join(save_dir, 'u-' + self.EXP_NAME + '.pt')
            s_path = os.path.join(save_dir, 's-' + self.EXP_NAME + '.pt')
            s_fig_path = os.path.join(save_dir, 'eigenvalue_spectrum-' + self.EXP_NAME + '.png')
            vT_path = os.path.join(save_dir, 'vT-' + self.EXP_NAME + '.pt')

            if os.path.exists(u_path) & os.path.exists(s_path) & os.path.exists(vT_path):
                print(f'!!!ALREADY SAMPLED LOCAL BASIS IDX : {local_basis_idx}!!!')
                continue

            else:
                print(f'!!!SAMPLE LOCAL BASIS IDX : {local_basis_idx}!!!')

            # get latent code
            xT = self.dataset[local_basis_idx]

            # xT -> xt
            if fix_xt:
                xt, _, _  = self.DDIMforwardsteps(xT, t_start_idx=0, t_end_idx=0)
                _, t, _ = self.DDIMforwardsteps(xT, t_start_idx=0, t_end_idx=h_t_idx)
            elif fix_t:
                _, t, _  = self.DDIMforwardsteps(xT, t_start_idx=0, t_end_idx=0)
                xt, _, _ = self.DDIMforwardsteps(xT, t_start_idx=0, t_end_idx=h_t_idx)
            else:
                xt, t, _ = self.DDIMforwardsteps(xT, t_start_idx=0, t_end_idx=h_t_idx)

            xt = xt.to(self.device)
            u, s, vT = self.unet.local_encoder_pullback_xt(
                x=xt, t=t, op=op, block_idx=block_idx, pca_rank=pca_rank,
                min_iter=10, max_iter=50, convergence_threshold=1e-4,
            )

            # save eigenvalues spectrum
            plt.scatter(range(s.size(0)), s.cpu().tolist(), s=1)
            plt.savefig(s_fig_path, dpi = 80)
            plt.close()

            # save semantic direction in h-space
            torch.save(u, u_path)
            torch.save(s, s_path)
            torch.save(vT, vT_path)

            del xT, xt, t, u, s, vT
            torch.cuda.empty_cache()
                
        return
    
    @torch.no_grad()
    def DDIMforwardsteps(
            self, xt, t_start_idx, t_end_idx, vis_psd=False, save_image=True, return_xt=True, performance_boosting=False,
        ):
        '''
        Args
            xt              : latent variable
            t_start_idx     : current timestep
            t_end_idx       : target timestep (t_end_idx = -1 for full forward)
            buffer_device   : device to store buffer
            vis_psd         : visualize psd
            save_image      : save image
            return_xt       : return xt
        '''
        print('start DDIMforward')
        # before start
        assert (t_start_idx < self.for_steps) & (t_end_idx <= self.for_steps)
        use_clipped_model_output = None
        num_inference_steps = self.for_steps
        eta = 0

        # set timestep (we do not use default scheduler set timestep method)
        if self.use_yh_custom_scheduler:
            self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        else:
            self.scheduler = DDIMScheduler.from_config(self.scheduler.config)
            self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps
        # print(f'timesteps : {timesteps}')

        # save traj
        xt_traj = [xt.clone()]
        et_traj = []

        ##################
        # denoising loop #
        ##################
        for i, t in enumerate(timesteps):
            # skip timestpe if not in range
            if t_end_idx == i:
                print('t_end_idx : ', i)
                return xt, t, i
        
            elif (i < t_start_idx): 
                continue

            elif t_start_idx == i:
                print('t_start_idx : ', i)

            if performance_boosting & (self.performance_boosting_t_idx <= i) & (self.performance_boosting_t_idx != len(timesteps)-1):
                eta = 1
            else:
                eta = 0

            # use buffer to avoid OOM
            xt = xt.to(device=self.buffer_device)
            if xt.size(0) // self.memory_bound == 0:
                xt_buffer = [xt]
            else:
                xt_buffer = list(xt.chunk(xt.size(0) // self.memory_bound))

            for buffer_idx, xt in enumerate(xt_buffer):
                xt = xt.to(device=self.device, dtype=self.dtype)

                # 1. predict noise model_output
                et = self.unet(xt, t)
                if not isinstance(et, torch.Tensor):
                    et = et.sample

                # 2. predict previous mean of xt x_t-1 and add variance depending on eta
                # eta corresponds to η in paper and should be between [0, 1]
                # do x_t -> x_t-1
                xt = self.scheduler.step(
                    et, t, xt, eta=eta, use_clipped_model_output=use_clipped_model_output, generator=None
                ).prev_sample

                # save xt in buffer
                xt_buffer[buffer_idx] = xt.to(self.buffer_device)

            # save traj
            if buffer_idx == 0:
                xt_traj.append(xt[0].detach().cpu().type(torch.float32).clone())
                et_traj.append(et[0].detach().cpu().type(torch.float32).clone())

            xt = torch.cat(xt_buffer, dim=0)
            del xt_buffer
            torch.cuda.empty_cache()

        if save_image:
            image = (xt / 2 + 0.5).clamp(0, 1)
            tvu.save_image(
                image, os.path.join(self.result_folder, f'x0_gen-{self.EXP_NAME}.png'), nrow = image.size(0),
            )

            # visualize alignment btw. et and et-1
            # et_alignment = [F.cosine_similarity(et_traj[t].view(-1), et_traj[t+1].view(-1), dim=0).item() for t in range(len(et_traj)-1)]
            # plt.plot(et_alignment)
            # plt.savefig(os.path.join(self.obs_folder, f'et_alignment-{self.EXP_NAME}.png'))
            # plt.close()
            
        # visualize power spectral density of zt_traj, et_traj
        if vis_psd:
            vis_power_spectral_density(
                et_traj, save_path=os.path.join(self.obs_folder, f'et_psd-{self.EXP_NAME}.png')
            )

            vis_power_spectral_density(
                xt_traj, save_path=os.path.join(self.obs_folder, f'xt_psd-{self.EXP_NAME}.png')
            )

        if return_xt:
            return xt

        return

    @torch.no_grad()
    def x_space_guidance(self, xt, t_idx, vk, single_edit_step):
        # necesary parameters
        t = self.scheduler.timesteps[t_idx]

        # edit xt with vk
        xt_edit = xt + single_edit_step * vk

        # predict the noise residual
        et = self.unet(
            torch.cat([xt, xt_edit], dim=0), t, 
        )
        if not isinstance(et, torch.Tensor):
            et = et.sample

        # DDS regularization
        et_null, et_edit = et.chunk(2)
        xt_edit = xt + self.x_space_guidance_scale * (et_edit - et_null)
        return xt_edit
    
####################
# Custom timesteps #
####################
from functools import partial
from typing import Union

def custom_set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None, inversion_flag: bool = False):
    """
    Sets the discrete timesteps used for the diffusion chain. Supporting function to be run before inference.
    Args:
        num_inference_steps (`int`):
            the number of diffusion steps used when generating samples with a pre-trained model.
    """

    if num_inference_steps > self.config.num_train_timesteps:
        raise ValueError(
            f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.config.train_timesteps`:"
            f" {self.config.num_train_timesteps} as the unet model trained with this scheduler can only handle"
            f" maximal {self.config.num_train_timesteps} timesteps."
        )

    self.num_inference_steps = num_inference_steps
    step_ratio = self.config.num_train_timesteps // self.num_inference_steps
    # creates integer timesteps by multiplying by ratio
    # casting to int to avoid issues when num_inference_step is power of 3
    # timesteps = (np.arange(0, num_inference_steps) * step_ratio).round().copy().astype(np.int64)
    timesteps = np.linspace(0, 1, num_inference_steps) * (self.config.num_train_timesteps-2) # T=999
    timesteps = timesteps + 1e-6
    timesteps = timesteps.round().astype(np.int64)
    # reverse timesteps except inverse diffusion
    # keep it numpy array
    if not inversion_flag:
        timesteps = np.flip(timesteps).copy()

    self.timesteps = torch.from_numpy(timesteps).to(device)
    self.timesteps += self.config.steps_offset