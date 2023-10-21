import argparse
import traceback
import logging
import yaml
import sys
import os
import torch
import numpy as np
import random
import shutil

from utils.utils import dict2namespace
from configs.params import X_SPACE_GUIDANCE_SCALE_DICT, X_SPACE_EDIT_STEP_SIZE_DICT

def parse_args():
    # parser = argparse.ArgumentParser(description=globals()['__doc__'])
    parser = argparse.ArgumentParser()

    # default setting 
    parser.add_argument('--sh_file_name',   type=str,   default='',      required=False, help="for logging")
    parser.add_argument('--device',         type=str,   default='',      required=False, help="'cuda', 'cpu'")
    parser.add_argument('--dtype',          type=str,   default='fp32',  required=False, help="'fp32', 'fp16'")
    parser.add_argument('--seed',           type=int,   default=0,       required=False, help='Random seed')
    parser.add_argument('--result_folder',  type=str,   default='./runs/', help='Path for saving running related data.')
    
    # model, dataset setting
    parser.add_argument('--model_name',     type=str,   default='',     required=False)
    parser.add_argument('--dataset_name',   type=str,   default='',     required=False)
    parser.add_argument('--num_imgs',       type=int,   default=100,    required=False)
    parser.add_argument('--image_size',     type=int,   default=256,    required=False)
    parser.add_argument('--c_in',           type=int,   default=3,      required=False)
    parser.add_argument('--sample_idx',     type=int,   default=0,      required=False)

    # args (prompt)
    parser.add_argument('--for_prompt',     type=str,   default='',     required=False)
    parser.add_argument('--inv_prompt',     type=str,   default='',     required=False)
    parser.add_argument('--neg_prompt',     type=str,   default='',     required=False)
    
    # args (diffusion schedule)
    parser.add_argument('--for_steps',      type=int,   default=100,    required=False)
    parser.add_argument('--inv_steps',      type=int,   default=100,    required=False)
    parser.add_argument('--performance_boosting_t',     type=float,     default=0.0,    required=False)
    parser.add_argument('--use_yh_custom_scheduler',    type=str2bool,  default='True', required=False, help='Use custom scheduler for better inversion quality')

    # args (guidance)
    parser.add_argument('--guidance_scale', type=float, default=0,  required=False)
    
    # args (h space edit)
    parser.add_argument('--edit_prompt',     type=str,  default='',      required=False)
    # parser.add_argument('--h_space_guidance_scale',     type=float, default=0,  required=False)

    parser.add_argument('--edit_xt',        type=str,   default='default',      required=False, help="'parallel-x' or 'parallel-h'")
    # parser.add_argument('--edit_ht',        type=str,   default='default',      required=False, help="'multiple-t' or 'single-t'")

    # parser.add_argument('--after_res',      type=str2bool,  default='False',    required=False)
    # parser.add_argument('--after_sa',       type=str2bool,  default='False',    required=False)

    # parser.add_argument('--use_dynamic_thresholding',   type=str2bool,  default='False',    required=False)
    # parser.add_argument('--dynamic_thresholding_q',     type=float,     default=0.8,        required=False)
    # parser.add_argument('--use_preserve_contrast',      type=str2bool,  default='False',    required=False)
    # parser.add_argument('--use_preserve_norm',          type=str2bool,  default='False',    required=False)
    # parser.add_argument('--use_repainting_reg',         type=str2bool,  default='False',    required=False)
    # parser.add_argument('--num_repainting',             type=int,       default=0,          required=False)
    # parser.add_argument('--single_edit_step_repainting',type=float,     default=0,          required=False)
    # parser.add_argument('--use_sega_reg',               type=str2bool,  default='False',    required=False)
    # parser.add_argument('--sega_reg_sigma',             type=float,     default=1.0,        required=False)
    # parser.add_argument('--use_match_prompt',           type=str2bool,  default='False',    required=False)

    parser.add_argument('--use_x_space_guidance',                       type=str2bool,  default='False',    required=False)
    parser.add_argument('--x_space_guidance_edit_step',                 type=float,     default=1,          required=False)
    parser.add_argument('--x_space_guidance_scale',                     type=float,     default=0,          required=False)
    parser.add_argument('--x_space_guidance_num_step',                  type=int,       default=0,          required=False)
    parser.add_argument('--x_space_guidance_use_edit_prompt',           type=str2bool,  default='True',     required=False)

    # parser.add_argument('--use_delta_denoising_score_reg',              type=str2bool,  default='False',    required=False)
    # parser.add_argument('--delta_denoising_score_reg_step',             type=float,     default=0,          required=False)
    # parser.add_argument('--delta_denoising_score_reg_scale',            type=float,     default=0,          required=False)

    # args (h space edit + currently not using. please refer main.py)
    parser.add_argument('--h_t',            type=float, default=0.8,            required=False)
    parser.add_argument('--edit_t',         type=float, default=1.0,            required=False, help="after no_edit_t_idx, do not apply edit")
    parser.add_argument('--no_edit_t',      type=float, default=0.5,            required=False, help="after no_edit_t_idx, do not apply edit")
    parser.add_argument('--h_edit_step_size', type=float, default=0,            required=False)
    parser.add_argument('--x_edit_step_size', type=float, default=0,            required=False)
    
    # memory
    parser.add_argument('--pca_device',     type=str,   default='cpu',      required=False)
    parser.add_argument('--buffer_device',  type=str,   default='cpu',      required=False)
    parser.add_argument('--save_result_as', type=str,   default='image',    required=False, help='image or tensor')

    # exp setting    
    parser.add_argument('--note',                                           type=str,       required=True)
    parser.add_argument('--run_cfg_forward',                                type=str2bool,  default='False', required=False)
    parser.add_argument('--run_mcg_forward',                                type=str2bool,  default='False', required=False)
    parser.add_argument('--run_pfg_forward',                                type=str2bool,  default='False', required=False)
    parser.add_argument('--run_ddim_forward',                               type=str2bool,  default='False', required=False)
    parser.add_argument('--run_ddim_inversion',                             type=str2bool,  default='False', required=False)

    parser.add_argument('--run_edit_local_encoder_pullback_zt',             type=str2bool,  default='False', required=False)
    parser.add_argument('--run_edit_local_decoder_pullback_zt',             type=str2bool,  default='False', required=False)
    parser.add_argument('--run_edit_local_x0_decoder_pullback_zt',          type=str2bool,  default='False', required=False)
    parser.add_argument('--run_edit_local_pca_zt',                          type=str2bool,  default='False', required=False)
    
    parser.add_argument('--run_edit_local_encoder_pullback_zt_with_various_prompt', type=str2bool,  default='False', required=False)
    parser.add_argument('--various_prompt_type',                            type=str,       default='',      required=False)
    parser.add_argument('--various_prompt_sample_idx',                      type=int,       default=0,       required=False)
    
    parser.add_argument('--run_sample_encoder_local_tangent_space_zt',      type=str2bool,  default='False', required=False)
    parser.add_argument('--fix_xt',                                         type=str2bool,  default='False', required=False)
    parser.add_argument('--fix_t',                                          type=str2bool,  default='False', required=False)

    parser.add_argument('--run_edit_global_frechet_mean_zt',                type=str2bool,  default='False', required=False)
    parser.add_argument('--frechet_mean_space',                             type=str,       default='',      required=False)
    parser.add_argument('--local_projection',                               type=str2bool,  default='False', required=False)
    parser.add_argument('--num_local_basis',                                type=int,       default=100,     required=False)

    parser.add_argument('--run_edit_parallel_transport',                    type=str2bool,  default='False', required=False)
    parser.add_argument('--sample_idx_0',                                   type=int,       default=0,       required=False)
    parser.add_argument('--sample_idx_1',                                   type=int,       default=0,       required=False)

    parser.add_argument('--run_edit_global_hungarian_mean_zt',              type=str2bool,  default='False', required=False)
    parser.add_argument('--hungarian_mean_space',                           type=str,       default='',      required=False)

    parser.add_argument('--run_edit_text_driven_direction',                 type=str2bool,  default='False', required=False)

    # deprecated
    parser.add_argument('--run_edit_global_pca_zt',                         type=str2bool,  default='False', required=False)

    # mode
    parser.add_argument('--debug_mode',                                     type=str2bool,  default='False', required=False)
    parser.add_argument('--sampling_mode',                                  type=str2bool,  default='False', required=False)
    args = parser.parse_args()
    return args

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('true'):
        return True
    elif v.lower() in ('false'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def preset(args):
    # reproducatibility
    seed_everything(args.seed)
    
    ###############
    # config file #
    ###############
    # parse config file (pretrained model)
    if 'stable-diffusion' in args.model_name:
        args.is_stable_diffusion = True

        # save path
        args.exp = f'Stable_Diffusion-{args.dataset_name}-{args.note}'
        args.exp_folder = os.path.join(args.result_folder, args.exp)
        
    else:
        args.is_stable_diffusion = False

        if args.model_name == 'CelebA_HQ':
            raise NotImplementedError('Model weight deprecated...')
        elif args.model_name in ["FFHQ_P2", "AFHQ_P2", "Flower_P2"]:
            raise NotImplementedError('Please download P2 weight from https://github.com/jychoi118/P2-weighting')
        elif args.model_name in ['LSUN_bedroom', 'LSUN_cat', 'LSUN_horse']:
            raise NotImplementedError('Please download P2 weight from https://github.com/jychoi118/P2-weighting')
        elif args.model_name in ['CelebA_HQ_HF', 'LSUN_church_HF', 'FFHQ_HF']:
            pass
        else:
            raise ValueError('model_name choice: [CelebA_HQ_HF, LSUN_church_HF, FFHQ_HF]')

        # save path
        args.exp = f'{args.model_name}-{args.dataset_name}-{args.note}'
        args.exp_folder = os.path.join(args.result_folder, args.exp)

    ##########
    # folder #    
    ##########
    os.makedirs(args.exp_folder, exist_ok=True)
    shutil.copy(os.path.join('scripts', args.sh_file_name), os.path.join(args.exp_folder, args.sh_file_name))
    shutil.copy(os.path.join('utils', 'define_argparser.py'), os.path.join(args.exp_folder, 'define_argparser.py'))
    shutil.copy('main.py', os.path.join(args.exp_folder, 'main.py'))

    args.obs_folder    = os.path.join(args.exp_folder, 'obs')
    args.result_folder = os.path.join(args.exp_folder, 'results')
    
    os.makedirs(args.obs_folder, exist_ok=True)
    os.makedirs(args.result_folder, exist_ok=True)

    ##################
    # dependent args #
    ##################
    args.device = torch.device(args.device)
    args.dtype = torch.float32 if args.dtype == 'fp32' else torch.float16
    print(f'device : {args.device}, dtype : {args.dtype}')

    # edit scale
    if args.use_x_space_guidance:
        if args.is_stable_diffusion:
            args.x_space_guidance_scale = X_SPACE_GUIDANCE_SCALE_DICT['stable-diffusion'][args.h_t]
        else:
            args.x_space_guidance_scale = X_SPACE_GUIDANCE_SCALE_DICT['uncond'][args.h_t]

    # input size, memory bound to avoid OOM
    if args.is_stable_diffusion:
        args.c_in = 4
        args.image_size = 64
        args.memory_bound = 5
    elif 'CIFAR10' in args.model_name:
        args.c_in = 3
        args.memory_bound = 50
        args.image_size = 32
    else:
        args.c_in = 3
        args.image_size = 256
        args.memory_bound = 50
        args.noise_schedule = 'linear'

    ##########
    # assert #
    ##########
    if args.is_stable_diffusion:
        assert args.use_yh_custom_scheduler
        # assert args.for_steps == 100
        assert args.performance_boosting_t <= 0
    else:
        assert args.use_yh_custom_scheduler
        assert args.for_steps == 100
        assert args.performance_boosting_t == 0.2

    return args

def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True