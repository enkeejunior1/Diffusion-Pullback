from utils.define_argparser import parse_args, preset

from modules.edit import (
    EditStableDiffusion,
    EditUncondDiffusion,
)

if __name__ == "__main__":
    ##########
    # preset #
    ##########
    # parse args
    args = parse_args()
    
    # preset
    args = preset(args)

    # get instance
    if args.is_stable_diffusion:
        print('is stable-diffusion')
        edit = EditStableDiffusion(args)

    else:
        print('is NOT stable-diffusion')
        edit = EditUncondDiffusion(args)
    
    ########################################
    # experiment : local direction editing #
    ########################################
    if args.run_edit_local_encoder_pullback_zt:
        edit.run_edit_local_encoder_pullback_zt(
            idx=args.sample_idx, op='mid', block_idx=0,
            vis_num=4, vis_num_pc=2, pca_rank=2, edit_prompt=args.edit_prompt,
        )

    if args.run_edit_parallel_transport:
        edit.run_edit_parallel_transport(
            sample_idx_0=args.sample_idx_0, sample_idx_1=args.sample_idx_1, op='mid', block_idx=0,
            vis_num=4, vis_num_pc=2, pca_rank=50, 
        )

    ###########################################
    # experiment : sample local tangent space #
    ###########################################
    if args.run_sample_encoder_local_tangent_space_zt:
        if False:
            EDIT_T_LIST = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
            for edit_t in reversed(EDIT_T_LIST):
                for edit_prompt in ['']:
                    edit.run_sample_encoder_local_tangent_space_zt(
                        h_t=edit_t, op=op, block_idx=block_idx, pca_rank=50, num_local_basis=5, use_edit_prompt=None, edit_prompt=edit_prompt,
                    )

        if args.is_stable_diffusion:
            # # exp : various layer
            # EDIT_T_LIST = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
            # OPS_BLOCKS_LIST = [['mid', 0], ['down', 3], ['down', 2]]
            # MS_COCO_PROMPT_LIST = get_ms_coco_prompt_list(num_captions=10)

            # exp : various prompts + various timestep
            EDIT_T_LIST = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
            OPS_BLOCKS_LIST = [['mid', 0]]
            MS_COCO_PROMPT_LIST = get_ms_coco_prompt_list(num_captions=50)

            # # exp : various prompts
            # EDIT_T_LIST = [1.0]
            # OPS_BLOCKS_LIST = [['mid', 0]]
            # MS_COCO_PROMPT_LIST = get_ms_coco_prompt_list(num_captions=100)

            # sample across various timesteps
            for (op, block_idx) in OPS_BLOCKS_LIST:
                for edit_t in EDIT_T_LIST:
                    for edit_prompt in MS_COCO_PROMPT_LIST:
                        edit.run_sample_encoder_local_tangent_space_zt(
                            h_t=edit_t, op=op, block_idx=block_idx, pca_rank=50, num_local_basis=args.num_local_basis, use_edit_prompt=None, edit_prompt=edit_prompt,
                        )

        else:
            # EDIT_T_LIST = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
            EDIT_T_LIST = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
            # EDIT_T_LIST = [
            #     0.02, 0.04, 0.06, 0.08, 0.1 , 0.12, 0.14, 0.16, 0.18, 0.2 , 0.22,
            #     0.24, 0.26, 0.28, 0.3 , 0.32, 0.34, 0.36, 0.38, 0.4 , 0.42, 0.44,
            #     0.46, 0.48, 0.5 , 0.52, 0.54, 0.56, 0.58, 0.6 , 0.62, 0.64, 0.66,
            #     0.68, 0.7 , 0.72, 0.74, 0.76, 0.78, 0.8 , 0.82, 0.84, 0.86, 0.88,
            #     0.9 , 0.92, 0.94, 0.96, 0.98, 1.  
            # ]
            for edit_t in reversed(EDIT_T_LIST):
                edit.run_sample_encoder_local_tangent_space_zt(
                    h_t=edit_t, op=op, block_idx=block_idx, pca_rank=50, num_local_basis=args.num_local_basis, fix_xt=args.fix_xt, fix_t=args.fix_t
                )

    #####################
    # simple experiment #
    #####################
    # experiment : forward (for debug diffusion model load)
    if args.run_ddim_forward:
        edit.run_DDIMforward(num_samples=5)
    
    # experiment : inversion
    if args.run_ddim_inversion:
        edit.run_DDIMinversion(idx=args.sample_idx)