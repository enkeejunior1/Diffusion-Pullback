# my method : local pullback (w/ Jacobian subspace iteration)

for t in 0.7 0.6
    do
    for sample_idx in 4 5
        do
        for edit_prompt in "tiger" "bear" "running" "standing" "sitting"
            do
            python main.py \
                --sh_file_name                          main_various_local_encoder_pullback_with_edit_prompt.sh   \
                --sample_idx                            $sample_idx                                 \
                --device                                cuda:0                                      \
                --dtype                                 fp32                                        \
                --seed                                  0                                           \
                --model_name                            stabilityai/stable-diffusion-2-1-base       \
                --dataset_name                          Examples                                    \
                --for_steps                             100                                         \
                --inv_steps                             100                                         \
                --for_prompt                            ""                                          \
                --neg_prompt                            ""                                          \
                --inv_prompt                            ""                                          \
                --edit_prompt                           $edit_prompt                                \
                --use_yh_custom_scheduler               True                                        \
                --x_space_guidance_edit_step            1                                           \
                --x_space_guidance_scale                0.5                                         \
                --x_space_guidance_num_step             64                                          \
                --x_space_guidance_use_edit_prompt      True                                        \
                --edit_t                                $t                                          \
                --run_edit_local_encoder_pullback_zt    True                                        \
                --note                                  "with_prompt"
            done
        done
    done
