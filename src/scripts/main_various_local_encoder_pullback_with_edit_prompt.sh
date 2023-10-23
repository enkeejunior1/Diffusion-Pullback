for t in 0.7 0.6
    do
    for sample_idx in 5
        do
        for edit_prompt in "sitting dog" "tiger"
            do
            python main.py \
                --sh_file_name                          main_various_local_encoder_pullback_with_edit_prompt.sh   \
                --device                                cuda:0                                      \
                --sample_idx                            $sample_idx                                 \
                --model_name                            stabilityai/stable-diffusion-2-1-base       \
                --dataset_name                          Examples                                    \
                --edit_prompt                           "$edit_prompt"                              \
                --x_space_guidance_scale                1                                           \
                --x_space_guidance_num_step             64                                          \
                --edit_t                                $t                                          \
                --run_edit_local_encoder_pullback_zt    True                                        \
                --note                                  "with_prompt"
            done
        done
    done
