# 
for t in 1.0 0.8 0.6
    do
    for sample_idx in 0 1 2 3 4
        do
        python main.py \
            --sh_file_name                          main_celeba_hf_local_encoder_pullback.sh    \
            --sample_idx                            $sample_idx                                 \
            --device                                cuda:1                                      \
            --dtype                                 fp32                                        \
            --seed                                  0                                           \
            --model_name                            CelebA_HQ_HF                                \
            --dataset_name                          CelebA_HQ                                   \
            --for_steps                             100                                         \
            --inv_steps                             100                                         \
            --use_yh_custom_scheduler               True                                        \
            --x_space_guidance_edit_step            1                                           \
            --x_space_guidance_scale                0.1                                         \
            --x_space_guidance_num_step             16                                          \
            --edit_t                                $t                                          \
            --performance_boosting_t                0.2                                         \
            --run_edit_local_encoder_pullback_zt    True                                        \
            --note                                  "Uncond"
        done
    done