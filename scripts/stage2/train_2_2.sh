python train_stage2_cloth.py --name clothpart_256p_2 \
--dataroot /home/wujinlin5/yqw_home/Motion_Transfer/DATASET/iper/train --dataset_mode cloth_2 --model cloth_2 --nThreads 16 \
--checkpoints_dir ./check_points/test2 \
--input_nc_T_2 9 --input_nc_S_2 12 --output_nc_2 3 --ngf 64 --n_downsample_cloth 3 --label_nc_2 6 --n_blocks 5 \
--resize_or_crop scaleHeight --loadSize 256 --random_drop_prob 0 --color_aug \
--gpu_ids 1 --n_gpus_gen 1 --batchSize 1 --max_frames_per_gpu 6 --display_freq 40 --print_freq 40 --save_latest_freq 1000 \
--niter 5 --niter_decay 5 --n_scales_temporal 3 --n_frames_D 2 \
--no_first_img --n_frames_total 12 --max_t_step 4 --tf_log --no_flow 

