python train_stage2.py --name clothpart_256p \
--dataroot /home/jovyan/Motion_Transfer/DATASET/iper/train --dataset_mode cloth --model cloth --nThreads 16 \
--checkpoints_dir ./check_points/test \
--input_nc_T_2 9 --input_nc_S_2 3 --output_nc_2 3 --ngf 64 --n_downsample_cloth 3 --label_nc_2 6 --grid_size 3 \
--resize_or_crop scaleHeight --loadSize 256 --random_drop_prob 0 --color_aug \
--gpu_ids 0 --n_gpus_gen 1 --batchSize 1 --max_frames_per_gpu 6 --display_freq 40 --print_freq 40 --save_latest_freq 1000 \
--niter 5 --niter_decay 5 --n_scales_temporal 3 --n_frames_D 2 \
--no_first_img --n_frames_total 12 --max_t_step 4 --tf_log --no_flow \
--continue_train --load_pretrain_2 ./check_points/test/clothpart_256p --which_epoch_2 2

