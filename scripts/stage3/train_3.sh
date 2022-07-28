python train_stage3.py --name composer_256p \
--dataroot /home/wujinlin5/yqw_home/Motion_Transfer/DATASET/iper/train --dataset_mode composer --model composer --nThreads 16 \
--checkpoints_dir ./check_points/test \
--input_nc_T_2 9 --input_nc_S_2 3 --output_nc_2 3 --input_nc_T_3 13 --input_nc_S_3 15 --ngf 64 --ndf 32 --add_face_disc --label_nc_3 6 --label_nc_2 6 --output_nc_3 3 --num_D 2 \
--resize_or_crop scaleHeight --loadSize 256 --random_drop_prob 0 --color_aug \
--gpu_ids 1 --n_gpus_gen 1 --batchSize 1 --max_frames_per_gpu 6 --display_freq 40 --print_freq 40 --save_latest_freq 1000 \
--niter 5 --niter_decay 5 --n_scales_temporal 3 \
--no_first_img --n_frames_total 12 --max_t_step 4 --tf_log --use_fusion --continue_train \
--load_pretrain_2 /home/wujinlin5/yqw_home/Motion_Transfer/C2F-iper/check_points/test/clothpart_256p --which_epoch_2 latest \
--load_pretrain_3 /home/wujinlin5/yqw_home/Motion_Transfer/C2F-iper/check_points/test/composer_256p --which_epoch_3 latest

