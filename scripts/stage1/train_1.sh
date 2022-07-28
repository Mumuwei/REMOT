python train_stage1.py --name parser_256p \
--dataroot /home/wujinlin5/yqw_home/Motion_Transfer/DATASET/iper/test --dataset_mode parser --model parser --nThreads 16 \
--input_nc_T_1 11 --input_nc_S_1 3 --ngf 64 --ndf 32 --label_nc_1 11 --output_nc_1 11 --num_D 2 \
--resize_or_crop scaleHeight --loadSize 256 --random_drop_prob 0 \
--gpu_ids 0 --n_gpus_gen 1 --batchSize 4 --max_frames_per_gpu 6 --display_freq 40 --print_freq 40 --save_latest_freq 1000 \
--niter 5 --niter_decay 5 --n_scales_temporal 3 --lambda_P 10 \
--no_first_img --n_frames_total 12 --max_t_step 4 --tf_log \
--load_pretrain_1 /home/wujinlin5/yqw_home/Motion_Transfer/C2F-iper/check_points/test2/parser_256p --which_epoch_1 latest


#n_frames_total the overall number of frames in a sequence to train with
#max_frames_per_gpu 每次加载到一个gpu的帧数
#max_t_step max spacing between neighboring sampled frames. If greater than 1, the network may randomly skip frames during training