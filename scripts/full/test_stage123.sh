python test_stage23.py --name output1 \
--dataroot /home/yangquanwei1/Motion_Transfer/DATASET/iper/test --dataset_mode test123 --model full --nThreads 16 \
--input_nc_T_1 12 --input_nc_S_1 3 --input_nc_T_2 9 --input_nc_S_2 3 --input_nc_P_2 10 --input_nc_T_3 13 --input_nc_S_3 15 \
--label_nc_1 12 --label_nc_2 6 --label_nc_3 6 --output_nc_1 12 --output_nc_2 3 --output_nc_3 3 \
--ngf 64 --grid_size 3 \
--resize_or_crop scaleHeight --loadSize 256 --random_drop_prob 0 \
--no_first_img --gpu_ids 0 --use_fusion \
--load_pretrain_2 ./check_points/test/clothpart_256p \
--which_epoch_2 latest \
--load_pretrain_3 ./check_points/test/composer_256p \
--which_epoch_3 latest

