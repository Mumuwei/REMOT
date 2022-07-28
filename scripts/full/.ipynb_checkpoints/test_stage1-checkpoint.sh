python test_stage1.py --name output1 \
--dataroot /home/yangquanwei1/Motion_Transfer/DATASET/iper/test --dataset_mode test1 --model parser --nThreads 16 \
--input_nc_T_1 11 --input_nc_S_1 3 --input_nc_T_2 9 --input_nc_S_2 3 --input_nc_P_2 10 --input_nc_T_3 13 --input_nc_S_3 15 \
--label_nc_1 11 --label_nc_2 6 --label_nc_3 6 --output_nc_1 11 --output_nc_2 3 --output_nc_3 3 \
--ngf 64 --grid_size 3 \
--resize_or_crop scaleHeight --loadSize 256 --random_drop_prob 0 \
--no_first_img --gpu_ids 0 \
--load_pretrain_1 ./check_points/test/parser_256p \
--which_epoch_1 5 \

