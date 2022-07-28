import os.path
import torchvision.transforms as transforms
import torch
from PIL import Image
import numpy as np

from data.base_dataset import BaseDataset, get_img_params, get_transform, get_transform_fixed, get_video_params, concat_frame
from data.image_folder import make_grouped_dataset, check_path_valid
from data.keypoint2img import read_keypoints

#dataset for full testing of all the stages
class Test123Dataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot 
        
        #self.dir_tpose = os.path.join(opt.dataroot, opt.phase + '_pose/target')
        #目标图像数据
        self.dir_tparsing = os.path.join(opt.dataroot, opt.phase + '_parsing_adj/target')
        self.dir_tfg = os.path.join(opt.dataroot, opt.phase + '_fg_adj/target')
        
        
        #源图像数据
        self.dir_spose = os.path.join(opt.dataroot, opt.phase + '_pose/source')
        self.dir_sparsing = "/home/yangquanwei1/Motion_Transfer/C2F-iper/results/full_256p/test_input_stage2_SP"#这个应该使用stage1的输出
        self.dir_sfg = os.path.join(opt.dataroot, opt.phase + '_fg/source')
        self.dir_simg = os.path.join(opt.dataroot, opt.phase + '_img/source')
        self.dir_bg = os.path.join(opt.dataroot, opt.phase + '_bg')
             
        self.tparsing_paths = sorted(make_grouped_dataset(self.dir_tparsing))
        self.tfg_paths = sorted(make_grouped_dataset(self.dir_tfg))
        
        self.spose_paths = sorted(make_grouped_dataset(self.dir_spose))
        self.sparsing_paths = sorted(make_grouped_dataset(self.dir_sparsing))
        self.sfg_paths = sorted(make_grouped_dataset(self.dir_sfg))
        self.simg_paths = sorted(make_grouped_dataset(self.dir_simg))
        
        self.init_frame_idx_full(self.simg_paths)

    def __getitem__(self, index):
        TPose, TParsing, TFG, TPose_uncloth, TParsing_uncloth, TFG_uncloth, TPose_cloth, TParsing_cloth, TFG_cloth, SPose, SParsing, SI, BG, BG_flag, seq_idx = self.update_frame_idx_full(self.simg_paths, index)
        simg_paths = self.simg_paths[seq_idx]        
        n_frames_total, start_idx, t_step = get_video_params(self.opt, self.n_frames_total, len(simg_paths), self.frame_idx)
        
        simg = Image.open(simg_paths[start_idx]).convert('RGB')     
        size = simg.size

        BigSizeFlag = True
        if size[0]/size[1] > 1:
            BigSizeFlag = True
        else:
            BigSizeFlag = False

        if BigSizeFlag:
            params = get_img_params(self.opt, (1920,1080))
        else:
            params = get_img_params(self.opt, size)

        tfg_path = self.tfg_paths[seq_idx][0]
        
        video_name = tfg_path.split("/")[-2]
        bg_path = self.dir_bg + '/' + video_name + '.png'
        BG_i, BG_flag = self.get_bg_image(bg_path, size, params, BigSizeFlag)



        frame_range = list(range(n_frames_total+2)) if (self.opt.isTrain or self.TPose is None) else [self.opt.n_frames_G-1+2]
        for i in frame_range:
            
            #目标数据读取
            #tpose_path = self.tpose_paths[seq_idx][start_idx + i * t_step]
            tparsing_path = self.tparsing_paths[seq_idx][start_idx + i * t_step]
            tfg_path = self.tfg_paths[seq_idx][start_idx + i * t_step]
            TParsing_i, TFG_i = self.get_TImage(tparsing_path, tfg_path, size, params, BigSizeFlag)
            
            TParsing_i = self.crop(TParsing_i)
            TFG_i = self.crop(TFG_i)
            
            TParsing = concat_frame(TParsing, TParsing_i, n_frames_total+2)
            TFG = concat_frame(TFG, TFG_i, n_frames_total+2)
        
            simg_path = simg_paths[start_idx + i * t_step]
            spose_path = self.spose_paths[seq_idx][start_idx + i * t_step]
            sparsing_path = self.sparsing_paths[seq_idx][start_idx + i * t_step]

            SPose_i, SParsing_i, SI_i = self.get_SImage(spose_path, simg_path, sparsing_path, size, params, BigSizeFlag, BG_flag)

            SI_i = self.crop(SI_i)
            SPose_i = self.crop(SPose_i)
            #SParsing_i = self.crop(SParsing_i)

            SPose = concat_frame(SPose, SPose_i, n_frames_total+2)
            SParsing = concat_frame(SParsing, SParsing_i, n_frames_total+2)
            SI = concat_frame(SI, SI_i, n_frames_total+2)
            BG = concat_frame(BG, BG_i, n_frames_total+2)
            
            #print("n_frames_total",n_frames_total)#3
        if not self.opt.isTrain:
            self.TParsing, self.TFG, self.SPose, self.SParsing, self.SI, self.BG, self.BG_flag = TParsing, TFG, SPose, SParsing, SI, BG, BG_flag
            self.frame_idx += 1            
        change_seq = False if self.opt.isTrain else self.change_seq
        #print('SParsing',SParsing.shape)
        
        return_list = { 'TParsing': TParsing, 'TFG': TFG, 'SPose': SPose, 'SParsing': SParsing, 'SI': SI, 'BG': BG, 'BG_flag': BG_flag, 'A_path': simg_path, 'change_seq': change_seq}
        
        
        return return_list

    def get_bg_image(self, bg_path, size, params, BigSizeFlag):
        if os.path.exists(bg_path):
            BG = Image.open(bg_path).convert('RGB')
            if BG.size == (1024,1024):
                BG = BG.resize((256,256), Image.BICUBIC)   
            transform_scale = get_transform_fixed(self.opt, params, normalize=True, method=Image.BICUBIC, color_aug=self.opt.color_aug)
            BG_scaled = transform_scale(BG)
            #print("xxx",BG_scaled.shape)
            BG_scaled = self.crop(BG_scaled)
            BG_flag = True
        else:
            
            print('No available background input found in: ' + bg_path)
            BG_scaled = -torch.ones(3, 256, 192)
            BG_flag = False
        return BG_scaled, BG_flag

    def get_SImage(self, spose_path, simg_path, sparsing_path, size, params, BigSizeFlag, BG_flag):          
        SI = Image.open(simg_path).convert('RGB')
        if SI.size != (1920,1080) and SI.size != (192,256) and BigSizeFlag:
            SI = SI.resize((1920,1080), Image.BICUBIC)
        elif not BigSizeFlag and SI.size != (192,256):
            SI = SI.resize((256,256), Image.BICUBIC)

        SParsing = Image.open(sparsing_path)
        
#         if SParsing.size != (1920,1080) and SParsing.size != (192,256) and BigSizeFlag:
#             SParsing = SParsing.resize((1920,1080), Image.NEAREST)
#         elif not BigSizeFlag and SParsing.size != (192,256):
#             SParsing = SParsing.resize((192,256), Image.NEAREST)
                 
        SParsing_np = np.array(SParsing)
        print(SParsing_np.shape)

        SParsing_new_np = np.zeros_like(SParsing_np)
        SParsing_new_np[(SParsing_np == 1) ] = 1
        SParsing_new_np[(SParsing_np == 2) ] = 2
        SParsing_new_np[(SParsing_np == 3) | (SParsing_np == 4) ] = 3
        
        SParsing_new_np[(SParsing_np == 7)| (SParsing_np == 8) | (SParsing_np == 9) | (SParsing_np == 10)] = 4 # 四肢
        SParsing_new_np[(SParsing_np == 5)| (SParsing_np == 6)] = 5#鞋子sing_new_np = np.zeros_like(SParsing_np)
        
        
        
#         TParsing_new_np[(TParsing_np == 9)] = 5
#         TParsing_new_np[(TParsing_np == 10)] = 6
#         TParsing_new_np[(TParsing_np == 12)] = 7
#         TParsing_new_np[(TParsing_np == 13)] = 8
#         TParsing_new_np[(TParsing_np == 14)] = 9
#         TParsing_new_np[(TParsing_np == 15)] = 10
       
        
#         1:上衣
#         2:下衣
#         3：头发
#         4：脸
#         5：14
#         6：15
#         7：16
#         8：17
#         9：10
#         10：18
#         11：19

        
        SParsing_new = Image.fromarray(SParsing_new_np)
        #print("SParsing_new",SParsing_new)
        
#         if BigSizeFlag:
#             transform_scale = get_transform(self.opt, params, normalize=False, method=Image.NEAREST)
#         else:
        transform_scale = get_transform_fixed(self.opt, params, normalize=False, method=Image.NEAREST)
        SParsing_scaled = transform_scale(SParsing_new)*255.0
        #print("SParsing_new",SParsing_new)
        
        if not BG_flag:
            SI_np = np.array(SI)
            SI_np[:,:,0][SParsing_np==0] = 0
            SI_np[:,:,1][SParsing_np==0] = 0
            SI_np[:,:,2][SParsing_np==0] = 0
            SI = Image.fromarray(SI_np)

        random_drop_prob = self.opt.random_drop_prob if self.opt.isTrain else 0
        #print(size)
        SPose_array, _ = read_keypoints(spose_path, size, random_drop_prob, self.opt.remove_face_labels, self.opt.basic_point_only)
        SPose = Image.fromarray(SPose_array)         
        if SPose.size == (1024,1024):
            SPose = SPose.resize((256,256), Image.NEAREST)

        if BigSizeFlag:
            transform_scale = get_transform(self.opt, params, normalize=True, method=Image.NEAREST)
        else:
            transform_scale = get_transform_fixed(self.opt, params, normalize=True, method=Image.NEAREST)
        SPose_scaled = transform_scale(SPose)

        if BigSizeFlag:
            transform_scale = get_transform(self.opt, params, normalize=True, method=Image.BICUBIC)
        else:
            transform_scale = get_transform_fixed(self.opt, params, normalize=True, method=Image.BICUBIC)
        SI_scaled = transform_scale(SI)

        return SPose_scaled, SParsing_scaled, SI_scaled

    def get_TImage(self, tparsing_path, timg_path, size, params, BigSizeFlag):          
        random_drop_prob = self.opt.random_drop_prob if self.opt.isTrain else 0

        TI = Image.open(timg_path).convert('RGB')
        if TI.size != (1920,1080) and BigSizeFlag:
            TI = TI.resize((1920,1080), Image.BICUBIC)
        elif not BigSizeFlag:
            TI = TI.resize((256,256), Image.BICUBIC)
        TFG_np = np.array(TI)

        TParsing = Image.open(tparsing_path)
        TParsing_size = TParsing.size
        if TParsing_size != (1920,1080) and TParsing_size != (192,256) and BigSizeFlag:
            TParsing = TParsing.resize((1920,1080), Image.NEAREST)
        elif not BigSizeFlag and TParsing_size != (192,256):
            TParsing = TParsing.resize((256,256), Image.NEAREST)
        TParsing_np = np.array(TParsing)

        TParsing_new_np = np.zeros_like(TParsing_np)
        
        TParsing_new_np[(TParsing_np == 4) | (TParsing_np == 16) | (TParsing_np == 17) | (TParsing_np == 5) | (TParsing_np == 7)] = 1#上衣
        TParsing_new_np[(TParsing_np == 8) | (TParsing_np == 6)] = 2#下衣
        TParsing_new_np[(TParsing_np == 1) | (TParsing_np == 2) | (TParsing_np == 3)|  (TParsing_np == 11)] = 3  #人脸
        TParsing_new_np[(TParsing_np == 12)| (TParsing_np == 13) | (TParsing_np == 14) | (TParsing_np == 15)] = 4 # 四肢
        TParsing_new_np[(TParsing_np == 9) | (TParsing_np == 10)] = 5#鞋子
        
        
        

        TParsing_new = Image.fromarray(TParsing_new_np)
        if TParsing_size != (192,256) and BigSizeFlag:
            transform_scale = get_transform(self.opt, params, normalize=False, method=Image.NEAREST, color_aug=False)
        else:
            transform_scale = get_transform_fixed(self.opt, params, normalize=False, method=Image.NEAREST, color_aug=False)
        TParsing_scaled = transform_scale(TParsing_new)*255.0

        if BigSizeFlag:
            transform_scale = get_transform(self.opt, params, normalize=True, method=Image.BICUBIC, color_aug=self.opt.color_aug)
        else:
            transform_scale = get_transform_fixed(self.opt, params, normalize=True, method=Image.BICUBIC, color_aug=self.opt.color_aug)
        


        TFG_scaled = transform_scale(Image.fromarray(TFG_np))

        return TParsing_scaled, TFG_scaled

    def crop(self, Ai):
        w = Ai.size()[2]
        base = 32
        x_cen = w // 2
        bs = int(w * 0.25) // base * base
        bs = 96
        return Ai[:,:,(x_cen-bs):(x_cen+bs)]

    def __len__(self):        
        return sum(self.frames_count)

    def name(self):
        return 'FullDataset'