import os.path
import torchvision.transforms as transforms
import torch
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import numpy as np

from data.base_dataset import BaseDataset, get_img_params, get_transform, get_transform_fixed, get_video_params, concat_frame
from data.image_folder import make_grouped_dataset, check_path_valid
from data.keypoint2img import read_keypoints

# dataset for the Composition GAN of stage 3，和solodance一样的
class ComposerDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot 

        
        #目标人体解析图，目标图像，源姿态，源解析图，源前景，源图像，背景
        
        self.dir_tparsing = os.path.join(opt.dataroot, opt.phase + '_parsing_adj/target')
        self.dir_tfg = os.path.join(opt.dataroot, opt.phase + '_fg_adj/target')
        
        self.dir_spose = os.path.join(opt.dataroot, opt.phase + '_pose/source')
        self.dir_sparsing = os.path.join(opt.dataroot, opt.phase + '_parsing/source')
        self.dir_sfg = os.path.join(opt.dataroot, opt.phase + '_fg/source')
        self.dir_simg = os.path.join(opt.dataroot, opt.phase + '_img/source')
        self.dir_bg = os.path.join(opt.dataroot, opt.phase + '_bg')
             
        self.tparsing_paths = sorted(make_grouped_dataset(self.dir_tparsing))
        self.tfg_paths = sorted(make_grouped_dataset(self.dir_tfg))
        self.spose_paths = sorted(make_grouped_dataset(self.dir_spose))
        self.sparsing_paths = sorted(make_grouped_dataset(self.dir_sparsing))
        self.sfg_paths = sorted(make_grouped_dataset(self.dir_sfg))
        self.simg_paths = sorted(make_grouped_dataset(self.dir_simg))

        self.init_frame_idx_composer(self.simg_paths)

    def __getitem__(self, index):
        TParsing, TFG, SPose, SParsing, SFG, SFG_full, BG, BG_flag, SI, seq_idx = self.update_frame_idx_composer(self.simg_paths, index)
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
            params = get_img_params(self.opt, (1024,1024))
        else:
            params = get_img_params(self.opt, size)

        
        tfg_path = self.tfg_paths[seq_idx][0]
        #/home/wujinlin5/yqw_home/data/SoloDance_upload/train/train_fg_adg/target/video16/000102.png
        #print("timg_path",timg_path)
        
        #video_name = timg_path[timg_path.rfind('video'):timg_path.rfind('/timg')]
        video_name = tfg_path.split("/")[-2]
        #print("video_name",video_name)
        bg_path = self.dir_bg + '/' + video_name + '.png'
        #print("bg_path",bg_path)
        BG_i, BG_flag = self.get_bg_image(bg_path, size, params, BigSizeFlag)
        #print("BG_i",BG_i.shape)

        

        frame_range = list(range(n_frames_total)) if (self.opt.isTrain or self.TPose is None) else [self.opt.n_frames_G-1]
        for i in frame_range:
            #目标路径
            tparsing_path = self.tparsing_paths[seq_idx][start_idx + i * t_step]
            tfg_path = self.tfg_paths[seq_idx][start_idx + i * t_step]
            TParsing_i, TFG_i = self.get_TImage(tparsing_path, tfg_path, size, params, BigSizeFlag)
            TParsing_i, TFG_i = self.crop(TParsing_i), self.crop(TFG_i)
            #print("TParsing_i, TFG_i",TParsing_i.shape, TFG_i.shape)
            
            #源路径
            simg_path = simg_paths[start_idx + i * t_step]
            sfg_path = self.sfg_paths[seq_idx][start_idx + i * t_step]
            spose_path = self.spose_paths[seq_idx][start_idx + i * t_step]
            sparsing_path = self.sparsing_paths[seq_idx][start_idx + i * t_step]


            SPose_i = self.get_SPose(spose_path,params)
            SParsing_i, SFG_i, SI_i = self.get_SImage(sparsing_path, sfg_path, simg_path, size, params, BigSizeFlag)


            SPose_i,SParsing_i = self.crop(SPose_i),self.crop(SParsing_i)
            SFG_i,SI_i = self.crop(SFG_i), self.crop(SI_i)
            #print("SPose_i, SParsing_i, SFG_i",SPose_i.shape, SParsing_i.shape, SFG_i.shape)
            
            
            TParsing = concat_frame(TParsing, TParsing_i, n_frames_total)
            TFG = concat_frame(TFG, TFG_i, n_frames_total)
            
            SPose = concat_frame(SPose, SPose_i, n_frames_total)
            SParsing = concat_frame(SParsing, SParsing_i, n_frames_total)
            SFG = concat_frame(SFG, SFG_i, n_frames_total)
            SI = concat_frame(SI, SI_i, n_frames_total)
            BG = concat_frame(BG, BG_i, n_frames_total)

        if not self.opt.isTrain:
            self.TParsing, self.TFG, self.SPose, self.SParsing, self.SFG, self.BG, self.BG_flag, self.SI = TParsing, TFG, SPose, SParsing, SFG, BG, BG_flag, SI
            self.frame_idx += 1            
        change_seq = False if self.opt.isTrain else self.change_seq
        
        #print("TParsing, TFG",TParsing.shape, TFG.shape) BG_flag：是否有背景
        return_list = {'TParsing': TParsing, 'TFG': TFG, 'SPose': SPose, 'SParsing': SParsing, 'SFG': SFG, 'BG': BG, 'BG_flag': BG_flag, 'SI': SI, 'A_path': simg_path, 'change_seq': change_seq}
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
            print("BG----")
            BG_scaled = -torch.ones(3, 256, 192)
            BG_flag = False
        return BG_scaled, BG_flag

    
    def get_SImage(self, sparsing_path, sfg_path, simg_path, size, params, BigSizeFlag):          
        SFG = Image.open(sfg_path).convert('RGB')
        SI = Image.open(simg_path).convert('RGB')
        
        #print("SI",SI.size)
        
        if SFG.size == (1024,1024):
            SFG = SFG.resize((256,256), Image.BICUBIC)   
        if SI.size == (1024,1024):
            SI = SI.resize((256,256), Image.BICUBIC)
            
        SFG_np = np.array(SFG)
        SI_np = np.array(SI)
        
        random_drop_prob = self.opt.random_drop_prob if self.opt.isTrain else 0

        SParsing = Image.open(sparsing_path)
        SParsing_size = SParsing.size
        if SParsing_size == (1024,1024):
            SParsing = SParsing.resize((256,256), Image.NEAREST)
        SParsing_np = np.array(SParsing)

        
        #print("SParsing_new_np.shape222",np.max(SParsing_np))
        if SParsing_size == (256,256):
            SParsing_new_np = SParsing_np
            #SParsing_new_np[(SParsing_np != 1) & (SParsing_np != 2)] = 0
        else:
            #print("xxx")
            #这个数组本身都是全0，然后想要的地方赋值为1，2
            SParsing_new_np = np.zeros_like(SParsing_np)
            SParsing_new_np[(SParsing_np == 4) | (SParsing_np == 16) | (SParsing_np == 17) | (SParsing_np == 5) | (SParsing_np == 7)] = 1#上衣
            SParsing_new_np[(SParsing_np == 8) | (SParsing_np == 6)] = 2#下衣
            SParsing_new_np[(SParsing_np == 1) | (SParsing_np == 2) | (SParsing_np == 3)|  (SParsing_np == 11)] = 3  #人脸
            SParsing_new_np[(SParsing_np == 12)| (SParsing_np == 13) | (SParsing_np == 14) | (SParsing_np == 15)] = 4 # 四肢
            SParsing_new_np[(SParsing_np == 9)| (SParsing_np == 10)] = 5#鞋子
            
           
            #划分成几个区域生成
            #4,16,17,5,7
            #8,6
            #1,2,3,11
            #12,13,14,15
            #9,10            
        #print("SParsing_new_np.shape",np.max(SParsing_new_np))
       
        #0-255--->0-1---->0-255
        transform_scale = get_transform_fixed(self.opt, params, normalize=False, method=Image.NEAREST, color_aug=False)
        SParsing_scaled = transform_scale(Image.fromarray(SParsing_new_np))*255.0

        
      
        transform_scale = get_transform_fixed(self.opt, params, normalize=True, method=Image.BICUBIC, color_aug=self.opt.color_aug)
        SFG_scaled = transform_scale(Image.fromarray(SFG_np))
        SI_scaled = transform_scale(Image.fromarray(SI_np))
        
        #torch.Size([1, 256, 448]) torch.Size([3, 256, 448]) torch.Size([3, 256, 448])
        #print("SParsing_scaled, SFG_scaled, SFG_full_scaled",SParsing_scaled.shape, SFG_scaled.shape, SFG_full_scaled.shape)
        #print("SParsing_scaled",torch.max(SParsing_scaled),torch.min(SParsing_scaled))
        
        return SParsing_scaled, SFG_scaled, SI_scaled

    def get_SPose(self, spose_path,params):
        random_drop_prob = self.opt.random_drop_prob if self.opt.isTrain else 0
        size=(1024,1024)#因为是用256x256分辨率检测出来的关键点，所有清晰度可能存在问题
        SPose_array, _ = read_keypoints(spose_path, size, random_drop_prob, self.opt.remove_face_labels, self.opt.basic_point_only)
        SPose = Image.fromarray(SPose_array) 
        if SPose.size == (1024,1024):
            SPose = SPose.resize((256,256), Image.NEAREST)
       
        SPose_np = np.array(SPose)
        transform_scale = get_transform_fixed(self.opt, params, normalize=True, method=Image.NEAREST, color_aug=False)
        SPose_scaled = transform_scale(Image.fromarray(SPose_np))
        
        return SPose_scaled
    
    def get_TImage(self, sparsing_path, sfg_path, size, params, BigSizeFlag):          
        SFG = Image.open(sfg_path).convert('RGB')
       
      
        if SFG.size == (1024,1024):
            SFG = SFG.resize((256,256), Image.BICUBIC)   
           
        SFG_np = np.array(SFG)

        
        random_drop_prob = self.opt.random_drop_prob if self.opt.isTrain else 0

        SParsing = Image.open(sparsing_path)
        SParsing_size = SParsing.size
        if SParsing_size == (1024,1024):
            SParsing = SParsing.resize((256,256), Image.NEAREST)
        SParsing_np = np.array(SParsing)

        
        #print("SParsing_new_np.shape222",np.max(SParsing_np))
        if SParsing_size == (256,256):
            SParsing_new_np = SParsing_np
            #SParsing_new_np[(SParsing_np != 1) & (SParsing_np != 2)] = 0
        else:
            #print("xxx")
            #这个数组本身都是全0，然后想要的地方赋值为1，2
            SParsing_new_np = np.zeros_like(SParsing_np)
            SParsing_new_np[(SParsing_np == 4) | (SParsing_np == 16) | (SParsing_np == 17) | (SParsing_np == 5) | (SParsing_np == 7)] = 1#上衣
            SParsing_new_np[(SParsing_np == 8) | (SParsing_np == 6)] = 2#下衣
            SParsing_new_np[(SParsing_np == 1) | (SParsing_np == 2) | (SParsing_np == 3)|  (SParsing_np == 11)] = 3  #人脸
            SParsing_new_np[(SParsing_np == 12)| (SParsing_np == 13) | (SParsing_np == 14) | (SParsing_np == 15)] = 4 # 四肢
            SParsing_new_np[(SParsing_np == 9)| (SParsing_np == 10)] = 5#鞋子
            
           
            #划分成几个区域生成
            #4,16,17,5,7
            #8,6
            #1,2,3,11
            #12,13,14,15
            #9,10            
        #print("SParsing_new_np.shape",np.max(SParsing_new_np))
       
        transform_scale = get_transform_fixed(self.opt, params, normalize=False, method=Image.NEAREST, color_aug=False)
        SParsing_scaled = transform_scale(Image.fromarray(SParsing_new_np))*255.0

        transform_scale = get_transform_fixed(self.opt, params, normalize=True, method=Image.BICUBIC, color_aug=self.opt.color_aug)
        SFG_scaled = transform_scale(Image.fromarray(SFG_np))

        
        #torch.Size([1, 256, 448]) torch.Size([3, 256, 448]) torch.Size([3, 256, 448])
        #print("SParsing_scaled, SFG_scaled, SFG_full_scaled",SParsing_scaled.shape, SFG_scaled.shape, SFG_full_scaled.shape)
        #print("SParsing_scaled",torch.max(SParsing_scaled),torch.min(SParsing_scaled))
        
        return SParsing_scaled, SFG_scaled

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
        return 'ComposerDataset'