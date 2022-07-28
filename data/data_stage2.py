import os.path
import torchvision.transforms as transforms
import torch
from PIL import Image
import numpy as np

from data.base_dataset import BaseDataset, get_img_params, get_transform, get_transform_fixed, get_video_params, concat_frame
from data.image_folder import make_grouped_dataset, check_path_valid
from data.keypoint2img import read_keypoints
#only use parsing,所有区域都用上了，和solodance一样


class ClothDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot 
        #print("xxxxxxx")
        
        #网络需要用到的数据的路径
        
        #加载对齐后的人体图像以及解析图
        self.dir_tparsing = os.path.join(opt.dataroot, opt.phase + '_parsing_adj/target')
        self.dir_timg = os.path.join(opt.dataroot, opt.phase + '_fg_adj/target')
        
        
        #源人物的人体解析图以及图像
        self.dir_sparsing = os.path.join(opt.dataroot, opt.phase + '_parsing/source')
        self.dir_simg = os.path.join(opt.dataroot, opt.phase + '_fg/source')
       
        #print(self.dir_sparsing,self.dir_simg)
    
        
        #暂时先顶替一下
#         self.dir_tparsing = self.dir_sparsing
#         self.dir_timg = self.dir_simg
        
        
        #对这4个路径进行排序
        self.tparsing_paths = sorted(make_grouped_dataset(self.dir_tparsing))
        self.timg_paths = sorted(make_grouped_dataset(self.dir_timg))
        
        self.sparsing_paths = sorted(make_grouped_dataset(self.dir_sparsing))
        self.simg_paths = sorted(make_grouped_dataset(self.dir_simg))
        
        
        #print("self.simg_paths",len(self.simg_paths))#164
#         print("self.simg_paths",self.simg_paths[0])
        
#         print("self.timg1_paths",len(self.timg1_paths[0]))
#         print("self.timg2_paths",len(self.timg2_paths))

        self.init_frame_idx_cloth(self.simg_paths)

        
    #加载数据
    def __getitem__(self, index):
       
        #序列的id号,只有seq_idx有数
        TParsing, TFG, SParsing, SFG, seq_idx = self.update_frame_idx_cloth(self.simg_paths, index)
        #print(seq_idx)#0-152
       
        #第seq_idx个视频的路径
        simg_paths = self.simg_paths[seq_idx] 
        
        #获取这个视频的一些参数
        n_frames_total, start_idx, t_step = get_video_params(self.opt, self.n_frames_total, len(simg_paths), self.frame_idx)
        #print("n_frames_total, start_idx, t_step",n_frames_total, start_idx, t_step)
        
        
        simg = Image.open(simg_paths[start_idx]).convert('RGB')     
        size = simg.size
        #print(size)#(1024 1024)
        

        BigSizeFlag = True
        if size[0]/size[1] > 1:
            BigSizeFlag = True
        else:
            BigSizeFlag = False

        if BigSizeFlag:
            params = get_img_params(self.opt, (1024,1024))
        else:
            params = get_img_params(self.opt, size)

            
            
#         tparsing_path = self.tparsing_paths[seq_idx][0]
#         timg_path = self.timg_paths[seq_idx][0]

#         TParsing, TFG = self.get_TImage(tparsing_path, timg_path, size, params, BigSizeFlag)
#         TParsing, TFG = self.crop(TParsing), self.crop(TFG)
        #print(params)
        frame_range = list(range(n_frames_total)) if (self.opt.isTrain or self.TParsing is None) else [self.opt.n_frames_G-1]
        for i in frame_range:
            
            #先加载源人物的相关数据吧
            simg_path = simg_paths[start_idx + i * t_step]
            sparsing_path = self.sparsing_paths[seq_idx][start_idx + i * t_step]
            
            
            #print(simg_path,sparsing_path)
            SParsing_i, SFG_i = self.get_SImage(sparsing_path, simg_path, size, params, BigSizeFlag)
            #print("xxx",SParsing_i.size())#256,256

            SParsing_i = self.crop(SParsing_i)
            SFG_i = self.crop(SFG_i)
            #print("yyy",SParsing_i.size())#256,192

            SParsing = concat_frame(SParsing, SParsing_i, n_frames_total)
            SFG = concat_frame(SFG, SFG_i, n_frames_total)
            
            
            #这里开始加载目标人物相关数据，两者实际上是一样的
            timg_path = self.timg_paths[seq_idx][start_idx + i * t_step]
            tparsing_path = self.tparsing_paths[seq_idx][start_idx + i * t_step]
            TParsing_i, TFG_i = self.get_SImage(tparsing_path, timg_path, size, params, BigSizeFlag)
            
            TParsing_i = self.crop(TParsing_i)
            TFG_i = self.crop(TFG_i)
            
            TParsing = concat_frame(TParsing, TParsing_i, n_frames_total)
            TFG = concat_frame(TFG, TFG_i, n_frames_total)
            
        #torch.Size([14, 256, 192]) torch.Size([42, 256, 192]) torch.Size([14, 256, 192]) torch.Size([42, 256, 192])
        #print("SParsing,SFG,TParsing,TFG",SParsing.shape,SFG.shape,TParsing.shape,TFG.shape)
         
        #如果不是训练阶段，则需要视频帧id逐渐+1
        if not self.opt.isTrain:
            self.T1, self.T2, self.SParsing, self.SFG = SParsing, SFG, T1, T2
            self.frame_idx += 1            
        change_seq = False if self.opt.isTrain else self.change_seq
        return_list = {'TFG': TFG, 'TParsing': TParsing, 'SFG': SFG, 'SParsing': SParsing, 'A_path': simg_path, 'change_seq': change_seq}
     
        return return_list

    
    #获取源人物的解析图和图像,simg就是前景
    def get_SImage(self, sparsing_path, simg_path, size, params, BigSizeFlag):          
        SI = Image.open(simg_path).convert('RGB')
        #print("SI",SI.size)
        if SI.size == (1024,1024):
            SI = SI.resize((256,256), Image.BICUBIC)
        SFG_np = np.array(SI)

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
    
    
                

#     def get_TImage(self, tparsing_path, timg_path, size, params, BigSizeFlag):          
#         random_drop_prob = self.opt.random_drop_prob if self.opt.isTrain else 0
#         TI = Image.open(timg_path).convert('RGB')

#         if TI.size != (1920,1080) and BigSizeFlag:
#             TI = TI.resize((1920,1080), Image.BICUBIC)
#         elif not BigSizeFlag:
#             TI = TI.resize((192,256), Image.BICUBIC)
#         TFG_np = np.array(TI)        

#         TParsing = Image.open(tparsing_path)
#         TParsing_size = TParsing.size

#         if TParsing_size != (1920,1080) and TParsing_size != (192,256) and BigSizeFlag:
#             TParsing = TParsing.resize((1920,1080), Image.NEAREST)
#         elif not BigSizeFlag and TParsing_size != (192,256):
#             TParsing = TParsing.resize((192,256), Image.NEAREST)
#         TParsing_np = np.array(TParsing)

#         TParsing_new_np = np.zeros_like(TParsing_np)

#         TParsing_new_np[(TParsing_np == 3) | (TParsing_np == 5) | (TParsing_np == 6) | (TParsing_np == 7) | (TParsing_np == 11)] = 1
#         TParsing_new_np[(TParsing_np == 8) | (TParsing_np == 9) | (TParsing_np == 12)] = 2

#         TParsing_new = Image.fromarray(TParsing_new_np)
#         if TParsing_size != (192,256) and BigSizeFlag:
#             transform_scale = get_transform(self.opt, params, normalize=False, method=Image.NEAREST, color_aug=False)
#         else:
#             transform_scale = get_transform_fixed(self.opt, params, normalize=False, method=Image.NEAREST, color_aug=False)
#         TParsing_scaled = transform_scale(TParsing_new)*255.0

#         if BigSizeFlag:
#             transform_scale = get_transform(self.opt, params, normalize=True, method=Image.NEAREST, color_aug=False)
#         else:
#             transform_scale = get_transform_fixed(self.opt, params, normalize=True, method=Image.NEAREST, color_aug=False)

#         if BigSizeFlag:
#             transform_scale = get_transform(self.opt, params, normalize=True, method=Image.BICUBIC, color_aug=self.opt.color_aug)
#         else:
#             transform_scale = get_transform_fixed(self.opt, params, normalize=True, method=Image.BICUBIC, color_aug=self.opt.color_aug)
#         TFG_np[:,:,0][(TParsing_new_np == 0)] = 0
#         TFG_np[:,:,1][(TParsing_new_np == 0)] = 0
#         TFG_np[:,:,2][(TParsing_new_np == 0)] = 0
#         TFG_scaled = transform_scale(Image.fromarray(TFG_np))

#         return TParsing_scaled, TFG_scaled

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
        return 'ClothDataset'