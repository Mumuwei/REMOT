import os.path
import torchvision.transforms as transforms
import torch
from PIL import Image
import numpy as np

from data.base_dataset import BaseDataset, get_img_params, get_transform, get_transform_fixed, get_video_params, concat_frame
from data.image_folder import make_grouped_dataset, check_path_valid
from data.keypoint2img import read_keypoints

# dataset for the Layout GAN of stage 1，阶段1的数据读取
class ParserDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot 

        #各个数据的路径
        opt.phase="test"
        self.dir_tparsing = os.path.join(opt.dataroot, opt.phase + '_parsing/target')#目标人物的parsing
        self.dir_spose = os.path.join(opt.dataroot, opt.phase + '_pose/source')    #源人物的pose
        self.dir_sparsing = os.path.join(opt.dataroot, opt.phase + '_parsing/source')#源人物的parsing
        self.dir_sfg = os.path.join(opt.dataroot, opt.phase + '_fg/source')       #源人物的前景区域 （只有人体部分）
        opt.phase="train"
        
        #获得该目录下的所有文件，video/frame  目标视频就1帧 ,源视频有300帧
        self.tparsing_paths = sorted(make_grouped_dataset(self.dir_tparsing))
        self.spose_paths = sorted(make_grouped_dataset(self.dir_spose))
        self.sparsing_paths = sorted(make_grouped_dataset(self.dir_sparsing))
        self.sfg_paths = sorted(make_grouped_dataset(self.dir_sfg))

        
        #这个函数在basedataset类中
        self.init_frame_idx_parser(self.sparsing_paths)

    def __getitem__(self, index):
        #加载数据,153个视频序列的id,0-153
        #print("index0",index)
        
        TParsing, SPose, SParsing, SFG, seq_idx = self.update_frame_idx_parser(self.sparsing_paths, index)
        #print("seq_idx",seq_idx)
        #某个视频序列的解析图路径和前景路径
        sparsing_paths = self.sparsing_paths[seq_idx]        
        sfg_paths = self.sfg_paths[seq_idx]
        
        n_frames_total, start_idx, t_step = get_video_params(self.opt, self.n_frames_total, len(sparsing_paths), self.frame_idx)
        
        #根据得到的参数加载数据
        sparsing = Image.open(sparsing_paths[start_idx])
        sfg = Image.open(sfg_paths[start_idx]).convert('RGB')
        size = sfg.size
        
        #print("size",size)#1920x1080,wxh
        
        BigSizeFlag = True
        if size[0]/size[1] > 1:
            BigSizeFlag = True
        else:
            BigSizeFlag = False

        #params {'new_size': (448, 256), 'crop_size': (0, 0), 'crop_pos': (0, 0), 'flip': False, 'color_aug': (25.47493445372909, 0.9425410920649194,               -9.5706724238375, 1.1142829309979536, 6.387742955341906)}
        
        #返回数据增强等相关参数
        if BigSizeFlag:
            params = get_img_params(self.opt, (1920, 1080))
            #print("params",params)
        else:
            params = get_img_params(self.opt, size)

        #target video中只有一个视频帧
        tparsing_path = self.tparsing_paths[seq_idx][0]

        #处理这个target视频帧
        TParsing = self.get_TImage(tparsing_path, size, params, BigSizeFlag)
        TParsing = self.crop(TParsing)
        #print(TParsing.shape)#torch.Size([1, 256, 192])
        
        
        #生成0-13的数字
        frame_range = list(range(n_frames_total)) if (self.opt.isTrain or self.TPose is None) else [self.opt.n_frames_G-1]
        #print(frame_range)
        for i in frame_range:
            #开始加载源视频帧序列
            #加载源视频的sparsing数据
            sparsing_path = sparsing_paths[start_idx + i * t_step]
            #加载源视频帧的pose数据
            spose_path = self.spose_paths[seq_idx][start_idx + i * t_step]
            #加载源视频帧的前景数据
            sfg_path = sfg_paths[start_idx + i * t_step]
            
            #对视频帧进行处理
            SPose_i, SParsing_i, SFG_i = self.get_SImage(spose_path, sparsing_path, sfg_path, size, params, BigSizeFlag)

            #对这3种数据进行裁剪 256，192
            SParsing_i = self.crop(SParsing_i)
            SPose_i = self.crop(SPose_i)
            SFG_i = self.crop(SFG_i)

            #把这14帧拼接起来
            SPose = concat_frame(SPose, SPose_i, n_frames_total)
            SParsing = concat_frame(SParsing, SParsing_i, n_frames_total)
            SFG = concat_frame(SFG, SFG_i, n_frames_total)
            
          
        #print("SPose,SParsing,SFG",SPose.shape,SParsing.shape,SFG.shape)    
            
        #print("------------------------------------------")
        if not self.opt.isTrain:
            self.TParsing, self.SPose, self.SParsing, self.SFG = TParsing, SPose, SParsing, SFG
            self.frame_idx += 1            
        change_seq = False if self.opt.isTrain else self.change_seq
        
        #torch.Size([42, 256, 192]) torch.Size([14, 256, 192]) torch.Size([42, 256, 192]) torch.Size([1, 256, 192])
        #print("SPose,SParsing,SFG,TParsing",SPose.shape,SParsing.shape,SFG.shape,TParsing.shape,sparsing_path)   
        
        return_list = {'TParsing': TParsing, 'SPose': SPose, 'SParsing': SParsing, 'SFG': SFG, 'A_path': sparsing_path, 'change_seq': change_seq}
        return return_list

    #获取源视频的一些变换后的数据，源视频帧即是驱动视频帧
    def get_SImage(self, spose_path, sparsing_path, sfg_path, size, params, BigSizeFlag): 
        #随机丢弃一部分数据
        random_drop_prob = self.opt.random_drop_prob if self.opt.isTrain else 0
        #读取前景区域
        SFG = Image.open(sfg_path).convert('RGB')
        #读取pose数据
        #(1920, 1080) 0.0 False False
        SPose_array, _ = read_keypoints(spose_path, size, random_drop_prob, self.opt.remove_face_labels, self.opt.basic_point_only)
        SPose = Image.fromarray(SPose_array)  
        
        #print("SFG,SPose",SFG.size,SPose.size)#1920,1080
        if SPose.size != (1920,1080) and BigSizeFlag:
            SPose = SPose.resize((1920,1080), Image.NEAREST)
        elif not BigSizeFlag:
            SPose = SPose.resize((256,256), Image.NEAREST)
        SPose_np = np.array(SPose)
        #print("SPose_np",SPose_np.shape)#(1080, 1920, 3)
        
        #读取解析图数据
        SParsing = Image.open(sparsing_path)
        SParsing_size = SParsing.size
        if SParsing_size != (1920,1080) and BigSizeFlag:
            SParsing = SParsing.resize((1920,1080), Image.NEAREST)
        elif not BigSizeFlag:
            SParsing = SParsing.resize((256,256), Image.NEAREST)
        SParsing_np = np.array(SParsing)
        #print("SParsing_np",SParsing_np.shape)

        if SFG.size != (1920,1080) and BigSizeFlag:
            SFG = SFG.resize((1920,1080), Image.BICUBIC)
        elif not BigSizeFlag and SFG.size != (256,256):
            SFG = SFG.resize((256,256), Image.BICUBIC)

        #print("SFG",SFG.size)#(1080, 1920)
        SParsing_new_np = np.zeros_like(SParsing_np)
        
       
        SParsing_new_np[(SParsing_np == 4) | (SParsing_np == 16) | (SParsing_np == 17) | (SParsing_np == 5) | (SParsing_np == 7)] = 1#上衣
        SParsing_new_np[(SParsing_np == 8) | (SParsing_np == 6)] = 2#下衣
        SParsing_new_np[(SParsing_np == 1) | (SParsing_np == 2)] = 3  #头发
        SParsing_new_np[(SParsing_np == 3) | (SParsing_np == 11)] = 4  #人脸

        SParsing_new_np[(SParsing_np == 9)] = 5
        SParsing_new_np[(SParsing_np == 10)] = 6
        SParsing_new_np[(SParsing_np == 12)] = 7
        SParsing_new_np[(SParsing_np == 13)] = 8
        SParsing_new_np[(SParsing_np == 14)] = 9
        SParsing_new_np[(SParsing_np == 15)] = 10
            
        
        ##划分成几个区域生成
        #1上衣         3,5,6,7,11 
        #2下衣         8,9,12
        #3头部         1,2+4,13 头发加面部
        #4四肢         10,14,15,16,17
        #5鞋子加袜子     18,19

        if BigSizeFlag:
            transform_scale = get_transform(self.opt, params, normalize=True, method=Image.NEAREST)
        else:
            transform_scale = get_transform_fixed(self.opt, params, normalize=True, method=Image.NEAREST)

        SPose_scaled = transform_scale(Image.fromarray(SPose_np))

        SParsing_new = Image.fromarray(SParsing_new_np)
        if BigSizeFlag:
            transform_scale = get_transform(self.opt, params, normalize=False, method=Image.NEAREST)
        else:
            transform_scale = get_transform_fixed(self.opt, params, normalize=False, method=Image.NEAREST)
        SParsing_scaled = transform_scale(SParsing_new)*255.0

        if BigSizeFlag:
            transform_scale = get_transform(self.opt, params, normalize=True, method=Image.BICUBIC)
        else:
            transform_scale = get_transform_fixed(self.opt, params, normalize=True, method=Image.BICUBIC)
        SFG_scaled = transform_scale(SFG)

        #print("SPose_scaled, SParsing_scaled, SFG_scaled",SPose_scaled.shape, SParsing_scaled.shape, SFG_scaled.shape)
        return SPose_scaled, SParsing_scaled, SFG_scaled

    
    #处理目标视频帧的人体解析图
    def get_TImage(self, tparsing_path, size, params, BigSizeFlag):  
        #随机drop的概率
        random_drop_prob = self.opt.random_drop_prob if self.opt.isTrain else 0
        
        TParsing = Image.open(tparsing_path)
        TParsing_size = TParsing.size
        #print("TParsing_size",TParsing_size) #1920,1080
        if TParsing_size != (1920,1080) and BigSizeFlag:
            TParsing = TParsing.resize((1920,1080), Image.NEAREST)
        elif not BigSizeFlag:
            TParsing = TParsing.resize((256,256), Image.NEAREST)
        TParsing_np = np.array(TParsing)

        #print("TParsing_np",TParsing_np.shape)#1080,1920
        TParsing_new_np = np.zeros_like(TParsing_np)
        TParsing_new_np[(TParsing_np == 4) | (TParsing_np == 16) | (TParsing_np == 17) | (TParsing_np == 5) | (TParsing_np == 7)] = 1#上衣
        TParsing_new_np[(TParsing_np == 8) | (TParsing_np == 6)] = 2#下衣
        TParsing_new_np[(TParsing_np == 1) | (TParsing_np == 2)] = 3  #头发
        TParsing_new_np[(TParsing_np == 3) | (TParsing_np == 11)] = 4  #人脸

        TParsing_new_np[(TParsing_np == 9)] = 5
        TParsing_new_np[(TParsing_np == 10)] = 6
        TParsing_new_np[(TParsing_np == 12)] = 7
        TParsing_new_np[(TParsing_np == 13)] = 8
        TParsing_new_np[(TParsing_np == 14)] = 9
        TParsing_new_np[(TParsing_np == 15)] = 10
        
        #print("TParsing_new_np",TParsing_new_np.shape)#1080,1920

        TParsing_new = Image.fromarray(TParsing_new_np)
        #print("TParsing_new",TParsing_new.size)#1920,1080
        
        
        #进行transform
        if BigSizeFlag:
            transform_scale = get_transform(self.opt, params, normalize=False, method=Image.NEAREST)
        else:
            transform_scale = get_transform_fixed(self.opt, params, normalize=False, method=Image.NEAREST)
        #print("transform_scale",transform_scale)
        TParsing_scaled = transform_scale(TParsing_new)*255.0
        
        #print("TParsing_scaled",TParsing_scaled.shape)#1, 256, 448

        if BigSizeFlag:
            transform_scale = get_transform(self.opt, params, normalize=True, method=Image.NEAREST)
        else:
            transform_scale = get_transform_fixed(self.opt, params, normalize=True, method=Image.NEAREST)

        return TParsing_scaled

    def crop(self, Ai):
        w = Ai.size()[2]
        base = 32
        x_cen = w // 2
        bs = int(w * 0.25) // base * base#448*0.25=114，114//32=3，3*32=96
        bs = 96
        return Ai[:,:,(x_cen-bs):(x_cen+bs)]

    def __len__(self):        
        return sum(self.frames_count) #298x153

    def name(self):
        return 'ParserDataset'