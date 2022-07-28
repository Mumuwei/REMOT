from util.util import add_dummy_to_tensor
import torch.utils.data as data
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import random

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

    
    #初始化帧id
    def init_frame_idx_parser(self, A_paths):
        #有多少个序列,153的序列
        self.n_of_seqs = len(A_paths)                                         # number of sequences to train
        #最大的序列的长度，好像都是300
        self.seq_len_max = max([len(A) for A in A_paths])                     # max number of frames in the training sequences

        self.seq_idx = 0                                                      # index for current sequence
        
        #如果是训练阶段，frame_idx=0，测试阶段为start_frame，这个只在test过程中有
        self.frame_idx = self.opt.start_frame if not self.opt.isTrain else 0  # index for current frame in the sequence
        self.frames_count = []                                                # number of frames in each sequence
        
        for path in A_paths:
            self.frames_count.append(len(path) - self.opt.n_frames_G + 1)
            #n_frame_G是生成器输入的样本数，n_frame_G-1是输入的前多少帧,300帧的序列，有效训练长度只有298帧
 
        tmp = np.array(self.frames_count)
        self.frames_cum = tmp.cumsum() 

        self.video_len = len(A_paths[0])

        self.folder_prob = [count / sum(self.frames_count) for count in self.frames_count]#n_frames_total，一个序列中训练的总帧数,每个序列被选中的概率
        #训练30帧
        
        self.n_frames_total = self.opt.n_frames_total if self.opt.isTrain else 1 
        self.TParsing, self.SPose, self.SParsing, self.SFG = None, None, None, None

    
    #初始化stage2的帧id，初始化部分
    def init_frame_idx_cloth(self, A_paths):
        self.n_of_seqs = len(A_paths)                         # number of sequences to train 164个
        
        self.seq_len_max = max([len(A) for A in A_paths])           # max number of frames in the training sequences 300帧视频帧
        #print("xxx",self.seq_len_max)#2506
        self.seq_len_min = min([len(A) for A in A_paths])  
        #print("xxx",self.seq_len_min)#408
        
        
        self.seq_idx = 0                                         # index for current sequence
        self.frame_idx = self.opt.start_frame if not self.opt.isTrain else 0      # index for current frame in the sequence 测试阶段开始视频帧
        self.frames_count = []                                     # number of frames in each sequence
        for path in A_paths:
            # print(len(path) - self.opt.n_frames_G + 1),298
            self.frames_count.append(len(path) - self.opt.n_frames_G + 1)
        
        tmp = np.array(self.frames_count)
        self.frames_cum = tmp.cumsum()   
            
            

        self.video_len = len(A_paths[0]) 
        self.folder_prob = [count / sum(self.frames_count) for count in self.frames_count]
        #print(self.folder_prob) #每个视频被选中的概率？
        
        self.n_frames_total = self.opt.n_frames_total if self.opt.isTrain else 1 
        #print(self.n_frames_total) #训练的长度，一次训练中包含的视频帧数，12
        
        self.T_img1, self.T_img2, self.SParsing, self.SFG_full = None, None, None, None
        
        #self.TParsing, self.TFG, self.SParsing, self.SFG, self.SFG_full = None, None, None, None, None

    def init_frame_idx_composer(self, A_paths):
        self.n_of_seqs = len(A_paths)                                         # number of sequences to train
        self.seq_len_max = max([len(A) for A in A_paths])                     # max number of frames in the training sequences

        self.seq_idx = 0                                                      # index for current sequence
        self.frame_idx = self.opt.start_frame if not self.opt.isTrain else 0  # index for current frame in the sequence
        self.frames_count = []                                                # number of frames in each sequence
        for path in A_paths:
            #print(len(path) - self.opt.n_frames_G + 1)
            self.frames_count.append(len(path) - self.opt.n_frames_G + 1)
            
        tmp = np.array(self.frames_count)
        self.frames_cum = tmp.cumsum()  

        self.video_len = len(A_paths[0])

        self.folder_prob = [count / sum(self.frames_count) for count in self.frames_count]
        self.n_frames_total = self.opt.n_frames_total if self.opt.isTrain else 1 
        self.TParsing, self.TFG, self.SPose, self.SParsing, self.SFG, self.SFG_full, self.BG, self.BG_flag, self.SI = None, None, None, None, None, None, None, None, None

    def init_frame_idx_full(self, A_paths):
        self.n_of_seqs = len(A_paths)                                         # number of sequences to train
        self.seq_len_max = max([len(A) for A in A_paths])                     # max number of frames in the training sequences

        self.seq_idx = 0                                                      # index for current sequence
        self.frame_idx = self.opt.start_frame if not self.opt.isTrain else 0  # index for current frame in the sequence
        self.frames_count = []                                                # number of frames in each sequence
        for path in A_paths:
            self.frames_count.append(len(path) - self.opt.n_frames_G + 1 - 2)

        self.video_len = len(A_paths[0])

        self.folder_prob = [count / sum(self.frames_count) for count in self.frames_count]
        self.n_frames_total = self.opt.n_frames_total if self.opt.isTrain else 1 
        self.SFG, self.TPose, self.TParsing, self.TFG, self.TPose_uncloth, self.TParsing_uncloth, self.TFG_uncloth, self.TPose_cloth, self.TParsing_cloth, self.TFG_cloth, self.SPose, self.SParsing, self.SI, self.BG, self.BG_flag = None, None, None, None, None, None, None, None, None, None, None, None, None, None, None

    #更新cloth的idx
    def get_id(self,x):
        for i, value in enumerate(self.frames_cum):
            if value>x:
                return i
            
        
    #更新paser的idx
    def update_frame_idx_parser(self, A_paths, index):
        if self.opt.isTrain:
             
            seq_idx = self.get_id(index)
            self.frame_idx = index-self.frames_cum[seq_idx-1]
            
            return None, None, None, None, seq_idx
        else:
            self.change_seq = self.frame_idx >= self.frames_count[self.seq_idx]
#             print("self.seq_idx",self.seq_idx)
#             print("self.frame_idx",self.frame_idx)
#             print("self.frames_count[self.seq_idx]",self.frames_count[self.seq_idx])
            if self.change_seq:
                self.seq_idx += 1
                self.frame_idx = 0
                self.TParsing, self.SPose, self.SParsing, self.SFG = None, None, None, None
            return self.TParsing, self.SPose, self.SParsing, self.SFG, self.seq_idx

    
    
    
    def update_frame_idx_cloth(self, A_paths, index):
        if self.opt.isTrain:
           
            seq_idx = self.get_id(index)
            self.frame_idx = index-self.frames_cum[seq_idx-1]
            
            #seq_idx = int(index / (self.video_len - self.opt.n_frames_G + 1))   #商    0-152
            #self.frame_idx = index % (self.video_len - self.opt.n_frames_G + 1) #余数   0-297   
            return None, None, None, None,seq_idx
        
        
        
        else:
            self.change_seq = self.frame_idx >= self.frames_count[self.seq_idx]
            if self.change_seq:
                self.seq_idx += 1
                self.frame_idx = 0
                self.TParsing, self.TFG, self.SPose, self.SParsing, self.SFG, self.SFG_full = None, None, None, None, None
            return self.TParsing, self.TFG, self.SPose, self.SParsing, self.SFG, self.SFG_full, self.seq_idx

    def update_frame_idx_composer(self, A_paths, index):
        if self.opt.isTrain:
            seq_idx = self.get_id(index)
            self.frame_idx = index-self.frames_cum[seq_idx-1]         
            return None, None, None, None, None, None, None, None, None, seq_idx
        else:
            self.change_seq = self.frame_idx >= self.frames_count[self.seq_idx]
            if self.change_seq:
                self.seq_idx += 1
                self.frame_idx = 0
                self.TParsing, self.TFG, self.SPose, self.SParsing, self.SFG, self.SFG_full, self.BG, self.BG_flag, self.SI = None, None, None, None, None, None, None, None, None
            return self.TParsing, self.TFG, self.SPose, self.SParsing, self.SFG, self.SFG_full, self.BG, self.BG_flag, self.SI, self.seq_idx

    def update_frame_idx_full(self, A_paths, index):
        self.change_seq = self.frame_idx >= self.frames_count[self.seq_idx]
        if self.change_seq:
            self.seq_idx += 1
            self.frame_idx = 0
            self.TPose, self.TParsing, self.TFG, self.SFG, self.TParsing_uncloth, self.TFG_uncloth, self.TPose_cloth, self.TParsing_cloth, self.TFG_cloth, self.SPose, self.SParsing, self.SI, self.BG, self.BG_flag = None, None, None, None, None, None, None, None, None, None, None, None, None, None
        return self.TPose, self.TParsing, self.TFG, self.TPose_uncloth, self.TParsing_uncloth, self.TFG_uncloth, self.TPose_cloth, self.TParsing_cloth, self.TFG_cloth, self.SPose, self.SParsing, self.SI, self.BG, self.BG_flag, self.seq_idx
    
    
    def update_frame_idx_stage1(self, A_paths, index):
        self.change_seq = self.frame_idx >= self.frames_count[self.seq_idx]
        print("self.frame_idx",self.frame_idx)
        print("self.frames_count[self.seq_idx]",self.frames_count[self.seq_idx])
        if self.change_seq:
            self.seq_idx += 1
            self.frame_idx = 0
            self.TParsing, self.SPose, self.SParsing, self.SFG = None, None, None, None
        return self.TParsing, self.SPose, self.SParsing, self.SFG, self.seq_idx
    
    
    
    
    

    def update_frame_idx(self, A_paths, index):
        if self.opt.isTrain:
            if self.opt.dataset_mode == 'pose':                
                seq_idx = np.random.choice(len(A_paths), p=self.folder_prob) # randomly pick sequence to train
                self.frame_idx = index
            else:    
                seq_idx = index % self.n_of_seqs            
            return None, None, None, seq_idx
        else:
            self.change_seq = self.frame_idx >= self.frames_count[self.seq_idx]
            if self.change_seq:
                self.seq_idx += 1
                self.frame_idx = 0
                self.A, self.B, self.I = None, None, None
            return self.A, self.B, self.I, self.seq_idx

    def init_data_params(self, data, n_gpus, tG):
        opt = self.opt
        _, n_frames_total, self.height, self.width = data['SI'].size()  # n_frames_total = n_frames_load * n_loadings + tG - 1        
        n_frames_total = n_frames_total // opt.output_nc_3
        n_frames_load = opt.max_frames_per_gpu * n_gpus                # number of total frames loaded into GPU at a time for each batch
        n_frames_load = min(n_frames_load, n_frames_total - tG + 1)
        self.t_len = n_frames_load + tG - 1                             # number of loaded frames plus previous frames
        return n_frames_total-self.t_len+1, n_frames_load, self.t_len

    
    #加载之前生成的结果
    def init_data_params_parser(self, data, n_gpus, tG):
        opt = self.opt
        #驱动视频序列的数量，h，w
        _, n_frames_total, self.height, self.width = data['SParsing'].size()  # n_frames_total = n_frames_load * n_loadings + tG - 1  
        #print("n_frames_total",n_frames_total)
        n_frames_load = opt.max_frames_per_gpu * n_gpus                # number of total frames loaded into GPU at a time for each batch
        #print("n_frames_load0",n_frames_load)
        n_frames_load = min(n_frames_load, n_frames_total - tG + 1)
        #print("n_frames_load1",n_frames_load)
        
        #一共需要加载的帧数
        self.t_len = n_frames_load + tG - 1                             # number of loaded frames plus previous frames
        #print("self.t_len",self.t_len)
        return n_frames_total-self.t_len+1, n_frames_load, self.t_len

    def init_data_params_cloth(self, data, n_gpus, tG):
        opt = self.opt
        _, n_frames_total, self.height, self.width = data['SParsing'].size()  # n_frames_total = n_frames_load * n_loadings + tG - 1        
        n_frames_load = opt.max_frames_per_gpu * n_gpus                # number of total frames loaded into GPU at a time for each batch
        n_frames_load = min(n_frames_load, n_frames_total - tG + 1)
        self.t_len = n_frames_load + tG - 1                             # number of loaded frames plus previous frames
        return n_frames_total-self.t_len+1, n_frames_load, self.t_len

    def init_data(self, t_scales):
        fake_B_last = None  # the last generated frame from previous training batch (which becomes input to the next batch)
        real_B_all, fake_B_all, flow_ref_all, conf_ref_all = None, None, None, None # all real/generated frames so far
        if self.opt.sparse_D:
            real_B_all, fake_B_all, flow_ref_all, conf_ref_all = [None]*t_scales, [None]*t_scales, [None]*t_scales, [None]*t_scales
        
        frames_all = real_B_all, fake_B_all, flow_ref_all, conf_ref_all        
        return fake_B_last, frames_all

    def init_data_parser(self, t_scales):
        fake_B_last = None  # the last generated frame from previous training batch (which becomes input to the next batch)
        real_B_all, fake_B_all, flow_ref_all, conf_ref_all, real_sfg_all = None, None, None, None, None # all real/generated frames so far
        if self.opt.sparse_D:
            real_B_all, fake_B_all = [None]*t_scales, [None]*t_scales
        
        frames_all = real_B_all, fake_B_all, flow_ref_all, conf_ref_all, real_sfg_all
        return fake_B_last, frames_all

    #生成一些none作为前几帧的输入
    def init_data_cloth(self, t_scales):
        fake_B_last = None  # the last generated frame from previous training batch (which becomes input to the next batch)
        real_B_all, fake_B_all, flow_ref_all, conf_ref_all, real_slo_all = None, None, None, None, None # all real/generated frames so far
        if self.opt.sparse_D:
            real_B_all, fake_B_all = [None]*t_scales, [None]*t_scales
        
        frames_all = real_B_all, fake_B_all, flow_ref_all, conf_ref_all, real_slo_all
        fake_B_last = None  # the last generated frame from previous training batch (which becomes input to the next batch)
        
        
        fake_B_last = [None for i in range(5)]
        
        frames_all = [frames_all for i in range(5)]
        
        return fake_B_last, frames_all

    def prepare_data(self, data, i, input_nc, output_nc):
        t_len, height, width = self.t_len, self.height, self.width
        # 5D tensor: batchSize, # of frames, # of channels, height, width
        input_A = (data['A'][:, i*input_nc:(i+t_len)*input_nc, ...]).view(-1, t_len, input_nc, height, width)
        input_B = (data['B'][:, i*output_nc:(i+t_len)*output_nc, ...]).view(-1, t_len, output_nc, height, width)                
        inst_A = (data['inst'][:, i:i+t_len, ...]).view(-1, t_len, 1, height, width) if len(data['inst'].size()) > 2 else None
        return [input_A, input_B, inst_A]

    def prepare_data_composer(self, data, i):
        t_len, height, width = self.t_len, self.height, self.width
        # 5D tensor: batchSize, # of frames, # of channels, height, width
        input_TParsing = (data['TParsing'][:, i*1:(i+t_len)*1, ...]).view(-1, t_len, 1, height, width)
        input_TFG = (data['TFG'][:, i*3:(i+t_len)*3, ...]).view(-1, t_len, 3, height, width)
        input_SPose = (data['SPose'][:, i*3:(i+t_len)*3, ...]).view(-1, t_len, 3, height, width)
        input_SParsing = (data['SParsing'][:, i*1:(i+t_len)*1, ...]).view(-1, t_len, 1, height, width)
        input_SFG = (data['SFG'][:, i*3:(i+t_len)*3, ...]).view(-1, t_len, 3, height, width)
        input_BG = (data['BG'][:, i*3:(i+t_len)*3, ...]).view(-1, t_len, 3, height, width)
        input_SI = (data['SI'][:, i*3:(i+t_len)*3, ...]).view(-1, t_len, 3, height, width)

        return [input_TParsing, input_TFG, input_SPose, input_SParsing, input_SFG, input_BG, input_SI]

    #给网络输入准备输入
    def prepare_data_parser(self, data, i):
        t_len, height, width = self.t_len, self.height, self.width
        # 5D tensor: batchSize, # of frames, # of channels, height, width
        
        
        #为什么没有和之前生成的视频帧合并呢
        input_TParsing = (data['TParsing']).view(-1, 1, 1, height, width)
        input_SPose = (data['SPose'][:, i*3:(i+t_len)*3, ...]).view(-1, t_len, 3, height, width)
        input_SParsing = (data['SParsing'][:, i*1:(i+t_len)*1, ...]).view(-1, t_len, 1, height, width)
        input_SFG = (data['SFG'][:, i*3:(i+t_len)*3, ...]).view(-1, t_len, 3, height, width)
        
        #torch.Size([4, 1, 1, 256, 192]) torch.Size([4, 8, 3, 256, 192]) torch.Size([4, 8, 1, 256, 192]) torch.Size([4, 8, 3, 256, 192])
        #print("input_TParsing, input_SPose, input_SParsing, input_SFG",input_TParsing.shape, input_SPose.shape, input_SParsing.shape, input_SFG.shape)

        return [input_TParsing, input_SPose, input_SParsing, input_SFG]

    #准备第二阶段的数据
#     def prepare_data_cloth(self, data, i):
#         t_len, height, width = self.t_len, self.height, self.width
#         # 5D tensor: batchSize, # of frames, # of channels, height, width
#         input_TParsing = (data['TParsing']).view(-1, 1, 1, height, width)
#         input_TFG = (data['TFG']).view(-1, 1, 3, height, width)
#         input_SParsing = (data['SParsing'][:, i*1:(i+t_len)*1, ...]).view(-1, t_len, 1, height, width)
#         input_SFG = (data['SFG'][:, i*3:(i+t_len)*3, ...]).view(-1, t_len, 3, height, width)
#         input_SFG_full = (data['SFG_full'][:, i*3:(i+t_len)*3, ...]).view(-1, t_len, 3, height, width)
#         return [input_TParsing, input_TFG, input_SParsing, input_SFG, input_SFG_full]

    # {'T1': T1, 'T2': T2, 'SParsing': SParsing, 'SFG': SFG, 'A_path': simg_path, 'change_seq': change_seq}
    def prepare_data_cloth(self, data, i):
        t_len, height, width = self.t_len, self.height, self.width
        # 5D tensor: batchSize, # of frames, # of channels, height, width
        input_TParsing = (data['TParsing'][:, i*1:(i+t_len)*1, ...]).view(-1, t_len, 1, height, width)
        input_TFG = (data['TFG'][:, i*3:(i+t_len)*3, ...]).view(-1, t_len, 3, height, width)
        
        
        input_SParsing = (data['SParsing'][:, i*1:(i+t_len)*1, ...]).view(-1, t_len, 1, height, width)
        input_SFG = (data['SFG'][:, i*3:(i+t_len)*3, ...]).view(-1, t_len, 3, height, width)
        #input_SPose = (data['SPose'][:, i*3:(i+t_len)*3, ...]).view(-1, t_len, 3, height, width)
        
        return [input_TParsing, input_TFG, input_SParsing, input_SFG]
    
    def prepare_data_cloth_2(self, data, i):
        t_len, height, width = self.t_len, self.height, self.width
        # 5D tensor: batchSize, # of frames, # of channels, height, width
        input_TParsing = (data['TParsing'][:, i*1:(i+t_len)*1, ...]).view(-1, t_len, 1, height, width)
        input_TFG = (data['TFG'][:, i*3:(i+t_len)*3, ...]).view(-1, t_len, 3, height, width)
        
        
        input_SParsing = (data['SParsing'][:, i*1:(i+t_len)*1, ...]).view(-1, t_len, 1, height, width)
        input_SFG = (data['SFG'][:, i*3:(i+t_len)*3, ...]).view(-1, t_len, 3, height, width)
        input_SPose = (data['SPose'][:, i*3:(i+t_len)*3, ...]).view(-1, t_len, 3, height, width)
        
        return [input_TParsing, input_TFG, input_SParsing, input_SFG, input_SPose]
    
    

def make_power_2(n, base=32.0):    
    return int(round(n / base) * base)

def get_img_params(opt, size):
    w, h = size
    new_h, new_w = h, w        
    if 'resize' in opt.resize_or_crop:   # resize image to be loadSize x loadSize
        new_h = new_w = opt.loadSize #256x256           
    elif 'scaleWidth' in opt.resize_or_crop: # scale image width to be loadSize
        new_w = opt.loadSize
        new_h = opt.loadSize * h // w
    
    #sparsing 是这个
    elif 'scaleHeight' in opt.resize_or_crop: # scale image height to be loadSize
        new_h = opt.loadSize       #256
        new_w = opt.loadSize * w // h #455
        #print(new_h,new_w)
    elif 'randomScaleWidth' in opt.resize_or_crop:  # randomly scale image width to be somewhere between loadSize and fineSize
        new_w = random.randint(opt.fineSize, opt.loadSize + 1)
        new_h = new_w * h // w
    elif 'randomScaleHeight' in opt.resize_or_crop: # randomly scale image height to be somewhere between loadSize and fineSize
        new_h = random.randint(opt.fineSize, opt.loadSize + 1)
        new_w = new_h * w // h
    new_w = int(round(new_w / 4)) * 4  #456
    new_h = int(round(new_h / 4)) * 4  #256  
    
    
    crop_x = crop_y = 0
    crop_w = crop_h = 0
    if 'crop' in opt.resize_or_crop or 'scaledCrop' in opt.resize_or_crop:
        if 'crop' in opt.resize_or_crop:      # crop patches of size fineSize x fineSize
            crop_w = crop_h = opt.fineSize
        else:
            if 'Width' in opt.resize_or_crop: # crop patches of width fineSize
                crop_w = opt.fineSize
                crop_h = opt.fineSize * h // w
            else:                              # crop patches of height fineSize
                crop_h = opt.fineSize
                crop_w = opt.fineSize * w // h

        crop_w, crop_h = make_power_2(crop_w), make_power_2(crop_h)        
        x_span = (new_w - crop_w) // 2
        crop_x = np.maximum(0, np.minimum(x_span*2, int(np.random.randn() * x_span/3 + x_span)))        
        crop_y = random.randint(0, np.minimum(np.maximum(0, new_h - crop_h), new_h // 8))
        #crop_x = random.randint(0, np.maximum(0, new_w - crop_w))
        #crop_y = random.randint(0, np.maximum(0, new_h - crop_h))        
    else:
        new_w, new_h = make_power_2(new_w), make_power_2(new_h)
        #print("11",new_w,new_h) #448,256

        
    new_w = 256
    new_h = 256
    # for color augmentation,生成随机数，数据增强
    h_b = random.uniform(-30, 30)
    s_a = random.uniform(0.8, 1.2)
    s_b = random.uniform(-10, 10)
    v_a = random.uniform(0.8, 1.2)
    v_b = random.uniform(-10, 10)    

    flip = (random.random() > 0.5) and (opt.dataset_mode != 'pose')
    return {'new_size': (new_w, new_h), 'crop_size': (crop_w, crop_h), 'crop_pos': (crop_x, crop_y), 'flip': flip,
            'color_aug': (h_b, s_a, s_b, v_a, v_b)}

def get_transform_fixed(opt, params, method=Image.BICUBIC, normalize=True, toTensor=True, color_aug=False):
    transform_list = []

    ### color augmentation
    if opt.isTrain and color_aug:
        transform_list.append(transforms.Lambda(lambda img: __color_aug(img, params['color_aug'])))    

    ### random flip
    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    if toTensor:
        transform_list += [transforms.ToTensor()]
    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def get_transform(opt, params, method=Image.BICUBIC, normalize=True, toTensor=True, color_aug=False):
    transform_list = []
    ### resize input image 
    if 'resize' in opt.resize_or_crop:
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Scale(osize, method))   
    else:
        transform_list.append(transforms.Lambda(lambda img: __scale_image(img, params['new_size'], method)))
        
    ### crop patches from image
    if 'crop' in opt.resize_or_crop or 'scaledCrop' in opt.resize_or_crop:
        transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_size'], params['crop_pos'])))    

    ### color augmentation
    if opt.isTrain and color_aug:
        transform_list.append(transforms.Lambda(lambda img: __color_aug(img, params['color_aug']))) 
        #print("color_aug")

    ### random flip
    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))
        
        
    #toTensor和进行归一化
    if toTensor:
        transform_list += [transforms.ToTensor()]
    if normalize:
        #print("xcxcxc")
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    
    return transforms.Compose(transform_list)

def toTensor_normalize():    
    transform_list = [transforms.ToTensor()]    
    transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def __scale_image(img, size, method=Image.BICUBIC):
    w, h = size    
    return img.resize((w, h), method)

#自己定义的数据增广方法
def __crop(img, size, pos):
    ow, oh = img.size
    tw, th = size
    x1, y1 = pos        
    if (ow > tw or oh > th):        
        return img.crop((x1, y1, min(ow, x1 + tw), min(oh, y1 + th)))
    return img

def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img

def __color_aug(img, params):
    h, s, v = img.convert('HSV').split()    
    h = h.point(lambda i: (i + params[0]) % 256)
    s = s.point(lambda i: min(255, max(0, i * params[1] + params[2])))
    v = v.point(lambda i: min(255, max(0, i * params[3] + params[4])))
    img = Image.merge('HSV', (h, s, v)).convert('RGB')
    return img

#得到视频的参数，cur_seq_len：这个视频序列的长度
def get_video_params(opt, n_frames_total, cur_seq_len, index):
    
    tG = opt.n_frames_G
    if opt.isTrain:  
        #min(19,298)
        #print("xxx",n_frames_total, cur_seq_len - tG + 1)
        n_frames_total = min(n_frames_total, cur_seq_len - tG + 1)
        
        #print("n_frames_total",n_frames_total)
        #每个batch的gpu个数
        n_gpus = opt.n_gpus_gen if opt.batchSize == 1 else 1       # number of generator GPUs for each batch
        #print("n_gpus",n_gpus)#1
        
        
        n_frames_per_load = opt.max_frames_per_gpu * n_gpus        # number of frames to load into GPUs at one time (for each batch)
        n_frames_per_load = min(n_frames_total, n_frames_per_load)
        #print("n_frames_per_load",n_frames_per_load)
        
        n_loadings = n_frames_total // n_frames_per_load           # how many times are needed to load entire sequence into GPUs 
        #print("n_loadings",n_loadings)#12//6
        
        
        n_frames_total = n_frames_per_load * n_loadings + tG - 1   # rounded overall number of frames to read from the sequence
        #print("n_frames_total",n_frames_total)#6*2+3-1=14
        
        max_t_step = min(opt.max_t_step, (cur_seq_len-1) // (n_frames_total-1))
        #print("step",opt.max_t_step, (cur_seq_len-1) // (n_frames_total-1))#4,299//13,23
        
        #随机选择一个步数1-4
        t_step = np.random.randint(max_t_step) + 1                    # spacing between neighboring sampled frames
        #print("t_step",t_step)
        
        #第一帧的最大可能索引,300-13*step
        offset_max = max(1, cur_seq_len - (n_frames_total-1)*t_step)  # maximum possible index for the first frame 
        #print("offset_max",offset_max)
        #print("index",index)       #index指的是self.frame_idx,是index%298，范围是0-297
        
        
        #为什么parsing是随机选择的呢
        if opt.dataset_mode == 'pose' or opt.dataset_mode == 'cloth':
            start_idx = index % offset_max          #求余数
        else:
            start_idx = np.random.randint(offset_max)                 # offset for the first frame to load
            
        if opt.debug:
            print("loading %d frames in total, first frame starting at index %d, space between neighboring frames is %d"
                % (n_frames_total, start_idx, t_step))
            
            
    else:
        n_frames_total = tG
        start_idx = index
        t_step = 1   
    return n_frames_total, start_idx, t_step

def concat_frame(A, Ai, nF):
    if A is None:
        A = Ai
    else:
        c = Ai.size()[0]
        if A.size()[0] == nF * c:#此时已经拼接结束了
            A = A[c:]
        A = torch.cat([A, Ai])
    return A