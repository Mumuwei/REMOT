### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import numpy as np
import math
import torch
import torch.nn.functional as F
import os
import sys
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from .base_model import BaseModel
from . import networks

class ClothPart(BaseModel):
    def name(self):
        return 'ClothPart'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain        
        if not opt.debug:
            torch.backends.cudnn.benchmark = True       
        
        # define net G                        
        self.n_scales = opt.n_scales_spatial        
        self.use_single_G = opt.use_single_G
        self.split_gpus = (self.opt.n_gpus_gen < len(self.opt.gpu_ids)) and (self.opt.batchSize == 1)

        #
        self.net = networks.define_part(opt.input_nc_S_2, opt.input_nc_T_2, opt.output_nc_2, opt.ngf, 
                                       opt.n_downsample_cloth, opt.norm, 0, self.gpu_ids, opt)

        print('----------Satge2G Networks initialized -------------') 
        print('-----------------------------------------------')

        # load networks
        net_name = 'ClothWarper'
        if not self.isTrain or opt.continue_train or opt.load_pretrain_2:  
            print("load pretrain model2G",opt.load_pretrain_2)
            self.load_network(self.net, net_name, opt.which_epoch_2, opt.load_pretrain_2)
                        
        # define training variables
        if self.isTrain:            
            self.n_gpus = self.opt.n_gpus_gen if self.opt.batchSize == 1 else 1 # number of gpus for running generator            
            self.n_frames_bp = 1 # number of frames to backpropagate the loss            
            self.n_frames_per_gpu = min(self.opt.max_frames_per_gpu, self.opt.n_frames_total // self.n_gpus) # number of frames in each GPU
            self.n_frames_load = self.n_gpus * self.n_frames_per_gpu   # number of frames in all GPUs            
            if self.opt.debug:
                print('training %d frames at once, using %d gpus, frames per gpu = %d' % (self.n_frames_load, 
                    self.n_gpus, self.n_frames_per_gpu))

        # set loss functions and optimizers
        if self.isTrain:            
            self.old_lr = opt.lr
          
            # initialize optimizer G
            params = list(self.net.parameters())

            if opt.TTUR:                
                beta1, beta2 = 0, 0.9
                lr = opt.lr / 2
            else:
                beta1, beta2 = opt.beta1, 0.999
                lr = opt.lr            
            self.optimizer = torch.optim.Adam(params, lr=lr, betas=(beta1, beta2))

    #重新对输入数据进行编码
    def encode_input(self, input_TParsing, input_TFG, input_SParsing, input_SFG=None):        
        size = input_SParsing.size()
        self.bs, tG, self.height, self.width = size[0], size[1], size[3], size[4]
      
        input_TParsing = input_TParsing.data.cuda()
        input_SParsing = input_SParsing.data.cuda()
        
        if input_SFG is not None:
            input_TFG = Variable(input_TFG.data.cuda())
            input_SFG = Variable(input_SFG.data.cuda())
        
        
        #还是生成one-hot label
        oneHot_size = (self.bs, tG, self.opt.label_nc_2, self.height, self.width)
        
        SParsing_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
        TParsing_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
        
#         print("label",self.opt.label_nc_2)
#         print("input_SParsing",input_SParsing.shape)#input_SParsing torch.Size([1, 3, 1, 256, 192])
#         print("input_TParsing",input_TParsing.shape)#input_TParsing torch.Size([1, 3, 1, 256, 192])
#         print("input_TParsing",input_TParsing.max())
#         print("input_TParsing",input_TParsing.min())
#         print("input_SParsing",input_SParsing.max())
#         print("input_SParsing",input_SParsing.min())
        SParsing_label = SParsing_label.scatter_(2, input_SParsing.long(), 1.0)
        TParsing_label = TParsing_label.scatter_(2, input_TParsing.long(), 1.0)
        
        
        input_TP = TParsing_label
        input_TP = Variable(input_TP)
        
        input_SP = SParsing_label
        input_SP = Variable(input_SP)

        
       
        return input_TP, input_TFG, input_SP, input_SFG

    
    
    def forward(self, input_TParsing, input_TFG, input_SParsing, input_SFG, part_total_prev_last, dummy_bs=0):
        tG = self.opt.n_frames_G           
        gpu_split_id = self.opt.n_gpus_gen + 1        

        real_input_TP, real_input_TFG, real_input_SP, real_SFG = self.encode_input(input_TParsing, input_TFG, input_SParsing, input_SFG)   
        
        #torch.Size([1, 8, 6, 256, 192]) torch.Size([1, 8, 3, 256, 192]) torch.Size([1, 8, 6, 256, 192]) torch.Size([1, 8, 3, 256, 192])
        #print("real_input_TP, real_input_TFG, real_input_SP, real_SFG",real_input_TP.shape, real_input_TFG.shape, real_input_SP.shape, real_SFG.shape)
        
        
        is_first_frame = part_total_prev_last[0] is None
        if is_first_frame: # at the beginning of a sequence; needs to generate the first frame
            part_total_prev_last = self.generate_first_part() 
            part_total_prev_last = [part_total_prev_last for i in range(real_input_TP.shape[2]-1)]
            
        
        #torch.Size([1, 2, 3, 256, 192])
        #print("part_total_prev",part_total_prev)
        net = torch.nn.parallel.replicate(self.net, self.opt.gpu_ids[:gpu_split_id]) if self.split_gpus else self.net
        start_gpu = self.gpu_ids[1] if self.split_gpus else real_input_TP.get_device()   
        
        #这个地方是逐帧进行训练
        #torch.Size([1, 2, 2, 256, 192])
        #print("flow_total_prev",flow_total_prev.shape)
        
        Fake_total_final,Fake_total_raw,Real_total,Real_SP,Flows,Weights = [],[],[],[],[],[]
        for i in range(real_input_TP.shape[2]-1):
            #分区域目标人体
            Tsemi = real_input_TP[:, :, i+1, :, :]
            Tsemi = torch.unsqueeze(Tsemi, 2)
            Tsemi = Tsemi.repeat(1, 1, real_input_TFG.size(2), 1, 1)
            real_input_TFGi = real_input_TFG.mul(Tsemi)
            
            #分区域源人体
            Ssemi = real_input_SP[:, :, i+1, :, :]
            real_input_SPi = torch.unsqueeze(Ssemi, 2)#判别器会用到
            Ssemi = real_input_SPi.repeat(1, 1, real_SFG.size(2), 1, 1)
            real_SFGi = real_SFG.mul(Ssemi)#计算loss时候会用到
            
            
           
            parti_total_prev = part_total_prev_last[i]
            #torch.Size([1, 8, 1, 256, 192]) torch.Size([1, 8, 3, 256, 192])
            #print("real_input_SPi,real_input_TFGi",real_input_SPi.shape,real_input_TFGi.shape)
        
            fake_part_final,fake_part_raw = self.generate_frame_train(net, 
                                            real_input_TFGi, real_input_SPi, parti_total_prev,start_gpu, is_first_frame)
            
            
        
            #下一轮要用到的前两帧生成图像
            part_total_prev_last[i] = fake_part_final[:, -tG+1:].detach()
            fake_part_total = fake_part_final[:,tG-1:]
            
            
            #全部append起来，这里先不用raw部分
            Fake_total_final.append(fake_part_total)
            Fake_total_raw.append(fake_part_raw)
            Real_total.append(real_SFGi[:,tG-1:])
            #真实输入的语义解析图
            Real_SP.append(real_input_SPi[:,tG-1:])
            
#             Flows.append(flows)
#             Weights.append(weights)
            
        #
        #print("Fake_total_final,Fake_total_raw,Real_total",Fake_total_final[0].shape,Fake_total_raw[0].shape,Real_total[0].shape)
        return Fake_total_final, Real_total, Real_SP, real_input_SP, real_SFG, real_input_TP, real_input_TFG, part_total_prev_last

    
    #逐帧进行生成
    def generate_frame_train(self, net, real_input_TFGi, real_input_SPi, parti_total_prev, start_gpu, is_first_frame):        
        tG = self.opt.n_frames_G        
        n_frames_load = self.n_frames_load
        use_raw_only = True
        dest_id = self.gpu_ids[0] if self.split_gpus else start_gpu        

        ### generate inputs
        fake_part_prev,fake_part_raw,flows,weights = None,None,None,None
        
        #torch.Size([1, 8, 1, 256, 192]) torch.Size([1, 8, 3, 256, 192])
        ### sequentially generate each frame
        for t in range(n_frames_load):
            gpu_id = (t // self.n_frames_per_gpu + start_gpu) if self.split_gpus else start_gpu # the GPU idx where we generate this frame
            net_id = gpu_id if self.split_gpus else 0                             # the GPU idx where the net is located

            #print(flow_total_prev.shape)
            real_input_Timg = real_input_TFGi[:, t+tG-1:t+tG,...].view(self.bs, -1, self.height, self.width).cuda(gpu_id)  #1,1,3,h,w 只有一个  
            real_input_Sp = real_input_SPi[:,t:t+tG,...].view(self.bs, -1, self.height, self.width).cuda(gpu_id) #1,3,1,h,w
            part_total_prevs = parti_total_prev[:, t:t+tG-1,...].view(self.bs, -1, self.height, self.width).cuda(gpu_id) #1,2,3,h,w
            
            
            #torch.Size([1, 3, 256, 192]) torch.Size([1, 3, 256, 192]) torch.Size([1, 6, 256, 192])
            #print("real_input_Timg,real_input_Sp,part_total_prevs",real_input_Timg.shape,real_input_Sp.shape,part_total_prevs.shape)
         
            #print("flow_total_prevs",flow_total_prevs.shape)
            if (t % self.n_frames_bp) == 0:
                part_total_prevs = part_total_prevs.detach()
                
         
            part_final, part_raw, flow, weight = net.forward(real_input_Timg, real_input_Sp, part_total_prevs,use_raw_only)
            
            #print("part_final",part_final.shape)#1,3,256,192
            
            parti_total_prev = self.concat([parti_total_prev, part_final.unsqueeze(1).cuda(dest_id)], dim=1)               
            fake_part_raw = self.concat([fake_part_raw, part_raw.unsqueeze(1).cuda(dest_id)], dim=1)
#             flows = self.concat([flows, flow.unsqueeze(1).cuda(dest_id)], dim=1)
#             weights = self.concat([weights, weight.unsqueeze(1).cuda(dest_id)], dim=1)
            
            
            
          
        #torch.Size([1, 8, 3, 256, 192]) torch.Size([1, 6, 3, 256, 192]) torch.Size([1, 6, 2, 256, 192]) torch.Size([1, 6, 1, 256, 192])   
        #print("parti_total_prev,fake_part_raw,flows,weights",parti_total_prev.shape,fake_part_raw.shape,flows.shape,weights.shape)
        return parti_total_prev, fake_part_raw

    
    #(TParsing, TFG, SParsing)
    def inference(self, input_TParsing, input_TFG, input_SParsing, input_SFG=None):
        with torch.no_grad():
            #torch.Size([1, 3, 1, 256, 192]) torch.Size([1, 3, 3, 256, 192]) torch.Size([1, 3, 1, 256, 192])
            #print("input_TParsing, input_TFG, input_SParsing",input_TParsing.shape, input_TFG.shape, input_SParsing.shape)
            real_input_TP, real_input_TFG, real_input_SP, real_SFG = self.encode_input(input_TParsing, input_TFG, input_SParsing, input_SFG) 
            #torch.Size([1, 3, 6, 256, 192]) torch.Size([1, 3, 3, 256, 192]) torch.Size([1, 3, 6, 256, 192])
            #print("real_input_TP, real_input_TFG, real_input_SP, real_SFG",real_input_TP.shape, real_input_TFG.shape, real_input_SP.shape)
                  
            self.is_first_frame = not hasattr(self, 'part_total_prevs') or self.part_total_prevs is None
                 
            if self.is_first_frame: # at the beginning of a sequence; needs to generate the first frame
                part_total_prev = self.generate_first_part() 
                self.part_total_prevs = [part_total_prev for i in range(real_input_TP.shape[2]-1)]
                
            Fake_total, Real_TP, Real_TFG, Real_SP, Real_SFG = self.generate_frame_infer(real_input_TP[:,-1:], real_input_TFG[:,-1:], real_input_SP,real_SFG)
            #torch.Size([1, 3, 256, 192]) torch.Size([1, 1, 256, 192]) torch.Size([1, 3, 256, 192]) torch.Size([1, 1, 256, 192]) torch.Size([1, 3, 256, 192])
            #print("Fake_total, Real_TP, Real_TFG, Real_SP, Real_SFG",Fake_total[0].shape, Real_TP[0].shape, Real_TFG[0].shape, Real_SP[0].shape, Real_SFG[0].shape)
            
            return Fake_total,Real_TP,Real_TFG,Real_SP, Real_SFG,real_input_TP, real_input_TFG, real_input_SP, real_SFG

        
        
    def generate_frame_infer(self, real_input_TP, real_input_TFG, real_input_SP,real_SFG):
        tG = self.opt.n_frames_G
        net = self.net
        #([1, 1, 6, 256, 192]) torch.Size([1, 1, 3, 256, 192]) torch.Size([1, 3, 6, 256, 192])
        #print("xxxreal_input_TP, real_input_TFG, real_input_SP, real_SFG",real_input_TP.shape, real_input_TFG.shape, real_input_SP.shape)
        Fake_total,Real_TP,Real_TFG,Real_SP, Real_SFG = [],[],[],[],[]
        for i in range(real_input_TP.shape[2]-1):
            #分区域目标人体
            Tsemi = real_input_TP[:, :, i+1, :, :]
            real_TPi = torch.unsqueeze(Tsemi, 2)
            Tsemi = real_TPi.repeat(1, 1, real_input_TFG.size(2), 1, 1)
            real_input_TFGi = real_input_TFG.mul(Tsemi)
            
            #分区域源人体
            Ssemi = real_input_SP[:, :, i+1, :, :]
            real_input_SPi = torch.unsqueeze(Ssemi, 2)#判别器会用到
            Ssemi = real_input_SPi.repeat(1, 1, real_SFG.size(2), 1, 1)
            real_SFGi = real_SFG.mul(Ssemi)#计算loss时候会用到
            
            
            real_input_Timg = real_input_TFGi.view(self.bs, -1, self.height, self.width)
            real_input_Sp = real_input_SPi.view(self.bs, -1, self.height, self.width)
            input_part_total_prevs = self.part_total_prevs[i].view(self.bs, -1, self.height, self.width)
     
      
            part_final, part_raw, flow, weight = net.forward(real_input_Timg, real_input_Sp, input_part_total_prevs)
            #print("part_final",part_final.shape)#torch.Size([1, 3, 256, 192])
            
            #print(self.part_total_prevs[i][1:].shape)
            self.part_total_prevs[i] = torch.cat([self.part_total_prevs[i][:,1:], part_final.unsqueeze(1)], dim=1)
            #print(self.part_total_prevs[i].shape)
            
            
            Fake_total.append(part_final)
            Real_TFG.append(real_input_TFGi.squeeze(1))
            Real_SFG.append(real_SFGi[:,-1])
            #真实输入的语义解析图
            Real_SP.append(real_input_SPi[:,-1])
            Real_TP.append(real_TPi.squeeze(1))
            
          
        return Fake_total,Real_TP,Real_TFG,Real_SP, Real_SFG
    
    
    

    def generate_first_frame(self, real_A, real_B, pool_map=None):
        tG = self.opt.n_frames_G
        if self.opt.no_first_img:          # model also generates first frame            
            fake_B_prev = Variable(self.Tensor(self.bs, tG-1, self.opt.output_nc, self.height, self.width).zero_())
        elif self.opt.isTrain or self.opt.use_real_img: # assume first frame is given
            fake_B_prev = real_B[:,:(tG-1),...]            
        elif self.opt.use_single_G:        # use another model (trained on single images) to generate first frame
            fake_B_prev = None
            if self.opt.use_instance:
                real_A = real_A[:,:,:self.opt.label_nc,:,:]
            for i in range(tG-1):                
                feat_map = self.get_face_features(real_B[:,i], pool_map[:,i]) if self.opt.dataset_mode == 'face' else None
                fake_B = self.netG_i.forward(real_A[:,i], feat_map).unsqueeze(1)                
                fake_B_prev = self.concat([fake_B_prev, fake_B], dim=1)
        else:
            raise ValueError('Please specify the method for generating the first frame')
            
        fake_B_prev = self.build_pyr(fake_B_prev)
        if not self.opt.isTrain:
            fake_B_prev = [B[0] for B in fake_B_prev]
        return fake_B_prev    

    def generate_first_flow(self, pool_map=None):
        tG = self.opt.n_frames_G
        if self.opt.no_first_img:          # model also generates first frame            
            flow_prev = Variable(self.Tensor(self.bs, tG-1, 2, self.height, self.width).zero_())
        else:
            raise ValueError('Please specify the method for generating the first frame')

        return flow_prev   
    
    def generate_first_part(self, pool_map=None):
        tG = self.opt.n_frames_G
        if self.opt.no_first_img:          # model also generates first frame            
            part_prev = Variable(self.Tensor(self.bs, tG-1, 3, self.height, self.width).zero_())
        else:
            raise ValueError('Please specify the method for generating the first frame')

        return part_prev   
    

    def return_dummy(self, input_A):
        h, w = input_A.size()[3:]
        t = self.n_frames_load
        tG = self.opt.n_frames_G  
        flow, weight = (self.Tensor(1, t, 2, h, w), self.Tensor(1, t, 1, h, w)) if not self.opt.no_flow else (None, None)
        return self.Tensor(1, t, 3, h, w), self.Tensor(1, t, 3, h, w), flow, weight, \
               self.Tensor(1, t, self.opt.input_nc, h, w), self.Tensor(1, t+1, 3, h, w), self.build_pyr(self.Tensor(1, tG-1, 3, h, w))

    def load_single_G(self): # load the model that generates the first frame
        opt = self.opt     
        s = self.n_scales
        if 'City' in self.opt.dataroot:
            single_path = 'checkpoints/label2city_single/'
            if opt.loadSize == 512:
                load_path = single_path + 'latest_net_G_512.pth'            
                netG = networks.define_G(35, 3, 0, 64, 'global', 3, 'instance', 0, self.gpu_ids, opt)                
            elif opt.loadSize == 1024:                            
                load_path = single_path + 'latest_net_G_1024.pth'
                netG = networks.define_G(35, 3, 0, 64, 'global', 4, 'instance', 0, self.gpu_ids, opt)                
            elif opt.loadSize == 2048:     
                load_path = single_path + 'latest_net_G_2048.pth'
                netG = networks.define_G(35, 3, 0, 32, 'local', 4, 'instance', 0, self.gpu_ids, opt)
            else:
                raise ValueError('Single image generator does not exist')
        elif 'face' in self.opt.dataroot:            
            single_path = 'checkpoints/edge2face_single/'
            load_path = single_path + 'latest_net_G.pth' 
            opt.feat_num = 16           
            netG = networks.define_G(15, 3, 0, 64, 'global_with_features', 3, 'instance', 0, self.gpu_ids, opt)
            encoder_path = single_path + 'latest_net_E.pth'
            self.netE = networks.define_G(3, 16, 0, 16, 'encoder', 4, 'instance', 0, self.gpu_ids)
            self.netE.load_state_dict(torch.load(encoder_path))
        else:
            raise ValueError('Single image generator does not exist')
        netG.load_state_dict(torch.load(load_path))        
        return netG

    def get_face_features(self, real_image, inst):                
        feat_map = self.netE.forward(real_image, inst)            
        #if self.opt.use_encoded_image:
        #    return feat_map
        
        load_name = 'checkpoints/edge2face_single/features.npy'
        features = np.load(load_name, encoding='latin1').item()                        
        inst_np = inst.cpu().numpy().astype(int)

        # find nearest neighbor in the training dataset
        num_images = features[6].shape[0]
        feat_map = feat_map.data.cpu().numpy()
        feat_ori = torch.FloatTensor(7, self.opt.feat_num, 1) # feature map for test img (for each facial part)
        feat_ref = torch.FloatTensor(7, self.opt.feat_num, num_images) # feature map for training imgs
        for label in np.unique(inst_np):
            idx = (inst == int(label)).nonzero() 
            for k in range(self.opt.feat_num): 
                feat_ori[label,k] = float(feat_map[idx[0,0], idx[0,1] + k, idx[0,2], idx[0,3]])
                for m in range(num_images):
                    feat_ref[label,k,m] = features[label][m,k]                
        cluster_idx = self.dists_min(feat_ori.expand_as(feat_ref).cuda(), feat_ref.cuda(), num=1)

        # construct new feature map from nearest neighbors
        feat_map = self.Tensor(inst.size()[0], self.opt.feat_num, inst.size()[2], inst.size()[3])
        for label in np.unique(inst_np):
            feat = features[label][:,:-1]                                                    
            idx = (inst == int(label)).nonzero()                
            for k in range(self.opt.feat_num):                    
                feat_map[idx[:,0], idx[:,1] + k, idx[:,2], idx[:,3]] = feat[min(cluster_idx, feat.shape[0]-1), k]
        
        return Variable(feat_map)

    def compute_mask(self, real_As, ts, te=None): # compute the mask for foreground objects
        _, _, _, h, w = real_As.size() 
        if te is None:
            te = ts + 1        
        mask_F = real_As[:, ts:te, self.opt.fg_labels[0]].clone()
        for i in range(1, len(self.opt.fg_labels)):
            mask_F = mask_F + real_As[:, ts:te, self.opt.fg_labels[i]]
        mask_F = torch.clamp(mask_F, 0, 1)
        return mask_F    

    def compute_fake_B_prev(self, real_B_prev, fake_B_last, fake_B):
        fake_B_prev = real_B_prev[:, 0:1] if fake_B_last[0] is None else fake_B_last[0][:, -1:]
        if fake_B.size()[1] > 1:
            fake_B_prev = torch.cat([fake_B_prev, fake_B[:, :-1].detach()], dim=1)
        return fake_B_prev

    def save(self, label):        
        self.save_network(self.net, 'ClothWarper', label, self.gpu_ids)                    