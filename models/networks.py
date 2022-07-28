### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import copy
#from .dconv.modules.modulated_deform_conv import ModulatedDeformConvPack
import time
import random

###############################################################################
# Functions
###############################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and hasattr(m, 'weight'):        
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def define_warper(input_nc_1, input_nc_2, input_nc_3, ngf, n_downsampling, norm, scale, gpu_ids=[], opt=[]):
    net = None    
    norm_layer = get_norm_layer(norm_type=norm)

    net = ClothWarper(opt, input_nc_1, input_nc_2, input_nc_3, ngf, n_downsampling, opt.n_blocks, opt.fg, opt.no_flow, norm_layer)

    #print_network(netG)
    if len(gpu_ids) > 0:
        net.cuda(gpu_ids[0])
    net.apply(weights_init)
    return net

#重新定义stage2
def define_part(input_nc_1, input_nc_2, output_nc, ngf, n_downsampling, norm, scale, gpu_ids=[], opt=[]):
    net = None    
    norm_layer = get_norm_layer(norm_type=norm)
#opt, input_nc_1, input_nc_2, output_nc, ngf, n_downsampling, n_blocks, use_fg_model=False, no_flow=False,norm_layer=nn.BatchNorm2d, padding_type='reflect'
    #3 9 3 64 3 9
#     print("input_nc_1, input_nc_2, output_nc, ngf, n_downsampling,opt.n_blocks",input_nc_1, input_nc_2, output_nc, ngf, n_downsampling,opt.n_blocks)
    
    net = Part_Cloth(opt, input_nc_1, input_nc_2, output_nc, ngf, n_downsampling, opt.n_blocks, opt.fg, opt.no_flow, norm_layer)
    #print(net)
    #print_network(netG)
    if len(gpu_ids) > 0:
        net.cuda(gpu_ids[0])
    net.apply(weights_init)
    return net


def define_composer(input_nc_1, input_nc_2, output_nc, prev_output_nc, ngf, n_downsampling, norm, scale, gpu_ids=[], opt=[]):
    net = None    
    norm_layer = get_norm_layer(norm_type=norm)

    net = Composer(opt, input_nc_1, input_nc_2, output_nc, prev_output_nc, ngf, n_downsampling, opt.n_blocks, opt.fg, opt.no_flow, norm_layer)

    #print_network(netG)
    if len(gpu_ids) > 0:
        net.cuda(gpu_ids[0])
    net.apply(weights_init)
    return net

def define_parser(input_nc_1, input_nc_2, output_nc, prev_output_nc, ngf, n_downsampling, norm, scale, gpu_ids=[], opt=[]):
    net = None    
    norm_layer = get_norm_layer(norm_type=norm)
    
    #print("opt.no_flow",opt.no_flow) 需要用到光流
    net = Parser(opt, input_nc_1, input_nc_2, output_nc, prev_output_nc, ngf, n_downsampling, opt.n_blocks, opt.fg, opt.no_flow, norm_layer)

    #print_network(netG)
    if len(gpu_ids) > 0:
        net.cuda(gpu_ids[0])
    net.apply(weights_init)
    return net

def define_D(input_nc, ndf, n_layers_D, norm='instance', num_D=1, getIntermFeat=False, gpu_ids=[]):        
    norm_layer = get_norm_layer(norm_type=norm)   
    netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, num_D, getIntermFeat)   
    #print_network(netD)
    if len(gpu_ids) > 0:    
        netD.cuda(gpu_ids[0])
    netD.apply(weights_init)
    return netD

def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def get_grid(batchsize, rows, cols, gpu_id=0, dtype=torch.float32):
    hor = torch.linspace(-1.0, 1.0, cols)
    hor.requires_grad = False
    hor = hor.view(1, 1, 1, cols)
    hor = hor.expand(batchsize, 1, rows, cols)
    ver = torch.linspace(-1.0, 1.0, rows)
    ver.requires_grad = False
    ver = ver.view(1, 1, rows, 1)
    ver = ver.expand(batchsize, 1, rows, cols)

    t_grid = torch.cat([hor, ver], 1)
    t_grid.requires_grad = False

    if dtype == torch.float16: t_grid = t_grid.half()
    return t_grid.cuda(gpu_id)

##############################################################################
# Classes
##############################################################################
class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def grid_sample(self, input1, input2):
        if self.opt.fp16: # not sure if it's necessary
            return torch.nn.functional.grid_sample(input1.float(), input2.float(), mode='bilinear', padding_mode='border').half()
        else:
            return torch.nn.functional.grid_sample(input1, input2, mode='bilinear', padding_mode='border')

    #计算光流warp
    def resample(self, image, flow, normalize=True):        
        b, c, h, w = image.size()        
        if not hasattr(self, 'grid') or self.grid.size() != flow.size():
            self.grid = get_grid(b, h, w, gpu_id=flow.get_device(), dtype=flow.dtype)            
        if normalize:
            flow = torch.cat([flow[:, 0:1, :, :] / ((w - 1.0) / 2.0), flow[:, 1:2, :, :] / ((h - 1.0) / 2.0)], dim=1)        
        final_grid = (self.grid + flow).permute(0, 2, 3, 1).cuda(image.get_device())
        output = self.grid_sample(image, final_grid)
        return output

##############################################################################
# Classes for coarse TPS warping
##############################################################################
class FeatureL2Norm(nn.Module):
    def __init__(self):
        super(FeatureL2Norm, self).__init__()

    def forward(self, feature):
        epsilon = 1e-6
        norm = torch.pow(torch.sum(torch.pow(feature,2),1)+epsilon,0.5).unsqueeze(1).expand_as(feature)
        return torch.div(feature,norm)
    
class FeatureCorrelation(nn.Module):
    def __init__(self):
        super(FeatureCorrelation, self).__init__()
    
    def forward(self, feature_A, feature_B, LO_A, LO_B):
        b,c,h,w = feature_A.size()
        n_class = LO_A.size()[1]
        # reshape features for matrix multiplication
        feature_A = feature_A.transpose(2,3).contiguous().view(b,c,h*w)
        feature_B = feature_B.view(b,c,h*w).transpose(1,2)
        # perform matrix mult.
        feature_mul = torch.bmm(feature_B,feature_A)
        #b,haxwa,hb,wb(b,wa,ha,hb,wb)
        LO_A = LO_A.transpose(2,3).contiguous().view(b,n_class,h*w)
        LO_B = LO_B.view(b,n_class,h*w).transpose(1,2)
        LO_mul = torch.bmm(LO_B, LO_A)

        feature_mul = feature_mul * LO_mul
        correlation_tensor = feature_mul.view(b,h,w,h*w).transpose(2,3).transpose(1,2)

        return correlation_tensor


       
##############################################################################
# Class for our C2F-FWN
##############################################################################
#新修改的v2v模块



class Part_Cloth(BaseNetwork):
    def __init__(self, opt, input_nc_1, input_nc_2, output_nc, ngf, n_downsampling, n_blocks, use_fg_model=False, no_flow=False,
                norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(Part_Cloth, self).__init__()                
        self.opt = opt
        self.n_downsampling = n_downsampling
        self.use_fg_model = use_fg_model
        self.no_flow = no_flow
        activation = nn.ReLU(True)
        """
        ngf：第一层卷积层的卷积核个数
        n_downsampling：下采样的个数，也是上采样的个数
        n_blocks：中间的resnetblock个数
        
        """
        
        ### flow and image generation
        ### downsample,对这三个输入分别进行下采样        
        model_down_1 = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc_1, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        for i in range(n_downsampling):
            mult = 2**i
            model_down_1 += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                               norm_layer(ngf * mult * 2), activation]  

        mult = 2**n_downsampling
        for i in range(n_blocks - n_blocks//2):
            model_down_1 += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        
        model_down_2 = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc_2, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        model_down_2 += copy.deepcopy(model_down_1[4:])
        
        #只输入两部分好了
#         model_down_lo = [nn.ReflectionPad2d(3), nn.Conv2d(prev_output_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
#         model_down_lo += copy.deepcopy(model_down_T[4:])
    
   
        ### resnet blocks 
        model_res_part = []
        for i in range(n_blocks//2):
            model_res_part += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        ### upsample
        model_up_part = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model_up_part += [nn.ConvTranspose2d(ngf*mult, ngf*mult//2, kernel_size=3, stride=2, padding=1, output_padding=1),norm_layer(ngf*mult//2), activation]  
            
        ### 最后再用卷积处理一下生成12个通道，最后再使用softmax处理生成粗糙的结果
        model_final_part = [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        #model_final_part = [nn.Conv2d(ngf, output_nc, kernel_size=1, padding=0), nn.Tanh()]
        
        #model_final_softmax = [nn.Softmax(dim=1)]
        #model_final_logsoftmax = [nn.LogSoftmax(dim=1)]

        
        #计算这三帧的flow
        model_res_flow = copy.deepcopy(model_res_part)
        model_up_flow = copy.deepcopy(model_up_part)
        model_final_flow = [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 2, kernel_size=7, padding=0)]                
        model_final_w = [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 1, kernel_size=7, padding=0), nn.Sigmoid()] 

        #将网络连接在一起
        self.model_down_1 = nn.Sequential(*model_down_1)        
        self.model_down_2 = nn.Sequential(*model_down_2)        
        #self.model_down_lo = nn.Sequential(*model_down_lo)        
        self.model_res_part = nn.Sequential(*model_res_part)
        self.model_up_part = nn.Sequential(*model_up_part)
        self.model_final_part = nn.Sequential(*model_final_part)
        #self.model_final_softmax = nn.Sequential(*model_final_softmax)
        #self.model_final_logsoftmax = nn.Sequential(*model_final_logsoftmax)
        self.model_res_flow = nn.Sequential(*model_res_flow)
        self.model_up_flow = nn.Sequential(*model_up_flow)
        self.model_final_flow = nn.Sequential(*model_final_flow)
        self.model_final_w = nn.Sequential(*model_final_w)
    
    #生成局部衣服
    def forward(self, t_part, s_parsing, t_prev,use_raw_only=True):
        #print("input_T, input_S, lo_prev, use_raw_only",input_T.shape, input_S.shape, lo_prev.shape, use_raw_only)
        #torch.Size([4, 12, 256, 192]) torch.Size([4, 9, 256, 192]) torch.Size([4, 24, 256, 192])
        gpu_id = t_part.get_device()
        #print(gpu_id)
        t_part = torch.cat((t_prev,t_part),axis=1)
        #print(t_part.shape)
        #print(s_parsing.shape)
        
        #对这两个数据分别进行下采样
        downsample_1 = self.model_down_1(s_parsing)   #3,
        downsample_2 = self.model_down_2(t_part)     #9
       
        #print(downsample_1.shape)
        #print(downsample_2.shape)
        
        #得到特征图并生成图像
        part_feat = self.model_up_part(self.model_res_part(downsample_1+downsample_2))
        part_raw = self.model_final_part(part_feat)
        
        #是否需要计算flow,先不计算flow了
        flow = weight = flow_feat = None
        if not self.no_flow:
            #print("flow")
            flow_feat = self.model_up_flow(self.model_res_flow(downsample_1))
            flow = self.model_final_flow(flow_feat) * 20
            weight = self.model_final_w(flow_feat)
        
        #是否需要warp
        if use_raw_only or self.no_flow:
            part_final = part_raw
        else:
            #print("warp")
            part_warp = self.resample(t_prev[:,-3:,...].cuda(gpu_id), flow).cuda(gpu_id)        
            weight_ = weight.expand_as(part_raw)
            part_final = part_raw * weight_ + part_warp * (1-weight_)

                
        #print(part_final.shape)      
        return part_final, part_raw, flow, weight
    
    
##############################################################################
# Class for the Composition GAN of stage 3
##############################################################################
"""
class Composer(BaseNetwork):
    def __init__(self, opt, input_nc_1, input_nc_2, output_nc, prev_output_nc, ngf, n_downsampling, n_blocks, use_fg_model=False, no_flow=False,
                norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(Composer, self).__init__()                
        self.opt = opt
        self.n_downsampling = n_downsampling
        self.use_fg_model = use_fg_model
        self.no_flow = no_flow
        activation = nn.ReLU(True)
        
        ### flow and image generation
        ### downsample        
        model_down_T = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc_1, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        for i in range(n_downsampling):
            mult = 2**i
            model_down_T += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                               norm_layer(ngf * mult * 2), activation]  

        mult = 2**n_downsampling
        for i in range(n_blocks - n_blocks//2):
            model_down_T += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        
        model_down_S = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc_2, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        model_down_S += copy.deepcopy(model_down_T[4:])
        model_down_fg = [nn.ReflectionPad2d(3), nn.Conv2d(prev_output_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        model_down_fg += copy.deepcopy(model_down_T[4:])
    
        ### resnet blocks
        model_res_fg = []
        for i in range(n_blocks//2):
            model_res_fg += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        model_res_sdfl = copy.deepcopy(model_res_fg)      

        
        ### upsample
        model_up_fg = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model_up_fg += [nn.ConvTranspose2d(ngf*mult, ngf*mult//2, kernel_size=3, stride=2, padding=1, output_padding=1),
                             norm_layer(ngf*mult//2), activation]                    
        model_final_fg = [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]

        model_up_sd = copy.deepcopy(model_up_fg)
        model_final_sd = [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 1, kernel_size=7, padding=0), nn.Sigmoid()]

        if not no_flow:
            model_up_flow = copy.deepcopy(model_up_fg)
            model_final_flow = [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 2, kernel_size=7, padding=0)]                
            model_final_w = [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 1, kernel_size=7, padding=0), nn.Sigmoid()] 

        self.model_down_T = nn.Sequential(*model_down_T)        
        self.model_down_S = nn.Sequential(*model_down_S)        
        self.model_down_fg = nn.Sequential(*model_down_fg)        
        self.model_res_fg = nn.Sequential(*model_res_fg)
        self.model_res_sdfl = nn.Sequential(*model_res_sdfl)
        self.model_up_fg = nn.Sequential(*model_up_fg)
        self.model_up_sd = nn.Sequential(*model_up_sd)
        self.model_final_fg = nn.Sequential(*model_final_fg)
        self.model_final_sd = nn.Sequential(*model_final_sd)

        if not no_flow:
            self.model_up_flow = nn.Sequential(*model_up_flow)                
            self.model_final_flow = nn.Sequential(*model_final_flow)                       
            self.model_final_w = nn.Sequential(*model_final_w)

    def forward(self, input_T, input_S, input_SFG, input_BG, img_prev, use_raw_only):
        
        gpu_id = input_T.get_device()
        #print("input_S",input_S.shape)#15x3
        input_smask = input_S[:, -self.opt.label_nc_3:-self.opt.label_nc_3+1]#-12:-11
        input_smask_cloth = input_S[:, -self.opt.label_nc_3+1:-self.opt.label_nc_3+self.opt.label_nc_2].sum(dim=1, keepdim=True)#-11:-9
#         print("self.opt.label_nc_3,self.opt.label_nc_2",self.opt.label_nc_3,self.opt.label_nc_2)#12,3
#         print("input_smask",input_smask.shape)#([1, 1, 256, 192])
#         print("input_smask_cloth",input_smask_cloth.shape)#([1, 1, 256, 192])
        
        #将头发和帽子去掉
        print("input_SFG111",input_SFG.max(),input_SFG.min())
        input_SFG[:,0][(input_smask_cloth[:,0]==0)] = -1
        input_SFG[:,1][(input_smask_cloth[:,0]==0)] = -1
        input_SFG[:,2][(input_smask_cloth[:,0]==0)] = -1
        
        #input_S_full：姿态图+解析图+衣服区域图像
        input_S_full = torch.cat([input_S, input_SFG], dim=1)
#         print("input_S_full",input_S_full.shape)#input_S_full torch.Size([1, 48, 256, 192])，3x（12+3）+3
#         print("input_T",input_T.shape)#torch.Size([1, 13, 256, 192]),10+3，为什么是10呢，12-2
#         print("img_prev",img_prev.shape)#torch.Size([1, 6, 256, 192]),3x2
        #三个输入分别进行下采样
        downsample_1 = self.model_down_T(input_T)
        downsample_2 = self.model_down_S(input_S_full)
        downsample_3 = self.model_down_fg(img_prev)
        
        fg_feat = self.model_up_fg(self.model_res_fg(downsample_1+downsample_2+downsample_3))
        fg_res = self.model_final_fg(fg_feat)
        #print("fg_res",fg_res.shape)#torch.Size([1, 3, 256, 192])
        
        res_sdfl = self.model_res_sdfl(downsample_2+downsample_3)
        sd_feat = self.model_up_sd(res_sdfl)
        #print("sd_feat",sd_feat.shape)#torch.Size([1, 64, 256, 192])
        print("fg_res",fg_res.max(),fg_res.min())#-1，1
        print("input_SFG",input_SFG.max(),input_SFG.min())#-1，1
        
        #将非衣服区域和衣服区域相加
        #input_smask_cloth 0 ，0：2 + -1  1，-1：1
        fg = (1-input_smask_cloth).expand_as(fg_res) * (fg_res + 1) + input_SFG
        
        #tet = (1-input_smask_cloth).expand_as(fg_res) * (fg_res + 1)
        #print("tet",tet.max(),tet.min()) #0,2
        #print("fg",fg.max(),fg.min())  #0,2
        #not sure if it is appropriate

        #将背景区域设置为-1
        fg[:,0][input_smask[:,0]==1] = -1
        fg[:,1][input_smask[:,0]==1] = -1
        fg[:,2][input_smask[:,0]==1] = -1

        sd = self.model_final_sd(sd_feat)
        BG = torch.zeros_like(input_BG).cuda(gpu_id)
        BG = sd.expand_as(input_BG) * (input_BG + 1)
        #print("BG",BG.max(),BG.min())#0,2
        #生成原始的图像
        img_raw = fg + BG 

        flow = weight = flow_feat = None
        if not self.no_flow:
            flow_feat = self.model_up_flow(res_sdfl)                                                              
            flow = self.model_final_flow(flow_feat) * 20
            weight = self.model_final_w(flow_feat) 
        if self.no_flow:
            img_final = img_raw
        else:
            img_warp = self.resample(img_prev[:,-3:,...].cuda(gpu_id), flow).cuda(gpu_id)        
            weight_ = weight.expand_as(img_raw)
            img_final = img_raw * weight_ + img_warp * (1-weight_)             

        return img_final, img_raw, fg_res, sd, fg, flow, weight
"""

###stage3

class Composer(BaseNetwork):
    def __init__(self, opt, input_nc_1, input_nc_2, output_nc, prev_output_nc, ngf, n_downsampling, n_blocks, use_fg_model=False, no_flow=False,
                norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(Composer, self).__init__()                
        self.opt = opt
        self.n_downsampling = n_downsampling
        self.use_fg_model = use_fg_model
        self.no_flow = no_flow
        activation = nn.ReLU(True)
        
        ### flow and image generation
        ### downsample  
        input_nc_1 = 27
        input_nc_2 = 9
        prev_output_nc = 3
      
        model_down_SP = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc_1, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        for i in range(n_downsampling):
            mult = 2**i
            model_down_SP += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                               norm_layer(ngf * mult * 2), activation]  

        mult = 2**n_downsampling
        for i in range(n_blocks - n_blocks//2):
            model_down_SP += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        
        model_down_Sfg = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc_2, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        model_down_Sfg += copy.deepcopy(model_down_SP[4:])
        model_down_Tfg = [nn.ReflectionPad2d(3), nn.Conv2d(prev_output_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        model_down_Tfg += copy.deepcopy(model_down_SP[4:])
            
        ### resnet blocks
        model_res_fg = []
        for i in range(n_blocks//2):
            model_res_fg += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        model_res_sdfl = copy.deepcopy(model_res_fg)      

        
        ### upsample
        model_up_fg = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model_up_fg += [nn.ConvTranspose2d(ngf*mult, ngf*mult//2, kernel_size=3, stride=2, padding=1, output_padding=1),
                             norm_layer(ngf*mult//2), activation]                    
        model_final_fg = [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]

        model_up_sd = copy.deepcopy(model_up_fg)
        model_final_sd = [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 1, kernel_size=7, padding=0), nn.Sigmoid()]

        if not no_flow:
            model_up_flow = copy.deepcopy(model_up_fg)
            model_final_flow = [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 2, kernel_size=7, padding=0)]                
            model_final_w = [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 1, kernel_size=7, padding=0), nn.Sigmoid()] 

        self.model_down_SP = nn.Sequential(*model_down_SP)        
        self.model_down_Sfg = nn.Sequential(*model_down_Sfg)        
        self.model_down_Tfg = nn.Sequential(*model_down_Tfg)        
        self.model_res_fg = nn.Sequential(*model_res_fg)
        self.model_res_sdfl = nn.Sequential(*model_res_sdfl)
        self.model_up_fg = nn.Sequential(*model_up_fg)
        self.model_up_sd = nn.Sequential(*model_up_sd)
        self.model_final_fg = nn.Sequential(*model_final_fg)
        self.model_final_sd = nn.Sequential(*model_final_sd)

        if not no_flow:
            self.model_up_flow = nn.Sequential(*model_up_flow)                
            self.model_final_flow = nn.Sequential(*model_final_flow)                       
            self.model_final_w = nn.Sequential(*model_final_w)
            
    def get_affinity(self, mk, qk):
        B, CK, h, w = mk.shape  #2,3,4,4
        mk = mk.flatten(start_dim=2)
        qk = qk.flatten(start_dim=2)

#         print("mk",mk.shape)#2,12,48
#         print("qk",qk.shape)#2,12,48

        m1 = torch.sqrt(mk.pow(2).sum(1).unsqueeze(2))#2,12,48--->2,48---->2,48,1
        q1 = torch.sqrt(qk.pow(2).sum(1).unsqueeze(1))#2,12,48--->2,48---->2,1,48
        mq = m1@q1   #2,48,1*2,1,48 = 2,48,48
      

        #计算余弦距离
        b = (mk.transpose(1, 2) @ qk)#2,48,12*2,12,48--->2,48,48
        
        affinity = b/ mq   # B, THW, HW
        #print("affinity",affinity.shape)

        # softmax operation; aligned the evaluation style
        x_exp = torch.exp(affinity)#2,48,48
        x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)#2,48,48-->2,1,48
   
        affinity = x_exp / x_exp_sum 

        #print(affinity.sum(1))

        return affinity

    #mv是补充信息
    def fusion(self,mv,qv):
        B, CV, H, W = qv.shape
        affinity = self.get_affinity(mv,qv)
        mo = mv.view(B, CV, H*W)#2,12,16
        mem = torch.bmm(mo, affinity)#2,12,16*2,16,16 = 2,12,16
        mem = mem.view(B, CV, H, W)
        mem_out = qv+mem

        return mem_out
        
       
    def forward(self, input_SP, input_SFG_2, input_TFG, input_BG, img_prev, use_raw_only=False, use_fusion=True): 
        gpu_id = input_SP.get_device()
        
        input_mask = input_SP[:,3:4]#
        #print("input_smask",input_smask.shape)
        input_SFG_full = torch.cat([input_SFG_2, img_prev], dim=1)
#         print("img_prev",img_prev.shape)#1，6，256，192
#         print("input_SP",input_SP.shape)#([1, 27, 256, 192])
#         print("input_SFG_full",input_SFG_full.shape)#([1, 9, 256, 192])
#         print("input_TFG",input_TFG.shape)#([1, 3, 256, 192])
        
        downsample_1 = self.model_down_SP(input_SP)
        downsample_2 = self.model_down_Sfg(input_SFG_full)
        downsample_3 = self.model_down_Tfg(input_TFG)
        #torch.Size([1, 512, 32, 24])
        #print("downsample_1,downsample_2,downsample_3",downsample_1.shape,downsample_2.shape,downsample_3.shape)
        
        #downsample_3.sum().backward()

        if use_fusion:
            #print("xxx")
            downsample_23 = self.fusion(downsample_3,downsample_2)
        else:
            downsample_23 = downsample_2 + downsample_3
            #print("xxx")
        
        #fg_feat = self.model_up_fg(self.model_res_fg(downsample_1+downsample_2+downsample_3))
        #fg_feat.sum().backward()
        
        fg_feat = self.model_up_fg(self.model_res_fg(downsample_1+downsample_23))
        fg = self.model_final_fg(fg_feat)
        #print("fg_res",fg_res.shape)#torch.Size([1, 3, 256, 192])
        #fg.sum().backward()
        
        
        res_sdfl = self.model_res_sdfl(downsample_1+downsample_2)
        #print("res_sdf1",res_sdfl.shape)
        
        sd_feat = self.model_up_sd(res_sdfl)
        #print("sd_feat",sd_feat.shape)#torch.Size([1, 64, 256, 192])
        #sd_feat.sum().backward()
         
        #将背景区域设置为-1

#         fg[:,0][input_mask[:,0]==1] = -1
#         fg[:,1][input_mask[:,0]==1] = -1
#         fg[:,2][input_mask[:,0]==1] = -1
        
#         fg.sum().backward()
#         print("xxx")
        
        sd = self.model_final_sd(sd_feat)
        BG = torch.zeros_like(input_BG).cuda(gpu_id)
        BG = sd.expand_as(input_BG) * (input_BG + 1)#
        #print("BG",BG.max(),BG.min())#0,2
        #生成原始的图像
        img_raw = fg + BG 
        #img_raw.sum().backward()

        flow = weight = flow_feat = None
        if not self.no_flow:
            flow_feat = self.model_up_flow(res_sdfl)        
            flow = self.model_final_flow(flow_feat) * 20
            weight = self.model_final_w(flow_feat) 
        if self.no_flow:
            img_final = img_raw
        else:
            img_warp = self.resample(img_prev[:,-3:,...].cuda(gpu_id), flow).cuda(gpu_id)        
            weight_ = weight.expand_as(img_raw)
            img_final = img_raw * weight_ + img_warp * (1-weight_)
            
        #img_final.sum().backward()
        #print("img_final",img_final)
        #预测图像，预测的粗糙图像，背景mask，前景图像，光流，权重
        return img_final, img_raw, sd, fg, flow, weight



##############################################################################
# Class for the Layout GAN of stage 1
##############################################################################
class Parser(BaseNetwork):
    def __init__(self, opt, input_nc_1, input_nc_2, output_nc, prev_output_nc, ngf, n_downsampling, n_blocks, use_fg_model=False, no_flow=False,
                norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(Parser, self).__init__()                
        self.opt = opt
        self.n_downsampling = n_downsampling
        self.use_fg_model = use_fg_model
        self.no_flow = no_flow
        activation = nn.ReLU(True)
        
        ### flow and image generation
        ### downsample,对这三个输入分别进行下采样        
        model_down_T = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc_1, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        for i in range(n_downsampling):
            mult = 2**i
            model_down_T += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                               norm_layer(ngf * mult * 2), activation]  

        mult = 2**n_downsampling
        for i in range(n_blocks - n_blocks//2):
            model_down_T += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        
        model_down_S = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc_2, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        model_down_S += copy.deepcopy(model_down_T[4:])
        model_down_lo = [nn.ReflectionPad2d(3), nn.Conv2d(prev_output_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        model_down_lo += copy.deepcopy(model_down_T[4:])
    
    
    
        ### resnet blocks 
        model_res_lo = []
        for i in range(n_blocks//2):
            model_res_lo += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        ### upsample
        model_up_lo = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model_up_lo += [nn.ConvTranspose2d(ngf*mult, ngf*mult//2, kernel_size=3, stride=2, padding=1, output_padding=1),
                             norm_layer(ngf*mult//2), activation]  
            
        ### 最后再用卷积处理一下生成12个通道，最后再使用softmax处理生成粗糙的结果
        model_final_lo = [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model_final_softmax = [nn.Softmax(dim=1)]
        #model_final_logsoftmax = [nn.LogSoftmax(dim=1)]

        
        #计算这三帧的flow
        model_res_flow = copy.deepcopy(model_res_lo)
        model_up_flow = copy.deepcopy(model_up_lo)
        model_final_flow = [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 2, kernel_size=7, padding=0)]                
        model_final_w = [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 1, kernel_size=7, padding=0), nn.Sigmoid()] 

        self.model_down_T = nn.Sequential(*model_down_T)        
        self.model_down_S = nn.Sequential(*model_down_S)        
        self.model_down_lo = nn.Sequential(*model_down_lo)        
        self.model_res_lo = nn.Sequential(*model_res_lo)
        self.model_up_lo = nn.Sequential(*model_up_lo)
        self.model_final_lo = nn.Sequential(*model_final_lo)
        self.model_final_softmax = nn.Sequential(*model_final_softmax)
        #self.model_final_logsoftmax = nn.Sequential(*model_final_logsoftmax)
        self.model_res_flow = nn.Sequential(*model_res_flow)
        self.model_up_flow = nn.Sequential(*model_up_flow)
        self.model_final_flow = nn.Sequential(*model_final_flow)
        self.model_final_w = nn.Sequential(*model_final_w)

    def forward(self, input_T, input_S, lo_prev, use_raw_only):
        #print("input_T, input_S, lo_prev, use_raw_only",input_T.shape, input_S.shape, lo_prev.shape, use_raw_only)
        #torch.Size([4, 12, 256, 192]) torch.Size([4, 9, 256, 192]) torch.Size([4, 24, 256, 192])
        gpu_id = input_T.get_device()
        downsample_1 = self.model_down_T(input_T)
        downsample_2 = self.model_down_S(input_S)
        downsample_3 = self.model_down_lo(lo_prev)
        #([4, 512, 32, 24]) torch.Size([4, 512, 32, 24]) torch.Size([4, 512, 32, 24])
        #print("downsample_1,downsample_2,downsample_3",downsample_1.shape,downsample_2.shape,downsample_3.shape)
        lo_feat = self.model_up_lo(self.model_res_lo(downsample_1+downsample_2+downsample_3))
        #lo_feat [4, 64, 256, 192]
        #print("lo_feat",lo_feat.shape)
        lo_raw = self.model_final_lo(lo_feat)
        
        #torch.Size([4, 12, 256, 192])
        #print("lo_raw",lo_raw.shape)
        lo_softmax_raw = self.model_final_softmax(lo_raw)
        lo_logsoftmax_raw = torch.log(torch.abs(lo_softmax_raw)+1e-6)

        #计算连续帧的flow
        flow_feat = self.model_up_flow(self.model_res_flow(downsample_2+downsample_3))
        flow = self.model_final_flow(flow_feat) * 20
        weight = self.model_final_w(flow_feat)
        
        #torch.Size([4, 2, 256, 192]) torch.Size([4, 1, 256, 192])
        #print("flow,weight",flow.shape,weight.shape)
        #print("flow",flow[0,0])

        #计算warp,只是前面一帧进行warp
        lo_softmax_warp = self.resample(lo_prev[:,-self.opt.label_nc_1:,...].cuda(gpu_id), flow).cuda(gpu_id)        
        weight_ = weight.expand_as(lo_softmax_raw)
        #print("lo_softmax_warp,weight_",lo_softmax_warp.shape, weight_.shape)
        
        
        #计算warp后的最终图片,这个地方不是很理解，回头要重点看看
        lo_softmax_final = lo_softmax_raw * weight_ + lo_softmax_warp * (1-weight_) 
        lo_logsoftmax_final = torch.log(torch.abs(lo_softmax_final)+1e-6)

        #print("lo_logsoftmax_final",lo_logsoftmax_final.shape)#torch.Size([4, 12, 256, 192])
        return lo_softmax_final, lo_softmax_raw, lo_logsoftmax_final, lo_logsoftmax_raw, flow, weight
        #fake_slo, fake_slo_raw, fake_slo_ls, fake_slo_raw_ls, flow, weight
    
    


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class ResnetBlock_deform(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock_deform, self).__init__()
        self.conv_block_1 = self.build_conv_block_1(dim, padding_type, norm_layer, activation, use_dropout)
        self.conv_block_2 = self.build_conv_block_2(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block_1(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [ModulatedDeformConvPack(dim, dim, kernel_size=(3,3), stride=1, padding=p, deformable_groups=1, bias=True),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        return nn.Sequential(*conv_block)

    def build_conv_block_2(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [ModulatedDeformConvPack(dim, dim, kernel_size=(3,3), stride=1, padding=p, deformable_groups=1, bias=True),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, input_list):
        x, input_LO = input_list
        res = self.conv_block_1([x, input_LO, input_LO])
        res = self.conv_block_2([res, input_LO, input_LO])
        out = x + res
        return out

class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, 
                 num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
        ndf_max = 64
     
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, min(ndf_max, ndf*(2**(num_D-1-i))), n_layers, norm_layer,
                                       getIntermFeat)
            if getIntermFeat:                                
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))                
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)        

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]            
            for i in range(len(model)):
                result.append(model[i](result[-1]))            
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):        
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))                                
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)                    
        return result

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)            

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)        

##############################################################################
# Losses
##############################################################################
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor        
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None        
        gpu_id = input.get_device()
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).cuda(gpu_id).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).cuda(gpu_id).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)                
                loss += self.loss(pred, target_tensor)
            return loss
        else:            
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)

class VGGLoss(nn.Module):
    def __init__(self, gpu_id=0):
        super(VGGLoss, self).__init__()        
        self.vgg = Vgg19().cuda(gpu_id)
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.downsample = nn.AvgPool2d(2, stride=2, count_include_pad=False)

    def forward(self, x, y):
        while x.size()[3] > 1024:
            x, y = self.downsample(x), self.downsample(y)
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())        
        return loss

class CrossEntropyLoss(nn.Module):
    def __init__(self, label_nc):
        super(CrossEntropyLoss, self).__init__()
        self.softmax = nn.LogSoftmax(dim=1)
        self.criterion = nn.NLLLoss2d()

    def forward(self, output, label):
        label = label.long().max(1)[1]        
        output = self.softmax(output)
        return self.criterion(output, label)

class PixelwiseSoftmaxLoss(nn.Module):
    def __init__(self):
        super(PixelwiseSoftmaxLoss, self).__init__()
        self.criterion = nn.NLLLoss2d()

    def forward(self, output, label):
        label = label.long().max(1)[1]
        return self.criterion(output, label)

class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, input, target, mask):        
        mask = mask.expand(-1, input.size()[1], -1, -1)
        loss = self.criterion(input * mask, target * mask)
        return loss

class MultiscaleL1Loss(nn.Module):
    def __init__(self, scale=5):
        super(MultiscaleL1Loss, self).__init__()
        self.criterion = nn.L1Loss()
        self.downsample = nn.AvgPool2d(2, stride=2, count_include_pad=False)
        #self.weights = [0.5, 1, 2, 8, 32]
        self.weights = [1, 0.5, 0.25, 0.125, 0.125]
        self.weights = self.weights[:scale]

    def forward(self, input, target, mask=None):
        loss = 0
        if mask is not None:
            mask = mask.expand(-1, input.size()[1], -1, -1)
        for i in range(len(self.weights)):
            if mask is not None:                
                loss += self.weights[i] * self.criterion(input * mask, target * mask)
            else:
                loss += self.weights[i] * self.criterion(input, target)
            if i != len(self.weights)-1:
                input = self.downsample(input)
                target = self.downsample(target)
                if mask is not None:
                    mask = self.downsample(mask)
        return loss

from torchvision import models
class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss,self).__init__()

    def forward(self,flow):
        batch_size = flow.size()[0]
        h_x = flow.size()[2]
        w_x = flow.size()[3]
        count_h = self._tensor_size(flow[:,:,1:,:])
        count_w = self._tensor_size(flow[:,:,:,1:])
        h_tv = torch.abs((flow[:,:,1:,:]-flow[:,:,:h_x-1,:])).sum()
        w_tv = torch.abs((flow[:,:,:,1:]-flow[:,:,:,:w_x-1])).sum()
        return (h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]