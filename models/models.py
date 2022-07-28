### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os
import torch
import torch.nn as nn
import numpy as np
import fractions
def lcm(a,b): return abs(a * b)/fractions.gcd(a,b) if a and b else 0

def wrap_model_parser(opt, modelG, modelD, flowNet):
    print("opt.n_gpus_gen,opt.gpu_ids",opt.n_gpus_gen,opt.gpu_ids)
    
    #给模型分配GPU
    if opt.n_gpus_gen == len(opt.gpu_ids):
        #这个时候三个模型在同样的gpu上
        modelG = myModel(opt, modelG)
        modelD = myModel(opt, modelD)
        flowNet = myModel(opt, flowNet)
    else:             
        if opt.batchSize == 1:
            gpu_split_id = opt.n_gpus_gen + 1
            modelG = nn.DataParallel(modelG, device_ids=opt.gpu_ids[0:1])                
        else:
            gpu_split_id = opt.n_gpus_gen
            modelG = nn.DataParallel(modelG, device_ids=opt.gpu_ids[:gpu_split_id])
        modelD = nn.DataParallel(modelD, device_ids=[opt.gpu_ids[0]] + opt.gpu_ids[gpu_split_id:])
        flowNet = nn.DataParallel(flowNet, device_ids=[opt.gpu_ids[0]] + opt.gpu_ids[gpu_split_id:])
    return modelG, modelD, flowNet

def wrap_model_cloth(opt, modelG, modelD):
    #print("opt.n_gpus_gen,opt.gpu_ids",opt.n_gpus_gen,opt.gpu_ids)
    
    #给模型分配GPU
    if opt.n_gpus_gen == len(opt.gpu_ids):
        #这个时候三个模型在同样的gpu上
        modelG = myModel(opt, modelG)
        modelD = myModel(opt, modelD)
        #flowNet = myModel(opt, flowNet)
    else:             
        if opt.batchSize == 1:
            gpu_split_id = opt.n_gpus_gen + 1
            modelG = nn.DataParallel(modelG, device_ids=opt.gpu_ids[0:1])                
        else:
            gpu_split_id = opt.n_gpus_gen
            modelG = nn.DataParallel(modelG, device_ids=opt.gpu_ids[:gpu_split_id])
        modelD = nn.DataParallel(modelD, device_ids=[opt.gpu_ids[0]] + opt.gpu_ids[gpu_split_id:])
        #flowNet = nn.DataParallel(flowNet, device_ids=[opt.gpu_ids[0]] + opt.gpu_ids[gpu_split_id:])
    return modelG, modelD

def wrap_model(opt, modelG, modelD, flowNet):
    if opt.n_gpus_gen == len(opt.gpu_ids):
        modelG = myModel(opt, modelG)
        modelD = myModel(opt, modelD)
        flowNet = myModel(opt, flowNet)
    else:             
        if opt.batchSize == 1:
            gpu_split_id = opt.n_gpus_gen + 1
            modelG = nn.DataParallel(modelG, device_ids=opt.gpu_ids[0:1])                
        else:
            gpu_split_id = opt.n_gpus_gen
            modelG = nn.DataParallel(modelG, device_ids=opt.gpu_ids[:gpu_split_id])
        modelD = nn.DataParallel(modelD, device_ids=[opt.gpu_ids[0]] + opt.gpu_ids[gpu_split_id:])
        flowNet = nn.DataParallel(flowNet, device_ids=[opt.gpu_ids[0]] + opt.gpu_ids[gpu_split_id:])
    return modelG, modelD, flowNet

def wrap_model_3(opt, modelG, ClothPart, modelD, flowNet):
    if opt.n_gpus_gen == len(opt.gpu_ids):
        modelG = myModel(opt, modelG)
        ClothPart = myModel(opt, ClothPart)
        modelD = myModel(opt, modelD)
        flowNet = myModel(opt, flowNet)
    else:             
        if opt.batchSize == 1:
            gpu_split_id = opt.n_gpus_gen + 1
            modelG = nn.DataParallel(modelG, device_ids=opt.gpu_ids[0:1]) 
            ClothPart = nn.DataParallel(ClothPart, device_ids=opt.gpu_ids[0:1])
        else:
            gpu_split_id = opt.n_gpus_gen
            modelG = nn.DataParallel(modelG, device_ids=opt.gpu_ids[:gpu_split_id])
            ClothPart = nn.DataParallel(ClothPart, device_ids=opt.gpu_ids[:gpu_split_id])
            
        modelD = nn.DataParallel(modelD, device_ids=[opt.gpu_ids[0]] + opt.gpu_ids[gpu_split_id:])
        flowNet = nn.DataParallel(flowNet, device_ids=[opt.gpu_ids[0]] + opt.gpu_ids[gpu_split_id:])
    return modelG, ClothPart, modelD, flowNet

# def wrap_model_cloth(opt, ClothWarper, ClothWarperLoss, flowNet):
#     if opt.n_gpus_gen == len(opt.gpu_ids):
#         ClothWarper = myModel(opt, ClothWarper)
#         ClothWarperLoss = myModel(opt, ClothWarperLoss)
#         flowNet = myModel(opt, flowNet)
#     else:             
#         if opt.batchSize == 1:
#             gpu_split_id = opt.n_gpus_gen + 1
#             ClothWarper = nn.DataParallel(ClothWarper, device_ids=opt.gpu_ids[0:1])                
#         else:
#             gpu_split_id = opt.n_gpus_gen
#             ClothWarper = nn.DataParallel(ClothWarper, device_ids=opt.gpu_ids[:gpu_split_id])
#         ClothWarperLoss = nn.DataParallel(ClothWarperLoss, device_ids=[opt.gpu_ids[0]] + opt.gpu_ids[gpu_split_id:])
#         flowNet = nn.DataParallel(flowNet, device_ids=[opt.gpu_ids[0]] + opt.gpu_ids[gpu_split_id:])
#     return ClothWarper, ClothWarperLoss, flowNet


class myModel(nn.Module):
    def __init__(self, opt, model):        
        super(myModel, self).__init__()
        self.opt = opt
        self.module = model
        #print(opt.gpu_ids)
        self.model = nn.DataParallel(model, device_ids=opt.gpu_ids)
        self.bs_per_gpu = int(np.ceil(float(opt.batchSize) / len(opt.gpu_ids))) # batch size for each GPU
        #print("self.bs_per_gpu",self.bs_per_gpu)#每个gpu分配的样本
        self.pad_bs = self.bs_per_gpu * len(opt.gpu_ids) - opt.batchSize 
        #print("self.pad_bs",self.pad_bs)#剩余的样本
    
    
    
    def forward(self, *inputs, **kwargs):
        inputs = self.add_dummy_to_tensor(inputs, self.pad_bs)
        outputs = self.model(*inputs, **kwargs, dummy_bs=self.pad_bs)
        if self.pad_bs == self.bs_per_gpu: # gpu 0 does 0 batch but still returns 1 batch
            return self.remove_dummy_from_tensor(outputs, 1)
        return outputs        

    def add_dummy_to_tensor(self, tensors, add_size=0):        
        if add_size == 0 or tensors is None: return tensors
        if type(tensors) == list or type(tensors) == tuple:
            return [self.add_dummy_to_tensor(tensor, add_size) for tensor in tensors]    
                
        if isinstance(tensors, torch.Tensor):            
            dummy = torch.zeros_like(tensors)[:add_size]
            tensors = torch.cat([dummy, tensors])
        return tensors

    def remove_dummy_from_tensor(self, tensors, remove_size=0):
        if remove_size == 0 or tensors is None: return tensors
        if type(tensors) == list or type(tensors) == tuple:
            return [self.remove_dummy_from_tensor(tensor, remove_size) for tensor in tensors]    
        
        if isinstance(tensors, torch.Tensor):
            tensors = tensors[remove_size:]
        return tensors

    
#创建stage3的模型
def create_model(opt):    
    print(opt.model)            
    if opt.model == 'composer':
        #stage3的网络
        from .modelG_stage3 import Vid2VidModelG
        modelG = Vid2VidModelG() 
        #stage2的网络
        from .model_stage2 import ClothPart
        ClothPart = ClothPart()
        
        if opt.isTrain:
            from .modelD_stage3 import Vid2VidModelD
            modelD = Vid2VidModelD()    
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)

    if opt.isTrain:
        from .flownet import FlowNet
        flowNet = FlowNet()
    
    modelG.initialize(opt)
    ClothPart.initialize(opt)
    
    
    
    if opt.isTrain:
        modelD.initialize(opt)
        flowNet.initialize(opt)        
        if not opt.fp16:
            modelG, ClothPart, modelD, flownet = wrap_model_3(opt, modelG, ClothPart, modelD, flowNet)
        return [modelG, ClothPart, modelD, flowNet]
    else:
        return modelG, ClothPart

    
#创建整个模型
def create_model_full(opt):    
    print(opt.model)            
    if opt.model == 'full':
        from .model_stage2 import ClothPart
        ClothPart = ClothPart()
        from .modelG_stage3 import Vid2VidModelG as Composer
        Composer = Composer()    
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    
    import copy
    opt_2 = copy.deepcopy(opt)
    opt_2.name = 'clothwarp_256p'
    opt_3 = copy.deepcopy(opt)
    opt_3.name = 'composer_256p'

    ClothPart.initialize(opt_2)
    Composer.initialize(opt_3)

    return ClothPart, Composer



#创建stage1的模型
def create_model_parser(opt):    
    print(opt.model)  
    #如果model名字正确的话创建相关模型
    if opt.model == 'parser':
        #创建生成模型
        from .modelG_stage1 import Vid2VidModelG
        modelG = Vid2VidModelG()
        #如果是训练阶段，则需要加载判别网络
        if opt.isTrain:
            from .modelD_stage1 import Vid2VidModelD
            modelD = Vid2VidModelD()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    
    #如果是训练阶段还需要加载光流模型
    if opt.isTrain:
        from .flownet import FlowNet
        flowNet = FlowNet()

        
    #初始化这几个模型
    modelG.initialize(opt)
    if opt.isTrain:
        modelD.initialize(opt)
        flowNet.initialize(opt)
        if not opt.fp16:
            #print("opt.fp16",opt.fp16)
            modelG, modelD, flowNet = wrap_model_parser(opt, modelG, modelD, flowNet)
        return [modelG, modelD, flowNet]
    else:
        return modelG

#定义stage2的模型，这个需要修改成v2v部分
def create_model_cloth(opt):    
    print(opt.model)              
    if opt.model == 'cloth':
        from .model_stage2 import ClothPart
        ClothPart = ClothPart()
        #定义新的loss
        if opt.isTrain:
            from .modelD_stage2 import Vid2VidModelD
            modelD = Vid2VidModelD()
            
    elif opt.model == 'cloth_2':
        from .model_stage2_cloth import ClothPart
        ClothPart = ClothPart()
        #定义新的loss
        if opt.isTrain:
            from .modelD_stage2 import Vid2VidModelD
            modelD = Vid2VidModelD()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
        
        
        

    
    ClothPart.initialize(opt)
    if opt.isTrain:
        modelD.initialize(opt)
        #flowNet.initialize(opt)        
        if not opt.fp16:
            ClothPart, modelD = wrap_model_cloth(opt, ClothPart, modelD)
        #模型，损失函数，光流模型
        return [ClothPart,modelD]
    else:
        return ClothPart





def create_optimizer(opt, models):
    modelG,modelD,flowNet = models
    optimizer_D_T = []    
    if opt.fp16:              
        from apex import amp
        for s in range(opt.n_scales_temporal):
            optimizer_D_T.append(getattr(modelD, 'optimizer_D_T'+str(s)))
        modelG, optimizer_G = amp.initialize(modelG, modelG.optimizer_G, opt_level='O1')
        modelD, optimizers_D = amp.initialize(modelD, [modelD.optimizer_D] + optimizer_D_T, opt_level='O1')
        optimizer_D, optimizer_D_T = optimizers_D[0], optimizers_D[1:]        
        modelG, modelD, flowNet = wrap_model(opt, modelG, modelD, flowNet)
    else:        
        optimizer_G = modelG.module.optimizer_G
        optimizer_D = modelD.module.optimizer_D        
        for s in range(opt.n_scales_temporal):
            optimizer_D_T.append(getattr(modelD.module, 'optimizer_D_T'+str(s)))
    return modelG, modelD, flowNet, optimizer_G, optimizer_D, optimizer_D_T

def create_optimizer_3(opt, models):
    modelG,ClothPart,modelD, flowNet = models
    
    #固定stage2的模型参数
    for param in ClothPart.parameters():
            param.requires_grad_(False)
            
    optimizer_D_T = []    
    if opt.fp16:              
        from apex import amp
        for s in range(opt.n_scales_temporal):
            optimizer_D_T.append(getattr(modelD, 'optimizer_D_T'+str(s)))
        modelG, optimizer_G = amp.initialize(modelG, modelG.optimizer_G, opt_level='O1')
        modelD, optimizers_D = amp.initialize(modelD, [modelD.optimizer_D] + optimizer_D_T, opt_level='O1')
        optimizer_D, optimizer_D_T = optimizers_D[0], optimizers_D[1:]        
        modelG, modelD, flowNet = wrap_model(opt, modelG, modelD, flowNet)
    else:        
        optimizer_G = modelG.module.optimizer_G
        optimizer_D = modelD.module.optimizer_D        
        for s in range(opt.n_scales_temporal):
            optimizer_D_T.append(getattr(modelD.module, 'optimizer_D_T'+str(s)))
    return modelG, ClothPart,modelD, flowNet, optimizer_G, optimizer_D, optimizer_D_T


#创建stage1的优化
def create_optimizer_parser(opt, models):
    modelG, modelD, flowNet = models
    optimizer_D_T = [] 
    
    #是否使用混合精度，默认是0
    if opt.fp16:  
        #print("hahahaha")
        from apex import amp
        for s in range(opt.n_scales_temporal):
            optimizer_D_T.append(getattr(modelD, 'optimizer_D_T'+str(s)))
        modelG, optimizer_G = amp.initialize(modelG, modelG.optimizer_G, opt_level='O1')
        modelD, optimizers_D = amp.initialize(modelD, [modelD.optimizer_D] + optimizer_D_T, opt_level='O1')
        optimizer_D, optimizer_D_T = optimizers_D[0], optimizers_D[1:]        
        modelG, modelD, flowNet = wrap_model_parser(opt, modelG, modelD, flowNet)
    else:        
        optimizer_G = modelG.module.optimizer_G
        optimizer_D = modelD.module.optimizer_D        
        for s in range(opt.n_scales_temporal):
            optimizer_D_T.append(getattr(modelD.module, 'optimizer_D_T'+str(s)))
    return modelG, modelD, flowNet, optimizer_G, optimizer_D, optimizer_D_T

#创建模型优化器,stage2，需要有生成器和判别器
def create_optimizer_cloth(opt, models):
    ClothPart, modelD = models  
    if opt.fp16:              
        from apex import amp
        ClothPart, optimizer_G = amp.initialize(ClothPart, ClothPart.optimizer, opt_level='O1')  
        modelD, optimizers_D = amp.initialize(modelD, [modelD.optimizer_D] + optimizer_D_T, opt_level='O1')
        optimizer_D, optimizer_D_T = optimizers_D[0], optimizers_D[1:] 
        ClothPart, modelD = wrap_model_cloth(opt, ClothPart,modelD)
    else:        
        optimizer_G = ClothPart.module.optimizer
        optimizer_D = modelD.module.optimizer_D 
    return ClothPart, modelD, optimizer_G, optimizer_D


#初始化一些参数
def init_params_composer(opt, modelG, modelD, data_loader):
    iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
    start_epoch, epoch_iter = 1, 0
    ### if continue training, recover previous states，
    if opt.continue_train:        
        if os.path.exists(iter_path):
            start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)        
        print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))   
        if start_epoch > opt.niter:
            modelG.module.update_learning_rate(start_epoch-1, 'G')
            modelD.module.update_learning_rate(start_epoch-1, 'D')
        if (opt.n_scales_spatial > 1) and (opt.niter_fix_global != 0) and (start_epoch > opt.niter_fix_global):
            modelG.module.update_fixed_params()

    
    #每个batch使用的gpu个数
    n_gpus = opt.n_gpus_gen if opt.batchSize == 1 else 1   # number of gpus used for generator for each batch
    
    #送入生成器的帧数，送入时许判别器的帧数
    tG, tD = opt.n_frames_G, opt.n_frames_D
    tDB = tD * 3 
    
    #空间尺度个数，判别器的空间尺度1，3
    s_scales = opt.n_scales_spatial
    t_scales = opt.n_scales_temporal
    
    #19，16
    input_nc_1 = opt.input_nc_T_3
    input_nc_2 = opt.input_nc_S_3

    #最小公约数
    print_freq = lcm(opt.print_freq, opt.batchSize)
    
    #如果是加载之前的模型的话
    total_steps = (start_epoch-1) * len(data_loader) + epoch_iter
    total_steps = total_steps // print_freq * print_freq  

    return n_gpus, tG, tD, tDB, s_scales, t_scales, input_nc_1, input_nc_2, start_epoch, epoch_iter, print_freq, total_steps, iter_path


#初始化stage2的参数
# def init_params(opt, ClothWarper, data_loader):
#     iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
#     start_epoch, epoch_iter = 1, 0
#     ### if continue training, recover previous states
#     if opt.continue_train:        
#         if os.path.exists(iter_path):
#             start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)        
#         print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))   
#         if start_epoch > opt.niter:
#             ClothWarper.module.update_learning_rate_cloth(start_epoch-1)

#     n_gpus = opt.n_gpus_gen if opt.batchSize == 1 else 1   # number of gpus used for generator for each batch
#     tG = opt.n_frames_G
#     tD = opt.n_frames_D

#     t_scales = opt.n_scales_temporal

#     input_nc_1 = opt.input_nc_T_2
#     input_nc_2 = opt.input_nc_S_2
#     input_nc_3 = opt.input_nc_P_2

#     print_freq = lcm(opt.print_freq, opt.batchSize)
#     total_steps = (start_epoch-1) * len(data_loader) + epoch_iter
#     total_steps = total_steps // print_freq * print_freq  

#     return n_gpus, tG, input_nc_1, input_nc_2, input_nc_3, start_epoch, epoch_iter, print_freq, total_steps, iter_path, tD, t_scales

def init_params(opt, ClothWarper, data_loader):
    iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
    start_epoch, epoch_iter = 1, 0
    ### if continue training, recover previous states
    if opt.continue_train:        
        if os.path.exists(iter_path):
            start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)        
        print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))   
        if start_epoch > opt.niter:
            ClothWarper.module.update_learning_rate_cloth(start_epoch-1)

    n_gpus = opt.n_gpus_gen if opt.batchSize == 1 else 1   # number of gpus used for generator for each batch
    tG = opt.n_frames_G
    tD = opt.n_frames_D

    t_scales = opt.n_scales_temporal

    input_nc_1 = opt.input_nc_T_2
    input_nc_2 = opt.input_nc_S_2
    output_nc_2 = opt.output_nc_2

    print_freq = lcm(opt.print_freq, opt.batchSize)
    total_steps = (start_epoch-1) * len(data_loader) + epoch_iter
    total_steps = total_steps // print_freq * print_freq  

    return n_gpus, tG, input_nc_1, input_nc_2, output_nc_2, start_epoch, epoch_iter, print_freq, total_steps, iter_path, tD, t_scales


def save_models_cloth(opt, epoch, epoch_iter, total_steps, visualizer, iter_path, ClothWarper, modelD, end_of_epoch=False):
    if not end_of_epoch:
        if total_steps % opt.save_latest_freq == 0:
            visualizer.vis_print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            ClothWarper.module.save('latest')
            modelD.module.save('latest')
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')
    else:
        if epoch % opt.save_epoch_freq == 0:
            visualizer.vis_print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))        
            ClothWarper.module.save('latest')
            modelD.module.save('latest')
            ClothWarper.module.save(epoch)
            modelD.module.save(epoch)
            np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

def save_models(opt, epoch, epoch_iter, total_steps, visualizer, iter_path, modelG, modelD, end_of_epoch=False):
    if not end_of_epoch:
        if total_steps % opt.save_latest_freq == 0:
            visualizer.vis_print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            modelG.module.save('latest')
            modelD.module.save('latest')
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')
    else:
        if epoch % opt.save_epoch_freq == 0:
            visualizer.vis_print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))        
            modelG.module.save('latest')
            modelD.module.save('latest')
            modelG.module.save(epoch)
            modelD.module.save(epoch)
            np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

def update_models_cloth(opt, epoch, ClothWarper, data_loader):
    ### linearly decay learning rate after certain iterations
    if epoch > opt.niter:
        ClothWarper.module.update_learning_rate_cloth(epoch)

def update_models(opt, epoch, modelG, modelD, data_loader):
    ### linearly decay learning rate after certain iterations
    if epoch > opt.niter:
        modelG.module.update_learning_rate(epoch, 'G')
        modelD.module.update_learning_rate(epoch, 'D')