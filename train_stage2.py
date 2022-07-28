### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import time
import os
import torch
from subprocess import call

from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model, create_optimizer, init_params, save_models_cloth, update_models_cloth, create_model_cloth, create_optimizer_cloth
import util.util as util
from util.visualizer import Visualizer
import warnings
warnings.filterwarnings("ignore")


def train():
    opt = TrainOptions().parse()
    if opt.debug:
        opt.display_freq = 1
        opt.print_freq = 1    
        opt.nThreads = 1

    ### initialize dataset，初始化数据集
    print("data prepare")
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)    
    print('#training frames = %d' % dataset_size)

    ### initialize models，初始化模型,从这里也要大改特改，加油
    models = create_model_cloth(opt)
    ClothPart, modelD, optimizer_G, optimizer_D = create_optimizer_cloth(opt, models)

    ### set parameters，设置参数   
    n_gpus, tG, input_nc_1, input_nc_2, output_nc_2, start_epoch, epoch_iter, print_freq, total_steps, iter_path, tD, t_scales = init_params(opt, ClothPart, data_loader)
    visualizer = Visualizer(opt)    

    #print("t_scales",t_scales)
    ### real training starts here，开始训练的地方 
    for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()    
        for idx, data in enumerate(dataset, start=epoch_iter):        
            if total_steps % print_freq == 0:
                iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            # whether to collect output images
            save_fake = total_steps % opt.display_freq == 0
            #save_fake = total_steps % 2 == 0
            
            n_frames_total, n_frames_load, t_len = data_loader.dataset.init_data_params_cloth(data, n_gpus, tG)
            #7 6 8
            #print(n_frames_total, n_frames_load, t_len)
            part_total_prev_last, frames_all = data_loader.dataset.init_data_cloth(t_scales)
            #print(part_total_prev_last)

            
            
            for i in range(0, n_frames_total, n_frames_load):
                is_first_frame = part_total_prev_last[0] is None
                #print("is_first_frame",is_first_frame)
                input_TParsing, input_TFG, input_SParsing, input_SFG = data_loader.dataset.prepare_data_cloth(data, i)
                
                #torch.Size([1, 8, 1, 256, 192]) torch.Size([1, 8, 3, 256, 192]) torch.Size([1, 8, 1, 256, 192]) torch.Size([1, 8, 3, 256, 192])
                #print("input_TParsing, input_TFG, input_SParsing, input_SFG",input_TParsing.shape, input_TFG.shape, input_SParsing.shape, input_SFG.shape)
               
                ###################################### Forward Pass ##########################
                ####### C2F-FWN 模型输入##############   
                #Fake_total_final, Real_total, Real_SP, real_input_SP, real_SFG, real_input_TP, real_input_TFG, part_total_prev_last
                Fake_total_final,Real_total,Real_SP,real_input_SP, real_SFG, real_input_TP, real_input_TFG, part_total_prev = ClothPart(input_TParsing, input_TFG, input_SParsing, input_SFG,part_total_prev_last)
                
                
                #fake_part1_total torch.Size([1, 6, 3, 256, 192])
                #print("fake_part1_total",fake_part1_total.shape)
                
                #torch.Size([1, 6, 3, 256, 192]) torch.Size([1, 6, 3, 256, 192]) torch.Size([1, 2, 3, 256, 192])
#                 print("Fake_total_final,Real_total,part_total_prev",Fake_total_final[0].shape,Real_total[0].shape,part_total_prev[0].shape)

                #torch.Size([1, 6, 1, 256, 192]) ,5
#                 print("Real_SP",Real_SP[0].shape)
#                 print("Real_SP2",len(Real_SP))
                
                #fake_slo_prev = modelG.module.compute_fake_B_prev(Real_total, part_total_prev, Fake_total_final)
                
                 
                part_total_prev_last = part_total_prev
                
                ####### compute losses,计算loss,同样很难，这个地方过于简单需要增加点loss
                losses = modelD(0, [Fake_total_final,Real_total,Real_SP])
                
                losses = [ torch.mean(x) if x is not None else 0 for x in losses ]
                loss_dict = dict(zip(modelD.module.loss_names, losses))
                #print("loss_dict",loss_dict)
          
                # collect losses
                loss_G, loss_D = modelD.module.get_losses(loss_dict)

          
                ###################################### Backward Pass #################################                 
                # update generator weights     
                # update generator weights     
                loss_backward(opt, loss_G, optimizer_G)                

                # update individual discriminator weights                
                loss_backward(opt, loss_D, optimizer_D)
                

                if i == 0: fg_dense_first = Fake_total_final[:][0]   # the first generated image in this sequence


            if opt.debug:
                call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]) 

            ############## Display results and errors ##########
            ### print out errors
            if total_steps % print_freq == 0:
                t = (time.time() - iter_start_time) / print_freq
                errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
                

                loss_names_vis = modelD.module.loss_names.copy()
                
                
                visualizer.print_current_errors_new(epoch, epoch_iter, errors, loss_names_vis, t)
                visualizer.plot_current_errors(errors, total_steps)
                
            ### display output images
            if save_fake:                
                visuals = util.save_all_tensors_cloth(opt,Fake_total_final,Real_total,Real_SP,real_input_SP, real_SFG,real_input_TP, real_input_TFG)            
                visualizer.display_current_results(visuals, epoch, total_steps)

            ### save latest model
            save_models_cloth(opt, epoch, epoch_iter, total_steps, visualizer, iter_path, ClothPart, modelD)  
            
            if epoch_iter > dataset_size - opt.batchSize:
                epoch_iter = 0
                break
           
        # end of epoch 
        iter_end_time = time.time()
        visualizer.vis_print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        
        #save_models(opt, epoch, epoch_iter, total_steps, visualizer, iter_path, modelG, modelD, end_of_epoch=True)

        ### save model for this epoch and update model params
        save_models_cloth(opt, epoch, epoch_iter, total_steps, visualizer, iter_path, ClothPart, modelD, end_of_epoch=True)
        update_models_cloth(opt, epoch, ClothPart, data_loader) 

def loss_backward(opt, loss, optimizer):
    optimizer.zero_grad()                
    if opt.fp16:
        from apex import amp
        with amp.scale_loss(loss, optimizer) as scaled_loss: 
            scaled_loss.backward()
    else:
        loss.backward()
    optimizer.step()

def reshape(tensors):
    if tensors is None: return None
    if isinstance(tensors, list):
        return [reshape(tensor) for tensor in tensors]    
    _, _, ch, h, w = tensors.size()
    return tensors.contiguous().view(-1, ch, h, w)

if __name__ == "__main__":
    train()