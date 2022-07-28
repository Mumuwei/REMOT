### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import time
import os
import torch
from subprocess import call

from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model_parser, create_optimizer_parser, save_models, update_models
from models.models import init_params_composer as init_params_parser
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

        
    ### initialize models，模型初始化
    models = create_model_parser(opt)
    modelG, modelD, flowNet, optimizer_G, optimizer_D, optimizer_D_T = create_optimizer_parser(opt, models)    
    
    ### initialize dataset，初始化数据集
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)    
    print('#training frames = %d' % dataset_size)

    

    ### set parameters，设置模型的一些参数    
    n_gpus, tG, tD, tDB, s_scales, t_scales, input_nc_1, input_nc_2, \
        start_epoch, epoch_iter, print_freq, total_steps, iter_path = init_params_parser(opt, modelG, modelD, data_loader)
    visualizer = Visualizer(opt)   
    
    #1 3 3 9 1 3 19 16 1 0 40.0 0.0 ./check_points/test/parser_256p/iter.txt
#     print(n_gpus, tG, tD, tDB, s_scales, t_scales, input_nc_1, input_nc_2, \
#         start_epoch, epoch_iter, print_freq, total_steps, iter_path)
    
    ### real training starts here  
    #开始进行这么多epoch运算
    for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()  
        #开始进行1个epoch的训练
        for idx, data in enumerate(dataset, start=epoch_iter):        
            if total_steps % print_freq == 0:
                iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            # whether to collect output images，是否采集输出数据
            save_fake = total_steps % opt.display_freq == 0
            
            n_frames_total, n_frames_load, t_len = data_loader.dataset.init_data_params_parser(data, n_gpus, tG)
            #print("n_frames_total, n_frames_load, t_len",n_frames_total, n_frames_load, t_len)
            fake_slo_prev_last, frames_all = data_loader.dataset.init_data_parser(t_scales)
            #print("fake_slo_prev_last, frames_all",fake_slo_prev_last, frames_all)

            #这又是在干啥,14帧只能运行两次，每次6个视频帧
            for i in range(0, n_frames_total, n_frames_load):
                #print(i)
                
                #torch.Size([4, 1, 1, 256, 192]) torch.Size([4, 8, 3, 256, 192]) torch.Size([4, 8, 1, 256, 192]) torch.Size([4, 8, 3, 256, 192])
                input_TParsing, input_SPose, input_SParsing, input_SFG = data_loader.dataset.prepare_data_parser(data, i)

                ###################################### Forward Pass ##########################
                ####### generator  
                #送入生成器
                
#                 if i==6:
#                     print("xxxfake_slo_prev_last",fake_slo_prev_last.shape)#([4, 2, 12, 256, 192])
                
                #ls指的是logsoftmax
                #fake_slo, fake_slo_raw, fake_slo_ls, fake_slo_raw_ls, flow, weight, real_input_T, real_input_S[:,tG-1:], real_slo[:,tG-2:], real_sfg[:,tG-2:], fake_slo_prev
                fake_slo, fake_slo_raw, fake_slo_ls, fake_slo_raw_ls, flow, weight, real_input_T, real_input_S, real_slop, real_sfgp, fake_slo_last \
                    = modelG(input_TParsing, input_SPose, input_SParsing, input_SFG, fake_slo_prev_last)
                
                
                #判别模型
                ####### discriminator            
                ### individual frame discriminator          
                real_slo_prev, real_slo = real_slop[:, :-1], real_slop[:, 1:]   # the collection of previous and current real frames
                real_sfg_prev, real_sfg = real_sfgp[:, :-1], real_sfgp[:, 1:]   # the collection of previous and current real frames
                flow_ref, conf_ref = flowNet(real_sfg, real_sfg_prev)
                fake_slo_prev = modelG.module.compute_fake_B_prev(real_slo_prev, fake_slo_prev_last, fake_slo)
                fake_slo_prev_last = fake_slo_last

                losses = modelD(0, [real_slo, fake_slo, fake_slo_raw, fake_slo_ls, fake_slo_raw_ls, real_input_T, real_input_S, real_slo_prev, fake_slo_prev, flow, weight, flow_ref, conf_ref])
                losses = [ torch.mean(x) if x is not None else 0 for x in losses ]
                loss_dict = dict(zip(modelD.module.loss_names, losses))

                ### temporal discriminator                
                # get skipped frames for each temporal scale
                frames_all, frames_skipped = modelD.module.get_all_skipped_frames(frames_all, \
                        real_slo, fake_slo, flow_ref, conf_ref, real_sfg, t_scales, tD, n_frames_load, i, flowNet)                                

                # run discriminator for each temporal scale
                loss_dict_T = []
                for s in range(t_scales):                
                    if frames_skipped[0][s] is not None:                        
                        losses = modelD(s+1, [frame_skipped[s] for frame_skipped in frames_skipped])
                        losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
                        loss_dict_T.append(dict(zip(modelD.module.loss_names_T, losses)))

                # collect losses
                loss_G, loss_D, loss_D_T, t_scales_act = modelD.module.get_losses(loss_dict, loss_dict_T, t_scales)

                ###################################### Backward Pass #################################                 
                # update generator weights     
                loss_backward(opt, loss_G, optimizer_G)                

                # update individual discriminator weights                
                loss_backward(opt, loss_D, optimizer_D)

                # update temporal discriminator weights
                for s in range(t_scales_act):                    
                    loss_backward(opt, loss_D_T[s], optimizer_D_T[s])

                if i == 0: fake_slo_first = fake_slo[0, 0]   # the first generated image in this sequence

            if opt.debug:
                call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]) 

            ############## Display results and errors ##########
            ### print out errors
            if total_steps % print_freq == 0:
                t = (time.time() - iter_start_time) / print_freq
                errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
                for s in range(len(loss_dict_T)):
                    errors.update({k+str(s): v.data.item() if not isinstance(v, int) else v for k, v in loss_dict_T[s].items()})            
                visualizer.print_current_errors_new(epoch, epoch_iter, errors, modelD.module.loss_names, t)
                visualizer.plot_current_errors(errors, total_steps)

            ### display output images
            if save_fake:                
                visuals = util.save_all_tensors_parser(opt, real_input_T, real_input_S, fake_slo, fake_slo_raw, fake_slo_first, real_slo, flow_ref, conf_ref, flow, weight)                
                visualizer.display_current_results(visuals, epoch, total_steps)

            ### save latest model
            save_models(opt, epoch, epoch_iter, total_steps, visualizer, iter_path, modelG, modelD)            
            if epoch_iter > dataset_size - opt.batchSize:
                epoch_iter = 0
                break
           
        # end of epoch 
        iter_end_time = time.time()
        visualizer.vis_print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        ### save model for this epoch and update model params
        save_models(opt, epoch, epoch_iter, total_steps, visualizer, iter_path, modelG, modelD, end_of_epoch=True)
        update_models(opt, epoch, modelG, modelD, data_loader) 

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