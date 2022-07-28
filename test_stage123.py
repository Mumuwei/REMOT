### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import time
import os
import numpy as np
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model_full
import util.util as util
from util.visualizer import Visualizer
from util import htmls
import torch
import flowiz as fz
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "0"

opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

#创建数据和模型
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
ClothWarper, Composer = create_model_full(opt)
visualizer = Visualizer(opt)

save_dir_input_1 = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, 'input_stage1'))
save_dir_input_2 = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, 'input_stage2'))
save_dir_input_3 = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, 'input_stage3'))
save_dir_output_1 = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, 'output_stage1'))
save_dir_output_2 = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, 'output_stage2'))
save_dir_output_3 = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, 'output_stage3'))
save_dir_output_4 = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, 'output_SI'))
save_dir_output_5 = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, 'output_SI_raw'))
save_dir_ref = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, 'input_source_ref'))


save_dir_input_2_img = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, 'input_stage2_img'))

print('Doing %d frames' % len(dataset))

SPose_cloth = torch.zeros(1, 3, 3, 256, 192).cuda(opt.gpu_ids[0])
SParsing_cloth = torch.zeros(1, 3, 1, 256, 192).cuda(opt.gpu_ids[0])
SParsing = torch.zeros(1, 3, 1, 256, 192).cuda(opt.gpu_ids[0])

for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break    
    if data['change_seq']:
        ClothWarper.part_total_prevs = None
        Composer.fake_SI_prev = None
          

    _, _, height, width = data['SI'].size()
    TParsing = Variable(data['TParsing']).view(1, -1, 1, height, width).cuda(opt.gpu_ids[0])
    TFG = Variable(data['TFG']).view(1, -1, 3, height, width).cuda(opt.gpu_ids[0])
    SPose = Variable(data['SPose']).view(1, -1, 3, height, width).cuda(opt.gpu_ids[0])
    SParsing = Variable(data['SParsing']).view(1, -1, 1, height, width).cuda(opt.gpu_ids[0])
    SI_ref = Variable(data['SI']).view(1, -1, 3, height, width).cuda(opt.gpu_ids[0])
    BG = Variable(data['BG']).view(1, -1, 3, height, width).cuda(opt.gpu_ids[0])

#     print("TParsing",TParsing.shape)
#     print("TFG",TFG.shape)
#     print("SPose",SPose.shape)
#     print("SParsing",SParsing.shape)
#     print("SI_ref",SI_ref.shape)
#     print("BG",BG.shape)

# TParsing torch.Size([1, 5, 1, 256, 192])
# TFG torch.Size([1, 5, 3, 256, 192])
# SPose torch.Size([1, 5, 3, 256, 192])
# SParsing torch.Size([1, 5, 1, 256, 192])
# SI_ref torch.Size([1, 5, 3, 256, 192])
# BG torch.Size([1, 5, 3, 256, 192])



    #stage1
    if data['change_seq'] or i == 0:
        for t in range(3):
            
            SParsing_sm_t, input_T_stage1, input_S_stage1 = Parser.inference(TParsing, SPose[:,t:t+3])
            #print("SParsing_sm_t",SParsing_sm_t.shape)#torch.Size([1, 12, 256, 192])
            SParsing_t = SParsing_sm_t[0].max(0, keepdim=True)[1]
       
    else:
        SParsing_sm_t, input_T_stage1, input_S_stage1 = Parser.inference(TParsing, SPose[:,-3:])
        SParsing_t = SParsing_sm_t[0].max(0, keepdim=True)[1]

    
    #stage2
    Fake_total,Real_TP, Real_TFG, Real_SP, Real_SFG, real_input_TP, real_input_TFG, real_input_SP,real_SFG = ClothWarper.inference(TParsing[:,-3:], TFG[:,-3:], SParsing[:,-3:],SI_ref[:,-3:])
    
    
    #stge3
    Fake_stage2_all = torch.zeros_like(Fake_total[0])
    for part in Fake_total:
        Fake_stage2_all = Fake_stage2_all + part.unsqueeze(1)
        
        #Fake_stage2_all, real_input_TP, real_input_TFG, input_SPose, real_input_SP, real_SFG, input_BG, input_SI
    fake_SI, fake_SI_raw, fake_sd, fake_SFG, flow, weight = Composer.inference(Fake_stage2_all, real_input_TP[:,-3:], real_input_TFG[:,-3:], SPose[:,-3:], real_input_SP[:,-3:], real_SFG[:,-3:], BG[:,-3:], SI_ref[:,-3:])
    #torch.Size([1, 3, 256, 192])
    
    
    #stage2
    input_tlo_ = real_input_TP[0,-1]
    input_tlo = util.tensor2lo(input_tlo_, opt.label_nc_2)
    
    input_slo_ = real_input_SP[0,-1]
    input_slo = util.tensor2lo(input_slo_, opt.label_nc_2)
    
    input_sp1 = Real_SP[0][0]#1,256,192
    input_sp2 = Real_SP[1][0]
    input_sp3 = Real_SP[2][0]
    input_sp4 = Real_SP[3][0]
    input_sp5 = Real_SP[4][0]
    
    
    input_sp1 = util.tensor2sp(input_sp1, 1)
    input_sp2 = util.tensor2sp(input_sp2, 1)
    input_sp3 = util.tensor2sp(input_sp3, 1)
    input_sp4 = util.tensor2sp(input_sp4, 1)
    input_sp5 = util.tensor2sp(input_sp5, 1)
    
    
    input_tp1 = Real_TP[0][0]#1,256,192
    input_tp2 = Real_TP[1][0]
    input_tp3 = Real_TP[2][0]
    input_tp4 = Real_TP[3][0]
    input_tp5 = Real_TP[4][0]
    
    input_tp1 = util.tensor2sp(input_tp1, 1)
    input_tp2 = util.tensor2sp(input_tp2, 1)
    input_tp3 = util.tensor2sp(input_tp3, 1)
    input_tp4 = util.tensor2sp(input_tp4, 1)
    input_tp5 = util.tensor2sp(input_tp5, 1)
    
    Parsing_all = []
    Parsing_all.append(input_sp1)
    Parsing_all.append(input_sp2)
    Parsing_all.append(input_sp3)
    Parsing_all.append(input_sp4)
    Parsing_all.append(input_sp5)
    Parsing_all.append(input_tp1)
    Parsing_all.append(input_tp2)
    Parsing_all.append(input_tp3)
    Parsing_all.append(input_tp4)
    Parsing_all.append(input_tp5)
    
    h = input_sp1.shape[0]
    w = input_sp1.shape[1]
    base_h = 300  #256
    base_w = 210  #192
    P_all = np.ones([base_h*2,base_w*5,3],dtype=np.uint8)
    for x in range(2):
        for y in range(5):
            P_all[x*base_h:x*base_h+h,y*base_w:y*base_w+w,:] = Parsing_all[x*5+y]
            
    
    
    
    real_img1 = Real_SFG[0][0]
    real_img2 = Real_SFG[1][0]
    real_img3 = Real_SFG[2][0]
    real_img4 = Real_SFG[3][0]
    real_img5 = Real_SFG[4][0]
       
    real_img1 = util.tensor2im(real_img1)
    real_img2 = util.tensor2im(real_img2)
    real_img3 = util.tensor2im(real_img3)
    real_img4 = util.tensor2im(real_img4)
    real_img5 = util.tensor2im(real_img5)
    
    ##
    fake_img1 = Fake_total[0][0]
    fake_img2 = Fake_total[1][0]
    fake_img3 = Fake_total[2][0]
    fake_img4 = Fake_total[3][0]
    fake_img5 = Fake_total[4][0]
    
    fake_img1 = util.tensor2im(fake_img1)
    fake_img2 = util.tensor2im(fake_img2)
    fake_img3 = util.tensor2im(fake_img3)
    fake_img4 = util.tensor2im(fake_img4)
    fake_img5 = util.tensor2im(fake_img5)
    
    fake_imgs = []
    fake_imgs.append(fake_img1)
    fake_imgs.append(fake_img2)
    fake_imgs.append(fake_img3)
    fake_imgs.append(fake_img4)
    fake_imgs.append(fake_img5)
    
    corse_SI = Fake_stage2_all[0,-1]
    corse_SI = util.tensor2im(corse_SI)
    
    
    fake_imgs.append(corse_SI)
    
    h = fake_img1.shape[0]
    w = fake_img1.shape[1]
    base_h = 300  #256
    base_w = 210  #192
    fake_img_all = np.ones([base_h*2,base_w*3,3],dtype=np.uint8)
    for x in range(2):
        for y in range(3):
            fake_img_all[x*base_h:x*base_h+h,y*base_w:y*base_w+w,:] = fake_imgs[x*3+y]
    
    
    
    #----------------stage3-----------------------
    input_bg = util.tensor2im(BG[0, -1])
    output_sd = util.tensor2im(fake_sd[0], normalize=False)#
    output_flow = fz.convert_from_flow(flow[0].permute(1,2,0).data.cpu().numpy())#
    
    
    input_TFG = util.tensor2im(real_input_TFG[0,-1].data.cpu())
    
    output_SFG = util.tensor2im(fake_SFG[0])
    output_SI = util.tensor2im(fake_SI[0])
    output_SI_raw = util.tensor2im(fake_SI_raw[0])
    
    source_ref = util.tensor2im(SI_ref[0, -1])

   

    visual_list_input_2 = [('input_tlo', input_tlo), 
                   ('input_slo', input_slo),
                   ('P_all',P_all)]

    visual_list_output_2 = [('fake_img_all', fake_img_all)]

    visual_list_input_3 = [('corse_SI', corse_SI),
                    ('input_bg', input_bg),
                    ('input_TFG', input_TFG) ]

    visual_list_output_3 = [('output_SFG', output_SFG), 
                   ('output_flow', output_flow),
                   ('output_sd', output_sd)]

    visual_list_output_4 = [('output_SI', output_SI)]
    visual_list_output_5 = [('output_SI_raw', output_SI_raw)]

    visual_list_ref = [('source_ref', source_ref)]

    
    visuals_input_2 = OrderedDict(visual_list_input_2) 
    visuals_output_2 = OrderedDict(visual_list_output_2) 
    visuals_input_3 = OrderedDict(visual_list_input_3) 
    visuals_output_3 = OrderedDict(visual_list_output_3) 
    visuals_output_4 = OrderedDict(visual_list_output_4) 
    visuals_output_5 = OrderedDict(visual_list_output_5) 
    visual_ref = OrderedDict(visual_list_ref)
    img_path = data['A_path']
    print('process image... %s' % img_path)

    visualizer.save_images(save_dir_input_2, visuals_input_2, img_path)
    visualizer.save_images(save_dir_output_2, visuals_output_2, img_path)
    visualizer.save_images(save_dir_input_3, visuals_input_3, img_path)
    visualizer.save_images(save_dir_output_3, visuals_output_3, img_path)
    visualizer.save_images(save_dir_output_4, visuals_output_4, img_path)
    visualizer.save_images(save_dir_output_5, visuals_output_5, img_path)
    visualizer.save_images(save_dir_ref, visual_ref, img_path)
