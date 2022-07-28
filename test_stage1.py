### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import time
import os
import numpy as np
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model_parser
import util.util as util
from util.visualizer import Visualizer
from util import htmls
import torch
import flowiz as fz
from PIL import Image

os.environ['CUDA_LAUNCH_BLOCKING'] = "0"

opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

#创建数据和模型
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
Parser = create_model_parser(opt)
visualizer = Visualizer(opt)

save_dir_input_1 = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, 'input_stage1'))
save_dir_output_1 = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, 'output_stage1'))
save_dir_input_2 = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, 'input_stage2_SP'))


print('Doing %d frames' % len(dataset))

SPose_cloth = torch.zeros(1, 3, 3, 256, 192).cuda(opt.gpu_ids[0])
SParsing_cloth = torch.zeros(1, 3, 1, 256, 192).cuda(opt.gpu_ids[0])
SParsing = torch.zeros(1, 3, 1, 256, 192).cuda(opt.gpu_ids[0])
#breakpoint()

print("----------------")
for i, data in enumerate(dataset):
    #print("i",i)
    #print("opt.how_many",opt.how_many)
    if i >= opt.how_many:
        break    
    
    #是否换了seq
    if data['change_seq']:
        Parser.fake_slo_prev = None
 
        
        

    _, _, height, width = data['SFG'].size()
    TParsing = Variable(data['TParsing']).view(1, -1, 1, height, width).cuda(opt.gpu_ids[0])
    SPose = Variable(data['SPose']).view(1, -1, 3, height, width).cuda(opt.gpu_ids[0])
    SParsing_ref = Variable(data['SParsing']).view(1, -1, 1, height, width).cuda(opt.gpu_ids[0])
    SI_ref = Variable(data['SFG']).view(1, -1, 3, height, width).cuda(opt.gpu_ids[0])

    
    '''
    print("TParsing",TParsing.shape)
    print("TParsing_uncloth",TParsing_uncloth.shape)
    print("TFG_uncloth",TFG_uncloth.shape)
    print("TParsing_cloth",TParsing_cloth.shape)
    print("TFG_cloth",TFG_cloth.shape)
    print("SPose",SPose.shape)
    print("SParsing_ref",SParsing_ref.shape)
    print("SI_ref",SI_ref.shape)
    print("BG",BG.shape)
    
    '''
    
    
    #print("SPose",SPose.shape)
    #fake_slo, real_input_T[0, -1], real_input_S[0, -1]
    if data['change_seq'] or i == 0:
        for t in range(3):
            
            SParsing_sm_t, input_T_stage1, input_S_stage1 = Parser.inference(TParsing, SPose[:,t:t+3])
            #print("SParsing_sm_t",SParsing_sm_t.shape)#torch.Size([1, 12, 256, 192])
            SParsing_t = SParsing_sm_t[0].max(0, keepdim=True)[1]
            #print("ttt",t)
#             print("SParsing_t",SParsing_t.shape)#torch.Size([1, 256, 192])
#             print("SParsing_t.max()",SParsing_t.max())
#             print("SParsing_t.min()",SParsing_t.min())
            
#             SParsing_t[SParsing_t == 1] = 100
#             SParsing_t[SParsing_t == 2] = 1
#             SParsing_t[SParsing_t == 3] = 2
#             SParsing_t[SParsing_t == 100] = 3
                    
          
    else:
        SParsing_sm_t, input_T_stage1, input_S_stage1 = Parser.inference(TParsing, SPose[:,-3:])
        SParsing_t = SParsing_sm_t[0].max(0, keepdim=True)[1]
        
 
    #stage1的输入
    input_spose_ = SPose[0,-1]
    input_spose_[input_spose_==-1] = 1
    input_spose = util.tensor2im(input_spose_)
    
    input_tlo = util.tensor2lo(input_T_stage1, opt.label_nc_1, old_type=True)
    
    #stage1的输出
    
    
    output_SLO_ =  SParsing_t
    output_SLO = util.tensor2lo(output_SLO_, opt.label_nc_1)
    #torch.Size([1, 256, 192])
    #print("output_SLO_",output_SLO_.shape)
    input_SLO_stage2 = util.tensor2input(output_SLO_)
    #print("input_SLO_stage2",input_SLO_stage2.shape)
    
    
    #stage2
    #Fake_total_final,Real_total,Real_SP,real_input_SP, real_SFG, real_input_TP, real_input_TFG, part_total_prev = ClothPart(input_TParsing, input_TFG, input_SParsing, input_SFG, part_total_prev_last)
    
  
    visual_list_input_1 = [('input_spose', input_spose),
                               ('input_tlo', input_tlo)]


    visual_list_output_1 = [('output_slo', output_SLO)]
    

    

    visuals_input_1 = OrderedDict(visual_list_input_1) 
    visuals_output_1 = OrderedDict(visual_list_output_1) 


    img_path = data['A_path']
    print('process image... %s' % img_path)
    visualizer.save_images(save_dir_input_1, visuals_input_1, img_path)
    visualizer.save_images(save_dir_output_1, visuals_output_1, img_path)
    visualizer.save_input2(save_dir_input_2, input_SLO_stage2, img_path)
    
    if i==0:
        img_path = save_dir_input_2+"/"+img_path[0].split("/")[-2]+"/"+img_path[0].split("/")[-1]
        img = Image.open(img_path)
        for j in range(4):
            save_path = img_path.replace("4.png",str(j)+".png")
            #print(save_path)
            img.save(save_path)
            
        

