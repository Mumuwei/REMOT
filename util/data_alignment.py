"""
1. pre-algin flat cloth to parsed cloth with a tunable parameter "align_factor"
2. obtain palms
3. obtain image gradient
"""
import os
import numpy as np
from PIL import Image
import cv2
import json
#import pycocotools.mask as maskUtils
from tqdm import tqdm
import math
import argparse
from matplotlib import pyplot as plt


#获取mask
def get_infor(parsing):
    parsing_array = np.array(parsing)
    mask = np.zeros_like(parsing_array)
    mask[parsing_array != 0] = 1
    index = np.where(mask!=0)
    
    if index[0].shape == (0,):
        return 0,0,0
    else:
        t,b = min(index[0]), max(index[0])
        l,r = min(index[1]), max(index[1])
        bbox_center = [int((l+r)/2), int((t+b)/2)]
        bbox_h = b - t
        bbox_w = r - l
    
    return bbox_h,bbox_w,bbox_center

def visual(img,bbox_center):
    draw = ImageDraw.Draw(img)
    vertex1 = [ x-10 for x in bbox_center ]
    vertex2 = [ x+10 for x in bbox_center ]
    vertex1.extend(vertex2)
    draw.ellipse(vertex1, fill=(0, 255, 0), outline=(255, 0, 0))
    display(img)

#对齐模块
def alignment(bbox_h1,bbox_w1,bbox_center1,bbox_h2,bbox_w2,bbox_center2,parsing,img):
    align_factor = 1
    ratio = bbox_h1 / bbox_h2
    scale_factor = ratio * align_factor
    #print(scale_factor)

    paste_x = int(bbox_center1[0] - bbox_center2[0]*scale_factor)
    paste_y = int(bbox_center1[1] - bbox_center2[1]*scale_factor)
    
    img_adj = img.resize((int(img.size[0]*scale_factor), int(img.size[1]*scale_factor)), Image.BILINEAR)
    parsing_adj = parsing.resize((int(img.size[0]*scale_factor), int(img.size[1]*scale_factor)), Image.NEAREST)
    
    blank_img = Image.fromarray(np.zeros((1024,1024,3), np.uint8))
    blank_img.paste(img_adj, (paste_x, paste_y))
    
    blank_mask = Image.fromarray(np.zeros((1024,1024), np.uint8)).convert("P")
    blank_mask.paste(parsing_adj, (paste_x, paste_y))
    
    return blank_img,blank_mask
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--solodance_root', type=str, default='/home/wujinlin5/yqw_home/Motion_Transfer/DATASET/iper/test',help='path to the MPV3D dataset')
    opt, _ = parser.parse_known_args()

    solo_root = opt.solodance_root

    # 目录
    img_root = os.path.join(solo_root, 'test_fg')
    parse_root = os.path.join(solo_root, 'test_parsing')
    
    # 源解析图和目标解析图
    s_parse_root = os.path.join(parse_root, 'source')
    t_parse_root = os.path.join(parse_root, 'target')
    t_img_root = os.path.join(img_root, 'target')

    # 保存路径
    img_save_dst = os.path.join(solo_root, 'test_fg_adj',"target")
    par_save_dst = os.path.join(solo_root, 'test_parsing_adj',"target")
    
    
    # 创建保存路径
    os.makedirs(img_save_dst, exist_ok=True)
    os.makedirs(par_save_dst, exist_ok=True)
    
    videos = os.listdir(s_parse_root)
    
    #print(videos)
    
    
    #遍历所有视频
    for video in videos[::-1]:
        save_video_dst = os.path.join(img_save_dst, video)
        save_parsing_dst = os.path.join(par_save_dst, video)
        
        os.makedirs(save_video_dst, exist_ok=True)
        os.makedirs(save_parsing_dst, exist_ok=True)
        
        
        s_video_path = os.path.join(s_parse_root, video)
        t_video_path = os.path.join(t_parse_root, video)
        #print(t_video_path,s_video_path)
        
        s_frames = os.listdir(s_video_path)
        t_frame = os.listdir(t_video_path)[0]
        #print(frames)
        
        t_path = os.path.join(t_parse_root, video, t_frame)
        t_parse = Image.open(t_path)
        t_bbox_h,t_bbox_w,t_bbox_center = get_infor(t_parse)
        
        
       
            
        ##target 图像
        timg_video_path = os.path.join(t_img_root, video)
        t_img_name = os.listdir(timg_video_path)[0]
        t_img_path = os.path.join(t_img_root, video, t_img_name)
        print(t_img_path)
        
        
        
        t_img = Image.open(t_img_path)
        if t_img.size != (1024, 1024):
            t_img = t_img.resize((1024,1024))
        
        
        
        for frame in s_frames:
            s_path = os.path.join(s_parse_root, video, frame)
            print(s_path)
            s_parse = Image.open(s_path)
            s_bbox_h,s_bbox_w,s_bbox_center = get_infor(s_parse)
            
            if t_bbox_h==0 or s_bbox_h==0:
                t_img_adj,t_parse_adj = t_parse,t_img
            else:
                t_img_adj,t_parse_adj = alignment(s_bbox_h,s_bbox_w,s_bbox_center,t_bbox_h,t_bbox_w,t_bbox_center,t_parse,t_img)


            t_img_adj.save(os.path.join(save_video_dst, frame))
            t_parse_adj.save(os.path.join(save_parsing_dst, frame))   
            
    ###对齐模块     
    print("----------------down---------------------------------")        