# REMOT


## Prerequisites
- Ubuntu
- Python 3
- NVIDIA GPU (>12GB memory) + CUDA10 cuDNN7
- PyTorch 1.0.0
### Other Dependencies
#### FlowNet_v2 (directly ported from the original [[flownet2]](https://github.com/NVIDIA/flownet2-pytorch) following the steps described in [[vid2vid]](https://github.com/NVIDIA/vid2vid))
    cd models/flownet2_pytorch
    bash install.sh

## Getting Started
It's a preview version of our source code. We will clean it up in the near future.  
### Notes
1. Main functions for training and testing can be found in "train_stage1.py", "train_stage2.py", "train_stage2.py", "test_all_stages.py";
2. Data preprocessings of all the stages can be found in "data" folder;
3. Model definitions of all the stages can be found in "models" folder;
4. Training and testing options can be found in "options" folder;
5. Training and testing scripts can be found in "scripts" folder;
6. Tool functions can be found in "util" folder.

### Data Preparation
Download all the data packages from [[google drive]](https://drive.google.com/drive/folders/1f6NEO1onLtf-K65bpms4_alBlNh5YIVW?usp=sharing) or [[baidu pan (extraction code:gle4]](https://pan.baidu.com/s/14oitDhULAeirGaojV_VYew), and uncompress them.
You should create a directory named 'iPER' in the root (i.e., 'iPER') of this project, and then put 'train' and 'test' folders to 'iPER' you just created.
The structure should look like this:  
-iPER
---SoloDance  
------train  
------test  

    
### Testing stage 1 
    bash scripts/full/test_stage1.sh
### Testing stage 123
    bash scripts/full/test_stage123.sh
    
## Acknowledgement
A large part of the code is borrowed from [NVIDIA/vid2vid](https://github.com/NVIDIA/vid2vid). Thanks for their wonderful works.


# REMOT
