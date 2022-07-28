import torch.utils.data
from data.base_data_loader import BaseDataLoader


def CreateDataset(opt):
    dataset = None
    #根据名称创建不同的dataloder
    if opt.dataset_mode == 'parser':
        from data.data_stage1 import ParserDataset
        dataset = ParserDataset()
        
    #自己创建的部分
    elif opt.dataset_mode == 'cloth':
        from data.data_stage2 import ClothDataset
        dataset = ClothDataset()
        
    elif opt.dataset_mode == 'cloth_2':
        from data.data_stage2_cloth import ClothDataset
        dataset = ClothDataset()
        
    elif opt.dataset_mode == 'composer':
        from data.data_stage3 import ComposerDataset
        dataset = ComposerDataset()
    elif opt.dataset_mode == 'full':
        from data.data_all_stages import FullDataset
        dataset = FullDataset()
        
    elif opt.dataset_mode == 'test1':
        from data.data_1_stage import Test1Dataset
        dataset = Test1Dataset()
        
    elif opt.dataset_mode == 'test23':
        from data.data_23_stage import Test23Dataset
        dataset = Test23Dataset()
        
    elif opt.dataset_mode == 'test123':
        from data.data_123_stage import Test123Dataset
        dataset = Test123Dataset()
        
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    print("dataset [%s] was created" % (dataset.name()))
    #初始化这个数据集
    dataset.initialize(opt)
    return dataset

#创建dataloader
class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    #这一部分被执行
    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
