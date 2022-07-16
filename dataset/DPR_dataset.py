from dataset.data_utils import img2tensor, imread
from dataset.transforms import multi_random_crop
from torch.utils import data as data 
import os
import os.path as osp
import torch
import numpy as np

def load_light(path):
    with open(path, 'r') as fr:
        lines = fr.readlines()
    light_params = np.array([float(l[:-1]) for l in lines])
    return light_params
class DPRDataset(data.Dataset):
    def __init__(self, opt):
        super(DPRDataset, self).__init__()
        self.opt = opt
        self.light_num = 5
       
        if opt.get('test_mode') == None:
            self.test_mode = False
        else:
            self.test_mode = opt['test_mode']

        self.input_root = opt['dataroot']
        self.input_folders = sorted(os.listdir(self.input_root))
        
        if self.test_mode:
            self.input_folders = self.input_folders[-100:]
        else:
            self.input_folders = self.input_folders[:-100]




    def __getitem__(self, index):
        # Load gt and input images. Dimension order: HWC; channel order: BGR;
        # image range: [-1, 1], float32.
        
        folder_idx = index // self.light_num
        input_light_idx = index % self.light_num if not self.test_mode else 0 
        folder_path = osp.join(self.input_root, self.input_folders[folder_idx])
        gt_light_idx = np.random.randint(0, 5) if not self.test_mode else 1

        input_path = osp.join(folder_path, f'{self.input_folders[folder_idx]}_{input_light_idx:02}.png')
        input_light_path = osp.join(folder_path, f'{self.input_folders[folder_idx]}_light_{input_light_idx:02}.txt')

        gt_path = osp.join(folder_path, f'{self.input_folders[folder_idx]}_{gt_light_idx:02}.png')
        gt_light_path = osp.join(folder_path, f'{self.input_folders[folder_idx]}_light_{gt_light_idx:02}.txt')

        light_input = torch.tensor(load_light(input_light_path)).float()
        light_gt =  torch.tensor(load_light(gt_light_path)).float()
        
        img_input = imread(input_path)
        img_gt = imread(gt_path)

        input_normal = imread(osp.join(folder_path, 'full_normal.png'))
        
        # crop imgs
        if self.test_mode == False and self.opt['gt_size'] > 0:
            (img_gt, img_input, input_normal), _=  multi_random_crop([img_gt, img_input, input_normal], self.opt['gt_size'])        

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_input, input_normal = img2tensor([img_gt, img_input, input_normal],
                                    bgr2rgb=True,
                                    float32=True)
        img_input = torch.cat([img_input, input_normal], axis=0)
       
        
        return {
            'input': img_input,
            'input_light': light_input,
            'gt': img_gt,
            'gt_light': light_gt,
            'input_path': input_path,
            'gt_path': gt_path
        }
        

    def __len__(self):
        return len(self.input_folders) * self.light_num


