import os
import torch
import numpy as np
from torch.utils import data as data
from base_utils.utils import load_depth
from dataset.data_utils import img2tensor, imread
from dataset.transforms import multi_random_crop, dir_augment


'''Image{idx}_{color}_{angle}.png'''
class Any2anyTrainingDataset(data.Dataset):
    def __init__(self, opt):
        super(Any2anyTrainingDataset, self).__init__()
        self.opt = opt
        self.root = opt['dataset_root']
        self.colors = ['2500', '3500', '4500', '5500', '6500']
        self.angles = ['E', 'W', 'S', 'N', 'NE', 'NW', 'SE', 'SW']
        self.multiply = len(self.colors) * len(self.angles)
        self.is_val = opt.get('is_val')
        self.mask = opt.get('mask')
        
        if isinstance(self.opt['auxiliary_dim'], list) == False:
            self.opt['auxiliary_dim'] = [self.opt['auxiliary_dim']]
        if isinstance(opt['dataroot_auxiliary'], list) == False:
            opt['dataroot_auxiliary'] = [opt['dataroot_auxiliary']]

        self.aux_roots = opt['dataroot_auxiliary']
        self.aux_dims = opt['auxiliary_dim']
        self.aux_type = opt['auxiliary_type']

        # amount of scene
        self.num_scene = len(os.listdir(self.root)) // self.multiply

       
    def __getitem__(self, idx):
        # addressing
        scene_id = idx // self.multiply
        idx -= scene_id * self.multiply
        color_id = idx // len(self.angles)
        angle_id = idx - color_id * len(self.angles)
        
        
        ref_scene_id = scene_id
        while ref_scene_id == scene_id:
            ref_scene_id = np.random.randint(0, self.num_scene)
        gt_scene_id = scene_id
        target_color_id = np.random.randint(0, len(self.colors))
        target_angle_id = np.random.randint(0, len(self.angles))

        if self.is_val:
            scene_id += 275
            ref_scene_id += 275
            gt_scene_id += 275

        input_path = os.path.join(self.root, 
                    f'Image{scene_id:0>3}_{self.colors[color_id]}_{self.angles[angle_id]}.png')
        ref_path = os.path.join(self.root, 
                    f'Image{ref_scene_id:0>3}_{self.colors[target_color_id]}_{self.angles[target_angle_id]}.png')
        gt_path =  os.path.join(self.root, 
                    f'Image{gt_scene_id:0>3}_{self.colors[target_color_id]}_{self.angles[target_angle_id]}.png')

        input_img = imread(input_path)
        ref_img = imread(ref_path)
        gt_img = imread(gt_path)

        if self.mask == True:
            mask_path = os.path.join(self.root, 'mask', f'Image{gt_scene_id:0>3}.npy')
            mask = np.load(mask_path)[:, :, np.newaxis]
        else:
            h, w, _ = input_img.shape
            mask = np.array([[1.0]]).repeat(w, 1).repeat(h, 0)[:, :, np.newaxis]

        input_auxs = []
        ref_auxs = []
        for dim, aux_root in zip(self.aux_dims, self.aux_roots):
            input_aux = os.path.join(aux_root, f'Image{scene_id:0>3}{self.aux_type}')
            ref_aux = os.path.join(aux_root, f'Image{ref_scene_id:0>3}{self.aux_type}')
            if dim == 1:
                input_aux = load_depth(input_aux)
                ref_aux = load_depth(ref_aux)
            else:
                input_aux = np.load(input_aux)
                ref_aux = np.load(ref_aux)

            if dim == 1:
                input_aux = input_aux[:, :, np.newaxis]
                ref_aux = ref_aux[:, :, np.newaxis]
            input_auxs.append(input_aux)
            ref_auxs.append(ref_aux)

        input_auxs = np.concatenate(input_auxs, axis=2)
        ref_auxs = np.concatenate(ref_auxs, axis=2)

        (input_img, ref_img, gt_img, 
            input_auxs, ref_auxs, mask), (angle_id, target_angle_id) = dir_augment([
            input_img, ref_img, gt_img, input_auxs, ref_auxs, mask], [angle_id, target_angle_id])

        if self.opt['gt_size'] > 0:
            (input_img, ref_img, gt_img, input_auxs, 
                ref_auxs, mask), _ =  multi_random_crop([
                input_img, ref_img, gt_img, input_auxs, ref_auxs, mask], self.opt['gt_size'], None)    
        
        input_img, ref_img, input_auxs, ref_auxs, gt_img, mask = img2tensor([
            input_img, ref_img, input_auxs, ref_auxs, gt_img, mask], True, True)

        input_img = torch.cat([input_img, input_auxs], 0)
        ref_img = torch.cat([ref_img, ref_auxs], 0)

        

        return {
            'input': input_img,
            'input_path': input_path,
            'ref': ref_img,
            'ref_path': ref_path,
            'gt': gt_img,
            'gt_path': gt_path,
            'input_angle': torch.tensor(angle_id),
            'target_angle': torch.tensor(target_angle_id),
            'input_color': torch.tensor(color_id),
            'target_color': torch.tensor(target_color_id),
            'mask': mask
        }

    def __len__(self):
        return self.num_scene * self.multiply

class Any2anyTrainingDataset2(data.Dataset):
    def __init__(self, opt):
        super(Any2anyTrainingDataset2, self).__init__()
        self.opt = opt
        self.root = opt['dataset_root']
        self.colors = ['2500', '3500', '4500', '5500', '6500']
        self.colors_code = torch.from_numpy(np.eye(5, dtype=np.float32))
        self.angles = ['E', 'W', 'S', 'N', 'NE', 'NW', 'SE', 'SW']
        self.angles_code = torch.from_numpy(np.eye(8, dtype=np.float32))
        self.multiply = len(self.colors) * len(self.angles)
        self.is_val = opt.get('is_val')
        
        if isinstance(self.opt['auxiliary_dim'], list) == False:
            self.opt['auxiliary_dim'] = [self.opt['auxiliary_dim']]
        if isinstance(opt['dataroot_auxiliary'], list) == False:
            opt['dataroot_auxiliary'] = [opt['dataroot_auxiliary']]

        self.aux_roots = opt['dataroot_auxiliary']
        self.aux_dims = opt['auxiliary_dim']
        self.aux_type = opt['auxiliary_type']

        # amount of scene
        self.num_scene = len(os.listdir(self.root)) // self.multiply

       
    def __getitem__(self, idx):
        # addressing
        scene_id = idx // self.multiply

        idx -= scene_id * self.multiply
        color_id = idx // len(self.angles)
        angle_id = idx - color_id * len(self.angles)

        

        ref_scene_id = scene_id
        while ref_scene_id == scene_id:
            ref_scene_id = np.random.randint(0, self.num_scene)
        gt_scene_id = scene_id
        target_color_id = np.random.randint(0, len(self.colors))
        target_angle_id = np.random.randint(0, len(self.angles))
        if self.is_val:
            scene_id += 275
            ref_scene_id += 275
            gt_scene_id += 275

        if self.is_val:
            input_path = os.path.join(self.root, f'Image{(+scene_id):0>3}_{self.colors[color_id]}_{self.angles[angle_id]}.png')
            ref_path = os.path.join(self.root, f'Image{ref_scene_id:0>3}_{self.colors[target_color_id]}_{self.angles[target_angle_id]}.png')
            gt_path =  os.path.join(self.root, f'Image{gt_scene_id:0>3}_{self.colors[target_color_id]}_{self.angles[target_angle_id]}.png')
        else: 
            input_path = os.path.join(self.root, f'Image{scene_id:0>3}_{self.colors[color_id]}_{self.angles[angle_id]}.png')
            ref_path = os.path.join(self.root, f'Image{ref_scene_id:0>3}_{self.colors[target_color_id]}_{self.angles[target_angle_id]}.png')
            gt_path =  os.path.join(self.root, f'Image{gt_scene_id:0>3}_{self.colors[target_color_id]}_{self.angles[target_angle_id]}.png')

        input_img = imread(input_path)
        ref_img = imread(ref_path)
        gt_img = imread(gt_path)

        input_auxs = []
        ref_auxs = []
        for dim, aux_root in zip(self.aux_dims, self.aux_roots):
            input_aux = os.path.join(aux_root, f'Image{scene_id:0>3}{self.aux_type}')
            ref_aux = os.path.join(aux_root, f'Image{ref_scene_id:0>3}{self.aux_type}')
            if dim == 1:
                input_aux = load_depth(input_aux)
                ref_aux = load_depth(ref_aux)
            else:
                input_aux = np.load(input_aux)
                ref_aux = np.load(ref_aux)

            if dim == 1:
                input_aux = input_aux[:, :, np.newaxis]
                ref_aux = ref_aux[:, :, np.newaxis]
            input_auxs.append(input_aux)
            ref_auxs.append(ref_aux)

        input_auxs = np.concatenate(input_auxs, axis=2)
        ref_auxs = np.concatenate(ref_auxs, axis=2)

        (input_img, ref_img, gt_img, input_auxs, ref_auxs), (angle_id, target_angle_id) = dir_augment([input_img, ref_img, gt_img, input_auxs, ref_auxs], [angle_id, target_angle_id])

        if self.opt['gt_size'] > 0:
            (input_img, ref_img, gt_img, input_auxs, ref_auxs), _ =  multi_random_crop([input_img, ref_img, gt_img, input_auxs, ref_auxs], self.opt['gt_size'], None)    
        
        input_img, ref_img, input_auxs, ref_auxs, gt_img = img2tensor([input_img, ref_img, input_auxs, ref_auxs, gt_img], True, True)

        input_img = torch.cat([input_img, input_auxs], 0)
        ref_img = torch.cat([ref_img, ref_auxs], 0)

        # print(input_img.shape)
        input_img = torch.nn.functional.interpolate(input_img.unsqueeze(0), scale_factor=0.5, mode='bilinear', align_corners=True).squeeze(0)
        ref_img = torch.nn.functional.interpolate(ref_img.unsqueeze(0), scale_factor=0.5, mode='bilinear', align_corners=True).squeeze(0)
        gt_img = torch.nn.functional.interpolate(gt_img.unsqueeze(0), scale_factor=0.5, mode='bilinear', align_corners=True).squeeze(0)

        return {
            'input': input_img,
            'input_path': input_path,
            'ref': ref_img,
            'ref_path': ref_path,
            'gt': gt_img,
            'gt_path': gt_path,
            'input_angle': self.angles_code[angle_id],
            'target_angle': self.angles_code[target_angle_id],
            'input_color': self.colors_code[color_id],
            'target_color': self.colors_code[target_color_id]
        }

    def __len__(self):
        return self.num_scene * self.multiply


'''Pair{idx}.png'''
class Any2anyTestingDataset(data.Dataset):
    def __init__(self, opt):
        super(Any2anyTestingDataset, self).__init__()
        self.opt = opt
        self.input_dir = opt['dataroot_input'][0]
        self.ref_dir = opt['dataroot_ref'][0]
        self.gt_dir = opt['dataroot_gt'][0]
        self.mask = opt.get('mask')

        
        if isinstance(self.opt['auxiliary_dim'], list) == False:
            self.opt['auxiliary_dim'] = [self.opt['auxiliary_dim']]
        if isinstance(opt['dataroot_auxiliary_input'], list) == False:
            opt['dataroot_auxiliary_input'] = [opt['dataroot_auxiliary_input']]
            opt['dataroot_auxiliary_ref'] = [opt['dataroot_auxiliary_ref']]


        self.input_aux_roots = opt['dataroot_auxiliary_input']
        self.ref_aux_roots = opt['dataroot_auxiliary_ref']
        self.aux_dims = opt['auxiliary_dim']
        self.aux_type = opt['auxiliary_type']
        self.num_scene = len(os.listdir(self.input_dir))

    def __getitem__(self, idx):
       

        input_path = os.path.join(self.input_dir, f'Pair{idx:0>3}.png')
        ref_path = os.path.join(self.ref_dir, f'Pair{idx:0>3}.png')
        gt_path =  os.path.join(self.gt_dir, f'Pair{idx:0>3}.png')

        input_img = imread(input_path)
        ref_img = imread(ref_path)
        gt_img = imread(gt_path)


        if self.mask == True:
            mask_path = os.path.join(self.input_dir[:-6], 'mask', f'Pair{idx:0>3}.npy')
            mask = np.load(mask_path)[:, :, np.newaxis]
        else:
            h, w, _ = input_img.shape
            mask = np.array([[1.0]]).repeat(w, 1).repeat(h, 0)[:, :, np.newaxis]


        input_auxs = []
        ref_auxs = []
        for dim, input_aux_root, ref_aux_root in zip(self.aux_dims, self.input_aux_roots, self.ref_aux_roots):
            input_aux = os.path.join(input_aux_root, f'Pair{idx:0>3}{self.aux_type}')
            ref_aux = os.path.join(ref_aux_root, f'Pair{idx:0>3}{self.aux_type}')
            if dim == 1:
                input_aux = load_depth(input_aux)
                ref_aux = load_depth(ref_aux)
            else:
                input_aux = np.load(input_aux)
                ref_aux = np.load(ref_aux)

            if dim == 1:
                input_aux = input_aux[:, :, np.newaxis]
                ref_aux = ref_aux[:, :, np.newaxis]
            input_auxs.append(input_aux)
            ref_auxs.append(ref_aux)


        input_auxs = np.concatenate(input_auxs, axis=2)
        ref_auxs = np.concatenate(ref_auxs, axis=2)

        input_img, ref_img, input_auxs, ref_auxs, gt_img, mask = img2tensor([input_img, ref_img, input_auxs, ref_auxs, gt_img, mask], True, True)

        input_img = torch.cat([input_img, input_auxs], 0)
        ref_img = torch.cat([ref_img, ref_auxs], 0)

        # mask = torch.nn.functional.interpolate(mask.unsqueeze(0), scale_factor=0.5, mode='bilinear', align_corners=True).squeeze(0)
        # input_img = torch.nn.functional.interpolate(input_img.unsqueeze(0), scale_factor=0.5, mode='bilinear', align_corners=True).squeeze(0)
        # ref_img = torch.nn.functional.interpolate(ref_img.unsqueeze(0), scale_factor=0.5, mode='bilinear', align_corners=True).squeeze(0)
        # gt_img = torch.nn.functional.interpolate(gt_img.unsqueeze(0), scale_factor=0.5, mode='bilinear', align_corners=True).squeeze(0)

        return {
            'input': input_img,
            'input_path': input_path,
            'ref': ref_img,
            'ref_path': ref_path,
            'gt': gt_img,
            'gt_path': gt_path,
            'mask': mask
        }


    def __len__(self):
        return self.num_scene


if __name__ == '__main__':
    opt = {
        'dataset_root': 'data/any2any/train/input', 
        'auxiliary_dim': [3, 1],
        'dataroot_auxiliary': ['data/any2any/train/normals', 'data/any2any/train/depth'], 
        'auxiliary_type': '.npy', 
        'gt_size': 0,
    }
    dataset = Any2anyTrainingDataset(opt)
    print(len(dataset))
    data = dataset.__getitem__(30*40)