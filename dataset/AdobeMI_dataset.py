import ast
import torch
import numpy as np
from torch.utils import data as data 
from dataset.data_utils import img2tensor, imread, parse_adobe_dataset, select_one2one_data
from dataset.transforms import augment, multi_random_crop

def showimg(img, name):
    import numpy as np
    import cv2
    img = (img + 1) / 2 * 255
    img = img.astype(np.uint8)
    cv2.imwrite(f'input_{name}.png', img)

def showimgs(imgs):
    import numpy as np
    import cv2
    for i in range(imgs.shape[0]):
        img = imgs[i].astype(np.uint8)
        cv2.imwrite(f'input_{i}.png', img)
    
def get_seg_dict(root):
    with open(root, 'r') as fr:
        dic_str = fr.readline()
    return ast.literal_eval(dic_str)

def seg2tensor(val, seg_dic):
    onehot = np.zeros(len(seg_dic), dtype=np.float32)
    class_idx = seg_dic[str(val)]
    onehot[class_idx] = 1.
    return onehot

# Image name format: {notation of scene}/dir_{light_dir}_mip2.jpg
# other component: probes, materials_mip2.png, meta.json, thumb.jpg, label_map.png
class AdobeMultiIlluminationOnetoOneDataset(data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.paths = parse_adobe_dataset(opt['dataroot'])
        self.paths = select_one2one_data(self.paths, self.opt['input_dir'], self.opt['gt_dir'])
        #self.seg_dic = get_seg_dict(opt['seg_root'])
        if opt.get('test_mode') == None:
            self.test_mode = False
        else:
            self.test_mode = opt['test_mode']


       
    def __maskprobe__(self, input, gt, chrome_pos, gray_pos):
        # input H x W x C
        chrome_begx, chrome_begy, chrome_endx, chrome_endy = chrome_pos
        gray_begx, gray_begy, gray_endx, gray_endy = gray_pos
        h, w, c = input.shape

        scale = 4

        chrome_begx = round(chrome_begx / scale)
        chrome_begy = round(chrome_begy / scale)
        chrome_endx = round(chrome_endx / scale)
        chrome_endy = round(chrome_endy / scale)
        gray_begx = round(gray_begx / scale)
        gray_begy = round(gray_begy / scale)
        gray_endx = round(gray_endx / scale)
        gray_endy = round(gray_endy / scale)

        
        mask = torch.ones((h,w,1))
        mask[chrome_begy: chrome_endy+1, chrome_begx: chrome_endx+1, :] = 0
        mask[gray_begy: gray_endy+1, gray_begx: gray_endx+1, :] = 0

        input[chrome_begy: chrome_endy+1, chrome_begx: chrome_endx+1, :] = -1.
        gt[chrome_begy: chrome_endy+1, chrome_begx: chrome_endx+1, :] = -1.
        input[gray_begy: gray_endy+1, gray_begx: gray_endx+1, :] = -1.
        gt[gray_begy: gray_endy+1, gray_begx: gray_endx+1, :] = -1.
        
        return mask


    def __getitem__(self, index):
        # Load gt and input images. Dimension order: HWC; channel order: BGR;
        # image range: [-1, 1], float32.

        data_dic = self.paths[index]
        gt_path = data_dic['gt']
        input_path = data_dic['input']
        #print(input_path)
        img_input = imread(input_path)
        img_gt = imread(gt_path)

        mask = self.__maskprobe__(img_input, img_gt, data_dic['chrome_pos'], data_dic['gray_pos'])


        scene_name, input_name = input_path.split('/')[-2:]
        gt_name = gt_path.split('/')[-1]
        
        input_path = f'{scene_name}_{input_name}'
        gt_path = f'{scene_name}_{gt_name}'
        
        # augmentation for training
        if self.opt['phase'] == 'train':
            img_gt, img_input, mask = augment([img_gt, img_input, mask], 
                        self.opt['use_flip'], self.opt['use_rot'], self.opt.get('use_color'))

        # crop imgs
            if self.opt['gt_size'] > 0:
                (img_gt, img_input, mask), _ =  multi_random_crop([img_gt, img_input, mask], self.opt['gt_size'])        

        if img_gt.shape[0] % 32 != 0 or img_gt.shape[1] % 32 != 0:
            # crop border
            # print("crop")
            mask = mask[4:-4, 14:-14, :]
            img_input = img_input[4:-4, 14:-14, :]
            img_gt = img_gt[4:-4, 14:-14, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        if self.test_mode:
            img_input = img2tensor([img_input],
                                    bgr2rgb=True,
                                    float32=True)
            img_gt = None
            gt_path = None
        else:
            img_gt, img_input = img2tensor([img_gt, img_input],
                                    bgr2rgb=True,
                                    float32=True)
        mask = mask.permute(2, 0, 1)
        if img_gt != None:
            return {
                'input': img_input,
                'gt': img_gt,
                'mask': mask,
                'input_path': input_path,
                'gt_path': gt_path
            }
        else:
            return {
                'input': img_input,
                'input_path': input_path,
                'mask': mask,
            }

    def __len__(self):
        return len(self.paths)

class AdobeMultiIlluminationWithSegOnetoOneDataset(data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.paths = parse_adobe_dataset(opt['dataroot'], True)
        self.paths = select_one2one_data(self.paths, self.opt['input_dir'], self.opt['gt_dir'])
        self.vec_seg2tensor = np.vectorize(seg2tensor, excluded=[1], signature="(n)->(m)")
        if opt.get('test_mode') == None:
            self.test_mode = False
        else:
            self.test_mode = opt['test_mode']


       
    def __maskprobe__(self, input, gt, onehot, chrome_pos, gray_pos):
        # input H x W x C
        chrome_begx, chrome_begy, chrome_endx, chrome_endy = chrome_pos
        gray_begx, gray_begy, gray_endx, gray_endy = gray_pos
        h, w, c = input.shape

        scale = 4

        chrome_begx = round(chrome_begx / scale)
        chrome_begy = round(chrome_begy / scale)
        chrome_endx = round(chrome_endx / scale)
        chrome_endy = round(chrome_endy / scale)
        gray_begx = round(gray_begx / scale)
        gray_begy = round(gray_begy / scale)
        gray_endx = round(gray_endx / scale)
        gray_endy = round(gray_endy / scale)

        
        mask = torch.ones((h,w,1))
        mask[chrome_begy: chrome_endy+1, chrome_begx: chrome_endx+1, :] = 0
        mask[gray_begy: gray_endy+1, gray_begx: gray_endx+1, :] = 0

        input[chrome_begy: chrome_endy+1, chrome_begx: chrome_endx+1, :] = -1.
        gt[chrome_begy: chrome_endy+1, chrome_begx: chrome_endx+1, :] = -1.
        input[gray_begy: gray_endy+1, gray_begx: gray_endx+1, :] = -1.
        gt[gray_begy: gray_endy+1, gray_begx: gray_endx+1, :] = -1.

        onehot[chrome_begy: chrome_endy+1, chrome_begx: chrome_endx+1, :] = 0
        onehot[gray_begy: gray_endy+1, gray_begx: gray_endx+1, :] = 0
        
        return mask


    def __getitem__(self, index):
        # Load gt and input images. Dimension order: HWC; channel order: BGR;
        # image range: [-1, 1], float32.

        data_dic = self.paths[index]
        gt_path = data_dic['gt']
        input_path = data_dic['input']
        label_path = data_dic['lab_img']
        
        img_input = imread(input_path)
        img_gt = imread(gt_path)
        label = imread(label_path, False)

        label_tensor = torch.LongTensor(label)
        onehot_tensor = torch.nn.functional.one_hot(label_tensor, 38)

        mask = self.__maskprobe__(img_input, img_gt, onehot_tensor, data_dic['chrome_pos'], data_dic['gray_pos'])

        # crop border
        onehot_tensor = onehot_tensor[4:-4, 6:-6, :]
        mask = mask[4:-4, 6:-6, :]
        img_input = img_input[4:-4, 6:-6, :]
        img_gt = img_gt[4:-4, 6:-6, :]

        scene_name, input_name = input_path.split('/')[-2:]
        gt_name = gt_path.split('/')[-1]
        
        input_path = f'{scene_name}_{input_name}'
        gt_path = f'{scene_name}_{gt_name}'
        
        # augmentation for training
        if self.opt['phase'] == 'train':
            img_gt, img_input, mask, onehot_tensor = augment([img_gt, img_input, mask, onehot_tensor], 
                                    self.opt['use_flip'], self.opt['use_rot'], self.opt.get('use_color'))

        # crop imgs
            if self.opt['gt_size'] > 0:
                (img_gt, img_input, mask, onehot_tensor
                    ), _ =  multi_random_crop([img_gt, img_input, mask, onehot_tensor], self.opt['gt_size'])        

        # BGR to RGB, HWC to CHW, numpy to tensor
        if self.test_mode:
            img_input = img2tensor([img_input],
                                    bgr2rgb=True,
                                    float32=True)
            img_gt = None
            gt_path = None
        else:
            img_gt, img_input = img2tensor([img_gt, img_input],
                                    bgr2rgb=True,
                                    float32=True)
        mask = mask.permute(2, 0, 1)
        onehot_tensor = onehot_tensor.permute(2, 0, 1)
        img_input = torch.cat([img_input, onehot_tensor], axis=0)
        if img_gt != None:
            return {
                'input': img_input,
                'gt': img_gt,
                'mask': mask,
                'input_path': input_path,
                'gt_path': gt_path
            }
        else:
            return {
                'input': img_input,
                'input_path': input_path,
                'mask': mask,
            }

    def __len__(self):
        return len(self.paths)


