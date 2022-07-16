import cv2
import numpy as np
import os
import math
from torch.utils import data
from torchvision.utils import make_grid
import torch
import json

def imread(path, float32=True):
    '''
    params:
        path: str
        float32: bool
    output:
        img: ndarray(uint8/float32) [-1, 1]
    description:
        read a image from "path". convert and normalize it when float32 is Ture.
    CAUTION:
        image format: HWC BGR
    '''
    
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if len(img.shape) == 3:
        img = img[:, : ,:3]
    if float32:
        img = img.astype(np.float32) / 255 * 2 - 1
    return img


def imwrite(img, save_path, mkdir=True):
    '''
    params:
        img: ndarray(uint8)
        save_path: str
        mkdir: bool
    description:
        write a image to "save_path". when "mkdir"==True && "save_path" doesn't
        exist, make a new directory.
    '''
    dir_name = os.path.abspath(os.path.dirname(save_path))
        
    if os.path.exists(dir_name) == False and mkdir:
        os.makedirs(dir_name, exist_ok=True)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(save_path, img)


def crop_border(imgs, crop_border):
    """
    params:
        imgs: list[ndarray] | ndarray (Images with shape (h, w, c))
        crop_border: int
    output:
        imgs: list[ndarray]
    description:
        crop borders of images.
    """
    if crop_border == 0:
        return imgs
    else:
        if isinstance(imgs, list):
            return [
                v[crop_border:-crop_border, crop_border:-crop_border, ...]
                for v in imgs
            ]
        else:
            return imgs[crop_border:-crop_border, crop_border:-crop_border,
                        ...]


def img2tensor(imgs, bgr2rgb=True, float32=True):
    '''
    params:
        imgs: list[ndarray] | ndarray.
        bgr2rgb: bool (Whether to change bgr to rgb)
        float32: bool (Whether to change to float32)
    output:
        list[tensor] | tensor
    description:
        numpy array to torch Tensor
    '''
    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays.
    After clamping to [min, max], values will be normalized to [0, 1].
    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.
    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list)
             and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(
            f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        
        if n_dim == 4:
            img_np = make_grid(
                _tensor, nrow=int(math.sqrt(_tensor.size(0))),
                normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError('Only support 4D, 3D or 2D tensor. '
                            f'But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result


def feat2img(tensor, out_type=np.uint8, min_max=(-1, 1)):
    
    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    
    for _tensor in tensor:
        # _tensor = _tensor.sum().squeeze()
        _tensor = _tensor.squeeze().float().detach().cpu()
        max_t = _tensor.max()
        print(max_t)
        min_t = _tensor.min()
        print(min_t)
        _tensor = (_tensor - min_t) / (max_t - min_t)
        img_np = _tensor.numpy()
        img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result


def paired_paths_from_folder(folders, keys, filename_tmpl):
    '''
    params:
        folders: list[str] )A list of folder path. The order of list should
            be [input_folder, gt_folder])
        keys: list[str] (A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['input', 'gt'])
        filename_tmpl: str (Template for each filename. Note that the
            template excludes the file extension. Usually the filename_tmpl is
            for files in the input folder)
    output:
        list[str] (returned path list).
    description
        generate paired paths from folders.
    '''
    # assert len(folders) == 2, (
    #     'The len of folders should be 2 with [input_folder, gt_folder]. '
    #     f'But got {len(folders)}')
    # assert len(keys) == 2, (
    #     'The len of keys should be 2 with [input_key, gt_key]. '
    #     f'But got {len(keys)}')
    enable_aux = False
    input_folder, gt_folder = folders[:2]
    if len(keys) == 3:
        input_key, gt_key, aux_key = keys
        enable_aux = True
    elif len(keys) == 2:
        input_key, gt_key = keys

    input_paths = list(os.listdir(input_folder))
    if gt_folder != None:
        gt_paths = list(os.listdir(gt_folder))
    if enable_aux:
        aux_dirs = folders[2]
    paths = []

    for input_path in input_paths:
        basename, ext = os.path.splitext(os.path.basename(input_path))
        input_name = f'{filename_tmpl.format(basename)}{ext}'
        input_path = os.path.join(input_folder, input_name)
        if gt_folder != None:
            gt_path = os.path.join(gt_folder, input_name)
        if enable_aux:
            aux_paths = [os.path.join(x, basename + '.npy') for x in aux_dirs]
        else:
            aux_paths = []
        assert input_name in input_paths, (f'{input_name} is not in '
                                            f'{input_key}_paths.')
        if gt_folder == None:
            gt_path = ''
            gt_key = 'nogt'
        paths.append(
            dict([(f'{input_key}_path', input_path),
                  (f'{gt_key}_path', gt_path),
                  ('aux_paths', aux_paths)
                  ]
            )
        )
    return paths


def parse_adobe_dataset(root, with_label=False):
    scenes = os.listdir(root)
    results = []
    for scene in scenes:
        scene = os.path.join(root, scene)
        imgs = [os.path.join(scene, x) for x in os.listdir(scene) if x.endswith('.jpg')]
        probes = [os.path.join(scene, 'probes', x) for x in os.listdir(f'{scene}/probes')]
        seg_img = os.path.join(scene, 'materials_mip2.png')
        info = os.path.join(scene, 'meta.json')
        lab_img = os.path.join(scene, 'label_map.png')
        chrome_pos, gray_pos = parse_meta_json(info)
        if with_label:
            results.append({'imgs': imgs, 'probes': probes, 'seg_img': seg_img, 'info': info, 
                        'chrome_pos': chrome_pos, 'gray_pos': gray_pos, 'lab_img':lab_img})
        else:
            results.append({'imgs': imgs, 'probes': probes, 'seg_img': seg_img, 'info': info, 
                        'chrome_pos': chrome_pos, 'gray_pos': gray_pos})
    return results


# Image name format: {notation of scene}/dir_{light_dir}_mip2.jpg
# other component: probes, materials_mip2.png, meta.json, thumb.jpg
def select_one2one_data(parsed_data, input_dir, gt_dir):
    for data_dic in parsed_data:
        imgs = data_dic.pop('imgs')
        data_root = os.path.split(imgs[0])[0]
        input_name = f'dir_{input_dir}_mip2.jpg'
        gt_name = f'dir_{gt_dir}_mip2.jpg'

        data_dic.pop('probes')
        input_gray_probe = os.path.join(data_root, 'probes', f'dir_{input_dir}_gray256.jpg')
        input_chrome_probe = os.path.join(data_root, 'probes', f'dir_{input_dir}_chrome256.jpg')
        gt_gray_probe = os.path.join(data_root, 'probes', f'dir_{gt_dir}_gray256.jpg')
        gt_chrome_probe = os.path.join(data_root, 'probes', f'dir_{gt_dir}_chrome256.jpg')

        
        data_dic['input'] = os.path.join(data_root, input_name)
        data_dic['gt'] = os.path.join(data_root, gt_name)
        data_dic['input_gray_probe'] = input_gray_probe
        data_dic['gt_gray_probe'] = gt_gray_probe
        data_dic['input_chrome_probe'] = input_chrome_probe
        data_dic['gt_chrome_probe'] = gt_chrome_probe


    return parsed_data

def parse_meta_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        data = json.loads(''.join(lines))

    chrome_begx = data['chrome']['bounding_box']['x']
    chrome_begy = data['chrome']['bounding_box']['y']
    chrome_endx = chrome_begx + data['chrome']['bounding_box']['w']
    chrome_endy = chrome_begy + data['chrome']['bounding_box']['h']

    gray_begx = data['gray']['bounding_box']['x']
    gray_begy = data['gray']['bounding_box']['y']
    gray_endx = gray_begx + data['gray']['bounding_box']['w']
    gray_endy = gray_begy + data['gray']['bounding_box']['h']

    return (chrome_begx, chrome_begy, chrome_endx, chrome_endy), (gray_begx, gray_begy, gray_endx, gray_endy)



