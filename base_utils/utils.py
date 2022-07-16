import numpy as np
import cv2
import random
import torch
import time
import os
from PIL import Image
from tqdm import tqdm

def get_time_str():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())


def set_random_seed(seed):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_depth(path):
    '''
    params:
        path: str
    output:
        ref_center_dis: float32(maybe useless)
        depth_map: ndarray(float32) [-1, 1]
    description:
        load depth info from "path"
    '''
    depth_map = np.load(path, allow_pickle=True)
    if len(depth_map.shape) != 0:
        return depth_map * 2 - 1
    return depth_map.item()['normalized_depth'] * 2 - 1


def stitch_images(inputs, *outputs, img_per_row=2):
    gap = 5
    columns = len(outputs) + 1

    width, height = inputs[0][:, :, 0].shape
    img = Image.new('RGB', (width * img_per_row * columns + gap * (img_per_row - 1), height * int(len(inputs) / img_per_row)))
    images = [inputs, *outputs]

    for ix in range(len(inputs)):
        xoffset = int(ix % img_per_row) * width * columns + int(ix % img_per_row) * gap
        yoffset = int(ix / img_per_row) * height

        for cat in range(len(images)):
            im = np.array((images[cat][ix]).cpu()).astype(np.uint8).squeeze()
            im = Image.fromarray(im)
            img.paste(im, (xoffset + cat * width, yoffset))

    return img


def mkdir_and_rename(path):
    """mkdirs. If path exists, rename it with timestamp and create a new one.
    Args:
        path (str): Folder path.
    """
    if os.path.exists(path):
        new_name = path + '_archived_' + get_time_str()
        print(f'Path already exists. Rename it to {new_name}', flush=True)
        os.rename(path, new_name)
    os.makedirs(path, exist_ok=True)


def make_exp_dirs(opt):
    print('mkdir')
    """Make dirs for experiments."""
    path_opt = opt['path'].copy()
    if opt['is_train']:
        mkdir_and_rename(path_opt.pop('experiments_root'))
    else:
        mkdir_and_rename(path_opt.pop('results_root'))
    print(path_opt)
    for key, path in path_opt.items():
        print(path)
        if ('strict_load' not in key) and ('pretrain_network'
                                           not in key) and ('resume'
                                                            not in key):
            os.makedirs(path, exist_ok=True)


def check_resume(opt, resume_iter):
    """Check resume states and pretrain_network paths.
    Args:
        opt (dict): Options.
        resume_iter (int): Resume iteration.
    """
    logger = get_root_logger()
    if opt['path']['resume_state']:
        # get all the networks
        networks = [key for key in opt.keys() if key.startswith('network_')]
        flag_pretrain = False
        for network in networks:
            if opt['path'].get(f'pretrain_{network}') is not None:
                flag_pretrain = True
        if flag_pretrain:
            logger.warning(
                'pretrain_network path will be ignored during resuming.')
        # set pretrained model paths
        for network in networks:
            name = f'pretrain_{network}'
            basename = network.replace('network_', '')
            if opt['path'].get('ignore_resume_networks') is None or (
                    basename not in opt['path']['ignore_resume_networks']):
                opt['path'][name] = os.path.join(
                    opt['path']['models'], f'net_{basename}_{resume_iter}.pth')
                logger.info(f"Set {name} to {opt['path'][name]}")


def gen_depth_img(depth_map, save_path)->None:
    '''
    params:
        depth_map: ndarray(float32)
        save_path: str
    description:
        convert a depth_map to a grayscale image(dark->bright, near->remote)
        and save it to "save_dir"
    '''
    depth_img = depth_map * 255.
    depth_img = depth_img.astype(np.uint8)
    cv2.imwrite(save_path, depth_img)


def cal_normal(d_im):
    '''
    params: 
        d_im: ndarray(float32)
    description:
        convert depth_map to surface normals map(input needs to multiply 2^16-1 before cal_normal)
    '''
    zy, zx = np.gradient(d_im)
    normal = np.dstack((-zx, -zy, np.ones_like(d_im)))
    n = np.linalg.norm(normal, axis=2)
    normal[:, :, 0] /= n
    normal[:, :, 1] /= n
    normal[:, :, 2] /= n
    return normal


def col_stitch(imgs):
    return np.concatenate(imgs, axis=1)


def stitch_images(inputs, *outputs, img_per_row=2):
    gap = 5
    columns = len(outputs) + 1

    width, height = inputs[0][:, :, 0].shape
    img = Image.new('RGB', (width * img_per_row * columns + gap * (img_per_row - 1), height * int(len(inputs) / img_per_row)))
    images = [inputs, *outputs]

    for ix in range(len(inputs)):
        xoffset = int(ix % img_per_row) * width * columns + int(ix % img_per_row) * gap
        yoffset = int(ix / img_per_row) * height

        for cat in range(len(images)):
            im = np.array((images[cat][ix]).cpu()).astype(np.uint8).squeeze()
            im = Image.fromarray(im)
            img.paste(im, (xoffset + cat * width, yoffset))

    return img


def get_mask(input_dir, output_dir):
    '''
    params:
        input_dir(str): depth map directory
        output_dir(str): saving directory
    output:
        0-1 mask indicates where is valid
    '''
    input_paths = os.listdir(input_dir)
    for path in tqdm(input_paths):
        full_path = os.path.join(input_dir, path)
        depth = load_depth(full_path)
        depth[depth>0] = 1
        depth.astype(np.uint8)
        np.save(os.path.join(output_dir, path[:-4]+'.npy'), depth)



# def find_ref(depth, surface_normal, input, target):
def cal_psnr_ssim(input, gt):
    from skimage.measure import compare_psnr, compare_ssim
    return compare_psnr(input, gt), compare_ssim(input, gt, multichannel=True)
    # print(f'psnr: {}, ssim: {}')



if __name__ == '__main__':
    x_code = np.array([float(x)/1024 for x in range(1024)])
    x_code = x_code[np.newaxis, :]
    x_code = np.concatenate([x_code]*1024, 0)
    from cv2 import imread, imwrite

    heat_map = paint_heatmap(imread('/home/paper99/media/Codes/NTIRE2021/results/HGNet_pos_enc/visualization/VIDIT/Image311_HGNet_pos_enc.png'), x_code)
    imwrite('./test.png', heat_map)    

    # import PIL.Image as Image
    # from imageio import imread
    # inputdir1 = '/home/paper99/media/Codes/Relighting-With-Depth/results/largedataDepthSurfacePyrSFTWDRN_L1/visualization/VIDIT/'
    # suffix1 = '_largedataDepthSurfacePyrSFTWDRN_L1.png'
    # inputdir2 = '/home/paper99/media/Codes/NTIRE2021/results/HGNet/visualization/VIDIT/'
    # suffix2 = '_HGNet.png'
    # gt_dir = '/media/ssd/CVPRW/VIDIT/validation/target/'
    # print('WDRN\tHGNet')
    # for i in range(300, 345):
    #     print(f'Image{i}:')
    #     input1 = imread(inputdir1+f'Image{i}'+suffix1)
    #     input2 = imread(inputdir2+f'Image{i}'+suffix2)
    #     gt = imread(gt_dir+f'Image{i}'+'.png')[:, :, :3]
    #     psnr1, ssim1 = cal_psnr_ssim(input1, gt)
    #     psnr2, ssim2 = cal_psnr_ssim(input2, gt)
    #     print(f'psnr: {psnr1}, {psnr2}')
    #     print(f'ssim: {ssim1}, {ssim2}')
    # # get_mask('/home/paper99/media/Datasets/VIDIT/unzip_any2any/depth', '/home/paper99/media/Datasets/VIDIT/unzip_any2any/mask')
    # # flip_img('/home/paper99/media/Datasets/VIDIT/unzip_any2any/raw_input', '/home/paper99/media/Datasets/VIDIT/unzip_any2any/target', '_4500_W.png')
    # # flip_depth('/home/paper99/media/Datasets/VIDIT/unzip_any2any/raw_depth', '/home/paper99/media/Datasets/VIDIT/unzip_any2any/depth')

        

