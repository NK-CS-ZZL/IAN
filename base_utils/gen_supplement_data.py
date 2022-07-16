import cv2 
from tqdm import tqdm
import numpy as np
import argparse
import os


args = argparse.ArgumentParser(description='Gen_Supplement_Data')
args.add_argument('--save_img_dir', type=str)
args.add_argument('--save_gt_dir', type=str)
args.add_argument('--input_img_dir', type=str)
args.add_argument('--save_depth_dir', type=str)
args.add_argument('--input_depth_dir', type=str)
args = args.parse_args()

def flip_img(input_dir, output_dir, tmplt):
    paths = [x for x in os.listdir(input_dir) if x.endswith(tmplt)]
    for path in tqdm(paths):
        img = cv2.imread(os.path.join(input_dir, path))
        cv2.flip(img, 1, img)
        output_path = os.path.join(output_dir, path[:-len(tmplt)] + '.png')
        cv2.imwrite(output_path, img)

def load_depth(path):
    '''
    params:
        path: str
    output:
        ref_center_dis: float32(maybe useless)
        depth_map: ndarray(float32)
    description:
        load depth info from "path"
    '''
    depth_map = np.load(path, allow_pickle=True).item()
    return depth_map['normalized_depth']

def flip_depth(input_dir, output_dir):
    paths = os.listdir(input_dir)
    paths = [x for x in paths if x.endswith('.npy')]
    for path in tqdm(paths):
        depth = load_depth(os.path.join(input_dir, path))
        depth = np.flip(depth, axis=1)
        depth_dict = dict()
        depth_dict['normalized_depth'] = depth
        np.save(os.path.join(output_dir, path), depth_dict)


if __name__ == '__main__':
    if not os.path.exists(args.save_img_dir):
        os.makedirs(args.save_img_dir)
    if not os.path.exists(args.save_depth_dir):
        os.makedirs(args.save_depth_dir)
    print('\tgenerating images...')
    flip_img(args.input_img_dir, args.save_img_dir, '_6500_N.png')
    print('\tgenerating gt...')
    flip_img(args.input_img_dir, args.save_gt_dir, '_4500_W.png')    
    print('\tgenerating depth maps...')
    flip_depth(args.input_depth_dir, args.save_depth_dir)