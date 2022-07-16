import numpy as np
import cv2
import random
import torch
import os
from tqdm import tqdm
import argparse
# from base_utils.utils import load_depth

# args = argparse.ArgumentParser(description='Gen_Surface_Normal')
# args.add_argument('--save_dir', type=str)
# args.add_argument('--input_dir', type=str)
# args = args.parse_args()
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


def cal_normal(d_im):
    d_im = d_im.astype(np.float32)
    # print(type(d_im[0][0]))
    # d_im = cv2.bilateralFilter(d_im, 9, 40, 40)
    zy, zx = np.gradient(d_im, 2)
    normal = np.dstack((-zx, -zy, np.ones_like(d_im)))
    n = np.linalg.norm(normal, axis=2)
    normal[:, :, 0] /= n
    normal[:, :, 1] /= n
    normal[:, :, 2] /= n
    
    return normal
    
if __name__ == '__main__':

    # input_dir = args.input_dir
    input_dir = "data/supplement/train/depth"
    output_dir = 'data/supplement/train/normals'
    # output_dir = args.save_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    def cal_normals(path):
        depth_path = os.path.join(input_dir, path)
        depth_map = load_depth(depth_path) 
        # depth_map_vis = ((depth_map+ 1) / 2 cd  * 255).astype(np.uint8)
        # cv2.imwrite(f'/home/vm411/Codes/NTIRE2021/base_utils/depth_vis/{path[:-4]+".png"}', depth_map_vis)

        depth_map = (load_depth(depth_path) + 1) / 2 * 65536
        normal = cal_normal(depth_map)
        np.save(os.path.join(output_dir, path), normal)


        # normal_vis = normal * 127.5 + 127.5
        # normal_vis = normal_vis.astype(np.uint8)
        # print(f"./normal_vis/{path[:-4]+'.png'}")
        # print(cv2.imwrite(f"/home/vm411/Codes/NTIRE2021/base_utils/normals_vis/{path[:-4]+'.png'}", normal_vis))
        

    depth_paths = os.listdir(input_dir)
    import multiprocessing
    
    p = multiprocessing.Pool(8)
    for _ in tqdm(p.imap(cal_normals, depth_paths), total=len(depth_paths)):
        continue
    p.close()
    p.join()

