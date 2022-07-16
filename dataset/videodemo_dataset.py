import math
import torch
import numpy as np
from torch.utils import data as data 
from dataset.data_utils import img2tensor, imread

def load_light(path):
    with open(path, 'r') as fr:
        lines = fr.readlines()
    light_params = np.array([float(l[:-1]) for l in lines])
    return light_params

def get_pos(l):
    def pos(x):
        if np.abs(x) <= l:
            return l - x
        else:
            return l
    return pos

def rot_sh(sh, degree):
    angle = degree / 180. * math.pi
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    rotmat0 = np.array([[1.]])
    rotmat1 = np.array(
        [[1., 0., 0.],
        [0, cos_a, sin_a],
        [0, -sin_a, cos_a]]
    )
    # 1, 0, -1 -> 0, 1, 2
    # 2 1 0 -1, -2 -> 0, 1, 2, 3, 4
    # (row, col)
    rotmat2 = np.zeros((5, 5))
    # corner (2, 2)
    rotmat2[0][0] = 0.5 * cos_a * rotmat1[0][0] + 0.5 * rotmat1[2][2]
    # corner (-2, -2)
    rotmat2[4][4] = 0.5 * rotmat1[0][0] + 0.5 * cos_a * rotmat1[2][2]
    
    l = 2
    pos2 = get_pos(l)
    pos1 = get_pos(l-1)
    pos0 = get_pos(l-2)

    for m in range(0, l):
        coff = np.sqrt(l*(l-0.5)/(l**2 - m**2))
        
        rotmat2[pos2(m)][pos2(l)] = coff * sin_a * rotmat1[pos1(m)][pos1(l-1)]
        rotmat2[pos2(-m)][pos2(-l)] = coff * sin_a * rotmat1[pos1(-m)][pos1(-l+1)]

        rotmat2[pos2(l)][pos2(m)] = (-1)**(m-l)*rotmat2[pos2(m)][pos2(l)]
        rotmat2[pos2(-l)][pos2(-m)] = (-1)**(l-m)*rotmat2[pos2(-m)][pos2(-l)]

    for m in range(0, l):
        for n in range(0, l):
            coff0 = l*(2*l-1)/np.sqrt((l**2-m**2)*(l**2-n**2))
            coff1 = m*n / (l*(l-1))
            coff2 =  ((l-1)**2-m**2)*((l-1)**2-n**2) / ((l-1)*(2*l-1)) 
            rotmat2[pos2(m)][pos2(n)] = coff0 * \
                                        (cos_a * rotmat1[pos1(m)][pos1(n)] \
                                        - coff1 * rotmat1[pos1(-m)][pos1(-n)] \
                                        - coff2 * rotmat0[pos0(m)][pos0(n)])

            rotmat2[pos2(n)][pos2(m)] = (-1)**(m-n)*rotmat2[pos2(m)][pos2(n)]
            rotmat2[pos2(-m)][pos2(-n)] = coff0 * \
                                        (cos_a * rotmat1[pos1(-m)][pos1(-n)] \
                                        - coff1 * rotmat1[pos1(m)][pos1(n)] \
                                        - coff2 * rotmat0[pos0(-m)][pos0(-n)])
            rotmat2[pos2(-n)][pos2(-m)] = (-1)**(n-m)*rotmat2[pos2(-m)][pos2(-n)]
            
    new_sh = np.zeros_like(sh)
    new_sh[:1] = sh[:1]
    new_sh[1:4] = np.matmul(rotmat1, sh[1:4])
    new_sh[4:] = np.matmul(rotmat2, sh[4:])
    return new_sh

class VideoDemoDataset(data.Dataset):
    def __init__(self, opt):
        super(VideoDemoDataset, self).__init__()
        self.opt = opt
        self.stride = opt['stride']
        self.frame_num = opt['frames']
        self.init_sh = load_light(opt['shroot'])# * 0.7
        self.demo_img = imread(opt['img_path'])
        self.normal = imread(opt['normal_path'])
        self.demo_img, self.normal = img2tensor([self.demo_img, self.normal],
                                    bgr2rgb=True,
                                    float32=True)
        self.input = torch.cat([self.demo_img, self.normal], axis=0)
       


    def __getitem__(self, index):
        # Load gt and input images. Dimension order: HWC; channel order: BGR;
        # image range: [-1, 1], float32.
        angle = index * self.stride
        curr_sh = torch.tensor(rot_sh(self.init_sh, angle)).float()

        return {
            'input': self.input,
            'input_light': self.init_sh,
            'gt': self.input[:3,:,:],
            'gt_light': curr_sh,
            'input_path': f'{index}.png',
            'gt_path': f'{index}.png'
        }
        

    def __len__(self):
        return self.frame_num




