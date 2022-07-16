from dataset.data_utils import paired_paths_from_folder, img2tensor, imread
from dataset.transforms import augment, paired_random_crop
from torch.utils import data as data
# from data_utils import paired_paths_from_folder, img2tensor, imread
# from transforms import augment, paired_random_crop
# from torch.utils import data as data

class PairedImageDataset(data.Dataset):
    """Paired image dataset for image restoration.
    Read input  and GT image pairs.
    params:
        opt: dict (Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_input (str): Data root path for input.
            gt_size (int): Cropped patched size for gt patches, enabled when it's greater than zero.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation)
            )
            scale: bool (Scale, which will be added automatically)
            phase: str ('train' or 'val')
    """

    def __init__(self, opt):
        super(PairedImageDataset, self).__init__()
        self.opt = opt
        self.paths = []
        if opt.get('dataroot_pinput') == None:
            opt['dataroot_pinput'] = opt['dataroot_input']

        if isinstance(opt['dataroot_pinput'], list) == False:
            opt['dataroot_pinput'] = [opt['dataroot_pinput']]
        
        for i in range(len(opt['dataroot_pinput'])):
            self.gt_folder, self.input_folder = opt['dataroot_gt'][i], opt['dataroot_pinput'][i]
            if 'filename_tmpl' in opt:
                self.filename_tmpl = opt['filename_tmpl']
            else:
                self.filename_tmpl = '{}'


            self.paths += paired_paths_from_folder(
                [self.input_folder, self.gt_folder], ['input', 'gt'],
                self.filename_tmpl)
            # print(self.paths)
       

    def __getitem__(self, index):
        scale = self.opt['scale']

        # Load gt and input images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_gt = imread(gt_path)
        input_path = self.paths[index]['input_path']
        img_input = imread(input_path)

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop while gt_size > 0
            if gt_size > 0:
                img_gt, img_input = paired_random_crop(img_gt, img_input, gt_size, scale,
                                                gt_path)
            # flip, rotation
            img_gt, img_input = augment([img_gt, img_input], self.opt['use_flip'],
                                     self.opt['use_rot'])


        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_input = img2tensor([img_gt, img_input],
                                    bgr2rgb=True,
                                    float32=True)

        return {
            'input': img_input,
            'gt': img_gt,
            'input_path': input_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)


