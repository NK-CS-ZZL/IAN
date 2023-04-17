# IAN (TIP 2022)
## Introduction
This repository is the official implementation of **Designing An Illumination-Aware Network for Deep Image Relighting**. [[Paper]](https://arxiv.org/abs/2207.10582) <a href="#demos">[Demos]</a>
> Designing An Illumination-Aware Network for Deep Image Relighting
> 
> [Zuo-Liang Zhu](https://github.com/NK-CS-ZZL), [Zhen Li](https://paper99.github.io/), [Rui-Xun Zhang](https://www.math.pku.edu.cn/teachers/ZhangRuixun%20/index.html), [Chun-Le Guo](https://mmcheng.net/clguo/), [Ming-Ming Cheng](https://mmcheng.net/)
>
> IEEE Transactions on Image Processing, 2022


<div align="center">
    <img src="https://user-images.githubusercontent.com/50139523/179220047-1b3256f3-b7eb-4943-928d-6a4de9803726.gif" width="384" height="256">
</div>


## Data preparation
### Datasets
+ VIDIT dataset [[Paper](https://arxiv.org/abs/2005.05460)] [[Download](https://github.com/majedelhelou/VIDIT)]
+ Multi-Illumination dataset [[Paper](https://arxiv.org/abs/1910.08131)] [[Download](https://projects.csail.mit.edu/illumination/)]
+ DPR dataset [[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhou_Deep_Single-Image_Portrait_Relighting_ICCV_2019_paper.pdf)] [[Download](https://drive.google.com/drive/folders/10luekF8vV5vo2GFYPRCe9Rm2Xy2DwHkT?usp=sharing)]
### Normal generation on the VIDIT dataset
+ Place the one2one training data into folders `./data/one2one/train/depth`, `./data/one2one/train/input`, `./data/one2one/train/target`
+ Place the any2any training data into folders `./data/any2any/train/depth` (all '.npy' files), `./data/any2any/train/input` (all RGB images)
+ Place the one2one validation data into folders `./data/validation/train/depth`, `./data/validation/train/input`, `./data/validation/train/target`
+ Run `gen_train_data.sh` to obtain full training and validation data.

## Quick Demo
+ Create the environment by `conda env create -f environment.yml`
+ Download the pretrained model on DPR dataset from the [link](https://drive.google.com/drive/folders/1oSvs6YEHdcc3-6xKeCy25uaB4FYxKdrx?usp=sharing) and place them into the folder 'pretrained'.
+ Run `python test.py -opt options/videodemo_opt.yml`.
+ Image results will be save in the folder `results`.
+ You can further utilize the `ffmpeg` to generate demo videos as `ffmpeg -f image2 -i [path_to_results] -vcodec libx264 -r 10 demo.mp4`.

## Train
```
    python train.py -opt [training config]
``` 
| Dataset            | Guidance | Config          | 
| -----------        | ----------- |      ---        | 
| VIDIT              | depth, normal, lpe*           | `options/train_opt4b.yml` | 
| Multi-Illumination | :x:           | `options/train_adobe_opt.yml` |
| DPR                | normal, lpe           | `options/trainany_opt4b.yml`  |   
| DPR                | :x:           | `options/trainany_opt4b_woaux.yml` | 

\* The `lpe' represents our proposed linear positional encoding. 
## Test
```
    python test.py -opt [testing config]
```
| Dataset            | Guidance | Config          | Pretrained |
| -----------        | ----------- |      ---        | --- | 
| VIDIT              | depth, normal, lpe           | `options/valid_opt.yml` | `pretrained/VIDITOne2One.pth` | 
| Multi-Illumination | :x:           | `options/valid_adobe_opt.yml` |`pretrained/MutliIllumination.pth` | 
| DPR                | normal, lpe           | `options/vaild_any_opt.yml`     | `pretrained/PortraitWithNormal.pth` | 
| DPR                | :x:           | `options/vaild_any_opt.yml`     | `pretrained/PortraitWithoutNormal.pth` |

You can download all pretrained models from this [Google Driver](https://drive.google.com/drive/folders/1oSvs6YEHdcc3-6xKeCy25uaB4FYxKdrx?usp=sharing) or [BaiduNetDisk](https://pan.baidu.com/s/1EvNlGgwBIe3tUYAml-33zw?pwd=5qtp) (pwd: 5qtp).

## Citation
```
 @article{zhu2022ian,
    author = {Zuo-Liang Zhu, Zhen Li, Rui-Xun Zhang, Chun-Le Guo, Ming-Ming Cheng},
    title = {Designing An Illumination-Aware Network for Deep Image Relighting},
    journal = {IEEE Transactions on Image Processing},
    year = {2022},
    doi = {10.1109/TIP.2022.3195366}
}
```
## Acknowledge
+ This repository is maintained by Zuo-Liang Zhu (`nkuzhuzl [AT] gmail.com`) and Zhen Li (`zhenli1031 [AT] gmail.com`).
+ Our code is based on a famous restoration toolbox [BasicSR](https://github.com/XPixelGroup/BasicSR).

## LICENSE
The code is released under [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) for non-commercial use only. Any commercial use should get formal permission first.


## References
+ AIM 2020: Scene Relighting and Illumination Estimation Challenge [[Webpage](https://competitions.codalab.org/competitions/24671#results)] [[Paper](https://arxiv.org/abs/2009.12798v1)]
+ NTIRE 2021 Depth Guided Image Relighting Challenge [[Webpage](https://competitions.codalab.org/competitions/28030#results)] [[Paper](https://arxiv.org/abs/2104.13365)]
+ Deep Single Portrait Image Relighting [[Github](https://github.com/zhhoper/DPR)] [[Paper](https://zhhoper.github.io/paper/zhou_ICCV2019_DPR.pdf)] [[Supp](https://zhhoper.github.io/paper/zhou_ICCV_2019_DPR_sup.pdf)]
+ Multi-modal Bifurcated Network for Depth Guided Image Relighting [[Github](https://github.com/weitingchen83/NTIRE2021-Depth-Guided-Image-Relighting-MBNet)] [[Paper](https://openaccess.thecvf.com/content/CVPR2021W/NTIRE/papers/Yang_Multi-Modal_Bifurcated_Network_for_Depth_Guided_Image_Relighting_CVPRW_2021_paper.pdf)]
+ Physically Inspired Dense Fusion Networks for Relighting [[Paper](https://arxiv.org/abs/2105.02209)]
+ LPIPS [[Github]](https://github.com/S-aiueo32/lpips-pytorch) [[Paper]](https://arxiv.org/abs/1801.03924)

<h2 id="demos"> More demos </h2>

https://user-images.githubusercontent.com/50139523/179356102-14bac41f-7caf-409c-b1c7-75eb315ef881.mp4


https://user-images.githubusercontent.com/50139523/179357378-ae446399-02c4-45a8-8223-66cc480d1fc9.mp4


https://user-images.githubusercontent.com/50139523/179356168-dab380d8-b844-45c1-a121-ef64233346d4.mp4


https://user-images.githubusercontent.com/50139523/179356170-69b23de2-911b-45f4-bd19-0e7fbf748feb.mp4


