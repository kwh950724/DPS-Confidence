# DPS-Confidence
Official PyTorch implementation of "Modeling Stereo-Confidence Out of the End-to-End Stereo-Matching Network via Disparity Plane Sweep", Jae Young Lee*, Woonghyun Ka*, Jaehyun Choi, and Junmo Kim (* equal contribution, alphabetical order), AAAI 2024. [[arXiv]](https://arxiv.org/abs/2401.12001)

## Citation
If you use this implementation for your research, please cite the following paper. 
```shell
@misc{lee2024modeling,
      title={Modeling Stereo-Confidence Out of the End-to-End Stereo-Matching Network via Disparity Plane Sweep}, 
      author={Jae Young Lee and Woonghyun Ka and Jaehyun Choi and Junmo Kim},
      year={2024},
      eprint={2401.12001},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## References
[1] Chang and Chen, "Pyramid Stereo Matching Network", CVPR, 2018. [[GitHub]](https://github.com/JiaRenChang/PSMNet?tab=readme-ov-file)<br/>
[2] Kim et al., "LAF-Net: Locally Adaptive Fusion Networks for Stereo Confidence Estimation", CVPR, 2019. [[GitHub]](https://github.com/seungryong/LAF)

## Setup
### Requirements
Experimental environment is as follows:
```shell
python==3.8.12
torch==1.12.0
cuda==10.2
numpy==1.21.2
opencv-python==4.8.0
scipy==1.9.1
```
You can simply set up the experimental environment by pulling the provided docker image.
```shell
$ docker pull kwh950724/pytorch:aaai24
```

### Datasets
For KITTI Stereo 2012 & 2015 datasets, only training datasets, which provide ground-truth disparity maps, are used.
Also, only data with filename ending in "_10" were used (e.g., 000000_10.png).<br/>

In case of Middlebury 2014 dataset, we used training dataset with quarter resolution.<br/>

Dataset directory structure should be as follows:
```shell
datasets
   ├ kitti-stereo-2012
   │   ├ colored_0
   │   ├ colored_1
   │   └ disp_occ
   ├ kitti-stereo-2015
   │   ├ image_2
   │   ├ image_3
   │   └ disp_occ_0
   └ middlebury-2014
       ├ im0
       ├ im1
       ├ disp0GT
       └ mask0nocc
```

### Reproducing Experimental Results
You can easily reproduce the experimental results of $\mathrm{Ours}$, $\mathrm{LAF^{*\dagger}}$, and $\mathrm{LAF^{\dagger}}$ with $\mathrm{PSMNet}$.<br/>

First, to evaluate confidence maps obtained from the proposed method ($\mathrm{Ours}$), run:
```shell
python main.py --dataset_type ["kitti2012" or "kitti2015" or "mid2014"] \
               --dataset_dir [dataset directory path]
```
Then, you can train $\mathrm{LAF^{*\dagger}}$ and $\mathrm{LAF^{\dagger}}$ using predicted disparity maps, confidence maps, and matching cost through the above process as follows:
```shell
python LAFNet/train.py --modal_type conf
                       --use_cost (optional)
```
To evaluate the trained model, run:
```shell
python LAFNet/evaluate.py --dataset_type ["kitti2012" or "kitti2015" or "mid2014"] \
                          --dataset_dir [dataset directory path] \
                          --weights_path [weight file (.pth) path] \
                          --modal_type conf
                          --use_cost (optional)
```
Pre-trained weights for $\mathrm{LAF^{*\dagger}}$ and $\mathrm{LAF^{\dagger}}$ can be found in `LAFNet/pretrained`.
