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

## Setup
### Datasets
```shell
datasets
└ kitti-stereo-2012
  └ colored_0
  └ colored_1
  └ disp_occ
└ kitti-stereo-2015
  └ image_2
  └ image_3
  └ disp_occ_0
└ middlebury-2014
  └ im0
  └ im1
  └ disp0GT
  └ mask0nocc
```

## References
[1] Chang and Chen, "Pyramid Stereo Matching Network", CVPR, 2018. [[GitHub]](https://github.com/JiaRenChang/PSMNet?tab=readme-ov-file)<br/>
[2] Kim et al., "LAF-Net: Locally Adaptive Fusion Networks for Stereo Confidence Estimation", CVPR, 2019. [[GitHub]](https://github.com/seungryong/LAF)
