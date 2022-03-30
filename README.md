# BiAlignNet
This repository contains the source code of BiAlignNet (Bidirectional Alignment Network) from the paper **FAST AND ACCURATE SCENE PARSING VIA BI-DIRECTION ALIGNMENT NETWORKS**, ICIP 2021. 


## Instructions
Please follow the instructions to run the code.

[BialignNet-Dfnet2](https://drive.google.com/file/d/1lToCzFmq3TGP_bkq8hLuI9ZWZahSX9ft/view?usp=sharing): mIoU 78.7

### Prerequisite
1. Read the DATASET.md to prepare the dataset you want to use (in this repo, we use cityscapes)
2. Set the dataset path in `config.py`
3. In this directory, `mkdir pretrained_models`
4. Download the pretrained [dfnet1](https://drive.google.com/file/d/1xkkmIjKUbMifcrKdWU7I_-Jx_1YQAXfN/view?usp=sharing) and [dfnet2](https://drive.google.com/file/d/1ZRRE99BPhbXwq-ZzO8A5GFmfCe7zxMsz/view?usp=sharing) and save them into `pretrained_models` directory

### Training
`sh scripts/train/train_cityscapes_bialign_dfnet2.sh`

### Evaluation in validation set
`sh scripts/evaluate_val/eval_cityscapes_bialign_dfnet2.sh /path/to/model /path/to/where/you/want/to/save/results`

### Submitting the test set result
`sh scripts/submit_test/submit_cityscapes_bialign.sh /path/to/model /path/to/where/you/want/to/save/results`


## Citation
```
@inproceedings{wu2021fast,
  title={Fast and Accurate Scene Parsing via Bi-Direction Alignment Networks},
  author={Wu, Yanran and Li, Xiangtai and Shi, Chen and Tong, Yunhai and Hua, Yang and Song, Tao and Ma, Ruhui and Guan, Haibing},
  booktitle={2021 IEEE International Conference on Image Processing (ICIP)},
  pages={2508--2512},
  year={2021},
  organization={IEEE}
}
```

## Acknowledgement
This repo is based on Semantic Segmentation from [SFSegNets](https://github.com/lxtGH/SFSegNets).
