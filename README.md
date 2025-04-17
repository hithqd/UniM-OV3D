# [IJCAI 2024] UniM-OV3D: Uni-Modality Open-Vocabulary 3D Scene Understanding with Fine-Grained Feature Representation

<!-- <br> -->
[Qingdong He<sup>1</sup>](https://scholar.google.com/citations?user=gUJWww0AAAAJ&hl=zh-CN), [Jinlong Peng<sup>1</sup>](https://scholar.google.com/citations?user=i5I-cIEAAAAJ&hl=zh-CN), [Zhengkai Jiang<sup>1</sup>](https://scholar.google.com/citations?user=ooBQi6EAAAAJ&hl=zh-CN), [Kai Wu<sup>1</sup>](https://scholar.google.com/citations?hl=zh-CN&user=ElfT3eoAAAAJ), [Xiaozhong Ji<sup>1</sup>](https://scholar.google.com/citations?user=iL2j_yAAAAAJ&hl=zh-CN&oi=ao), [Jiangning Zhang<sup>1</sup>](https://zhangzjn.github.io/), [Yabiao Wang<sup>1</sup>](https://scholar.google.com/citations?hl=zh-CN&user=xiK4nFUAAAAJ), [Chengjie Wang<sup>1</sup>](https://scholar.google.com/citations?hl=zh-CN&user=fqte5H4AAAAJ), [Mingang Chen<sup>2</sup>](https://scholar.google.com/citations?user=FBfC58EAAAAJ&hl=zh-CN&oi=ao), Yunsheng Wu<sup>1</sup>. 
<!-- <br> -->

<sup>1</sup>Youtu Lab, Tencent,
<sup>2</sup>Shanghai Development Center of Computer Software Technology

[![arXiv](https://img.shields.io/badge/arXiv-2312.05767-b31b1b.svg)](https://arxiv.org/abs/2401.11395)

![Image description](https://github.com/hithqd/UniM-OV3D/blob/main/docs/framework.png)

3D open-vocabulary scene understanding aims to recognize arbitrary novel categories beyond the base label space. However, existing works not only fail to fully utilize all the available modal information in the 3D domain but also lack sufficient granularity in representing the features of each modality. In this paper, we propose a unified multimodal 3D open-vocabulary scene understanding network, namely UniM-OV3D, which aligns point clouds with image, language and depth. To better integrate global and local features of the point clouds, we design a hierarchical point cloud feature extraction module that learns comprehensive fine-grained feature representations. Further, to facilitate the learning of coarse-to-fine point-semantic representations from captions, we propose the utilization of hierarchical 3D caption pairs, capitalizing on geometric constraints across various viewpoints of 3D scenes. Extensive experimental results demonstrate the effectiveness and superiority of our method in open-vocabulary semantic and instance segmentation, which achieves state-of-the-art performance on both indoor and outdoor benchmarks such as ScanNet, ScanNet200, S3IDS and nuScenes.

# Requirements
All the codes are tested in the following environment:
- Python 3.7+
- PyTorch 1.8
- CUDA 11.1
- [spconv v2.x](https://github.com/traveller59/spconv)

#### Install dependent libraries
a. Clone this repository.
```bash
git clone https://github.com/hithqd/UniM-OV3D.git
git fetch -all
git checkout main
```

b. Install the dependent libraries as follows:

* Install the dependent Python libraries (Please note that you need to install the correct version of `torch` and `spconv` according to your CUDA version): 
    ```bash
    pip install -r requirements.txt 
    ```

* Install [SoftGroup](https://github.com/thangvubk/SoftGroup) following its [official guidance](https://github.com/thangvubk/SoftGroup/blob/main/docs/installation.md).
    ```bash
    cd pcseg/external_libs/softgroup_ops
    python3 setup.py build_ext develop
    cd ../../..
    ```

* Install [pcseg](../pcseg)
    ```bash
    python3 setup.py develop
    ```
The dataset configs are located within [tools/cfgs/dataset_configs](../tools/cfgs/dataset_configs), and the model configs are located within [tools/cfgs](../tools/cfgs) for different settings.

# Datasets
#### ScanNet Dataset
- Please download the [ScanNet Dataset](http://www.scan-net.org/) and follow [PointGroup](https://github.com/dvlab-research/PointGroup/blob/master/dataset/scannetv2/prepare_data_inst.py) to pre-process the dataset as follows or directly download the pre-processed data [here](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/u3007346_connect_hku_hk/EpTBva1Ev0BLu7TYz_03UUQBpLnyFlijK9z645tavor68w?e=liM2HD).
- Additionally, please download the caption data [here](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/u3007346_connect_hku_hk/EpTBva1Ev0BLu7TYz_03UUQBpLnyFlijK9z645tavor68w?e=liM2HD). If you want to generate captions on your own, please download image data ([scannet_frames_25k]((http://www.scan-net.org/))) from ScanNet and follow scripts [generate_caption.py](../tools/process_tools/generate_caption.py) and [generate_caption_idx.py](../tools/process_tools/generate_caption_idx.py).

- The directory organization should be as follows:

    ```
    ├── data
    │   ├── scannetv2
    │   │   │── train
    │   │   │   │── scene0000_00.pth
    │   │   │   │── ...
    │   │   │── val
    │   │   │── text_embed
    │   │   │── caption_idx
    │   │   │── scannetv2_train.txt
    │   │   │── scannetv2_val.txt
    │   │   │—— scannet_frames_25k (optional, only for caption generation)
    ├── pcseg
    ├── tools
    ```

#### S3DIS Dataset
- Please download the [S3DIS Dataset](http://buildingparser.stanford.edu/dataset.html#Download) and follow [dataset/s3dis/preprocess.py](../dataset/s3dis/preprocess.py) to pre-process the dataset as follows or directly download the pre-processed data [here](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/u3007346_connect_hku_hk/EoNAsU5f8YRGtQYV8ewhwvQB7QPbxT-uwKqTk8FPiyUTtQ?e=wq58H7).
    ```bash
    python3 pcseg/datasets/s3dis/preprocess.py 
    ```
    
- Additionally, please download the caption data [here](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/u3007346_connect_hku_hk/EoNAsU5f8YRGtQYV8ewhwvQB7QPbxT-uwKqTk8FPiyUTtQ?e=wq58H7). If you want to generate captions on your own, please download image data [here](https://github.com/alexsax/2D-3D-Semantics) and follows scripts here: [generate_caption.py](../tools/process_tools/generate_caption.py) and [generate_caption_idx.py](../tools/process_tools/generate_caption_idx.py).
 
- The directory organization should be as follows:

    ```
    ├── data
    │   ├── s3dis
    │   │   │── stanford_indoor3d_inst
    │   │   │   │── Area_1_Conference_1.npy
    │   │   │   │── ...
    │   │   │── text_embed
    │   │   │── caption_idx
    │   │   │—— s3dis_2d (optional, only for caption generation)
    ├── pcseg
    ├── tools
    ```

#### nuScenes Dataset
* Please download the official [NuScenes 3D object detection dataset](https://www.nuscenes.org/download) and organize the downloaded files as follows: 
* Additionally, please download the caption data [here](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/jhyang13_connect_hku_hk/Eh9qCTiV0VBEuJDLRai7_MUBGOepuHr3F9y-VVnIjhyELw?e=a4yXcB).
```
├── data
│   ├── nuscenes
│   │   │── text_embed
│   │   │── v1.0-trainval (or v1.0-mini if you use mini)
│   │   │   │── samples
│   │   │   │── sweeps
│   │   │   │── maps
│   │   │   │── caption_idx
│   │   │   │── v1.0-trainval
├── pcseg
├── tools
```

* Install the `nuscenes-devkit` with version `1.0.5` by running the following command: 
```shell script
pip install nuscenes-devkit==1.0.5
```

# Model Zoo
#### 3D Semantic Segmentation

- Semantic segmentation on four datasets

    | Dataset | Partition | Path |
    |:---:|:---:|:---:|
    | ScanNet | B15/N4 | [ckpt](https://huggingface.co/QingdongHe/UniM-OV3D/blob/main/semantic_b15n4.pth) |
    | ScanNet | B12/N7 | [ckpt](https://huggingface.co/QingdongHe/UniM-OV3D/blob/main/semantic_b12n7.pth) |
    | ScanNet | B10/N9 | [ckpt](https://huggingface.co/QingdongHe/UniM-OV3D/blob/main/semantic_b10n9.pth) |
    | S3DIS | B8/N4 | [ckpt](https://huggingface.co/QingdongHe/UniM-OV3D/blob/main/semantic_b8n4_s3dis.pth) |
    | S3DIS | B6/N6 | [ckpt](https://huggingface.co/QingdongHe/UniM-OV3D/blob/main/semantic_b6n6_s3dis.pth) |
    | ScanNet200 | B170/N30 | [ckpt](https://huggingface.co/QingdongHe/UniM-OV3D/blob/main/semantic_b170n30_ScanNet200.pth) |
    | ScanNet200 | B150/N50 | [ckpt](https://huggingface.co/QingdongHe/UniM-OV3D/blob/main/semantic_b150n50_ScanNet200.pth) |
    | nuScenes | B12/N3 |  [ckpt](https://huggingface.co/QingdongHe/UniM-OV3D/blob/main/semantic_b12n3_nuScenes.pth) |
    | nuScenes | B10/N5 |  [ckpt](https://huggingface.co/QingdongHe/UniM-OV3D/blob/main/semantic_b10n5_nuScenes.pth) |
    
#### 3D Instance Segmentation
- Instance segmentation on two datasets
  
    | Dataset | Partition | Path |
    |:---:|:---:|:---:|
    | ScanNet | B13/N4 | [ckpt](https://huggingface.co/QingdongHe/UniM-OV3D/blob/main/instance_b13n4_scannet.pth) |
    | ScanNet | B10/N7 | [ckpt](https://huggingface.co/QingdongHe/UniM-OV3D/blob/main/instance_b10n7_scannet.pth) |
    | ScanNet | B8/N9 | [ckpt](https://huggingface.co/QingdongHe/UniM-OV3D/blob/main/instance_b8n9_scannet.pth) |
    | S3DIS | B8/N4 | [ckpt](https://huggingface.co/QingdongHe/UniM-OV3D/blob/main/instance_b8n4_s3dis.pth) |
    | S3DIS | B6/N6 | [ckpt](https://huggingface.co/QingdongHe/UniM-OV3D/blob/main/instance_b6n6_s3dis.pth) |

# Training

```bash
cd tools
sh scripts/dist_train.sh ${NUM_GPUS} --cfg_file ${CONFIG_FILE} ${PY_ARGS}
```

# Inference

```bash
cd tools
sh scripts/dist_test.sh ${NUM_GPUS} --cfg_file ${CONFIG_FILE} --ckpt ${CKPT_PATH}
```

# Citation
```
@article{he2024unim,
  title={UniM-OV3D: Uni-Modality Open-Vocabulary 3D Scene Understanding with Fine-Grained Feature Representation},
  author={He, Qingdong and Peng, Jinlong and Jiang, Zhengkai and Wu, Kai and Ji, Xiaozhong and Zhang, Jiangning and Wang, Yabiao and Wang, Chengjie and Chen, Mingang and Wu, Yunsheng},
  journal={33rd International Joint Conference on Artificial Intelligence (IJCAI)},
  year={2024}
}
```
