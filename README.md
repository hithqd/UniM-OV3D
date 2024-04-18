# [IJCAI 2024] UniM-OV3D: Uni-Modality Open-Vocabulary 3D Scene Understanding with Fine-Grained Feature Representation

<!-- <br> -->
[Qingdong He<sup>1</sup>](https://scholar.google.com/citations?user=gUJWww0AAAAJ&hl=zh-CN), [Jinlong Peng<sup>1</sup>](https://scholar.google.com/citations?user=i5I-cIEAAAAJ&hl=zh-CN), [Zhengkai Jiang<sup>1</sup>](https://scholar.google.com/citations?user=ooBQi6EAAAAJ&hl=zh-CN), [Kai Wu<sup>1</sup>](https://scholar.google.com/citations?hl=zh-CN&user=ElfT3eoAAAAJ), [Xiaozhong Ji<sup>1</sup>](https://scholar.google.com/citations?user=iL2j_yAAAAAJ&hl=zh-CN&oi=ao), [Jiangning Zhang<sup>1</sup>](https://zhangzjn.github.io/), [Yabiao Wang<sup>1</sup>](https://scholar.google.com/citations?hl=zh-CN&user=xiK4nFUAAAAJ), [Chengjie Wang<sup>1</sup>](https://scholar.google.com/citations?hl=zh-CN&user=fqte5H4AAAAJ), [Mingang Chen<sup>2</sup>](https://scholar.google.com/citations?user=FBfC58EAAAAJ&hl=zh-CN&oi=ao), Yunsheng Wu<sup>1</sup>. 
<!-- <br> -->

<sup>1</sup>Youtu Lab, Tencent
<sup>2</sup>Shanghai Development Center of Computer Software Technology

[![arXiv](https://img.shields.io/badge/arXiv-2312.05767-b31b1b.svg)](https://arxiv.org/abs/2401.11395)

![Image description](https://github.com/hithqd/UniM-OV3D/blob/main/docs/framework.png)

3D open-vocabulary scene understanding aims to recognize arbitrary novel categories beyond the base label space. However, existing works not only fail to fully utilize all the available modal information in the 3D domain but also lack sufficient granularity in representing the features of each modality. In this paper, we propose a unified multimodal 3D open-vocabulary scene understanding network, namely UniM-OV3D, which aligns point clouds with image, language and depth. To better integrate global and local features of the point clouds, we design a hierarchical point cloud feature extraction module that learns comprehensive fine-grained feature representations. Further, to facilitate the learning of coarse-to-fine point-semantic representations from captions, we propose the utilization of hierarchical 3D caption pairs, capitalizing on geometric constraints across various viewpoints of 3D scenes. Extensive experimental results demonstrate the effectiveness and superiority of our method in open-vocabulary semantic and instance segmentation, which achieves state-of-the-art performance on both indoor and outdoor benchmarks such as ScanNet, ScanNet200, S3IDS and nuScenes.

# TODO
* Code is coming soon.

# Citation
```
@article{he2024unim,
  title={UniM-OV3D: Uni-Modality Open-Vocabulary 3D Scene Understanding with Fine-Grained Feature Representation},
  author={He, Qingdong and Peng, Jinlong and Jiang, Zhengkai and Wu, Kai and Ji, Xiaozhong and Zhang, Jiangning and Wang, Yabiao and Wang, Chengjie and Chen, Mingang and Wu, Yunsheng},
  journal={arXiv preprint arXiv:2401.11395},
  year={2024}
}
```
