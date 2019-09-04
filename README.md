# Densely Residual Laplacian Super-resolution
This repository is for Densely Residual Laplacian Network (DRLN) introduced in the following paper

[Saeed Anwar](https://saeed-anwar.github.io/), [Nick Barnes], "Densely Residual Laplacian Super-resolution", [[arXiv]](https://arxiv.org/abs/1906.12021)

The code and models will be available soon here.

The model is built in PyTorch and tested on Ubuntu 14.04/16.04 environment (Python3.6, PyTorch_0.4.0/pyTorch_1.1.0, CUDA9.0, cuDNN5.1).

## Contents
1. [Introduction](#introduction)
2. [Network](#network)
2. [Train](#train)
3. [Test](#test)
4. [Results](#results)
5. [Citation](#citation)
6. [Acknowledgements](#acknowledgements)

## Introduction
Super-Resolution convolutional neural networks have recently demonstrated high-quality restoration for single images.
However, existing algorithms often require very deep architectures and long training times. Furthermore, current convolutional neural
networks for super-resolution are unable to exploit features at multiple scales and weigh them equally, limiting their learning capability. In this exposition, we present a compact and accurate super-resolution algorithm namely, Densely Residual Laplacian Network
(DRLN). The proposed network employs cascading residual on the residual structure to allow the flow of low-frequency information to
focus on learning high and mid-level features. In addition, deep supervision is achieved via the densely concatenated residual blocks
settings, which also helps in learning from high-level complex features. Moreover, we propose Laplacian attention to model the crucial
features to learn the inter and intra-level dependencies between the feature maps. Furthermore, comprehensive quantitative and
qualitative evaluations on low-resolution, noisy low-resolution, and real historical image benchmark datasets illustrate that our DRLN
algorithm performs favorably against the state-of-the-art methods visually and accurately.

## Network

![Net](/Figs/Net.PNG)
The architecture of our proposed densely residual Laplacian attention network (DRLN) with densely residual laplacian modules (DRLM).
![LapAtt](/Figs/LapAtt.PNG)
Laplacian attention architecture.




## Results
### Quantitative Results
![PSNR_SSIM_BI](/Figs/23_table.PNG)
![PSNR_SSIM_BI](/Figs/48_table.PNG)
![PSNR_SSIM_BI](/Figs/BD_table.PNG)
![PSNR_SSIM_BI](/Figs/noiseplot.PNG)

Quantitative results with BI degradation model. Best and second best results are highlighted and underlined

For more results, please refer to our [main papar](https://arxiv.org/abs/1807.02758) and [supplementary file](http://yulunzhang.com/papers/ECCV-2018-RCAN_supp.pdf).
### Visual Results
![Visual_PSNR_SSIM_BI](/Figs/4x.PNG)
Visual results with Bicubic (BI) degradation (4×) on “img 074” from Urban100


![Visual_PSNR_SSIM_BI](/Figs/8x.PNG)
![Visual_PSNR_SSIM_BI](/Figs/3x.PNG)
![Visual_PSNR_SSIM_BI](/Figs/BSDNoisy.PNG)
![Visual_PSNR_SSIM_BI](/Figs/lamaNoisy.PNG)
![Visual_PSNR_SSIM_BI](/Figs/real.PNG)

Visual comparison for 4× SR with BI model

## Citation
If you find the code helpful in your resarch or work, please cite the following papers.
```
@article{anwar2019drln,
  title={Densely Residual Laplacian Super-Resolution},
  author={Anwar, Saeed and Barnes, Nick},
  journal={arXiv preprint arXiv:1906.12021},
  year={2019}
}

@article{anwar2019deepSR,
  title={A Deep Journey into Super-resolution: A survey},
  author={Anwar, Saeed and Khan, Salman and Barnes, Nick},
  journal={arXiv preprint arXiv:1904.07523},
  year={2019}
}
```
## Acknowledgements
This code is built on [RCAN (PyTorch)](https://github.com/yulunzhang/RCAN) and [EDSR (PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch). We thank the authors for sharing their codes.
