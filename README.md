# Densely Residual Laplacian Super-resolution
This repository is for Densely Residual Laplacian Network (DRLN) introduced in the following paper

[Saeed Anwar](https://saeed-anwar.github.io/), [Nick Barnes], "Densely Residual Laplacian Super-resolution", [[arXiv]](https://arxiv.org/abs/1906.12021)

The model is built in PyTorch 1.1.0 and tested on Ubuntu 14.04/16.04 environment (Python3.6, CUDA9.0, cuDNN5.1).

Our DRLN is also available in PyTorch 0.4.0 and 0.4.1. You can download this version from [Google Drive](https://drive.google.com/open?id=1I91VGposSoFq6UrWBuxCKbst6sMi1yhR) or [here](https://icedrive.net/0/cb986Jh8rh). 

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

![PSNR_SSIM_BI](/Figs/front.PNG)
Sample results on URBAN100 with Bicubic (BI) degradation for 4x on “img 074” and for 8x on “img 040”.

## Network

![Net](/Figs/Net.PNG)
The architecture of our proposed densely residual Laplacian attention network (DRLN) with densely residual laplacian modules (DRLM).
![LapAtt](/Figs/LapAtt.PNG)
Laplacian attention architecture.

## Train
**Will be added later**

## Test
### Quick start
1. Download the trained models for our paper and place them in '/TestCode/TrainedModels'.

    All the models (BIX2/3/4/8, BDX3) can be downloaded from [Google Drive](https://drive.google.com/open?id=1MwRNAcUOBcS0w6Q7gGNZWYO_AP_svi7i) [Baidu](https://pan.baidu.com/s/1lAWlfQJHBJc3u9okpOL3lA) or [here](https://icedrive.net/0/a81sqSW91R). The total size for all models is 737MB.

2. Cd to '/TestCode/code', run the following scripts.

    **You can use scripts in file 'TestDRLN_All' to produce results for our paper or You can also use individual scripts such as TestDRLN_2x.sh**

    ```bash
    # No self-ensemble: DRLN
    # BI degradation model, x2, x3
    # x2
   CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 2 --model DRLN --n_feats 64 --pre_train ../TrainedModels/DRLN_BIX2/DRLN_BIX2.pt --test_only --save_results --chop --save 'DRLN_Set5' --testpath ../LR/LRBI --testset Set5
   
   CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 2 --model DRLN --n_feats 64 --pre_train ../TrainedModels/DRLN_BIX2/DRLN_BIX2.pt --test_only --save_results --chop --save 'DRLN_Set14' --testpath ../LR/LRBI --testset Set14
    # x3
   CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 3 --model DRLN --n_feats 64 --pre_train ../TrainedModels/DRLN_BIX3/DRLN_BIX3.pt --test_only --save_results --chop --save 'DRLN_Set5' --testpath ../LR/LRBI --testset Set5
   
   CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 3 --model DRLN --n_feats 64 --pre_train ../TrainedModels/DRLN_BIX3/DRLN_BIX3.pt --test_only --save_results --chop --save 'DRLN_Set14' --testpath ../LR/LRBI --testset Set14
   
    # x3 Blur-downgrade 
   CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 3 --model DRLN --n_feats 64 --pre_train ../TrainedModels/DRLN_BDX3/DRLN_BDX3.pt --test_only --save_results --chop --save 'DRLN_BD_Set5' --testpath ../LR/LRBD --testset Set5
    ```


## Results
**All the results for DRLN can be downloaded from [GoogleDrive](https://drive.google.com/open?id=1NJ20pHYolkzTBDB2UUy7pvY9F9sIyqO2) or [here](https://icedrive.net/0/bcATKQGntn). The size of the results is 2.41GB** 

### Quantitative Results
![PSNR_SSIM_BI](/Figs/23_table.PNG)
![PSNR_SSIM_BI](/Figs/48_table.PNG)
The performance of state-of-the-art algorithms on widely used publicly available five datasets (SET5, SET14, BSD100, URBAN100, MANGA109), in terms of PSNR (in dB) and SSIM. The best results are highlighted with red color while the blue color represents the second best super-resolution method.


![PSNR_SSIM_BI](/Figs/BD_table.PNG)
Quantitative results on  blur-down degradations for 3x. The best results are highlighted with red color while the blue color represents the second best.


![PSNR_SSIM_BI](/Figs/noiseplot.PNG)

The plot shows the average PSNR as functions of noise sigma. Our method consistently improves over specific
noisy super-resolution methods and CNN for all noise levels.

For more information, please refer to our [papar](https://arxiv.org/pdf/1906.12021.pdf)

### Visual Results
![Visual_PSNR_SSIM_BI](/Figs/4x.PNG)
Visual results with Bicubic (BI) degradation (4x) on "img 076" and "img_044" from URBAN100 as well as YumeiroCooking from MANGA109.


![Visual_PSNR_SSIM_BI](/Figs/8x.PNG)
Comparisons on images with fine details for a high upsampling factor of 8x on URBAN100 and MANGA109. The best results are in bold.

![Visual_PSNR_SSIM_BI](/Figs/3x.PNG)
Comparison on Blur-Downscale (BD) degraded images with sharp edges and texture, taken from URBAN100 and SET14
datasets for the scale of 3x. The sharpness of the edges on the objects and textures restored by our method is the best.

![Visual_PSNR_SSIM_BI](/Figs/BSDNoisy.PNG)
Noisy SR visual Comparison on BSD100. Textures on the birds are much better reconstructed, and the noise removed by our method as
compared to the IRCNN and RCAN for sigma = 10.

![Visual_PSNR_SSIM_BI](/Figs/lamaNoisy.PNG)
Noisy visual comparison on Llama. Textures on the fur, and on rocks in the background are much better reconstructed in our result as
compared to the conventional BM3D-SR and BM3D-SRNI.

![Visual_PSNR_SSIM_BI](/Figs/real.PNG)

Comparison on real-world images. In these cases, neither the downsampling blur kernels nor the ground-truth images are available.


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
