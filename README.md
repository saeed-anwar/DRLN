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
The architecture of our proposed densely residual Laplacian attention network (DRLN) with densely residual modules.
![LapAtt](/Figs/LapAtt.PNG)
Laplacian attention (CA) architecture.
