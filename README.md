# Lift-the-flap: what, where and when for context reasoning

Authors: Mengmi Zhang, Claire Tseng, Karla Montejo, Joseph Kwon, Gabriel Kreiman

This repository contains an implementation of a recurrent attention deep learning model for contextual reasoning in natural scenes. Our paper is currently under review.

Access to our unofficial manuscript [HERE](https://arxiv.org/abs/1902.00163).

## Project Description

Context reasoning is critical in a wide variety of applications where current inputs need to be interpreted in the light of previous experience and knowledge. Both spatial and temporal contextual information play a critical role in the domain of visual recognition. Here we investigate spatial constraints (what image features provide contextual information and where they are located), and temporal constraints (when different contextual cues matter) for visual recognition. The task is to reason about the scene context and infer what a target object hidden behind a flap is in a natural image. To tackle this problem, we first describe an online human psychophysics experiment recording active sampling via mouse clicks in lift-the-flap games and identify clicking patterns and features which are diagnostic for high contextual reasoning accuracy. As a proof of the usefulness of these clicking patterns and visual features, we extend a state-of-the-art recurrent model capable of attending to salient context regions, dynamically integrating useful information, making inferences, and predicting class label for the target object over multiple clicks. The proposed model achieves human-level contextual reasoning accuracy, shares human-like sampling behavior and learns interpretable features for contextual reasoning.


| [![Stimulus](gif/ori.jpg)](gif/ori.jpg)  | [![Human Clicks](gif/humans_clicks.gif)](gif/humans_clicks.gif) |[![Clicks by model](gif/model_clicks.gif)](gif/model_clicks.gif)  | [![Attended regions by model](gif/model_attention.gif)](gif/model_attention.gif) |
|:---:|:---:|:---:|:---:|
| Stimulus | Human Clicks | Clicks predicted by model | Attention predicted by model | 

## Environment

The code has been successfully tested in Ubuntu 18.04 with one GPU (NVIDA RTX 2080 Ti). It requires the following:
- PyTorch = 1.1.0 
- python = 2.7
- CUDA = 10.2
- torchvision = 0.3.0

Dependencies:
- numpy
- opencv
- scipy
- matplotlib
- skimage

Refer to [link](https://www.anaconda.com/distribution/) for Anaconda installation.  

After Anaconda installation, create a conda environment:
```
conda create -n pytorch27 python=2.7
```
Activate the conda environment:
```
conda activate pytorch27
```
In the conda environment, refer to [link](https://pytorch.org/get-started/locally/) for Pytorch installation.

Download our repository:
```
git clone https://github.com/kreimanlab/lift-the-flap-clicknet.git
```

Download our pre-trained model from [HERE](https://drive.google.com/open?id=1vNczaSc2MbuZ2OqqO2-BeENlPZFrZ_fL) and place the downloaded model ```checkpoint_2.pth.tar``` in folder ```/src/Models/```




