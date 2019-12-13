"""
Created on Sat Aug 24 13:30:20 2019

@author: mengmi
"""

from torch.utils.data import Dataset
import os
from scipy.misc import imread, imresize
import numpy as np
import cv2

class DualLoadDatasets(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, imgsize, txt_folder, img_folder, bin_folder, split, Gfiltersz, Gblursigma, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        :param Gfiltersz: image gaussian blur filter size
        :param Gblursigma: image gaussian blur variance
        """
        self.split = split
        self.imgsize = imgsize
        self.Gfiltersz = Gfiltersz
        self.Gblursigma = Gblursigma
        
        assert self.split in {'train', 'val', 'test','testhuman','testRprior'}

        with open(os.path.join(txt_folder, self.split + 'Color_img.txt'),'rb') as f:
            self.imglist = [os.path.join(img_folder,line.strip()) for line in f]
            
        with open(os.path.join(txt_folder, self.split + 'Color_binimg.txt'),'rb') as f:
            self.binlist = [os.path.join(bin_folder,line.strip()) for line in f]
             
        with open(os.path.join(txt_folder, self.split + 'Color_label.txt'),'rb') as f:
            self.labellist = [int(line) for line in f]


        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.imglist)

    def __getitem__(self, i):
                
        # Read images
        #print(self.imglist[i])
        #print(self.binlist[i])
        #print(self.labellist[i])
        img = imread(self.imglist[i])
        
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
            img = np.concatenate([img, img, img], axis=2)
        img = imresize(img, (self.imgsize, self.imgsize))           
        assert np.max(img) <= 255       
        
        if self.transform is not None:
            img = self.transform(img)  
            
        blur = cv2.GaussianBlur(img,(self.Gfiltersz,self.Gfiltersz),self.Gblursigma,self.Gblursigma,-1)
        
        # Read binimg
        binimg = imread(self.binlist[i],'L')
        binimg = imresize(binimg, (self.imgsize, self.imgsize))
        
        label = self.labellist[i]
    
        #transpose images
        #img = img.transpose(2, 0, 1)
        #assert img.shape == (3, self.imgsize, self.imgsize)     
        if self.split == 'testhuman' or self.split == 'testRprior':
            return img, binimg, blur, label, self.imglist[i]
        else:
            return img, binimg, blur, label
        

    def __len__(self):
        return self.dataset_size
