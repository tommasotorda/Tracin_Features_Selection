#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="5"
from nibabel.testing import data_path
import nibabel as nib
from glob import glob
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import random


# In[2]:


from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle


# In[4]:


from dltk.io.augmentation import *
from dltk.io.preprocessing import *
from scipy.ndimage.filters import gaussian_filter
import SimpleITK as sitk


# In[5]:


file_list = glob(os.path.join("/home/tordatom/Dati_Imaging/BraTs_19/MICCAI_BraTS_2019_Data_Training/HGG", "*", ""))


# In[6]:


def load_example(file_list):
    
    exam_path = [np.sort(glob(os.path.join(file_list[i], "*.nii.gz"))) for i in range(len(file_list))]
    seg = np.zeros(np.append(len(file_list),nib.load(exam_path[0][0]).shape))
    channels = np.zeros(np.append(seg.shape,4))
    
    for i in range(len(file_list)):
    
        seg[i] = nib.load(exam_path[i][1]).dataobj
    
        for j,k in enumerate(list(set(exam_path[i]) - set([exam_path[i][1]]))):
            
            a = nib.load(k)
            channels[i,:,:,:,j] = np.array(a.dataobj)
    

    return channels, seg


# In[7]:


def normalize(t1):
    for i in range(len(t1[:,0,0,0,0])):
        for j in range(len(t1[0,0,0,0,:])):
            t1[i,:,:,:,j] = 2*t1[i,:,:,:,j]/np.max(t1[i,:,:,:,j])-1
    return t1


# In[10]:


def elastic_transform1(image1,image2, alpha, sigma):
    
    assert len(alpha) == len(sigma),         "Dimensions of alpha and sigma are different"

    channelbool = image1.ndim - len(alpha)
    out = np.zeros((len(alpha) + channelbool, ) + image1.shape)

    # Generate a Gaussian filter, leaving channel dimensions zeroes
    for jj in range(len(alpha)):
        array = (np.random.rand(*image1.shape) * 2 - 1)
        out[jj] = gaussian_filter(array, sigma[jj],
                                  mode="constant", cval=0) * alpha[jj]

    # Map mask to indices
    shapes = list(map(lambda x: slice(0, x, None), image1.shape))
    grid = np.broadcast_arrays(*np.ogrid[shapes])
    indices = list(map((lambda x: np.reshape(x, (-1, 1))), grid + np.array(out)))

    # Transform image based on masked indices
    transformed_image1 = map_coordinates(image1, indices, order=0,
                                        mode='reflect').reshape(image1.shape)
    transformed_image2 = map_coordinates(image2, indices, order=0,
                                        mode='reflect').reshape(image2.shape)

    return transformed_image1, transformed_image2


# In[ ]:


#The 3D RMI is transformed into a batch of 2D images 

def Twod(channels):
    new = np.zeros((channels.shape[0]*channels.shape[3], channels.shape[1], channels.shape[2], channels.shape[4]))
    for i in range(channels.shape[0]):
        for j in range(channels.shape[3]):
            new[10*i+j] = channels[i,:,:,j,:]
    return new


# In[11]:


from sklearn.model_selection import train_test_split
train_to_test_ratio=0.8


# In[12]:


channels1, seg1 = load_example(file_list)
#we only take the slices of the RMI that are in the centre of the volume. 
#In total we save 10 slices per patient on the z-axis and we crop the images to 192x192.
channels, seg = channels1[:,27:219, 27:219, 90:100, :], seg1[:,27:219, 27:219, 90:100]
channels = normalize(channels)
seg = seg.reshape(np.append(seg.shape,1))
#the original label are 0,1,2,4 we shift the last one to 3. New label: 0,1,2,3
seg[seg == 4] = 3
seg = keras.utils.to_categorical(seg)


# In[14]:


X_train,X_test,Y_train,Y_test=train_test_split(np.array(channels),
                                               np.array(seg),train_size=train_to_test_ratio,
                                               shuffle=True, random_state=1234)

X_train, Y_train = Twod(X_train),  Twod(Y_train)
X_test, Y_test = Twod(X_test), Twod(Y_test)
            


# In[62]:


X = X_train
Y = Y_train
for i in range(5):
    #elastic transformation can not be applied during the training process.
    #not compatible with GPU process
    a,b = elastic_transform1(X_train, Y_train, alpha=[0,2e5, 2e5,0], sigma=[0,23, 23, 0])
    X = np.append(X, a, axis = 0)
    Y = np.append(Y, b, axis = 0)
X_train = X
Y_train = Y


# In[15]:


BATCH = 1


# In[35]:


os.mkdir("/home/tordatom/Dati_Imaging/BraTs_19/Segmentation2D/DataTracin")


# In[ ]:


os.mkdir('DataCgan/Train')
os.mkdir('DataCgan/Test')


# In[36]:


for i in np.arange(0,X_train.shape[0], BATCH):
    
    file_name_train = 'Train_{0}'.format(i//BATCH)
    if i+BATCH>=X_train.shape[0]:
        np.savez(os.path.join('/home/tordatom/Dati_Imaging/BraTs_19/Segmentation2D/DataTracin' , file_name_train), 
                 X_train = X_train[i:], Y_train = Y_train[i:])
        
    np.savez(os.path.join('/home/tordatom/Dati_Imaging/BraTs_19/Segmentation2D/DataTracin' , file_name_train),
             X_train = X_train[i:i+BATCH], Y_train = Y_train[i:i+BATCH])
        


# In[16]:


for i in np.arange(0,X_test.shape[0], BATCH):
    
    file_name_test = 'Test_{0}'.format(i//BATCH)
    if i+BATCH>=X_test.shape[0]:
        np.savez(os.path.join('/home/tordatom/Dati_Imaging/BraTs_19/Segmentation2D/DataTracin_test' , file_name_test), 
                 X_test = X_test[i:], Y_test = Y_test[i:])
        
    np.savez(os.path.join('/home/tordatom/Dati_Imaging/BraTs_19/Segmentation2D/DataTracin_test' , file_name_test),
             X_test = X_test[i:i+BATCH], Y_test = Y_test[i:i+BATCH])


# In[ ]:


for i in np.arange(0,X_test.shape[0], BATCH):
    
    file_name_test = 'Test_{0}'.format(i//BATCH)
    
    if i+BATCH>=X_train.shape[0]:
        np.savez(os.path.join('Data/Test' , file_name_test), 
                 X_test = X_test[i:], Y_test = Y_test[i:])
        
    np.savez(os.path.join('Data/Test' , file_name_test), X_test = X_test[i:i+BATCH], Y_test = Y_test[i:i+BATCH])
        


# In[ ]:


with np.load("Data/Train/Train_0.npz") as data:

    X=np.array(data["X_train"])
    Y=np.array(data["Y_train"])


# In[ ]:


X_train.shape, Y_train.shape


# In[ ]:


files = glob('Data/Train/*')
for f in files:
    os.remove(f)
files = glob('Data/Test/*')
for f in files:
    os.remove(f)

