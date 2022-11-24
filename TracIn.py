import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="2"
from nibabel.testing import data_path
import nibabel as nib
from glob import glob
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import time
import datetime
import json

import itertools
from keras.models import load_model
from keras.utils.vis_utils import plot_model
from sklearn import preprocessing


search_dir = "/home/tordatom/Dati_Imaging/BraTs_19/Segmentation2D/gradients/train_dice1/" #In this example we evaluate tracin only for label 1

os.chdir(search_dir)
file_list_gradtrain = filter(os.path.isfile, os.listdir(search_dir))
file_list_gradtrain = [os.path.join(search_dir, f) for f in file_list_gradtrain] # add path to each file
file_list_gradtrain.sort(key=lambda x: os.path.getmtime(x))


search_dir = "/home/tordatom/Dati_Imaging/BraTs_19/Segmentation2D/gradients/test_dice1/"
os.chdir(search_dir)
file_list_gradtest = filter(os.path.isfile, os.listdir(search_dir))
file_list_gradtest = [os.path.join(search_dir, f) for f in file_list_gradtest] # add path to each file
file_list_gradtest.sort(key=lambda x: os.path.getmtime(x))


search_dir = "/home/tordatom/Dati_Imaging/BraTs_19/Segmentation2D/DataTracin/"
os.chdir(search_dir)
file_list_train = filter(os.path.isfile, os.listdir(search_dir))
file_list_train = [os.path.join(search_dir, f) for f in file_list_train] # add path to each file
file_list_train.sort(key=lambda x: os.path.getmtime(x))


search_dir = "/home/tordatom/Dati_Imaging/BraTs_19/Segmentation2D/DataTracin_test/"
os.chdir(search_dir)
file_list_test = filter(os.path.isfile, os.listdir(search_dir))
file_list_test = [os.path.join(search_dir, f) for f in file_list_test] # add path to each file
file_list_test.sort(key=lambda x: os.path.getmtime(x))


file = glob(os.path.join('/home/tordatom/Dati_Imaging/BraTs_19/Segmentation2D/DataTracin/', "*"))


def load_grad_train(file):
    data = np.load(file, allow_pickle=True)
    index = int(file[84:len(file)-4])
    return index, data['grad_train']

def load_grad_test(file):
    data = np.load(file, allow_pickle=True)
    index = int(file[83:len(file)-4])
    return index, data['grad_test']


def load_image_train(image_file):
    data = np.load(image_file)
    index = int(image_file[69:len(image_file)-4])
    return index, data['X_train'], data['Y_train']

def load_image_test(image_file):
    data = np.load(image_file)
    index = int(image_file[73:len(image_file)-4])
    return index, data['X_test'], data['Y_test']


def test_reshape(index, X,Y):
    X = tf.cast(X, tf.float32)
    Y = tf.cast(Y, tf.float32)
    X = tf.reshape(X, [192,192,4])
    Y = tf.reshape(Y, [192,192,4])
    return index,X,Y

BATCH_SIZE = 1

#load all the data

train_grad = tf.data.Dataset.list_files(file_list_gradtrain, shuffle = False)
train_grad = train_grad.map(lambda item: tf.numpy_function(
          load_grad_train, [item], [tf.int64, tf.float32]),
          num_parallel_calls=tf.data.AUTOTUNE)


test_grad = tf.data.Dataset.list_files(file_list_gradtest, shuffle = False)
test_grad = test_grad.map(lambda item: tf.numpy_function(
          load_grad_test, [item], [tf.int64, tf.float32]),
          num_parallel_calls=tf.data.AUTOTUNE)


train_dataset = tf.data.Dataset.list_files(file_list_train, shuffle = False)
train_dataset = train_dataset.map(lambda item: tf.numpy_function(
          load_image_train, [item], [tf.int64, tf.double, tf.double]),
          num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.map(test_reshape)
train_dataset = train_dataset.batch(BATCH_SIZE)


test_dataset = tf.data.Dataset.list_files(file_list_test, shuffle = False)
test_dataset = test_dataset.map(lambda item: tf.numpy_function(
          load_image_test, [item], [tf.int64, tf.double, tf.double]),
          num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.map(test_reshape)
test_dataset = test_dataset.batch(BATCH_SIZE)

#track the order of the example
map_test = {}
map_train = {}

for i,j in enumerate(train_dataset):
    map_train[int(j[0])] = i
for i,j in enumerate(test_dataset):
    map_test[int(j[0])] = i
    


def tracin(grad1,grad2):
    s = grad1@grad2.transpose()
    s = np.array(s, dtype = float)
    
    return s


def my_function(train_grad, test_grad):
    
    score = {}
    
    for i,j in enumerate(train_grad):
        score[int(j[0])] = np.trace(tracin(np.array(test_grad), np.array(j[1])))
        if i%100 == 1 : print("x", end =" " )
    
    return score
    


final_score = {}
for i,j in enumerate(test_grad):
        final_score[int(j[0])] = my_function(train_grad, j[1])
        print(f"test_{j[0]}")

tracin_score = {}
#Regarding the fact that we work in 2D, in this case, proponent and opponent are not the specific patients, but one brain slice of the patients. If we want to have as prop and opp the specific patient, we must compare the comulative Tracin score for all the slices of their brain. 

for k in range(len(final_score)):
    print(f"test_{k}") #we select a test example
    c = {}
    for j in range(0, len(final_score[k]), 10):
        m = 0 
        a = 0
        index = j
        
        for i in range(j,j+10):
            a += final_score[k][i] #we sum all the slice of a training patient
        
            if abs(final_score[k][i]) > m:
                m = abs(final_score[k][i]) #we keep only the slice of the training patients with maximum tracin score. This help for plotting the results
                index = i
                
        print(f"max : {m}, index :{index}")
        c[index] = a
    print("\t")
    tracin_score[k] = c 


np.save("/home/tordatom/Dati_Imaging/BraTs_19/Segmentation2D/Radiomic_Features/score_dice1.npy", tracin_score) 

tracin_score = np.load("/home/tordatom/Dati_Imaging/BraTs_19/Segmentation2D/Radiomic_Features/score_dice1.npy",allow_pickle='TRUE').item()

#we sort the tracin score and took the first, and last 20 example as proponents and opponents
proponents_score = [sorted(b[i].items(), key=lambda x: x[1], reverse = True) for i in range(len(tracin_score))][:20]
opponents_score = [sorted(b[i].items(), key=lambda x: x[1]) for i in range(len(tracin_score))][:20]


#extract the index of the examples
proponents_index = np.zeros((len(proponents_score), 207))
for i,j in enumerate(proponents_score):
    for k,h in enumerate(proponents_score[i]):
        proponents_index[i][k] = int(h[0])

opponents_index = np.zeros((len(opponents_score), 207))
for i,j in enumerate(opponents_score):
    for k,h in enumerate(opponents_score[i]):
        opponents_index[i][k] = int(h[0])

np.savez(f"/home/tordatom/Dati_Imaging/BraTs_19/Segmentation2D/Radiomic_Features/prop_index_dice1.npz", proponents = proponents_index)
np.savez(f"/home/tordatom/Dati_Imaging/BraTs_19/Segmentation2D/Radiomic_Features/opp_index_dice1.npz", opponents = opponents_index)

def dice0(y_true, y_pred, smooth = 1e-7):
    y_true_f = tf.reshape(tf.cast(y_true[:,:,:,0], 'float32'), [len(y_true), -1]) 
    y_pred_f = tf.reshape(tf.cast(y_pred[:,:,:,0], 'float32'), [len(y_true), -1])
    return (2*tf.reduce_sum(tf.abs(y_true_f*y_pred_f), axis = 1))/(tf.reduce_sum(
        y_true_f**2 + y_pred_f**2, axis = 1)+smooth)
def dice1(y_true, y_pred, smooth = 1e-7):  
    y_true_f = tf.reshape(tf.cast(y_true[:,:,:,1], 'float32'), [len(y_true), -1]) 
    y_pred_f = tf.reshape(tf.cast(y_pred[:,:,:,1], 'float32'), [len(y_true), -1])
    return (2*tf.reduce_sum(tf.abs(y_true_f*y_pred_f), axis = 1))/(tf.reduce_sum(
        y_true_f**2 + y_pred_f**2, axis = 1)+smooth)

def dice2(y_true, y_pred, smooth = 1e-7):
    y_true_f = tf.reshape(tf.cast(y_true[:,:,:,2], 'float32'), [len(y_true), -1]) 
    y_pred_f = tf.reshape(tf.cast(y_pred[:,:,:,2], 'float32'), [len(y_true), -1])
    return (2*tf.reduce_sum(tf.abs(y_true_f*y_pred_f), axis = 1))/(tf.reduce_sum(
        y_true_f**2 + y_pred_f**2, axis = 1)+smooth)

def dice3(y_true, y_pred, smooth = 1e-7):  
    y_true_f = tf.reshape(tf.cast(y_true[:,:,:,3], 'float32'), [len(y_true), -1]) 
    y_pred_f = tf.reshape(tf.cast(y_pred[:,:,:,3], 'float32'), [len(y_true), -1])
    return (2*tf.reduce_sum(tf.abs(y_true_f*y_pred_f), axis = 1))/(tf.reduce_sum(
        y_true_f**2 + y_pred_f**2, axis = 1)+smooth)

def dice_loss(y_true, y_pred):
    a0 = 0
    a1 = 1
    a2 = 1
    a3 = 1
    return 1-(a0*dice0(y_true,y_pred)+a1*dice1(y_true,y_pred)+a2*dice2(
        y_true,y_pred)+a3*dice3(y_true,y_pred))/(a0+a1+a2+a3)




checkpoint_dir = '/home/tordatom/Dati_Imaging/BraTs_19/checkpoints/training_checkpoints_Unet'
file_list_ckpt = glob(os.path.join(checkpoint_dir, "*"))
file_list_ckpt.sort()




model = tf.keras.models.load_model(file_list_ckpt[-1], custom_objects={'dice0': dice0, 
                                                          'dice1': dice1,
                                                          'dice2': dice2, 
                                                          'dice3': dice3, 
                                                          "dice_loss":dice_loss})




os.mkdir("/home/tordatom/Dati_Imaging/BraTs_19/Segmentation2D/Tracin_Images")
os.mkdir("/home/tordatom/Dati_Imaging/BraTs_19/Segmentation2D/Tracin_Images/totdice")
os.mkdir("/home/tordatom/Dati_Imaging/BraTs_19/Segmentation2D/Tracin_Images/test")
os.mkdir("/home/tordatom/Dati_Imaging/BraTs_19/Segmentation2D/Tracin_Images/dice1")
os.mkdir("/home/tordatom/Dati_Imaging/BraTs_19/Segmentation2D/Tracin_Images/dice2")
os.mkdir("/home/tordatom/Dati_Imaging/BraTs_19/Segmentation2D/Tracin_Images/dice3")

#we plot the opponents and proponents for a given test example, choosing the label that we want to monitoring.

for i in range(70,len(proponents_index), 10):
    print(f"proponents_for_test_{i}")
    os.mkdir(f"/home/tordatom/Dati_Imaging/BraTs_19/Segmentation2D/Tracin_Images/dice3/proponents_of_test_{i}")
    for index in proponents_index[i]: 
        index = int(index)
        channel = next(itertools.islice(train_dataset, map_train[index], map_train[index]+1))[1]
        seg = next(itertools.islice(train_dataset, map_train[index], map_train[index]+1))[2]
        plt.figure(figsize=(12, 8))
        plt.subplot(131)
        plt.title(f"Channel: 3, index: {index}")
        plt.imshow(channel[0,:,:,3])
        plt.subplot(132)
        plt.title("Segmentation")
        plt.imshow(np.argmax(seg[0], axis = 2))
        plt.subplot(133)
        plt.title("Pred Segmentation")
        plt.imshow(np.argmax(model.predict(channel)[0], axis = 2))
        plt.savefig(f"/home/tordatom/Dati_Imaging/BraTs_19/Segmentation2D/Tracin_Images/dice3/proponents_of_test_{i}/train_proponent_{index}")
        plt.show()
    if i == 110: break


# In[ ]:


for i in range(70,110, 10):
    channel = next(itertools.islice(test_dataset, map_test[i], map_test[i]+1))[1]
    seg = next(itertools.islice(test_dataset, map_test[i], map_test[i]+1))[2]
    plt.figure(figsize=(12, 8))
    plt.subplot(131)
    plt.title(f"Channel 3, index: {i}")
    plt.imshow(channel[0,:,:,3])
    plt.subplot(132)
    plt.title("Segmentation")
    plt.imshow(np.argmax(seg[0], axis = 2))
    plt.subplot(133)
    plt.title("Pred Segmentation")
    plt.imshow(np.argmax(model.predict(channel)[0], axis = 2))
    plt.savefig(f"/home/tordatom/Dati_Imaging/BraTs_19/Segmentation2D/Tracin_Images/test/test_example_{i}")
    plt.show()


# In[ ]:


for i in range(70,len(opponents_index),10):
    print(f"opponents_for_test_{i}")
    os.mkdir(f"/home/tordatom/Dati_Imaging/BraTs_19/Segmentation2D/Tracin_Images/dice3/opponents_of_test_{i}")
    for index in opponents_index[i]: 
        index = int(index)
        channel = next(itertools.islice(train_dataset, map_train[index], map_train[index]+1))[1]
        seg = next(itertools.islice(train_dataset, map_train[index], map_train[index]+1))[2]
        plt.figure(figsize=(12, 8))
        plt.subplot(131)
        plt.title(f"Channel: 3, index: {index}")
        plt.imshow(channel[0,:,:,3])
        plt.subplot(132)
        plt.title("Segmentation")
        plt.imshow(np.argmax(seg[0], axis = 2))
        plt.subplot(133)
        plt.title("Pred Segmentation")
        plt.imshow(np.argmax(model.predict(channel)[0], axis = 2))
        plt.savefig(f"/home/tordatom/Dati_Imaging/BraTs_19/Segmentation2D/Tracin_Images/dice3/opponents_of_test_{i}/train_opponents_{index}")
        plt.show()
    if i == 110: break









