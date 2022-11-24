import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from nibabel.testing import data_path
import nibabel as nib
from glob import glob
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import time
import datetime
import json

from keras.models import load_model
from keras.utils.vis_utils import plot_model
from sklearn import preprocessing

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
        
#This is an example of calculation of gradients for TracIn application. In this case we choose to evaluate the
#gradient of the last layer of the model on the training set considering only 10 central epochs of the training process. 
#The loss function is the complete dice (the weigthed average of all the label), but we can choose different setup
#for example we can focus only on a specific label choosing the correct "loss".

checkpoint_dir = '/home/tordatom/Dati_Imaging/BraTs_19/checkpoints/training_checkpoints_Unet'
model = tf.keras.models.load_model(checkpoint_dir+"/ckpt_Unet_01", custom_objects={'dice0': dice0, 'dice1': dice1, 'dice2': dice2, 'dice3': dice3, "dice_loss":dice_loss})


file_list_ckpt = glob(os.path.join(checkpoint_dir, "*"))
file_list_ckpt.sort()


#when we load the example we want to store the name of the example in order to reconstruct to which example the specific #gradient belongs

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


search_dir = "/home/tordatom/Dati_Imaging/BraTs_19/Segmentation2D/DataTracin_train/"
os.chdir(search_dir)
file_list_train = filter(os.path.isfile, os.listdir(search_dir))
file_list_train = [os.path.join(search_dir, f) for f in file_list_train] # add path to each file
file_list_train.sort(key=lambda x: os.path.getmtime(x))

BATCH_SIZE = 1

train_dataset = tf.data.Dataset.list_files(file_list_train, shuffle = False)
train_dataset = train_dataset.map(lambda item: tf.numpy_function(
          load_image_test, [item], [tf.int64, tf.double, tf.double]),
          num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.map(test_reshape)
train_dataset = train_dataset.batch(BATCH_SIZE)


def get_grad(X,Y, model):

    with tf.GradientTape() as tape:
        loss = dice_loss(model(X, training = False) , Y)
    grad = tape.gradient(loss, model.trainable_variables[80])
    grad = np.array(grad, dtype = "float32").reshape([-1])

    return grad


os.mkdir("/home/tordatom/Dati_Imaging/BraTs_19/Segmentation2D/gradients/train_totdice")

grad_train = np.zeros((len(file_list_ckpt[2:12]), 256), dtype = "float32")

for k,h in enumerate(train_dataset):
    print(f"gradient of index: {h[0][0]}")
    for i,j in enumerate(file_list_ckpt[2:12]):
        print(f"checkpoint: {i}")
        model = tf.keras.models.load_model(j, custom_objects={'dice0': dice0, 
                                                                  'dice1': dice1,
                                                                  'dice2': dice2, 
                                                                  'dice3': dice3, 
                                                                  "dice_loss":dice_loss})
        grad_train[i] = get_grad(h[1], h[2], model)
        

    np.savez(f"/home/tordatom/Dati_Imaging/BraTs_19/Segmentation2D/gradients/test_totdice/gradtrain_{h[0][0]}.npz", grad_train = grad_train)
    print(f"gradtrain_{h[0][0]}.npz")
    print("\t")
    
    

