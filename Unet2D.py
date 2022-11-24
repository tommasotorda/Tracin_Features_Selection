#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from nibabel.testing import data_path
import nibabel as nib
from glob import glob
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import time
import datetime
import json


# In[2]:


from dltk.io.augmentation import *
from dltk.io.preprocessing import *
from scipy.ndimage.filters import gaussian_filter
import SimpleITK as sitk


# In[18]:


file_list_train = glob(os.path.join("/home/tordatom/Dati_Imaging/BraTs_19/DataCgan/Train", "*"))
file_list_test = glob(os.path.join("/home/tordatom/Dati_Imaging/BraTs_19/Data/Test", "*"))

X = np.array(np.load(file_list_train[0])['X_train'])
Y = np.array(np.load(file_list_train[0])['Y_train'])


# In[19]:


IMG_HEIGHT = X.shape[1]
IMG_WIDTH  = X.shape[2]
IMG_CHANNELS = X.shape[3]


# In[20]:


def load_image_train(image_file):
    data = np.load(image_file)
    return data['X_train'], data['Y_train']

def load_image_test(image_file):
    data = np.load(image_file)
    return data['X_test'], data['Y_test']


# In[21]:


def resize(input_image, real_image, height, width):
    shape = tf.shape(input_image)
    input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image


# In[22]:


def random_crop(input_image, real_image):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 4])

    return cropped_image[0], cropped_image[1]


# In[23]:


def random_translation(X,Y):
    i = tf.random.uniform(shape = [], minval=-15, maxval=15, dtype=tf.dtypes.int32)
    j = tf.random.uniform(shape = [], minval=-15, maxval=15, dtype=tf.dtypes.int32)
    Y = tf.roll(Y, [i,j], axis = [0,1])
    X = tf.roll(X, [i,j], axis = [0,1])
    return X,Y
    


# In[24]:


@tf.function()
def random_jitter(input_image, real_image):
    input_image = tf.cast(input_image, tf.float32)
    input_image = tf.reshape(input_image, [192,192,4])
    real_image = tf.reshape(real_image, [192,192,4])
    
    if tf.random.uniform(())> 0.5:
        input_image, real_image = random_translation(input_image, real_image)
        
    input_image, real_image = resize(input_image, real_image, 250, 250)

    input_image, real_image = random_crop(input_image, real_image)

    if tf.random.uniform(()) > 0.5:
        # Random mirroring
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)
        

    return input_image, real_image

def test_reshape(X,Y):
    X = tf.cast(X, tf.float32)
    X = tf.reshape(X, [192,192,4])
    Y = tf.reshape(Y, [192,192,4])
    return X,Y
    


# In[25]:


BATCH_SIZE = 10
BUFFER_SIZE = 500


# In[26]:


train_dataset = tf.data.Dataset.list_files('Data/PreTrain/*.npz')
train_dataset = train_dataset.map(lambda item: tf.numpy_function(
          load_image_train, [item], [tf.float32, tf.float32]),
          num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.map(random_jitter, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)


# In[27]:


train_dataset = tf.data.Dataset.list_files('DataCgan/Train/*.npz')
train_dataset = train_dataset.map(lambda item: tf.numpy_function(
          load_image_train, [item], [tf.float64, tf.float32]),
          num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.map(random_jitter, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)


# In[28]:


test_dataset = tf.data.Dataset.list_files('Data/Test/*.npz')
test_dataset = test_dataset.map(lambda item: tf.numpy_function(
          load_image_test, [item], [tf.float64, tf.float32]),
          num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.map(test_reshape)
test_dataset = test_dataset.batch(BATCH_SIZE)


# In[29]:


def conv_block(input, num_filters):
    x = tf.keras.layers.Conv2D(num_filters, 3, padding="same")(input)
    x = tf.keras.layers.BatchNormalization()(x)   #Not in the original network. 
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv2D(num_filters, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)  #Not in the original network
    x = tf.keras.layers.Activation("relu")(x)

    return x

#Encoder block: Conv block followed by maxpooling


def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = tf.keras.layers.MaxPool2D((2, 2))(x)
    return x, p   

#Decoder block
#skip features gets input from encoder for concatenation

def decoder_block(input, skip_features, num_filters):
    x = tf.keras.layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = tf.keras.layers.Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

#Build Unet using the blocks
def build_unet(input_shape, n_classes):
    inputs = tf.keras.layers.Input(input_shape)

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1024) #Bridge

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    if n_classes == 1:  #Binary
      activation = 'sigmoid'
    else:
      activation = 'softmax'

    outputs = tf.keras.layers.Conv2D(n_classes, 1, padding="same", activation=activation)(d4)  #Change the activation based on n_classes
    print(activation)

    model = tf.keras.Model(inputs, outputs, name="U-Net")
    return model


# In[30]:


input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
input_shape


# In[31]:


unet = build_unet(input_shape,Y.shape[-1])
tf.keras.utils.plot_model(unet, show_shapes=True, dpi=64);


# In[32]:


def generate_images(model, test_input, tar):
  prediction = model(test_input, training=True)
  plt.figure(figsize=(15, 15))

  display_list = [test_input[0,:,:,1], np.argmax(tar[0], axis =2), np.argmax(prediction[0], axis = 2)]
  title = ['Input Image', 'Ground Truth', 'Predicted Image']

  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    # Getting the pixel values in the [0, 1] range to plot.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.show()


# In[33]:


#unet_optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0)
unet_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


# In[34]:


from IPython import display


# In[35]:


def dice0(y_true, y_pred, smooth = 1e-7):
    y_true_f = tf.reshape(tf.cast(y_true[:,:,:,0], 'float32'), [-1]) 
    y_pred_f = tf.reshape(tf.cast(y_pred[:,:,:,0], 'float32'), [-1])
    return (2*tf.reduce_sum(tf.abs(y_true_f*y_pred_f)))/(tf.reduce_sum(
        y_true_f**2 + y_pred_f**2)+smooth)

def dice1(y_true, y_pred, smooth = 1e-7):  
    y_true_f = tf.reshape(tf.cast(y_true[:,:,:,1], 'float32'), [-1]) 
    y_pred_f = tf.reshape(tf.cast(y_pred[:,:,:,1], 'float32'), [-1])
    return (2*tf.reduce_sum(tf.abs(y_true_f*y_pred_f)))/(tf.reduce_sum(
        y_true_f**2 + y_pred_f**2)+smooth)

def dice2(y_true, y_pred, smooth = 1e-7):
    y_true_f = tf.reshape(tf.cast(y_true[:,:,:,2], 'float32'), [-1]) 
    y_pred_f = tf.reshape(tf.cast(y_pred[:,:,:,2], 'float32'), [-1])
    return (2*tf.reduce_sum(tf.abs(y_true_f*y_pred_f)))/(tf.reduce_sum(
        y_true_f**2 + y_pred_f**2)+smooth)

def dice3(y_true, y_pred, smooth = 1e-7):  
    y_true_f = tf.reshape(tf.cast(y_true[:,:,:,3], 'float32'), [-1]) 
    y_pred_f = tf.reshape(tf.cast(y_pred[:,:,:,3], 'float32'), [-1])
    return (2*tf.reduce_sum(tf.abs(y_true_f*y_pred_f)))/(tf.reduce_sum(
        y_true_f**2 + y_pred_f**2)+smooth)

def dice_loss1(y_true, y_pred):
    a0 = 1
    a1 = 1
    a2 = 1
    a3 = 1
    return 1-(a0*dice0(y_true,y_pred)+a1*dice1(y_true,y_pred)+a2*dice2(
        y_true,y_pred)+a3*dice3(y_true,y_pred))/(a0+a1+a2+a3)


# In[36]:


epochs = 30
checkpoint_dir = './training_checkpoints_Unet'
checkpoint_filepath = os.path.join(checkpoint_dir, "ckpt_Unet_{epoch:02d}")
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    verbose = 1,
    save_weights_only=False,
    monitor="val_loss",
    mode='min',
    save_best_only=False)


# In[37]:


unet.compile(optimizer=unet_optimizer, loss=dice_loss1, metrics=[dice0,dice1,dice2,dice3])


# In[38]:


history = unet.fit(train_dataset,
                    verbose=1, 
                    epochs=epochs,
                    #steps_per_epoch =X_train.shape[0]//BATCH_SIZE-1,
                    #validation_steps = X.shape[0]//BATCH_SIZE-1,
                    validation_data=test_dataset, 
                    callbacks=[model_checkpoint_callback],
                    shuffle=True)


# In[ ]:


model.save('/content/drive/MyDrive/Colab Notebooks/saved_models/tutorial119_sandstone_50epochs.hdf5')


# In[ ]:


unet.save('./Modelli/Unet_Allexp_60_120slice_50epch_Alldice_DataCgan_09_02_2022.hdf5')


# In[40]:


with open('./History/unet_05_04_2022.json', 'w') as handle: # saving the history of the model
    json.dump(history.history, handle)


# In[5]:


path = 'History/unet_05_04_2022.json'


# In[6]:


history = json.load(open(path, 'r'))


# In[7]:


loss = history['loss']
val_loss = history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
xposition = [15,16,17,18,19]
for xc in xposition:
    plt.axvline(x=xc, color='orange', linestyle='--')
    if xc == len(xposition): plt.axvline(x=xc, color='orange', linestyle='--', label = "ckpt")
    else: plt.axvline(x=xc, color='orange', linestyle='--')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.ylim([0,0.5])
plt.legend(loc = "upper right")
plt.legend()
plt.savefig('Images/Unet_centrlckpt')
plt.show()


# In[31]:


loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
xposition = [27,28,29]
for xc in xposition:
    plt.axvline(x=xc, color='orange', linestyle='--')
    if xc == len(xposition): plt.axvline(x=xc, color='orange', linestyle='--', label = "ckpt")
    else: plt.axvline(x=xc, color='orange', linestyle='--')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xlim([0.8,30.5])
plt.ylim([0,0.5])
plt.legend(loc = "upper right")
plt.legend()
plt.savefig("Images/Unet_history_finalckpt")
plt.show()


# In[ ]:


loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xlim([0.8,10.5])
plt.ylim([0,1.4])
plt.legend(loc = "upper right")
plt.legend()
plt.show()

rec = history.history['dice0']
val_rec = history.history['val_dice0']

plt.plot(epochs, rec, 'y', label='Training Dice 0 ')
plt.plot(epochs, val_rec, 'r', label='Validation Dice 0')
plt.title('Training and Validation Dice 0')
plt.xlabel('Epochs')
plt.ylabel('Dice0')
plt.legend()
plt.savefig("Images/Unet_Dice0")
plt.show()

rec1 = history.history['dice1']
val_rec1 = history.history['val_dice1']

plt.plot(epochs, rec1, 'y', label='Training Dice 1 ')
plt.plot(epochs, val_rec1, 'r', label='Validation Dice 1')
plt.title('Training and Validation Dice 1')
plt.xlabel('Epochs')
plt.ylabel('Dice1')
plt.xlim([0.8,30.5])
plt.ylim([0.25,1.0])
plt.legend()
plt.savefig("Images/Unet_Dice1")
plt.show()

rec2 = history.history['dice2']
val_rec2 = history.history['val_dice2']

plt.plot(epochs, rec2, 'y', label='Training Dice 2')
plt.plot(epochs, val_rec2, 'r', label='Validation Dice 2')
plt.title('Training and Validation Dice 2')
plt.xlabel('Epochs')
plt.ylabel('Dice2')
plt.xlim([0.8,30.5])
plt.ylim([0.6,1.0])
plt.legend()
plt.savefig("Images/Unet_Dice2")
plt.show()


rec3 = history.history['dice3']
val_rec3 = history.history['val_dice3']

plt.plot(epochs, rec3, 'y', label='Training Dice 3')
plt.plot(epochs, val_rec3, 'r', label='Validation Dice 3')
plt.title('Training and Validation Dice 3')
plt.xlabel('Epochs')
plt.ylabel('Dice3')
plt.xlim([0.8,30.5])
plt.ylim([0.6,1.0])
plt.legend()
plt.savefig("Images/Unet_Dice3")
plt.show()


# In[6]:



model = keras.models.load_model("Modelli/Unet_Allexp_60_120slice_50epch_Alldice_DataCgan_09_02_2022.hdf5",
                                custom_objects={'dice0': dice0, 'dice1': dice1, 'dice2': dice2, 'dice3': dice3, "dice_loss1":dice_loss1})


# In[7]:


model.summary()


# In[ ]:


for i,j in test_dataset:
    y = np.argmax(model.predict(i), axis = 3)
    a = np.argmax(j, axis = 3)
    for k in range(y.shape[0]):
        plt.figure(figsize=(12, 8))
        plt.subplot(231)
        plt.title("Channel 0")
        plt.imshow(i[k,:,:,0])
        plt.subplot(232)
        plt.title("Predicted Seg")
        plt.imshow(y[k])
        plt.subplot(233)
        plt.title("Ground Truth ")
        plt.imshow(a[k])
        plt.show()


# In[ ]:


for i,j in test_dataset:
    y = np.argmax(model.predict(i), axis = 3)
    a = np.argmax(j, axis = 3)
    a = np.ma.masked_where(a == 0, a)

    for k in range(y.shape[0]):
        plt.figure(figsize=(5, 5))
        plt.title("BraTs segmentation")
        plt.imshow(i[k,:,:,0], cmap = "gist_gray")
        #plt.title("Predicted Seg")
        #plt.imshow(y[k], alpha = 0.1)
        #plt.title("Ground Truth ")
        plt.imshow(a[k], alpha = 0.5)
        plt.show()


# In[8]:


tf.keras.utils.plot_model(model)


# In[ ]:




