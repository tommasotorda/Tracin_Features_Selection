{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "0f1a2ec7",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import os\n",
    "os.environ[\"CUDA_DEVICE_ORDER\"]=\"PCI_BUS_ID\"   # see issue #152\n",
    "os.environ[\"CUDA_VISIBLE_DEVICES\"]=\"0\"\n",
    "\n",
    "from glob import glob\n",
    "import matplotlib.pyplot as plt\n",
    "import tensorflow as tf\n",
    "from tensorflow import keras\n",
    "import time\n",
    "import datetime\n",
    "import json"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "43460034",
   "metadata": {},
   "outputs": [],
   "source": [
    "import itertools\n",
    "from keras.models import load_model\n",
    "from keras.utils.vis_utils import plot_model\n",
    "from sklearn import preprocessing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fbe5329c",
   "metadata": {},
   "outputs": [],
   "source": [
    "file_list = glob(os.path.join(\"/home/tordatom/Dati_Imaging/BraTs_19/MICCAI_BraTS_2019_Data_Training/HGG\", \"*\", \"\"))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 119,
   "id": "58d24162",
   "metadata": {},
   "outputs": [],
   "source": [
    "def dice0(y_true, y_pred, smooth = 1e-7):\n",
    "    y_true_f = tf.reshape(tf.cast(y_true[:,:,:,0], 'float32'), [len(y_true), -1]) \n",
    "    y_pred_f = tf.reshape(tf.cast(y_pred[:,:,:,0], 'float32'), [len(y_true), -1])\n",
    "    return (2*tf.reduce_sum(tf.abs(y_true_f*y_pred_f), axis = 1)+smooth)/(tf.reduce_sum(\n",
    "        y_true_f**2 + y_pred_f**2, axis = 1)+smooth)\n",
    "\n",
    "def dice1(y_true, y_pred, smooth = 1e-7):  \n",
    "    y_true_f = tf.reshape(tf.cast(y_true[:,:,:,1], 'float32'), [len(y_true), -1]) \n",
    "    y_pred_f = tf.reshape(tf.cast(y_pred[:,:,:,1], 'float32'), [len(y_true), -1])\n",
    "    return (2*tf.reduce_sum(tf.abs(y_true_f*y_pred_f), axis = 1)+smooth)/(tf.reduce_sum(\n",
    "        y_true_f**2 + y_pred_f**2, axis = 1)+smooth)\n",
    "\n",
    "def dice2(y_true, y_pred, smooth = 1e-7):\n",
    "    y_true_f = tf.reshape(tf.cast(y_true[:,:,:,2], 'float32'), [len(y_true), -1]) \n",
    "    y_pred_f = tf.reshape(tf.cast(y_pred[:,:,:,2], 'float32'), [len(y_true), -1])\n",
    "    return (2*tf.reduce_sum(tf.abs(y_true_f*y_pred_f), axis = 1)+smooth)/(tf.reduce_sum(\n",
    "        y_true_f**2 + y_pred_f**2, axis = 1)+smooth)\n",
    "\n",
    "def dice3(y_true, y_pred, smooth = 1e-7):  \n",
    "    y_true_f = tf.reshape(tf.cast(y_true[:,:,:,3], 'float32'), [len(y_true), -1]) \n",
    "    y_pred_f = tf.reshape(tf.cast(y_pred[:,:,:,3], 'float32'), [len(y_true), -1])\n",
    "    return (2*tf.reduce_sum(tf.abs(y_true_f*y_pred_f), axis = 1)+smooth)/(tf.reduce_sum(\n",
    "        y_true_f**2 + y_pred_f**2, axis = 1)+smooth)\n",
    "\n",
    "def dice_loss1(y_true, y_pred):\n",
    "    a0 = 0\n",
    "    a1 = 1\n",
    "    a2 = 1\n",
    "    a3 = 1\n",
    "    return 1-(a0*dice0(y_true,y_pred)+a1*dice1(y_true,y_pred)+a2*dice2(\n",
    "        y_true,y_pred)+a3*dice3(y_true,y_pred))/(a0+a1+a2+a3)\n",
    "\n",
    "def cce(y_true, y_pred):\n",
    "    return tf.keras.losses.categorical_crossentropy(y_true, y_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "034709e0",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2023-11-25 10:11:07.797662: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1510] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 21114 MB memory:  -> device: 0, name: Tesla V100-SXM2-32GB, pci bus id: 0000:06:00.0, compute capability: 7.0\n"
     ]
    }
   ],
   "source": [
    "checkpoint_dir = './Train_SGD/model2'\n",
    "file_list_ckpt = glob(os.path.join(checkpoint_dir, \"*\"))\n",
    "file_list_ckpt.sort()\n",
    "\n",
    "model = tf.keras.models.load_model(file_list_ckpt[-1], \n",
    "                                   custom_objects={'dice0': dice0, 'dice1': dice1, \n",
    "                                                   'dice2': dice2, 'dice3': dice3,\n",
    "                                                   \"cce\": cce,\n",
    "                                                   \"dice_loss1\":dice_loss1})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "1cba0641",
   "metadata": {},
   "outputs": [],
   "source": [
    "def load_image_train(image_file):\n",
    "    data = np.load(image_file)\n",
    "    image_file = str(image_file)\n",
    "    index = \"/\".join(image_file.split(\"/\", image_file.count(\"/\"))[image_file.count(\"/\"):])\n",
    "    index = int(index[6:len(index)-5])\n",
    "    return index, data['X_train'], data['Y_train']\n",
    "\n",
    "def load_image_test(image_file):\n",
    "    data = np.load(image_file)\n",
    "    image_file = str(image_file)\n",
    "    index = \"/\".join(image_file.split(\"/\", image_file.count(\"/\"))[image_file.count(\"/\"):])\n",
    "    index = int(index[5:len(index)-5])\n",
    "    return index, data['X_test'], data['Y_test']\n",
    "\n",
    "\n",
    "def t_reshape(index, X,Y):\n",
    "    X = tf.cast(X, tf.float32)\n",
    "    Y = tf.cast(Y, tf.float32)\n",
    "    X = tf.reshape(X, [192,192,4])\n",
    "    Y = tf.reshape(Y, [192,192,4])\n",
    "    return index,X,Y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "fb4244e0",
   "metadata": {},
   "outputs": [],
   "source": [
    "checkpoint_dir = \"./Data/DataTracin_train\"\n",
    "file_list_train = glob(os.path.join(checkpoint_dir, \"*\"))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "da25f77b",
   "metadata": {},
   "outputs": [],
   "source": [
    "BATCH_SIZE = 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "55880210",
   "metadata": {},
   "outputs": [],
   "source": [
    "train_dataset = tf.data.Dataset.list_files(file_list_train, shuffle = False)\n",
    "train_dataset = train_dataset.map(lambda item: tf.numpy_function(\n",
    "          load_image_train, [item], [tf.int64, tf.double, tf.double]),\n",
    "          num_parallel_calls=tf.data.AUTOTUNE)\n",
    "train_dataset = train_dataset.map(t_reshape)\n",
    "train_dataset = train_dataset.batch(BATCH_SIZE)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "e45be77a",
   "metadata": {},
   "outputs": [],
   "source": [
    "checkpoint_dir = \"./Data/DataTracin_test\"\n",
    "file_list_test = glob(os.path.join(checkpoint_dir, \"*\"))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "03f17da5",
   "metadata": {},
   "outputs": [],
   "source": [
    "test_dataset = tf.data.Dataset.list_files(file_list_test, shuffle = False)\n",
    "test_dataset = test_dataset.map(lambda item: tf.numpy_function(\n",
    "          load_image_test, [item], [tf.int64, tf.double, tf.double]),\n",
    "          num_parallel_calls=tf.data.AUTOTUNE)\n",
    "test_dataset = test_dataset.map(t_reshape)\n",
    "test_dataset = test_dataset.batch(BATCH_SIZE)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "604b204c",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2023-11-25 10:15:54.387959: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:185] None of the MLIR Optimization Passes are enabled (registered 2)\n"
     ]
    }
   ],
   "source": [
    "map_train = {}\n",
    "for i,j in enumerate(train_dataset):\n",
    "    map_train[int(j[0])] = i"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "3aa77318",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "conv2d (3, 3, 4, 64)\n",
      "conv2d_1 (3, 3, 64, 64)\n",
      "conv2d_2 (3, 3, 64, 128)\n",
      "conv2d_3 (3, 3, 128, 128)\n",
      "conv2d_4 (3, 3, 128, 256)\n",
      "conv2d_5 (3, 3, 256, 256)\n",
      "conv2d_6 (3, 3, 256, 512)\n",
      "conv2d_7 (3, 3, 512, 512)\n",
      "conv2d_8 (3, 3, 512, 1024)\n",
      "conv2d_9 (3, 3, 1024, 1024)\n",
      "conv2d_transpose (2, 2, 512, 1024)\n",
      "conv2d_10 (3, 3, 1024, 512)\n",
      "conv2d_11 (3, 3, 512, 512)\n",
      "conv2d_transpose_1 (2, 2, 256, 512)\n",
      "conv2d_12 (3, 3, 512, 256)\n",
      "conv2d_13 (3, 3, 256, 256)\n",
      "conv2d_transpose_2 (2, 2, 128, 256)\n",
      "conv2d_14 (3, 3, 256, 128)\n",
      "conv2d_15 (3, 3, 128, 128)\n",
      "conv2d_transpose_3 (2, 2, 64, 128)\n",
      "conv2d_16 (3, 3, 128, 64)\n",
      "conv2d_17 (3, 3, 64, 64)\n",
      "conv2d_18 (1, 1, 64, 4)\n"
     ]
    }
   ],
   "source": [
    "for layer in model.layers:\n",
    " # check for convolutional layer\n",
    "    if 'conv' not in layer.name:\n",
    "        continue\n",
    " # get filter weights\n",
    "    filters, biases = layer.get_weights()\n",
    "    print(layer.name, filters.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "0c35e023",
   "metadata": {},
   "outputs": [],
   "source": [
    "filters, biases = model.layers[1].get_weights()\n",
    "f_min, f_max = filters.min(), filters.max()\n",
    "filters = (filters - f_min) / (f_max - f_min)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "82e93943",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 24 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# plot first few filters\n",
    "n_filters, ix = 6, 1\n",
    "for i in range(n_filters):\n",
    "    # get the filter\n",
    "    f = filters[:, :, :, i]\n",
    "    # plot each channel separately\n",
    "    for j in range(4):\n",
    " # specify subplot and turn of axis\n",
    "        ax = plt.subplot(n_filters, 4, ix)\n",
    "        ax.set_xticks([])\n",
    "        ax.set_yticks([])\n",
    "        # plot filter channel in grayscale\n",
    "        plt.imshow(f[:, :, j], cmap='gray')\n",
    "        ix += 1\n",
    "# show the figure\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "2c1e682d",
   "metadata": {},
   "outputs": [],
   "source": [
    "#defining a model with output equal to the penultimate layer of the previous one\n",
    "outputs = [layer.output for layer in model.layers]\n",
    "model2 = tf.keras.Model(inputs=model.inputs, outputs=outputs[:-1])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "aca6b009",
   "metadata": {},
   "source": [
    "# Feature Maps"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f82d6f49",
   "metadata": {},
   "outputs": [],
   "source": [
    "l = 0\n",
    "for i in train_dataset:\n",
    "    l += 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6ba6cfc5",
   "metadata": {},
   "outputs": [],
   "source": [
    "#evaluation of feature maps on tumour class pixels only\n",
    "feature_maps_train = np.zeros((3,l,192, 192, 64))\n",
    "for j,i in enumerate(train_dataset):\n",
    "    pred = model1.predict(i[1])[66]\n",
    "    for dice in range(3):\n",
    "        for k in range(64):\n",
    "            feature_maps_train[dice,j,:,:,k] = pred[0,:,:,k]*i[2][0,:,:,dice+1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c51f14c6",
   "metadata": {},
   "outputs": [],
   "source": [
    "l = 0\n",
    "for i in test_dataset:\n",
    "    l += 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5179300f",
   "metadata": {},
   "outputs": [],
   "source": [
    "feature_maps_test = np.zeros((3,l,192, 192, 64))\n",
    "for j,i in enumerate(test_dataset):\n",
    "    pred = model1.predict(i[1])[66]\n",
    "    for dice in range(3):\n",
    "        for k in range(64):\n",
    "            feature_maps_test[dice,j,:,:,k] = pred[0,:,:,k]*i[2][0,:,:,dice+1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a4cc73a3",
   "metadata": {},
   "outputs": [],
   "source": [
    "np.savez(f\"/home/tordatom/Dati_Imaging/BraTs_19/Segmentation2D/feature_maps_test_SGD2.npz\", \n",
    "         feature_maps_test = feature_maps_test)\n",
    "\n",
    "np.savez(f\"/home/tordatom/Dati_Imaging/BraTs_19/Segmentation2D/feature_maps_train_SGD2.npz\", \n",
    "         feature_maps_train = feature_maps_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "286abf01",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "# plot all 64 maps in an 8x8 squares\n",
    "import matplotlib as mpl\n",
    "square = 8\n",
    "ix = 1\n",
    "f = feature_maps_train[110]\n",
    "for _ in range(square):\n",
    "    plt.figure(figsize=(15, 15))\n",
    "    for _ in range(square):\n",
    "# specify subplot and turn of axis\n",
    "        ax = plt.subplot(square, square, ix)\n",
    "        ax.set_xticks([])\n",
    "        ax.set_yticks([])\n",
    "# plot filter channel in grayscale\n",
    "        plt.imshow(f[ix-1, :, :], cmap='magma')\n",
    "                   #norm = mpl.colors.Normalize(vmin=-1, vmax=1))\n",
    "        ix += 1\n",
    "# show the figure\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7bcd0168",
   "metadata": {},
   "source": [
    "# Kernels selection"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5745ed13",
   "metadata": {},
   "outputs": [],
   "source": [
    "def load_image_train(image_file):\n",
    "    data = np.load(image_file)\n",
    "    return data['X_train'], data['Y_train']\n",
    "\n",
    "def load_image_test(image_file):\n",
    "    data = np.load(image_file)\n",
    "    return data['X_test'], data['Y_test']\n",
    "def test_reshape(X,Y):\n",
    "    X = tf.cast(X, tf.double)\n",
    "    X = tf.reshape(X, [192,192,4])\n",
    "    Y = tf.reshape(Y, [192,192,4])\n",
    "    return X,Y\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f35da0ef",
   "metadata": {},
   "outputs": [],
   "source": [
    "BATCH_SIZE = 1\n",
    "BUFFER_SIZE = 500\n",
    "\n",
    "train_dataset = tf.data.Dataset.list_files('./Data/DataTracin_train/*.npz')\n",
    "train_dataset = train_dataset.map(lambda item: tf.numpy_function(\n",
    "          load_image_train, [item], [tf.double, tf.double]),\n",
    "          num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)\n",
    "train_dataset = train_dataset.map(test_reshape)\n",
    "train_dataset = train_dataset.batch(BATCH_SIZE)\n",
    "\n",
    "test_dataset = tf.data.Dataset.list_files('./Data/DataTracin_test/*.npz')\n",
    "test_dataset = test_dataset.map(lambda item: tf.numpy_function(\n",
    "          load_image_test, [item], [tf.double, tf.double]),\n",
    "          num_parallel_calls=tf.data.AUTOTUNE)\n",
    "test_dataset = test_dataset.map(test_reshape)\n",
    "test_dataset = test_dataset.batch(BATCH_SIZE)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c14afdfb",
   "metadata": {},
   "outputs": [],
   "source": [
    "#evaluating the influence of each kernel\n",
    "_,_,d1,d2,d3,_= model.evaluate(train_dataset)\n",
    "\n",
    "score = np.zeros((64,3))\n",
    "for k in range(64):\n",
    "    dice_1 = []\n",
    "    dice_2 = []\n",
    "    dice_3 = []\n",
    "    l = [k]\n",
    "    ind = np.in1d(np.arange(64), l)\n",
    "    \n",
    "    for i in train_dataset:\n",
    "\n",
    "        y_true = i[1]\n",
    "\n",
    "        \n",
    "        feature_maps = model2.predict(i[0])\n",
    "        feature_maps[-1][0,:,:,ind] = feature_maps[-1][0,:,:,ind]*np.zeros((len(l),192,192))\n",
    "        y_pred = model.layers[-1](feature_maps[-1], training = False)\n",
    "        dice_1.append(dice1(y_true, y_pred))\n",
    "        dice_2.append(dice2(y_true, y_pred))\n",
    "        dice_3.append(dice3(y_true, y_pred))\n",
    "        \n",
    "    score[k] = [np.mean(dice_1), np.mean(dice_2), np.mean(dice_3)] \n",
    "    \n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "92681d6e",
   "metadata": {},
   "outputs": [],
   "source": [
    "#plot of the influence of each kernel, each point is coloured based on their selection\n",
    "inf = [d1,d2,d3]-score\n",
    "kernels = [1,6,8,10,14,17,20,25,26,30,31,32,36,38,40,42,43,45,47,50,54,56,59,60,61,62]\n",
    "ind = np.in1d(np.arange(64), kernels)\n",
    "\n",
    "plt.figure(figsize = (25,5))\n",
    "plt.subplot(1,3,1)\n",
    "plt.scatter(kernels, inf[ind,0], c = \"r\")\n",
    "\n",
    "\n",
    "kernels1 = [j for j in range(64) if j not in kernels]\n",
    "ind = np.in1d(np.arange(64), kernels1)\n",
    "plt.scatter(kernels1, inf[ind,0], c = \"b\")\n",
    "plt.xlabel(\"Kernels\"\n",
    "plt.ylabel(\"Influence\")\n",
    "plt.title(\"Kernels influece on Dice 1\")\n",
    "\n",
    "plt.ylim(-0.009, 0.1)\n",
    "plt.subplot(1,3,2)\n",
    "\n",
    "kernels = [10,14,18,50,52,56,60]\n",
    "ind = np.in1d(np.arange(64), kernels)\n",
    "plt.scatter(kernels, inf[ind,1], c = \"r\")\n",
    "\n",
    "kernels1 = [j for j in range(64) if j not in kernels]\n",
    "ind = np.in1d(np.arange(64), kernels1)\n",
    "plt.scatter(kernels1, inf[ind,1], c = \"b\")\n",
    "plt.xlabel(\"Kernels\")\n",
    "plt.ylabel(\"Influence\")\n",
    "plt.title(\"Kernels influece on Dice 2\")\n",
    "plt.ylim(-0.009, 0.1)\n",
    "plt.subplot(1,3,3)\n",
    "\n",
    "kernels = [6,8,10,14,17,20,26,25,30,31,36,38,40,42,43,45,50,51,52,53,54,56,59,60,61,62]\n",
    "ind = np.in1d(np.arange(64), kernels)\n",
    "plt.scatter(kernels, inf[ind,2], c = \"r\", label = \" Selected kernels from TFI\")\n",
    "\n",
    "kernels1 = [j for j in range(64) if j not in kernels]\n",
    "ind = np.in1d(np.arange(64), kernels1)\n",
    "plt.scatter(kernels1, inf[ind,2], c = \"b\", label = \" Non-Selected kernels from TFI\")\n",
    "plt.xlabel(\"Kernels\")\n",
    "plt.ylabel(\"Influence\")\n",
    "plt.title(\"Kernels influece on Dice 3\")\n",
    "plt.legend(loc = (-0.95,-0.3))\n",
    "plt.ylim(-0.009, 0.1)\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.9"
  },
  "toc": {
   "base_numbering": 1,
   "nav_menu": {},
   "number_sections": true,
   "sideBar": true,
   "skip_h1_title": false,
   "title_cell": "Table of Contents",
   "title_sidebar": "Contents",
   "toc_cell": false,
   "toc_position": {},
   "toc_section_display": true,
   "toc_window_display": false
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
