import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
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
from radiomics import featureextractor, firstorder, shape


import itertools
from keras.models import load_model
from keras.utils.vis_utils import plot_model
from sklearn import preprocessing
import SimpleITK as sitk


file_list = glob(os.path.join("/home/tordatom/Dati_Imaging/BraTs_19/MICCAI_BraTS_2019_Data_Training/HGG", "*", ""))


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

def normalize(t1):
    for i in range(len(t1[:,0,0,0,0])):
        for j in range(len(t1[0,0,0,0,:])):
            t1[i,:,:,:,j] = 2*t1[i,:,:,:,j]/np.max(t1[i,:,:,:,j])-1
    return t1

from sklearn.model_selection import train_test_split
train_to_test_ratio=0.8


channels1, seg1 = load_example(file_list)
channels, seg = channels1[:,27:219, 27:219, 60:120, :], seg1[:,27:219, 27:219, 60:120]
channels = normalize(channels)
seg = seg.reshape(np.append(seg.shape,1))
seg[seg == 4] = 3
seg = keras.utils.to_categorical(seg)
X_train,X_test,Y_train,Y_test=train_test_split(np.array(channels),
                                               np.array(seg),train_size=train_to_test_ratio,
                                               shuffle=False, random_state=1234)

#check if there exist some slice with empty mask or mask too small for pyradiomics, and delete it.

new_X = {}
new_Y = {}
for case_id in range(X_train.shape[0]):
    mask = np.argmax(Y_train[case_id], axis = 3)
    if int(np.unique(mask).shape[0]) == 1:
        print(f"empty mask! patient: {case_id}")
        continue
    if len(mask[mask != 0]) <= 5:
        print(f"mask too small! patient: {case_id}")
    else: 
        new_X[case_id] = X_train[case_id]
        new_Y[case_id] = np.argmax(Y_train[case_id], axis = 3)
    


extractor = featureextractor.RadiomicsFeatureExtractor()
# hang on to all our features
features = {}

for i in list(new_X.keys()):
    arr = np.zeros(X_train.shape[-1], dtype = "object")
    print(f"paziente: {case_id}")
               
    for channel in range(X_train.shape[-1]):
        print(f"channel: {channel}")
        image = sitk.GetImageFromArray(X_train[case_id,:,:,:,channel])
        y = np.argmax(Y_train[case_id], axis = 3)
        
        for i in np.unique(y):
            if i == 0: continue
            else:  y[y == i] = 1
          
        mask = sitk.GetImageFromArray(y)
        
        arr[channel] = extractor.execute(image, mask) #extract radiomic features

    features[case_id] = arr
    print("\n")


feature_names = list(sorted(filter ( lambda k: k.startswith("original_"), features[0][0] )))
samples = np.zeros((len(new_X.keys()), 4, len(feature_names)))


for i,case_id in enumerate(list(new_X.keys())):
    for channel in range(X_test.shape[-1]):
        a = np.array([])
        for feature_name in feature_names:
            a = np.append(a, features[case_id][channel][feature_name])
        samples[i,channel,:] = a
    
# May have NaNs
samples = np.nan_to_num(samples)



np.savez(f"/home/tordatom/Dati_Imaging/BraTs_19/Segmentation2D/Radiomic_Features/features_train2.npz", samples = samples)
np.savez(f"/home/tordatom/Dati_Imaging/BraTs_19/Segmentation2D/Radiomic_Features/train_examples2.npz", train_list = list(new_X.keys()))


np.savez(f"/home/tordatom/Dati_Imaging/BraTs_19/Segmentation2D/Radiomic_Features/features_test2.npz", samples = samples)
np.savez(f"/home/tordatom/Dati_Imaging/BraTs_19/Segmentation2D/Radiomic_Features/test_examples2.npz", test_list = list(new_X.keys()))


samples_train = np.load("/home/tordatom/Dati_Imaging/BraTs_19/Segmentation2D/Radiomic_Features/features_train2.npz")["samples"]
feature_names = np.load("/home/tordatom/Dati_Imaging/BraTs_19/Segmentation2D/Radiomic_Features/feature_names.npz")["feature_names"]
train_list = np.load("/home/tordatom/Dati_Imaging/BraTs_19/Segmentation2D/Radiomic_Features/train_examples2.npz")["train_list"]
samples_test = np.load("/home/tordatom/Dati_Imaging/BraTs_19/Segmentation2D/Radiomic_Features/features_test2.npz")["samples"]
test_list = np.load("/home/tordatom/Dati_Imaging/BraTs_19/Segmentation2D/Radiomic_Features/test_examples2.npz")["test_list"]
prop = np.load(f"/home/tordatom/Dati_Imaging/BraTs_19/Segmentation2D/Radiomic_Features/prop_index.npz")["proponents"]
map_test = np.load(f"/home/tordatom/Dati_Imaging/BraTs_19/Segmentation2D/Radiomic_Features/map_test.npz", allow_pickle= True)["map_test"]

tracin_score = np.load("/home/tordatom/Dati_Imaging/BraTs_19/Segmentation2D/Radiomic_Features/score_dice1.npy",allow_pickle='TRUE').item()


#we choose 1 channel to analyse, is a test choice. 
samples_train = samples_train[:,1,:]
samples_test = samples_test[:,1,:]


from scipy import stats

tot = np.concatenate((samples_test, samples_train))
zscore_tot = stats.zscore(tot, axis = 0)
zscore_test, zscore_train = np.split(zscore_tot, [samples_test.shape[0]])


map_test = map_test.reshape(1)
samples = np.einsum('ik,jk->ijk',zscore_test,zscore_train) #product features_test x features_train

#we want to extract the correct proponents for the given test example
proponents = np.zeros(len(prop), dtype = object)
opponents = np.zeros(len(prop), dtype = object)
neutrals = np.zeros(len(prop), dtype = object)

for i,h in enumerate(prop):
    proponents[i] = h[:69]
    neutrals[i] = h[69:138]
    opponents[i] = h[138:]

p = np.zeros(len(prop), dtype = "object")
o = np.zeros(len(prop), dtype = "object")
n = np.zeros(len(prop), dtype = "object")
index = np.zeros((len(prop), len(train_list)))

for j in range(len(index)):
    s = []
    for k,i in enumerate(proponents[j]):
        try:
            s.append(int(np.array(np.where(train_list == int(i/10)))))
        except:
            pass
    p[j] = s
for j in range(len(index)):
    s = []
    for k,i in enumerate(opponents[j]):
        try:
            s.append(int(np.array(np.where(train_list == int(i/10)))))
        except:
            pass
    o[j] = s
for j in range(len(index)):
    s = []
    for k,i in enumerate(neutrals[j]):
        try:
            s.append(int(np.array(np.where(train_list == int(i/10)))))
        except:
            pass
    n[j] = s
    
for i in range(len(index)):
    for j in p[i]:
        index[i][j] = 1
    for j in o[i]:
        index[i][j] = -1



index2 = np.zeros((52,207))
for i in range(int(len(index)/10)):
    
    index2[i] = index[i] 



#plot of the product of features ordered on the base of the tracin score
for j in range(len(samples)):
    for i in range(len(feature_names)):
        fig = plt.figure(1)
        ax = plt.axes([0., 0., 1., 1.])

        s1 = 100
        f = i
        c = 0
        test = j
        a, b = np.polyfit(range(len(n[test])+len(p[test])+len(o[test])), samples[j,p[test]+o[test]+n[test],f], 1)


        plt.scatter(range(len(p[test])), samples[j,p[test],f], color='pink', alpha=1.0, s=s1, lw=1, label = "proponent")
        plt.scatter(range(len(n[test])+len(p[test]), len(n[test])+len(p[test])+len(o[test])), samples[j,o[test], f], color='orange', alpha=1.0, s=s1, lw=1, label = "opponent")
        plt.scatter(range(len(p[test]), len(n[test])+len(p[test])), samples[j,n[test],f], color='green', alpha=1.0, s=s1, lw=1, label = "neutral")
        plt.plot(range(len(n[test])+len(p[test])+len(o[test])), a*range(len(n[test])+len(p[test])+len(o[test]))+b)

        plt.text(0,max(samples[j,:,f])+0.01*max(samples[j,:,f]), 'y = '  + '{:.10f}'.format(a) + 'x'+ '+ {:.2f}'.format(b))

        plt.legend(scatterpoints=1, loc=5, shadow=False)
        plt.title(f"scatter plot della feature: {feature_names[i]}, test: {j}")
        plt.show()

 
#we want to calculate the correlation between the tracin score and the product of radiomic features, ordered by tracin score

X = np.zeros((int((len(tracin_score)/10)), len(tracin_score[0])))
for i in range(int((len(tracin_score)/10))):
    for j,k in enumerate(proponents[i].tolist()+neutrals[i].tolist()+opponents[i].tolist()):
        X[i][j] = tracin_score[i][k]


cor = np.zeros((int((len(tracin_score)/10)), len(feature_names)))
for i in range(len(cor)):
    cor[i] = [np.corrcoef(X[i], samples[i,p[i]+n[i]+o[i],f], rowvar = False)[1,0] for f in range(107)]


#plot of the distribution of correlation's coefficients for each training example
for i in range(len(cor)):
    plt.hist(np.abs(cor[i]), bins = 20)
    plt.title(f"Distribuzione dei coef di cor per il paziente: {i}")
    plt.show()


#we can use the angular coefficient between the ordered product of features and the tracin_score, as an importance metric in order to extract witch features are the most important

coeff = np.zeros((len(samples), len(feature_names)))
for i in range(len(coeff)):
    for f in range(len(feature_names)):
        coeff[i,f] = np.polyfit(range(len(n[i])+len(p[i])+len(o[i])), samples[i,p[i]+o[i]+n[i],f], 1)[0]



for i in range(len(coeff[0])):
    plt.hist(coeff[:,i], bins = 20)
    plt.title(f"Distribuzione dei coeff angolari per la feature: {feature_names[i]}")
    plt.show()



for i in range(len(coeff)):
    plt.hist(np.abs(coeff[i]), bins = 20)
    plt.title(f"Distribuzione dei coefficienti angolari per il paziente: {i}")
    plt.show()






