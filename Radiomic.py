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
from radiomics import featureextractor, firstorder, shape


# In[2]:


import itertools
from keras.models import load_model
from keras.utils.vis_utils import plot_model
from sklearn import preprocessing


# In[3]:


import SimpleITK as sitk


# In[4]:


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

def normalize(t1):
    for i in range(len(t1[:,0,0,0,0])):
        for j in range(len(t1[0,0,0,0,:])):
            t1[i,:,:,:,j] = 2*t1[i,:,:,:,j]/np.max(t1[i,:,:,:,j])-1
    return t1

from sklearn.model_selection import train_test_split
train_to_test_ratio=0.8


# In[ ]:


channels1, seg1 = load_example(file_list)
#channels, seg = channels1[:,27:219, 27:219, 60:120, :], seg1[:,27:219, 27:219, 60:120]
channels, seg = channels1[:,27:219, 27:219, 60:120, :], seg1[:,27:219, 27:219, 60:120]
channels = normalize(channels)
seg = seg.reshape(np.append(seg.shape,1))
seg[seg == 4] = 3
seg = keras.utils.to_categorical(seg)
X_train,X_test,Y_train,Y_test=train_test_split(np.array(channels),
                                               np.array(seg),train_size=train_to_test_ratio,
                                               shuffle=True, random_state=1234)


# In[ ]:


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
    


# In[ ]:


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
        
        arr[channel] = extractor.execute(image, mask)

    features[case_id] = arr
    print("\n")


# In[ ]:


feature_names = list(sorted(filter ( lambda k: k.startswith("original_"), features[0][0] )))


# In[ ]:


samples = np.zeros((len(new_X.keys()), 4, len(feature_names)))


for i,case_id in enumerate(list(new_X.keys())):
    for channel in range(X_test.shape[-1]):
        a = np.array([])
        for feature_name in feature_names:
            a = np.append(a, features[case_id][channel][feature_name])
        samples[i,channel,:] = a
    
# May have NaNs
samples = np.nan_to_num(samples)


# In[ ]:


np.savez(f"/home/tordatom/Dati_Imaging/BraTs_19/Segmentation2D/Radiomic_Features/features_train2.npz", samples = samples)
#np.savez(f"/home/tordatom/Dati_Imaging/BraTs_19/Segmentation2D/Radiomic_Features/feature_names.npz", feature_names = feature_names)
np.savez(f"/home/tordatom/Dati_Imaging/BraTs_19/Segmentation2D/Radiomic_Features/train_examples2.npz", train_list = list(new_X.keys()))




# In[ ]:


np.savez(f"/home/tordatom/Dati_Imaging/BraTs_19/Segmentation2D/Radiomic_Features/features_test2.npz", samples = samples)
#np.savez(f"/home/tordatom/Dati_Imaging/BraTs_19/Segmentation2D/Radiomic_Features/feature_names.npz", feature_names = feature_names)
np.savez(f"/home/tordatom/Dati_Imaging/BraTs_19/Segmentation2D/Radiomic_Features/test_examples2.npz", test_list = list(new_X.keys()))


# In[165]:


samples_train = np.load("/home/tordatom/Dati_Imaging/BraTs_19/Segmentation2D/Radiomic_Features/features_train2.npz")["samples"]
feature_names = np.load("/home/tordatom/Dati_Imaging/BraTs_19/Segmentation2D/Radiomic_Features/feature_names.npz")["feature_names"]
train_list = np.load("/home/tordatom/Dati_Imaging/BraTs_19/Segmentation2D/Radiomic_Features/train_examples2.npz")["train_list"]
samples_test = np.load("/home/tordatom/Dati_Imaging/BraTs_19/Segmentation2D/Radiomic_Features/features_test2.npz")["samples"]
test_list = np.load("/home/tordatom/Dati_Imaging/BraTs_19/Segmentation2D/Radiomic_Features/test_examples2.npz")["test_list"]
prop = np.load(f"/home/tordatom/Dati_Imaging/BraTs_19/Segmentation2D/Radiomic_Features/prop_index.npz")["proponents"]
map_test = np.load(f"/home/tordatom/Dati_Imaging/BraTs_19/Segmentation2D/Radiomic_Features/map_test.npz", allow_pickle= True)["map_test"]

tracin_score = np.load("/home/tordatom/Dati_Imaging/BraTs_19/Segmentation2D/Radiomic_Features/score_dice1.npy",allow_pickle='TRUE').item()


# In[11]:


samples_train = samples_train[:,1,:]
samples_test = samples_test[:,1,:]


# In[12]:


from scipy import stats

tot = np.concatenate((samples_test, samples_train))
zscore_tot = stats.zscore(tot, axis = 0)
zscore_test, zscore_train = np.split(zscore_tot, [samples_test.shape[0]])


# In[ ]:


map_test = map_test.reshape(1)


# In[13]:


samples2 = np.einsum('ik,jk->ijk',zscore_test,zscore_train)


# In[ ]:


samples = samples2[7]


# In[166]:


proponents = np.zeros(len(prop), dtype = object)
opponents = np.zeros(len(prop), dtype = object)
neutrals = np.zeros(len(prop), dtype = object)

for i,h in enumerate(prop):
    proponents[i] = h[:69]
    neutrals[i] = h[69:138]
    opponents[i] = h[138:]


# In[167]:


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


# In[168]:


index2 = np.zeros((52,207))
for i in range(int(len(index)/10)):
    
    index2[i] = index[i] 


# In[ ]:


from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA

similarities = euclidean_distances(samples[:,:])


seed = np.random.RandomState(seed=3)

mds = manifold.MDS(n_components=2, max_iter=5000, eps=1e-12, random_state=seed,
                   n_init=10,
                   dissimilarity="precomputed", n_jobs=1, metric=False)
pos = mds.fit_transform(similarities)


# In[97]:


from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.cm as cm


fig = plt.figure(1)
ax = plt.axes([0., 0., 1., 1.])

s1 = 100
test = 70



plt.scatter(pos[p[test],0], pos[p[test],1], color='pink', alpha=1.0, s=s1, lw=1, label = "proponent")
plt.scatter(pos[o[test],0], pos[o[test],1], color='orange', alpha=1.0, s=s1, lw=1, label = "opponent")
plt.scatter(pos[n[test],0], pos[n[test],1], color='green', alpha=1.0, s=s1, lw=1, label = "neutral")
#for i, txt in enumerate(train_list):
#    ax.annotate(txt, (pos[i,0], pos[i,1]), xytext = (pos[i,0]+0.015,0.015+pos[i,1]))

plt.legend(scatterpoints=1, loc=5, shadow=False)

plt.title("Scatter plot dimensione ridotte = 2")

similarities = similarities.max() / similarities * 100
similarities[np.isinf(similarities)] = 0
plt.show()


# In[ ]:


#idea di trovare i proponenti e opponenti per epoche diverse e estrarre le feature per queste influenze.
#In questo modo possiamo vedere che feature apprende la rete durante il training


# In[ ]:


from sklearn.cluster import KMeans
from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA

similarities = euclidean_distances(index[:,:])


seed = np.random.RandomState(seed=3)

mds = manifold.MDS(n_components=2, max_iter=5000, eps=1e-12, random_state=seed,
                   n_init=10,
                   dissimilarity="precomputed", n_jobs=1, metric=False)
pos = mds.fit_transform(similarities)


# In[ ]:


k = KMeans(n_clusters=3, random_state=0).fit_predict(index[:,:])


# In[ ]:


fig = plt.figure(1)
ax = plt.axes([0., 0., 1., 1.])
for i in np.unique(k):
    plt.scatter(pos[k == i, 0], pos[k == i, 1], label = i)
plt.legend(scatterpoints=1, loc=5, shadow=False)

plt.title("Scatter plot dimensione ridotte = 2")


# In[ ]:


s = np.mean(samples_train, axis = 1)
for i in range(len(feature_names)):
    plt.hist(samples[p[7],i], alpha = 0.8, bins = 30, label = "proponents")
    plt.hist(samples[o[7],i], bins = 30, alpha = 0.8, color = "red", label = "opponents")
    plt.hist(samples[n[7],i], bins = 30, alpha = 0.8, color = "green", label = "neutral")
    plt.legend(scatterpoints=1, loc=5, shadow=False)
    plt.title(f"histogram of feature: {feature_names[i]}")
    plt.show()


# In[98]:


for j in range(len(samples2)):
    for i in range(len(feature_names)):
        fig = plt.figure(1)
        ax = plt.axes([0., 0., 1., 1.])

        s1 = 100
        f = i
        c = 0
        test = j
        a, b = np.polyfit(range(len(n[test])+len(p[test])+len(o[test])), samples2[j,p[test]+o[test]+n[test],f], 1)


        plt.scatter(range(len(p[test])), samples2[j,p[test],f], color='pink', alpha=1.0, s=s1, lw=1, label = "proponent")
        plt.scatter(range(len(n[test])+len(p[test]), len(n[test])+len(p[test])+len(o[test])), samples2[j,o[test], f], color='orange', alpha=1.0, s=s1, lw=1, label = "opponent")
        plt.scatter(range(len(p[test]), len(n[test])+len(p[test])), samples2[j,n[test],f], color='green', alpha=1.0, s=s1, lw=1, label = "neutral")
        plt.plot(range(len(n[test])+len(p[test])+len(o[test])), a*range(len(n[test])+len(p[test])+len(o[test]))+b)

        plt.text(0,max(samples2[j,:,f])+0.01*max(samples2[j,:,f]), 'y = '  + '{:.10f}'.format(a) + 'x'+ '+ {:.2f}'.format(b))

        plt.legend(scatterpoints=1, loc=5, shadow=False)
        plt.title(f"scatter plot della feature: {feature_names[i]}, test: {j}")
        plt.show()


# In[156]:


X = np.zeros((int((len(tracin_score)/10)), len(tracin_score[0])))
for i in range(int((len(tracin_score)/10))):
    for j,k in enumerate(proponents[i].tolist()+neutrals[i].tolist()+opponents[i].tolist()):
        X[i][j] = tracin_score[i][k]


# In[102]:


for j in range(len(samples2)):
    for i in range(len(feature_names)):
        fig = plt.figure(1)
        ax = plt.axes([0., 0., 1., 1.])

        s1 = 100
        f = i
        c = 0
        test = j
        a, b = np.polyfit(X[j], samples2[j,p[test]+n[test]+o[test],f], 1)


        plt.scatter(X[j, :len(proponents[j])], samples2[j,p[test],f], color='pink', alpha=1.0, s=s1, lw=1, label = "proponent")
        plt.scatter(X[j, len(proponents[j])+len(neutrals[j]):], samples2[j,o[test], f], color='orange', alpha=1.0, s=s1, lw=1, label = "opponent")
        plt.scatter(X[j, len(proponents[j]):len(proponents[j])+len(neutrals[j])], samples2[j,n[test],f], color='green', alpha=1.0, s=s1, lw=1, label = "neutral")
        plt.plot(X[j], a*X[j]+b)

        plt.text(0,max(samples2[j,:,f])+0.01*max(samples2[j,:,f]), 'y = '  + '{:.10f}'.format(a) + 'x'+ '+ {:.2f}'.format(b))

        plt.legend(scatterpoints=1, loc=5, shadow=False)
        plt.title(f"scatter plot della feature: {feature_names[i]}, test: {j}")
        plt.show()


# In[123]:


cor = np.zeros((int((len(tracin_score)/10)), len(feature_names)))
for i in range(len(cor)):
    cor[i] = [np.corrcoef(X[i], samples2[i,p[i]+n[i]+o[i],f], rowvar = False)[1,0] for f in range(107)]


# In[175]:


from scipy import stats

cor_spear = np.zeros((int((len(tracin_score)/10)), len(feature_names)))
for i in range(len(cor_spear)):
    cor_spear[i] = [stats.spearmanr(range(len(n[test])+len(p[test])+len(o[test])), samples2[i,p[i]+n[i]+o[i],f])[0] for f in range(107)]


# In[177]:


for i in range(len(cor_spear)):
    plt.hist(np.abs(cor_spear[i]), bins = 20)
    plt.title(f"Distribuzione dei coef di cor per il paziente: {i}")
    plt.show()


# In[124]:


for i in range(len(cor[0])):
    plt.hist(cor[:,i], bins = 20)
    plt.title(f"Distribuzione del coeff cor per la feature: {feature_names[i]}")
    plt.show()


# In[125]:


for i in range(len(cor)):
    plt.hist(np.abs(cor[i]), bins = 20)
    plt.title(f"Distribuzione dei coef di cor per il paziente: {i}")
    plt.show()


# In[ ]:


a = np.zeros((len(samples2), len(feature_names)))
for i in range(len(a)):
    for f in range(len(feature_names)):
        a[i,f] = np.polyfit(range(len(n[i])+len(p[i])+len(o[i])), samples2[i,p[i]+o[i]+n[i],f], 1)[0]


# In[ ]:


for i in range(len(a[0])):
    plt.hist(a[:,i], bins = 20)
    plt.title(f"Distribuzione dei coeff angolari per la feature: {feature_names[i]}")
    plt.show()


# In[ ]:


for i in range(len(a)):
    plt.hist(np.abs(a[i]), bins = 20)
    plt.title(f"Distribuzione dei coefficienti angolari per il paziente: {i}")
    plt.show()


# In[187]:


d2 = np.zeros(len(samples2), dtype = "object")
for i in range(len(cor)):
    x = {}
    for k,j in enumerate(cor[i]):
        x[feature_names[k]] = j
    d2[i] = x
    


# In[190]:


ang = [{k: v for k, v in sorted(d2[i].items(), key=lambda item: abs(item[1]), reverse = True)} for i in range(len(cor))]


# In[191]:


ang = np.array(ang)


# In[200]:


for i in range(len(ang)):
    print(f"paziente: {i}")
    for j in list(ang[i].keys()):
        if ang[i][j]>=0.20:
            print(j)
            


# In[198]:


for i in range(len(ang)):
    for j in list(ang[i].keys())[0:5]:
        fig = plt.figure(1)
        ax = plt.axes([0., 0., 1., 1.])

        s1 = 100
        f = np.int(np.where(feature_names == j)[0])
        c = 0
        test = 70
        a, b = np.polyfit(range(len(n[test])+len(p[test])+len(o[test])), samples2[i,p[test]+o[test]+n[test],f], 1)


        plt.scatter(range(len(p[test])), samples2[i,p[test],f], color='pink', alpha=1.0, s=s1, lw=1, label = "proponent")
        plt.scatter(range(len(n[test])+len(p[test]), len(n[test])+len(p[test])+len(o[test])), samples2[i,o[test], f], color='orange', alpha=1.0, s=s1, lw=1, label = "opponent")
        plt.scatter(range(len(p[test]), len(n[test])+len(p[test])), samples2[i,n[test],f], color='green', alpha=1.0, s=s1, lw=1, label = "neutral")
        plt.plot(range(len(n[test])+len(p[test])+len(o[test])), a*range(len(n[test])+len(p[test])+len(o[test]))+b)

        plt.text(0,max(samples2[i,:,f])+0.01*max(samples2[i,:,f]), 'y = '  + '{:.10f}'.format(a) + 'x'+ '+ {:.2f}'.format(b))

        plt.legend(scatterpoints=1, loc=5, shadow=False)
        plt.title(f"scatter plot del test: {i}, della feature: {j}")
        plt.show()


# In[ ]:




