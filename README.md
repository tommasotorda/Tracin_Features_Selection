# Tracin in Semantic Segmentation of tumor brains in MRI, an extended approach


## Abstract
In recent years, thanks to improved computational power and the availability of big data, AI has become a fundamental tool in basic research and industry. Despite this very rapid development, deep neural networks (DNN) remain black boxes that are difficult to explain. While a multitude of explainability (xAI) methods have been developed, their effectiveness and usefulness in realistic use cases are understudied. This is a major limitation in the application of these algorithms in sensitive fields such as clinical diagnosis, where the robustness, transparency and reliability of the algorithm are indispensable for its use. In addition, the majority of works have focused on feature attribution (e.g., saliency maps) techniques, neglecting other interesting families of xAI methods such as data influence methods as TracIn. The aim of this work is to implement, extend and test, for the first time, this data influence functions in a challenging clinical problem, namely, the segmentation of tumor brains in Magnetic Resonance Images (MRI). We present a new methodology to calculate TracIn that is generalizable for all semantic segmentation tasks where the different labels are mutually exclusive, which is the standard framework for these tasks. We also provide an interpretation map of the explanation, based on the cosine similarity between feature maps extracted from the neural network, which we use to prove the faithfulness of the algorithm with respect to the decision-making process of DNN. We show that TracIn can be used to make feature selection. The goodness of selection can then be evaluated on how the selected internal network kernels afflict the network prediction.


## Dataset BraTs19:

- **Channels**: a) native (T1) and b) post-contrast T1-weighted (T1Gd), c) T2-weighted (T2), and d) T2 Fluid Attenuated
Inversion Recovery (T2-FLAIR) volumes.
- **Label**: 1) the "enhancing tumor" (ET), 2) the "tumor core" (TC), and 3) the "whole tumor" (WT).

BraTS2019 utilizes multi-institutional pre-operative MRI scans. The training dataset is com- posed of 259 cases of high-grade gliomas (HGG) and 76 cases of low-grade gliomas (LGG), manually annotated by both clinicians and board-certified radiologists. For each patient, four MRI scans taken with different modalities are provided: T1, T1Gd, T2, T2-FLAIR. with an image’s shape of voxels 240 × 240 × 155. We focused only on HGG patients, dividing the dataset into 207 train patients and 52 validation patients

## Neural Network
To solve the segmentation task, we chose a popular and well-established neural network, the [Unet2D.ipynb](https://gitlab.com/mucca1/BraTs19/-/blob/main/Unet2D.ipynb) for 2D segmentation.


## Preprocessing
In [Preprocessing.ipynb](https://gitlab.com/mucca1/BraTs19/-/blob/main/Preprocessing.ipynb) data are prepared for training. The intensities are normalised between [-1,1] and the data are made two-dimensional. The patients are then divided into train and test and each 2D MRI is saved on a separate file.


## Gradients and Tracin
In [Gradients.ipynb](https://gitlab.com/mucca1/BraTs19/-/blob/main/Gradients.ipynb) we calculate the gradients for test and train. We can decide whether to specialise Tracin's explanation for the total loss function or for only a certain label, e.g. excluding the background. The gradients are then saved and loaded into [TracIn.ipynb](https://gitlab.com/mucca1/BraTs19/-/blob/main/TracIn.ipynb), here, proponents and opponents are extracted for each test example. 



## Feature maps

In [Vis_filter.ipynb](https://gitlab.com/mucca1/BraTs19/-/blob/main/Vis_filter.ipynb) feature maps are extracted from the train and test examples. These features are extracted only from the pixels belonging to a certain tumour class and are extracted taking into consideration only the penultimate layer of the network. In total, we have 64 filters to analyse for each of the 3 tumour labels.Here is also the evaluation of the goodness of the feature selection.

## Consistency tests
In [consistency_tests.ipynb](https://gitlab.com/mucca1/BraTs19/-/blob/main/consistency_tests.ipynb.ipynb) consistency tests are evaluated. The self-influence matrix, robustness and cross-validation tests and finally the feature selection on the basis of the Tracin score are reported.



