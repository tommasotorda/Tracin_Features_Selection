# Tracin vs Radiomica per BraTs19


## Descrizione
**Task di Med-1**: xAI tool per applicazioni mediche, in particolare per segmentazione di cervelli in MRI.

**Dataset BraTs19**:

- **Channels**: a) native (T1) and b) post-contrast T1-weighted (T1Gd), c) T2-weighted (T2), and d) T2 Fluid Attenuated
Inversion Recovery (T2-FLAIR) volumes.
- **Label**: 1) the "enhancing tumor" (ET), 2) the "tumor core" (TC), and 3) the "whole tumor" (WT).

Abbiamo utilizzato 259 scans di glioblastoma (HGG) tagliando il volume in slice bidimensionali lungo l’asse z (60 slice centrali
per paziente).

La segmentazione è ottenuta tramite l'utilizzo di una [Unet2D](https://gitlab.com/mucca1/BraTs19/-/blob/main/Unet2D.py).

Il metodo di xAI proposto  è basato su 2 distinti metodi: **TracIn** e le **Feature Radiomiche** estratte dalle immagini mediche.

## Preprocessing
in [Preprocessing.py](https://gitlab.com/mucca1/BraTs19/-/blob/main/Preprocessing.py)vengono preparati i dati per il training. Vengono normalizzate le intensità tra [-1,1], si applicano trasformazioni elestiche sul dataset di train che richiedono l'utilizzo della CPU e si rende bidimensionale il dato. I pazienti vengono poi divisi in train e test e ogni MRI in 2D è salvata su un file separato.

## Unet2D
in [Unet2D.py](https://gitlab.com/mucca1/BraTs19/-/blob/main/Unet2D.py) viene definito il modello, una semplice Unet con l'aggiunta di BatchNorm in ogni blocco convolutivo. Viene applicata anche della dataugmentation semplice come random flip e crop. 

## Gradients and Tracin
in [Gradients.py](https://gitlab.com/mucca1/BraTs19/-/blob/main/Gradients.py) calcoliamo i gradienti per test e train. Possiamo decidere se specializzare la spiegazione di Tracin per la loss function totale o per solo una determinata label, ecludendo ad esempio il background. I gradienti vengono poi salvati e caricati in [TracIn.py](https://gitlab.com/mucca1/BraTs19/-/blob/main/TracIn.py), qui si estraggono proponenti e opponenti per ogni esempi di test. 

## Radiomica
infine vegono estratte le feature radiomiche tramite la libreria pyradiomics in [Radiomica.py](https://gitlab.com/mucca1/BraTs19/-/blob/main/Radiomica.py). Le feature radiomiche del train e test vengono normalizzate tramite zscore e poi ne viene fatto il prodotto. Questo nuovo oggetto è plottato in funzione della score di tracin. 

## Caricamento del modello

Per caricare il modello e i dati ci dobbiamo connettere al mio container ``tordatom``. Per il modello dobbiamo usare le custom loss:

```
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
```
Prendiamo poi i checkpoint e carichiamo il modello
```
file_list_ckpt = glob(os.path.join(checkpoint_dir, "*"))
file_list_ckpt.sort()

model = tf.keras.models.load_model(file_list_ckpt[0], 
                                   custom_objects={'dice0': dice0, 'dice1': dice1, 
                                                   'dice2': dice2, 'dice3': dice3,
                                                   "dice_loss1":dice_loss1})
```
Per caricare i dati

```
def load_image_train(image_file):
    data = np.load(image_file)
    index = int(image_file[69:len(image_file)-4])
    return index, data['X_train'], data['Y_train']

def load_image_test(image_file):
    data = np.load(image_file)
    index = int(image_file[73:len(image_file)-4])
    return index, data['X_test'], data['Y_test']


def t_reshape(index, X,Y):
    X = tf.cast(X, tf.float32)
    Y = tf.cast(Y, tf.float32)
    X = tf.reshape(X, [192,192,4])
    Y = tf.reshape(Y, [192,192,4])
    return index,X,Y
```




