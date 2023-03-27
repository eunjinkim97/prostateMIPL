# python Train_and_Extract_DL_features.py --fold 1 --mode train
# python Train_and_Extract_DL_features.py --fold 1 --mode extract


import tensorflow as tf
from tensorflow import keras

import os,time,datetime,random
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from skimage.transform import rescale, resize, downscale_local_mean

import nibabel as nib # from nilearn.image import resample_to_img
from scipy import stats, ndimage
import SimpleITK as sitk
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--fold', type=int, default=1, help='fold 1 ~ 5')
parser.add_argument('--mode', type=str, default='train', help='train / extract')

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

def load_data(path,size_3d,mode,batch_size=2):
    import tensorflow as tf
    import numpy as np
    size=size_3d[0]
    size_z=size_3d[2]
    feature_description = { #tf.FixedLenFeature([3], tf.int64, default_value=[0,0,0]),
        'img': tf.io.FixedLenFeature([64,64,3], tf.float32),
        'gt': tf.io.FixedLenFeature([1], tf.int64),
        'duration': tf.io.FixedLenFeature([1], tf.int64),
        'ID': tf.io.FixedLenFeature([1], tf.int64),
    }
    def augment(data):
        tf.random.set_seed(1234)
        img = data['img']
        print('expected [bs,64, 64, 3, 1],...',img.shape)
        return img, data['gt'], data['duration'], data['ID']
    
    def _parse_example(example):
      # Parse the input `tf.Example` proto using the dictionary above.
        return tf.io.parse_single_example(example, feature_description)


    def to_4d(data):
        return data['img'], data['gt'], data['duration'], data['ID']

    def resize(data):
        data['img'] = tf.image.resize(data['img'],[224,224])
        return data

    if mode == 'training':
        dataset_test = tf.data.TFRecordDataset(path)
        train_ds = (
            dataset_test
            .map(_parse_example,num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .batch(batch_size).cache()
            .map(resize)
            .map(to_4d,num_parallel_calls=tf.data.experimental.AUTOTUNE)#.repeat()
        )
        return train_ds
        
    elif mode == 'valid':
        dataset_test = tf.data.TFRecordDataset(path)
        valid_ds = (
            dataset_test
            .map(_parse_example,num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .batch(batch_size).cache()
            .map(resize)
            .map(to_4d,num_parallel_calls=tf.data.experimental.AUTOTUNE)
            #.repeat()
        )
        return valid_ds
    
    elif mode == 'test':
        dataset_test = tf.data.TFRecordDataset(path)
        test_ds = (
            dataset_test
            .map(_parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .batch(batch_size).cache()
            .map(resize)
            .map(to_4d, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            #.repeat()
        )
        return test_ds
    
def split_ds(ds):
    x, y, duration, index = [], [], [], []
    for img, gt, dur, idx in ds:
        x.append(img[0])
        y.append(gt[0])
        duration.append(dur[0])
        index.append(idx[0])
    return x, y, duration, index

def define_model(ver_compile=True):
    model1=tf.keras.applications.efficientnet.EfficientNetB0(input_shape=(224,224,3),
                                                   include_top=False,
                                                   weights='imagenet')
    model2=tf.keras.applications.efficientnet.EfficientNetB0(input_shape=(224,224,3),
                                                   include_top=False,
                                                   weights='imagenet')
    model3=tf.keras.applications.efficientnet.EfficientNetB0(input_shape=(224,224,3),
                                                   include_top=False,
                                                   weights='imagenet')
    for layer in model2.layers:
        layer._name = layer.name + str("_2")
    for layer in model3.layers:
        layer._name = layer.name + str("_3")

    fine_tune_at = 10 # -180 : multi
    for layer in model1.layers[:fine_tune_at]:
        layer.trainable =  False # frozen
    for layer in model2.layers[:fine_tune_at]:
        layer.trainable =  False # frozen
    for layer in model3.layers[:fine_tune_at]:
        layer.trainable =  False # frozen
    gap1 = tf.keras.layers.GlobalAveragePooling2D()(model1.layers[-5].output) #320
    gap2 = tf.keras.layers.GlobalAveragePooling2D()(model2.layers[-5].output) #320
    gap3 = tf.keras.layers.GlobalAveragePooling2D()(model3.layers[-5].output) #320

    mergedOut = Concatenate()([gap1,gap2,gap3]) #960
    
    mergedOut = Dense(256)(mergedOut)
    mergedOut = Activation('relu')(mergedOut)
    mergedOut = Dense(128)(mergedOut)
    mergedOut = Activation('relu')(mergedOut)
    mergedOut = Dense(1)(mergedOut)#,activation='sigmoid'
    newModel = tf.keras.models.Model([model1.input,model2.input,model3.input], mergedOut)
    if ver_compile == True:
        print("Number of layers in the base model: ", len(newModel.trainable_variables))
        newModel.compile(tf.keras.optimizers.Adam(0.00001),\
                         loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),metrics=['accuracy'])#
    return newModel


def main():
    args = parser.parse_args()

    tr_dwi = load_data(args.fold+'_'+'train_DWI.tfrecords', (64,64,3), 'training', batch_size=1)
    te_dwi = load_data(args.fold+'_'+'test_DWI.tfrecords', (64,64,3), 'test', batch_size=1)

    tr_dce = load_data(args.fold+'_'+'train_DCE.tfrecords', (64,64,3), 'training', batch_size=1)
    te_dce = load_data(args.fold+'_'+'test_DCE.tfrecords', (64,64,3), 'test', batch_size=1)

    tr_t2 = load_data(args.fold+'_'+'train_T2.tfrecords', (64,64,3), 'training', batch_size=1)
    te_t2 = load_data(args.fold+'_'+'test_T2.tfrecords', (64,64,3), 'test', batch_size=1)

    x_tr_dwi, y_tr_dwi, dur_tr_dwi, idx_tr_dwi = split_ds(tr_dwi)
    x_te_dwi, y_te_dwi, dur_te_dwi, idx_te_dwi = split_ds(te_dwi)

    x_tr_dce, y_tr_dce, dur_tr_dce, idx_tr_dce = split_ds(tr_dce)
    x_te_dce, y_te_dce, dur_te_dce, idx_te_dce = split_ds(te_dce)

    x_tr_t2, y_tr_t2, dur_tr_t2, idx_tr_t2 = split_ds(tr_t2)
    x_te_t2, y_te_t2, dur_te_t2, idx_te_t2 = split_ds(te_t2)

    
    if args.mode == 'train':

        model = define_model()
        model.fit(x=[x_tr_dwi, x_tr_t2, x_tr_dce], y=y_tr_dwi, 
                validation_data=([x_te_dwi, x_te_t2, x_te_dce], y_te_dwi), 
                epochs=50, batch_size=8, verbose=2)
        
        model.save('{}fold_DL_weights_last'.format(args.fold))
    
    else:
        model = tf.keras.models.load_model('{}fold_DL_weights_last'.format(args.fold))

        feature_extractor = tf.keras.Model(
            inputs=model.inputs,
            outputs=model.layers[-6].output,
        )

        tr_features = feature_extractor([x_tr_dwi, x_tr_t2, x_tr_dce])
        te_features = feature_extractor([x_te_dwi, x_te_t2, x_te_dce])

        tr_features = pd.DataFrame(tr_features)
        idx, gt, dur = pd.DataFrame(idx_tr_dwi), pd.DataFrame(y_tr_dwi), pd.DataFrame(dur_tr_dwi)
        tr_features = pd.concat([idx, gt, dur, tr_features], axis=1)
        tr_features.to_csv('{}fold_train_960features.csv'.format(args.fold))

        te_features = pd.DataFrame(te_features)
        idx, gt, dur = pd.DataFrame(idx_te_dwi), pd.DataFrame(y_te_dwi), pd.DataFrame(dur_te_dwi)
        te_features = pd.concat([idx, gt, dur, te_features], axis=1)
        te_features.to_csv('{}fold_train_960features.csv'.format(args.fold))


if __name__ == "__main__":
    main()