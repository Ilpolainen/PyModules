# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 14:31:05 2018

@author: Ilpo
"""
import sys
import cv2
import numpy as np
import tensorflow as tf
#from random import shuffle
#import glob
#shuffle_data = True  # shuffle the addresses before saving
#cat_dog_train_path = 'Cat vs Dog/train/*.jpg'

# read addresses and labels from the 'train' folder

#addrs = glob.glob(cat_dog_train_path)
#labels = [0 if 'cat' in addr else 1 for addr in addrs]  # 0 = Cat, 1 = Dog

# to shuffle data
#if shuffle_data:
#    c = list(zip(addrs, labels))
#    shuffle(c)
#    addrs, labels = zip(*c)
    
# Divide the hata into 60% train, 20% validation, and 20% test

#train_addrs = addrs[0:int(0.6*len(addrs))]
#train_labels = labels[0:int(0.6*len(labels))]
#val_addrs = addrs[int(0.6*len(addrs)):int(0.8*len(addrs))]
#val_labels = labels[int(0.6*len(addrs)):int(0.8*len(addrs))]
#test_addrs = addrs[int(0.8*len(addrs)):]
#test_labels = labels[int(0.8*len(labels)):]

def load_image(addr,size=(224,224)):
    im = cv2.imread(addr)
    im = cv2.resize(im, (size),interpolation=cv2.INTER_CUBIC)
    im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    im = im.astype(np.float32)
    return im

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value])) 

def write_as_TFRecords(filename,addrs,labels=None):
  # address to save the TFRecords file
# open the TFRecords file
    writer = tf.python_io.TFRecordWriter(filename)
    for i in range(len(addrs)):
    # print how many images are saved every 1000 images
        if not i % 1000:
            print('Data: {}/{}'.format(i, len(addrs)))
            sys.stdout.flush()
    # Load the image
        img = load_image(addrs[i])
        if labels != None:
            label = labels[i]
            feature = {'label': int64_feature(label),
                       'image': bytes_feature(tf.compat.as_bytes(img.tostring()))}
        else:
            feature = {'image' : bytes_feature(tf.compat.as_bytes(img.tostring()))}
            
        # address to save the TFRecords file
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))
    
        # Serialize to string and write on the file
        writer.write(example.SerializeToString())
    
    writer.close()
    sys.stdout.flush()
    
        