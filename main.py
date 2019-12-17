# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os
import glob
from imageio import imread, imsave
import cv2
import argparse

#设置使用的gpu
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#占用GPU40%
#config.gpu_options.per_process_gpu_memory_fraction = 0.4

parser = argparse.ArgumentParser()
parser.add_argument('--no_makeup', type=str, default=os.path.join('imgs', 'no_makeup', 'xfsy_0068.png'), help='path to the no_makeup image')

args = parser.parse_args()

def preprocess(img):
    return (img / 255. - 0.5) * 2

def deprocess(img):
    return (img + 1) / 2

batch_size = 1
img_size = 256
no_makeup = cv2.resize(imread(args.no_makeup), (img_size, img_size))
X_img = np.expand_dims(preprocess(no_makeup), 0)
makeups = glob.glob(os.path.join('imgs', 'makeup', '*.*'))
makeups.sort()
print(makeups)

print((2 * img_size, (len(makeups) + 1) * img_size, 3))

result = np.ones((2 * img_size, (len(makeups) + 1) * img_size, 3))
print("**"*10)
#print("result.shape",result.shape)
#print("(result[img_size: 2 *  img_size, :img_size]).shape",(result[img_size: 2 *  img_size, :img_size]).shape)
#print("[img_size: 2 *  img_size, :img_size]",result[img_size: 2 *  img_size, :img_size])

print("no_makeup.shape",no_makeup.shape)
result[img_size: 2 *  img_size, :img_size] = no_makeup / 255.
#print(result)
#imsave('result_0.jpg', result)
#exit(0)

tf.reset_default_graph()

#sess = tf.Session(config=config)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.import_meta_graph(os.path.join('model', 'model.meta'))
saver.restore(sess, tf.train.latest_checkpoint('model'))

graph = tf.get_default_graph()
X = graph.get_tensor_by_name('X:0')
Y = graph.get_tensor_by_name('Y:0')
Xs = graph.get_tensor_by_name('generator/xs:0')

#for i in range(len(makeups)):
#    makeup = cv2.resize(imread(makeups[i]), (img_size, img_size))
#    Y_img = np.expand_dims(preprocess(makeup), 0)
#    Xs_ = sess.run(Xs, feed_dict={X: X_img, Y: Y_img})
#    Xs_ = deprocess(Xs_)
#    result[:img_size, (i + 1) * img_size: (i + 2) * img_size] = makeup / 255.
#    result[img_size: 2 * img_size, (i + 1) * img_size: (i + 2) * img_size] = Xs_[0]
#    
#imsave('result.jpg', result)
print((img_size,2*img_size, 3))
result = np.ones((img_size,2*img_size, 3))

print(result.shape)
print("---")
print(result[0, img_size:1].shape)
print((no_makeup / 255.).shape)

#exit(0)

makeup = cv2.resize(imread(makeups[6]), (img_size, img_size))
print("makeup",makeup.shape)
makeup = cv2.resize(imread("img_0.jpg"), (img_size, img_size))
print("makeup2",makeup.shape)
Y_img = np.expand_dims(preprocess(makeup), 0)
print("Y_img.shape",Y_img.shape)
print("X_img.shape",X_img.shape)
Xs_ = sess.run(Xs, feed_dict={X: X_img, Y: Y_img})
Xs_ = deprocess(Xs_)
result[:,0:img_size] = makeup / 255.
result[:,img_size:2*img_size] = Xs_[0]
imsave('result2.jpg',result)