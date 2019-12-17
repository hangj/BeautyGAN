# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os
import glob
from imageio import imread, imsave
import cv2
import argparse
from tqdm import tqdm

#设置使用的gpu
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#占用GPU40%
#config.gpu_options.per_process_gpu_memory_fraction = 0.4

parser = argparse.ArgumentParser()
parser.add_argument('--src_dir', type=str, required=True, help='path to the src image dir')
parser.add_argument('--target_dir', type=str, default=os.path.join('beauty_gan_target_dir') , help='path to the target image dir')
parser.add_argument('--makeup_num', type=int, default=6 , help='number of makeup')

args = parser.parse_args()

def preprocess(img):
    return (img / 255. - 0.5) * 2

def deprocess(img):
    return (img + 1) / 2

batch_size = 1
img_size = 256

#所有上装的图片
#makeups = glob.glob(os.path.join('imgs', 'makeup', '*.*'))
#makeups.sort()
#print(makeups)

#查找待上装图片
all_no_makeups = glob.glob(os.path.join(args.src_dir,'*.*'))

all_img_len=len(all_no_makeups)
print("---"*50,all_img_len)

#for x in all_no_makeups:
#    print(x)
#    print(os.path.split(x)[-1])
#    target_file_path = os.path.join(args.target_dir, os.path.split(x)[-1])
#    print(target_file_path)

if not os.path.exists(args.target_dir):
    os.makedirs(args.target_dir)

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

#makeup = cv2.resize(imread(makeups[args.makeup_num]), (img_size, img_size))

makeup = cv2.resize(imread(os.path.join('img_0.jpg')), (img_size, img_size))
tq=tqdm(all_no_makeups)
for no_makeup in tq:
    #print(no_makeup)
    #continue
    target_file_path = os.path.join(args.target_dir, os.path.split(no_makeup)[-1])
    no_makeup = cv2.resize(imread(no_makeup), (img_size, img_size))
    X_img = np.expand_dims(preprocess(no_makeup), 0)
    result = np.ones((img_size,img_size,3))
    #print(result.shape)
    Y_img = np.expand_dims(preprocess(makeup), 0)
    Xs_ = sess.run(Xs, feed_dict={X: X_img, Y: Y_img})
    Xs_ = deprocess(Xs_)
    #result[:, img_size:2 * img_size] = Xs_[0]
    result=Xs_[0]
    #print(result.shape)
    #print(result)
    #print(target_file_path)
    #result = result*255
    #因为result.dtype=float32
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    #result = result.astype(np.uint8)
    #print(result)
    imsave(target_file_path, result)



sess.close()
