#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import numpy as np
import cv2
import tensorflow as tf
from time import time

vh = 240
vw = 320

val_split = 10

FLAGS = tf.app.flags.FLAGS

if __name__ == '__main__':
   
    tf.app.flags.DEFINE_string("output_dir", "tfrecords/", "")
    tf.app.flags.DEFINE_string("arl_root", "/mnt/f3be6b3c-80bb-492a-98bf-4d0d674a51d6/ARL/labelbox/", "")
    tf.app.flags.DEFINE_integer("num_files", 2, "Num files to write for train dataset. More files=better randomness")
    tf.app.flags.DEFINE_boolean("debug", False, "")
    tf.app.flags.DEFINE_boolean("wheels", True, "")
    

    if FLAGS.debug:
        from matplotlib import pyplot as plt
        from matplotlib.patches import Patch
        plt.ion()
        imdata = None

def writeFileList(dirName):
    im_list = [] # list of all files with full path
    lab_list = [] # list of all files with full path
    for dirname, dirnames, filenames in os.walk(os.path.join(dirName,'rgb')):
        for filename in filenames:
            if filename.endswith('.jpg'):	
                fileName = os.path.join(dirname, filename)
                im_list.append(fileName)
                png = filename.split('.jpg')[0] + ".png"
                d = dirname.split('rgb')[0]
                lab_list.append([os.path.join(d, 'car' , png),
                    os.path.join(d, 'car_wheels' , png)])
                
    return im_list, lab_list


def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def generate():
   
    if not os.path.isdir(FLAGS.output_dir):
        os.mkdir(FLAGS.output_dir)

    train_writers = []
    for ii in range(FLAGS.num_files):
        train_writers.append(None if FLAGS.debug else \
                tf.python_io.TFRecordWriter(FLAGS.output_dir + "train_data%d.tfrecord" % ii))
    val_writer = None if FLAGS.debug else \
            tf.python_io.TFRecordWriter(FLAGS.output_dir + "validation_data.tfrecord")
    mean = np.zeros((3))
    im_list, lab_list = writeFileList(FLAGS.arl_root)
    count = 0
    for i in range(len(im_list)):
        im_fl = im_list[i]
        lab_fls = lab_list[i]

        print("Working on sample %d" % i)

        image = cv2.resize(cv2.imread(im_fl), (vw, vh))
        if FLAGS.wheels:
            car = cv2.resize(cv2.imread(lab_fls[0], 
                cv2.IMREAD_GRAYSCALE), (vw, vh),
                interpolation = cv2.INTER_NEAREST)[..., np.newaxis]

            wheels = cv2.resize(cv2.imread(lab_fls[1], 
                cv2.IMREAD_GRAYSCALE), (vw, vh), 
                interpolation = cv2.INTER_NEAREST)[..., np.newaxis]

            lab = np.uint8(np.logical_or(car, wheels))
        else:
            lab = cv2.imread(lab_fls[0], 
                cv2.IMREAD_GRAYSCALE)[..., np.newaxis]

        mask_label = np.zeros((vh, vw, 2), dtype=np.bool)
        mask_label[:, :, 1:2] = lab
        if np.any(mask_label[:,:,1]):
            # calculate bbox
            contours = cv2.findContours(lab, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            bbox = cv2.boundingRect(contours[0])
            mean += np.mean(image / 255.0, axis=(0,1))
            mask_label[:, :, 0] = np.logical_not(mask_label[:, :, 1])
                
            if FLAGS.debug:
                mask = np.argmax(mask_label, axis=-1)
                rgb = np.zeros((vh, vw, 3))

                legend = []
                np.random.seed(0)
                for i in range(2):
                    c = np.random.rand(3)
                    case = mask==i
                    if np.any(case):
                        legend.append(Patch(facecolor=tuple(c), edgecolor=tuple(c),
                                    label='background' if i==0 else 'car'))

                    rgb[case, :] = c
                
                _image = cv2.resize(image, (vw, vh)) / 255.0

                _image = 0.3 * _image + 0.7 * rgb
                cv2.rectangle(_image, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), 
                        (1.0, 0, 0), 2)

                global imdata
                if imdata is None:
                    imdata = plt.imshow(_image)
                    f = plt.gca()
                    f.axes.get_xaxis().set_ticks([])
                    f.axes.get_yaxis().set_ticks([])
                else:
                    imdata.set_data(_image)

                lgd = plt.legend(handles=legend, loc='upper left', bbox_to_anchor=(1.0, 1))
                
                plt.pause(1e-9)
                plt.draw()
                plt.pause(3)

            else:
                bbox = np.int64(bbox)
                # make bbox center oriented
                bbox[0] = bbox[0] + bbox[2] // 2
                bbox[1] = bbox[1] + bbox[2] // 2

                # 1 where bbox center is
                bbox_mask = np.zeros((vh, vw), np.uint8)
                bbox_mask[max(min(bbox[1], vh-1), 0), max(min(bbox[0], vw-1), 0)] = 1

                features_ = {
                    'bbox': bytes_feature(tf.compat.as_bytes(bbox.tostring())),
                    'bbox_mask': bytes_feature(tf.compat.as_bytes(bbox_mask.tostring())),
                    'img': bytes_feature(tf.compat.as_bytes(image.tostring())),
                    'label': bytes_feature(tf.compat.as_bytes(mask_label.astype(np.uint8).tostring()))
                }
                example = tf.train.Example(features=tf.train.Features(feature=features_))

                if np.random.randint(0,100) < val_split:
                    val_writer.write(example.SerializeToString())
                else:
                    train_writers[np.random.randint(0,FLAGS.num_files)].write(example.SerializeToString())
            count += 1
        else:
            print("No cars. Skipping")
    with open('mean.txt', 'w') as f:
        s = ''
        mean = mean / count
        for m in mean:
            s+= str(m) + ' '
        f.write(s)
    print("Done. Sample count =", count)
def main(argv):
    del argv
    generate()


if __name__ == "__main__":
    tf.app.run()
