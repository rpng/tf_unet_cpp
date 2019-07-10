#!/usr/bin/env python3

import os
import sys
import datetime
import tensorflow as tf
from tensorflow.contrib import slim

import numpy as np

from time import time

import utils

from gen_tfrecords import vw as __vw
from gen_tfrecords import vh as __vh

N_CLASSES = 2
vh = 240
vw = 320

FLAGS = tf.app.flags.FLAGS
if __name__ == '__main__':
    tf.app.flags.DEFINE_string("mode", "train", "train or predict")

    tf.app.flags.DEFINE_string("model_dir", "model", "Estimator model_dir")

    tf.app.flags.DEFINE_integer("steps", 100000, "Training steps")
    tf.app.flags.DEFINE_string(
        "hparams", "",
        "A comma-separated list of `name=value` hyperparameter values. This flag "
        "is used to override hyperparameter settings when manually "
        "selecting hyperparameters.")

    tf.app.flags.DEFINE_integer("batch_size", 20, "Size of mini-batch.")
    tf.app.flags.DEFINE_boolean("fake_fisheye", False, "If true the data will be warped")
    tf.app.flags.DEFINE_boolean("pretraining", False, "If true, bbox is ignored")
    tf.app.flags.DEFINE_string("input_dir", "tfrecords_arl_wheels_bbox/", "tfrecords dir")
    tf.app.flags.DEFINE_string("image", "", "Image to predict on")

def create_input_fn(split, batch_size):
    """Returns input_fn for tf.estimator.Estimator.

    Reads tfrecord file and constructs input_fn for training

    Args:
    tfrecord: the .tfrecord file
    batch_size: The batch size!

    Returns:
    input_fn for tf.estimator.Estimator.

    Raises:
    IOError: If test.txt or dev.txt are not found.
    """

    def input_fn():
        """input_fn for tf.estimator.Estimator."""
        
        indir = FLAGS.input_dir
        #tfrecord = 'train_data*.tfrecord' if split=='train' else 'validation_data.tfrecord'
        tfrecord = 'validation_data.tfrecord'

        def parser(serialized_example):


            features_ = {}
            features_['img'] = tf.FixedLenFeature([], tf.string)
            features_['label'] = tf.FixedLenFeature([], tf.string)
            if not FLAGS.pretraining:
                features_['bbox_mask'] = tf.FixedLenFeature([], tf.string)
                features_['bbox'] = tf.FixedLenFeature([], tf.string)

            fs = tf.parse_single_example(
                serialized_example,
                features=features_
            )
            
            #if split=='train':
            #    fs['img'] = tf.reshape(tf.cast(tf.decode_raw(fs['img'], tf.uint8),
            #        tf.float32) / 255.0, [__vh,__vw,3])
            #    fs['label'] = tf.reshape(tf.cast(tf.decode_raw(fs['label'], tf.uint8),
            #        tf.float32), [__vh,__vw,N_CLASSES])
            #else:
            fs['img'] = tf.reshape(tf.cast(tf.decode_raw(fs['img'], tf.uint8),
                    tf.float32) / 255.0, [vh, vw, 3])
            fs['label'] = tf.reshape(tf.cast(tf.decode_raw(fs['label'], tf.uint8),
                    tf.float32), [vh, vw, N_CLASSES])
            if not FLAGS.pretraining:
                fs['bbox_mask'] = tf.reshape(tf.cast(tf.decode_raw(fs['bbox_mask'], tf.uint8),
                        tf.float32), [vh,vw,1])
                fs['bbox'] = tf.cast(tf.decode_raw(fs['bbox'], tf.int64), tf.float32)
            else:
                fs['bbox_mask'] = tf.zeros_like(fs['img'])
                fs['bbox'] = tf.zeros([4], tf.float32)
            return fs

        #if split=='train':
        #    files = tf.data.Dataset.list_files(indir + tfrecord, shuffle=True,
        #            seed=np.int64(time()))
        #else:
        files = [indir + tfrecord]
            
        dataset = tf.data.TFRecordDataset(files)
        if split == 'train':
            dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(400,
                seed=np.int64(time())))
        dataset = dataset.apply(tf.data.experimental.map_and_batch(parser, batch_size,
                    num_parallel_calls=2))
        dataset = dataset.prefetch(buffer_size=2)

        return dataset

    return input_fn


def unet(images, is_training=False):
    
    # Variational Semantic Segmentator
    with tf.variable_scope("UNet"): 
        images = tf.identity(images, name='images')
        _mean = np.fromfile('mean.txt', np.float32, 3, ' ').reshape(1,1,1,3)
        mean = tf.placeholder_with_default(_mean, _mean.shape, name='mean')
        images = images - mean
        with slim.arg_scope(
            [slim.conv2d],
            normalizer_fn=None,
            activation_fn=lambda x: tf.nn.elu(x),
            padding='SAME'):
            
            ### Encoder ####################################

            d11 = slim.conv2d(images, 64, [3,3])
            d12 = slim.conv2d(d11, 64, [3,3])
            p1 = tf.layers.max_pooling2d(d12, [2,2], 2, padding='same')

            d21 = slim.conv2d(p1, 128, [3,3])
            d22 = slim.conv2d(d21, 128, [3,3])
            p2 = tf.layers.max_pooling2d(d22, [2,2], 2, padding='same')
            
            d31 = slim.conv2d(p2, 256, [3,3])
            d32 = slim.conv2d(d31, 256, [3,3])
            p3 = tf.layers.max_pooling2d(d32, [2,2], 2, padding='same')

            d41 = slim.conv2d(p3, 512, [3,3])
            d42 = slim.conv2d(d41, 512, [3,3])
            p4 = tf.layers.max_pooling2d(d42, [2,2], 2, padding='same')
            
            d51 = slim.conv2d(p4, 1024, [3,3])
            d52 = slim.conv2d(d51, 1024, [3,3])

            ### Decoder ####################################
            
            u41 = slim.conv2d(tf.depth_to_space(d52, 2), 512, [3,3])
            u42 = slim.conv2d(tf.concat([u41, d42], axis=-1), 512, [3,3])
            u43 = slim.conv2d(u42, 512, [3,3])
            
            u31 = slim.conv2d(tf.depth_to_space(u43, 2), 128, [3,3])
            u32 = slim.conv2d(tf.concat([u31, d32], axis=-1), 128, [3,3])
            u33 = slim.conv2d(u32, 128, [3,3])

            u21 = slim.conv2d(tf.depth_to_space(u33, 2), 64, [3,3])
            u22 = slim.conv2d(tf.concat([u21, d22], axis=-1), 64, [3,3])
            u23 = slim.conv2d(u22, 64, [3,3])

            u11 = slim.conv2d(tf.depth_to_space(u23, 2), 32, [3,3])
            u12 = slim.conv2d(tf.concat([u11, d12], axis=-1), 32, [3,3])
            u13 = slim.conv2d(u12, 32, [3,3])

            feat = slim.conv2d(u13, 5, [1,1],
                normalizer_fn=None,
                activation_fn=None,
                padding='SAME') # [0,1] for segmentation, [2] for bbox center, [3,4] for w, h
            
            pred = tf.nn.softmax(feat[:,:,:,:2], name='pred')
            mask = tf.argmax(pred, axis=-1, name='mask')
            sh = tf.shape(feat)
            # contiiguous bbox location along xy flattened
            bbox_loc_cont = tf.argmax(tf.reshape(feat[:,:,:,2], [sh[0], -1]),
                    axis=-1, output_type=tf.int32)
            # [batch_size 2] list of bbox centers for each sample
            bbox_loc = tf.transpose(tf.unravel_index(bbox_loc_cont, [vh, vw]), [1, 0])

            n = vw * vh
            row_inds = tf.range(0, sh[0],
                    dtype=tf.int32) * (n-1)
            buffer_inds = row_inds + bbox_loc_cont # contiguous indexing

            # now retrieve the bbox wh
            wh_flat = tf.reshape(feat[:,:,:,3:], [-1, 2]) # [batch_size*h*w 2]
            bbox_wh = tf.cast(tf.round(
                tf.nn.embedding_lookup(wh_flat, buffer_inds)), tf.int32) # [batch_size 2]
            '''
            print(buffer_inds.get_shape())
            print(bbox_loc_cont.get_shape())
            print(bbox_loc.get_shape())
            print(bbox_wh.get_shape())
            exit()
            '''
            # concat and convert bboxes to opencv rect format
            bbox = tf.concat([bbox_loc - tf.cast(bbox_wh/2, tf.int32), bbox_wh], 
                    axis=-1, name='bbox')

            return feat, mask, bbox

def model_fn(features, labels, mode, hparams):
   
    del labels
    
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    
    
    x = tf.concat([features['img'], features['label']], axis=-1)
    # TODO add bbox here and uncomment
    
    x = tf.image.random_flip_left_right(x)

    # uncomment for fake fisheye
    '''
    if is_training and FLAGS.fake_fisheye:
        x = tf.contrib.image.rotate(x, tf.random.normal([FLAGS.batch_size]))
        x = tf.image.random_crop(x, [FLAGS.batch_size, vh, vw, 5])
        x = utils.distort(x, tf.placeholder_with_default([-0.0247903, 0.05102395,
            -0.03482873, 0.00815826], [4]))

    x = tf.image.resize_images(x, [vh, vw])

    #features['img'] = x[:,:,:,:3]
    #features['label'] = tf.cast(x[:,:,:,3:], tf.bool)
    '''
    images = features['img']
    labels = features['label']
    bbox_vec = features['bbox'] # shape: [? 4] ==> [x y w h]
    bbox_mask = features['bbox_mask'] 

    feat, mask, _bbox = unet(images, is_training)
    #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prob_feat,
    #                labels=labels))
    seg = tf.clip_by_value(tf.nn.softmax(feat[:,:,:,:2]), 1e-6, 1.0)
    region = feat[:,:,:,2] # raw features for bbox location
    bbox = feat[:,:,:,3:] # [w, h]

    labels = tf.cast(labels, tf.float32)
    
    seg_loss = tf.reduce_mean(  
         -tf.reduce_sum(labels * tf.log(seg), axis=-1))
    batch_size = tf.shape(labels)[0]
    bbox_loc_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            labels = tf.reshape(bbox_mask, [batch_size, -1]), 
            logits = tf.reshape(region, [batch_size, -1])))

    # width and height regression
    bbox_wh_loss = tf.reduce_mean(
            tf.square(tf.reshape(bbox_vec[:, 2:4], [-1, 1, 1, 2]) - bbox))
    '''
    seg_loss = tf.Print(seg_loss, [seg_loss], "seg")
    bbox_loc_loss = tf.Print(bbox_loc_loss, [bbox_loc_loss], "bbox_loc")
    bbox_wh_loss = tf.Print(bbox_wh_loss, [bbox_wh_loss], "bbox_wh")
    '''

    if not FLAGS.pretraining:
        loss = 10. * seg_loss + bbox_loc_loss + 0.01 * bbox_wh_loss
    else:
        loss = seg_loss

    with tf.variable_scope("stats"):
        tf.summary.scalar("loss", loss)

    eval_ops = {
              "Test Error": tf.metrics.mean(loss),
    }
    
    def touint8(img):
        return tf.cast(img * 255.0, tf.uint8)
    im = touint8(images[0])
    
    to_return = {
          "loss": loss,
          "eval_metric_ops": eval_ops,
          'pred': mask[0],
          'bbox': _bbox[0],
          'im': im,
          'label': tf.argmax(labels[0], axis=-1)
    }

    predictions = {
        'mask': mask,
    }
    
    to_return['predictions'] = predictions

    utils.display_trainable_parameters()

    return to_return

def _default_hparams():
    """Returns default or overridden user-specified hyperparameters."""

    hparams = tf.contrib.training.HParams(
          learning_rate=1.0e-5,
    )
    if FLAGS.hparams:
        hparams = hparams.parse(FLAGS.hparams)
    return hparams


def main(argv):
    del argv
    
    tf.logging.set_verbosity(tf.logging.ERROR)

    hparams = _default_hparams()
    
    if FLAGS.mode == 'train':
        utils.train_and_eval(
            model_dir=FLAGS.model_dir,
            model_fn=model_fn,
            input_fn=create_input_fn,
            hparams=hparams,
            steps=FLAGS.steps,
            batch_size=FLAGS.batch_size,
       )
    elif FLAGS.mode == 'predict':
        import cv2

        tf.reset_default_graph()
        with tf.gfile.GFile('cpp/unet.pb', 'rb') as f:
            gd = tf.GraphDef()
            gd.ParseFromString(f.read())
        g = tf.Graph()
        with tf.Session(graph=g) as sess:
            im_t, mask_t = tf.import_graph_def(gd, return_elements=['UNet/images:0', 'UNet/mask:0'])
            dset_it = create_input_fn('val', 1)().make_one_shot_iterator()
            __dset = dset_it.get_next()
            try:
                j = 0
                if not os.path.isdir('plots'):
                    os.mkdir('plots')
                while 1:
                    print("Working on sample %d" % j)
                    dset = sess.run(__dset)
                    lab = dset['label']
                    im = dset['img']
                    im = cv2.resize(np.squeeze(im), (vw, vh))
                    mask = sess.run(mask_t, 
                            feed_dict={im_t:im[np.newaxis,...]})
                    mask = np.squeeze(mask)
                    lab = np.squeeze(lab)
                    lab = np.argmax(cv2.resize(lab, (vw, vh)), axis=2)
                    rgb_mask = np.zeros((vh, vw, 3))
                    rgb_lab = np.zeros((vh, vw, 3))
                    d = 'plots/%d' % j
                    j += 1
                    if not os.path.isdir(d):
                        os.mkdir(d)
                    for i in range(2):
                        c = np.array([0,0,0] if i==0 else [1,0,0])
                        rgb_mask[mask==i, :] = c
                        rgb_lab[lab==i, :] = c
                    for fn, t in [('im', im), ('mask', rgb_mask), ('lab', rgb_lab)]:
                        cv2.imwrite(os.path.join(d, fn) + '.jpg', np.uint8(255*t))
            except tf.errors.OutOfRangeError:
                print("Done")
    else:
        raise ValueError("Unknown mode: %s" % FLAGS.mode)

if __name__ == "__main__":
    sys.excepthook = utils.colored_hook(
        os.path.dirname(os.path.realpath(__file__)))
    tf.app.run()
