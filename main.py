# -*- coding: utf8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import sys
import argparse
import tensorflow as tf
import numpy as np
from scipy import misc
from flask import Flask, jsonify, Response
from ngface.utils import prewhiten
from facenet.align import detect_face
from facenet import facenet as FN
from ngface import face_models


sess = tf.Session()
app = Flask(__name__)


@app.route('/version')
def version():
    return jsonify(version='ngface version 0.1')


@app.route('/verify')
def verify():
    image_paths = [
        '/tmp/1.jpg',
        '/tmp/2.jpg'
    ]
    images = load_and_align_data(image_paths)

    print(images)

    dist_str = ''
    with tf.Graph().as_default():
        with tf.Session() as sess:
            model_dir = '/Users/zouying/Models/pretrained_models/Facenet/20170216-091149'
            meta_file, ckpt_file = face_models.get_model_filenames(model_dir)
            
            print('Meta file: %s' % meta_file)
            print('Checkpoint file: %s' % ckpt_file)
            FN.load_model(model_dir, meta_file, ckpt_file)

             # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Run forward pass to calculate embeddings
            feed_dict = { images_placeholder: images, phase_train_placeholder:False }
            emb = sess.run(embeddings, feed_dict=feed_dict)
     
            # nrof_images = len(args.image_files)
            dist = np.sqrt(np.sum(np.square(np.subtract(emb[0,:], emb[1,:]))))

            dist_str = '%1.4f' % dist
            print('Distance: ', dist_str)
            
            # return jsonify(result=dist_str)
            

    return jsonify(result=dist_str)


def load_and_align_data(image_paths):

    image_size = 160
    margin = 44
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    
    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
  
    nrof_samples = len(image_paths)
    img_list = [None] * nrof_samples
    for i in xrange(nrof_samples):
        img = misc.imread(os.path.expanduser(image_paths[i]))
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        det = np.squeeze(bounding_boxes[0,0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img_size[1])
        bb[3] = np.minimum(det[3]+margin/2, img_size[0])
        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        prewhitened = prewhiten(aligned)
        img_list[i] = prewhitened
    images = np.stack(img_list)
    return images


def main(args):
    app.run(host=args.host, port=args.port)


def parse_arguments(argv):
    """Init argvs and parse

    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--host', type=str, default="127.0.0.1",
                        help="Listen host. Default is 127.0.0.1")
    parser.add_argument('--port', type=int, default=8080,
                        help="Listen port. Default is 8080")
    parser.add_argument('--model_dir', type=str,
                        help="Directory of models")

    return parser.parse_args(argv)


if __name__ == '__main__':
    arg_parser = parse_arguments(sys.argv[1:])

    main(arg_parser)
