# -*- coding: utf8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import sys
import time
import argparse
import tensorflow as tf
import numpy as np
from scipy import misc
from flask import Flask, jsonify, Response, request
from ngface.utils import prewhiten
from facenet.align import detect_face
from ngface import face_models
from ngface import tfgraph, tfsession
from ngface import caffe_model
from ngface import utils



graph = tfgraph.get_graph()
# sess = tf.Session(graph=graph)
sess = tfsession.get_session()

app = Flask(__name__)


@app.route('/')
@app.route('/index')
def index():
    msg = 'curl -X POST -F "img1=@img1.jpg" -F "img2=@img2.jpg"' \
          ' \'http://192.168.31.188:8080/verify\''

    return jsonify(message=msg)


@app.route('/version')
def version():
    return jsonify(version='ngface version 0.1')


@app.route('/verify', methods=['POST'])
def verify():
    # verify time_used
    start = time.time()

    # read image from request
    img_list = utils.get_images_from_request(request.files, ['img1', 'img2'])

    np_images = load_and_align_data(img_list)

    dist_str = ''
    dist = 0
    with graph.as_default():
        # Get input and output tensors
        images_placeholder = graph.get_tensor_by_name("input:0")
        embeddings = graph.get_tensor_by_name("embeddings:0")
        phase_train_placeholder = graph.get_tensor_by_name("phase_train:0")

        # Run forward pass to calculate embeddings
        feed_dict = {images_placeholder: np_images,
                     phase_train_placeholder: False}
        emb = sess.run(embeddings, feed_dict=feed_dict)

        # nrof_images = len(args.image_files)
        dist = np.sqrt(np.sum(np.square(np.subtract(emb[0, :], emb[1, :]))))

        dist_str = '%1.4f' % dist
        print('Distance: ', dist_str)
        threshold = 0.9
        print('Threshold: ', threshold)

    time_used = time.time() - start
    return jsonify(is_same_person=str(dist<threshold),
                   time_used=str(time_used))


def load_and_align_data(img_list):

    image_size = 160
    margin = 44
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor

    pnet = caffe_model.get_pnet()
    rnet = caffe_model.get_rnet()
    onet = caffe_model.get_onet()

    np_images = []
    for img in img_list:
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
        np_images.append(prewhitened)

    images = np.stack(np_images)    
    return images


def main(args):
    # init models
    # init caffe model
    caffe_model.init_caffe_model()
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
