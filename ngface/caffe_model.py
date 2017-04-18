# -*- coding: utf8 -*-


import time
import tensorflow as tf
from facenet.align import detect_face


# pnet, rnet, onet is caffemodel
_pnet = None
_rnet = None
_onet = None


def init_caffe_model(model_path):
    """Init caffe model for detect face.

    """

    print('Creating networks and loading parameters')
    print('Load models path: ', model_path)

    start = time.time()  # measure load caffe model

    with tf.Graph().as_default():
        # TODO: GUI accelerate
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, model_path)
            global _pnet
            _pnet = pnet
            global _rnet
            _rnet = rnet
            global _onet
            _onet = onet

    print('time used: ', time.time()-start)


def get_pnet():
    global _pnet

    if _pnet == None:
        print('Caffe model not init. [pnet]')
        init_caffe_model()

    return _pnet


def get_rnet():
    global _rnet

    if _rnet == None:
        print('Caffe model not init. [rnet]')        
        init_caffe_model()

    return _rnet


def get_onet():
    global _onet

    if _onet == None:
        print('Caffe model not init. [onet]')
        init_caffe_model()

    return _onet
        