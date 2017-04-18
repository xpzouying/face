# -*- coding: utf8 -*-


import os
import tensorflow as tf
from ngface.face_models import load_model
from ngface.tfgraph import get_graph

_sess = None
_model_path = "~/Models/pretrained_models/Facenet/20170216-091149"


def init_session():
    global _sess

    # single session for facenet
    _sess = tf.Session(graph=get_graph())
    # load_model(sess, model_dir)
    exp_model_path = os.path.expanduser(_model_path)
    load_model(_sess, exp_model_path)


def get_session():
    global _sess

    if _sess == None:
        print('Session is None. Initialize session.')
        init_session()

    return _sess
