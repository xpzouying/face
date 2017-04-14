# -*- coding: utf8 -*-

import time
import os
import re
import tensorflow as tf
from ngface import tfgraph


def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) == 0:
        raise ValueError('No meta file found in the model directory (%s)'
                         % model_dir)
    elif len(meta_files) > 1:
        raise ValueError('There should not be more than one meta file in' +
                         'the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups()) >= 2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file


def load_model(sess, model_dir):
    """Load facenet model
    """

    # measure time used for loading model
    start = time.time()

    model_dir_exp = os.path.expanduser(model_dir)
    meta_file, ckpt_file = get_model_filenames(model_dir)
    meta_file_full_path = os.path.join(model_dir_exp, meta_file)
    ckpt_file_full_path = os.path.join(model_dir_exp, ckpt_file)
    print('Model meta file: ', meta_file_full_path)
    print('Model ckpt file: ', ckpt_file_full_path)
    print('Loading models. Waiting...')

    from ngface.tfgraph import get_graph
    g = get_graph()
    with g.as_default():
        saver = tf.train.import_meta_graph(meta_file_full_path)
        saver.restore(sess, ckpt_file_full_path)

    print('Models loaded. time_used: ', time.time()-start)

