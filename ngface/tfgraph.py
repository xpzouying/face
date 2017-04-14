# -*- coding: utf8 -*-

import tensorflow as tf


_graph = None

def get_graph():
    global _graph

    if _graph == None:
        _graph = tf.Graph()
    
    return _graph