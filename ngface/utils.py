# -*- coding: utf8 -*-

import numpy as np


def prewhiten(x):
    """Prewhiten. Copy from facenet.facenet.prewhiten

    """
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y

