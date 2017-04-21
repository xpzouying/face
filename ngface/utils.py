# -*- coding: utf-8 -*-

import numpy as np
import sys

if sys.version_info[0] == 3:
    # python27
    from io import StringIO
else:
    # python3
    import StringIO  # python27
from scipy import misc


def prewhiten(x):
    """Prewhiten. Copy from facenet.facenet.prewhiten

    """
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y


def get_images_from_request(request_file, names):
    """get pillow images from flask request

    @input: request_file: request.files
    @input: names: image name list for read
    @output: type ndarray. The array obtained by reading the image.
    """

    img_list = []
    for name in names:
        # get upload file
        f = request_file.get(name)
        if f is None:
            continue

        img = misc.imread(f)
        img_list.append(img)

    return img_list
