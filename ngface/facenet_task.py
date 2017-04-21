# -*- coding: utf-8 -*-

import base64
import sys

import cv2
import numpy as np
import tensorflow as tf
from scipy import misc

from ngface import caffe_model
from ngface.tfgraph import get_graph
from ngface.tfsession import get_session
from ngface.utils import prewhiten
from tfcore import detect_face

if sys.version_info[0] == 3:
    # python27
    from io import StringIO
else:
    # python3
    import StringIO  # python27




def verify(align_imgs):
    """Verify images after align

    @input: image after align
    @output: distance
    """

    g = get_graph()
    with g.as_default():
        # Get input and output tensors
        images_placeholder = g.get_tensor_by_name("input:0")
        embeddings = g.get_tensor_by_name("embeddings:0")
        phase_train_placeholder = g.get_tensor_by_name("phase_train:0")

        # Run forward pass to calculate embeddings
        feed_dict = {images_placeholder: align_imgs,
                     phase_train_placeholder: False}
        sess = get_session()
        emb = sess.run(embeddings, feed_dict=feed_dict)

        dist = np.sqrt(np.sum(np.square(np.substract(emb[0,:], emb[1,:]))))
        print('distance: %1.4f' % dist)

    return '%1.4f' % dist


def detect_face_task(img):
    """Detect faces from image

    @input: image
    @output:
        - all faces information
    """

    # paramter for detect
    # image_size = 160
    # margin = 44
    minsize = 20 # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709 # scale factor

    # caffe model
    pnet = caffe_model.get_pnet()
    rnet = caffe_model.get_rnet()
    onet = caffe_model.get_onet()

    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    print('detect bounding: ', bounding_boxes)
    print('Find faces: ', bounding_boxes.shape[0])

    # all_faces is faces information list, include face bytes, face position
    all_faces = []
    for face_position in bounding_boxes:
        face_position = face_position.astype(int)
        print('face position: ', face_position)

        # each face information, include position, face image
        head_rect = face_position[:4].tolist()  # numpy array to python list
        head_img = misc.toimage(img).crop(head_rect)
        head_img_io = StringIO.StringIO()
        head_img.save(head_img_io, format='JPEG')
        head_img_b64 = base64.b64encode(head_img_io.getvalue())

        # construct response
        face_info = {}
        face_info['rect'] = head_rect
        face_info['image'] = head_img_b64

        all_faces.append(face_info)

    return all_faces
