# -*- coding: utf8 -*-

import tensorflow as tf
import numpy as np
from scipy import misc
from tfcore import detect_face
from ngface.tfsession import get_session
from ngface.tfgraph import get_graph
from ngface.utils import prewhiten
from ngface import caffe_model


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
        face_info = {}
        face_info['rect'] = face_position[:4].tolist()  # numpy array to python list
        face_info['image'] = 'Not now'

        all_faces.append(face_info)

    return all_faces


# def load_and_align_images(imgs):
#     """Align images to numpy array
#     """
#     image_size = 182
#     margin = 44
#     minsize = 20    # minimum size of face
#     threshold = [0.6, 0.7, 0.7]  # three step's threshold
#     factor = 0.709  # scale factor

#     # g = get_graph()
#     # with g.as_default():
#     with tf.Graph().as_default():
#         # sess = get_session()
#         gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
#         sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
#         with sess.as_default():
#             pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

#     img_list = [None] * len(imgs)
#     index = 0
#     for img in imgs:

#         # img = misc.imread(path)
#         img_size = np.asarray(img.shape)[0:2]
#         bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
#         det = np.squeeze(bounding_boxes[0, 0:4])
#         bb = np.zeros(4, dtype=np.int32)
#         bb[0] = np.maximum(det[0] - margin / 2, 0)
#         bb[1] = np.maximum(det[1] - margin / 2, 0)
#         bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
#         bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
#         cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
#         aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
#         prewhitened = prewhiten(aligned)
#         # img_list.append(prewhitened)
#         img_list[index] = prewhitened
#         index += 1

#     images = np.stack(img_list)
#     return images

