# -*- coding: utf-8 -*-


import json
import time

import cv2
import numpy as np
from flask import jsonify, request
from scipy import misc

from ngface import app, caffe_model, facenet_task,\
    tfgraph, tfsession, utils, errors
from ngface.utils import prewhiten
from tfcore import detect_face


@app.route('/init')
def init():

    # measure time_used for init
    start = time.time()

    graph = tfgraph.get_graph()
    # sess = tf.Session(graph=graph)
    sess = tfsession.get_session()

    return jsonify(time_used=time.time()-start)


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
    graph = tfgraph.get_graph()
    with graph.as_default():
        # Get input and output tensors
        images_placeholder = graph.get_tensor_by_name("input:0")
        embeddings = graph.get_tensor_by_name("embeddings:0")
        phase_train_placeholder = graph.get_tensor_by_name("phase_train:0")

        # Run forward pass to calculate embeddings
        feed_dict = {images_placeholder: np_images,
                     phase_train_placeholder: False}
        sess = tfsession.get_session()
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


@app.route('/detect', methods=['POST'])
def detect():
    """Detect faces in POST image

    @input: flask request files: img
    @output: 
        {head_rect: {lt, rb},
         image: base64 head photo
        }
    """

    # measure detect time_used
    start = time.time()

    img_list = utils.get_images_from_request(request.files, ['img'])
    if len(img_list) == 0:
        # if img not in post
        resp = {}
        resp['faces'] = []
        resp['time_used'] = time.time() - start
        resp['error'] = str(errors.MissArgsError('img'))

        return jsonify(resp)

    all_faces_info = facenet_task.detect_face_task(img_list[0])
    print('all faces info: ', all_faces_info)

    resp = {}
    resp['faces'] = all_faces_info
    resp['time_used'] = time.time() - start

    # return Response(json.dumps(resp), mimetype='application/json')
    return jsonify(resp)


@app.route('/detect2', methods=['POST'])
def detect2():
    """Detect faces in POST image

    Convert image to gray to detect

    @input: flask request files: img
    @output: 
        {head_rect: {lt, rb},
         image: base64 head photo
        }
    """

    # measure detect time_used
    start = time.time()

    img_list = utils.get_images_from_request(request.files, ['img'])
    img = img_list[0]  # only one image
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    all_faces_info = facenet_task.detect_face_task(img)
    print('all faces info: ', all_faces_info)

    resp = {}
    resp['faces'] = all_faces_info
    resp['time_used'] = time.time() - start

    # return Response(json.dumps(resp), mimetype='application/json')
    return jsonify(resp)


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
        bounding_boxes, points = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

        print('detect bounding: ', bounding_boxes)
        print('detect points: ', points)
        print('detect bounding dir: ', dir(bounding_boxes))
        print('Find faces: ', bounding_boxes.shape[0])

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
