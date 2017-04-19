# -*- coding: utf8 -*-

"""Open the first camera, detect faces 

"""

import cv2
import requests
from requests_toolbelt import MultipartEncoder
from matplotlib import pyplot as plt


plt.ion()
cap = cv2.VideoCapture(0)

while(True):
    # read video frame from camera. frame is numpy.ndarray
    _, frame = cap.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # convert frame(numpy.ndarray) to image type
    img_str = cv2.imencode('.jpg', frame)[1].tostring()

    # construct post request
    multipart_data = MultipartEncoder(
        fields={
            'img': ('img', img_str, 'application/text')
        }
    )
    # r = requests.post('http://localhost:8080/detect', files=payload)
    r = requests.post('http://localhost:8080/detect',
                      data=multipart_data,
                      headers={'Content-Type': multipart_data.content_type})
    print(r.json())

    # get detect result
    faces_info = r.json().get('faces')
    print('faces info: ', faces_info)
    for face in faces_info:
        l, t, r, b = face.get('rect')
        # draw rectangle for face
        cv2.rectangle(img, (l, t), (r, b), (0, 255, 0), 5)

    plt.imshow(img)
    plt.draw()
    plt.pause(1)
