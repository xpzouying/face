# -*- coding: utf8 -*-

"""Open the first camera, detect faces 

"""


import cv2
from matplotlib import pyplot as plt


plt.ion()
cap = cv2.VideoCapture(0)


while(True):
    _, frame = cap.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    plt.imshow(img)
    plt.draw()
    plt.pause(1)


