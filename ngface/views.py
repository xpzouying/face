# -*- coding: utf8 -*-

from ngface import app, facenet_task
# from PIL import Image
from scipy import misc


@app.route('/version')
def version():
    return "ngfacenet version 0.1"


@app.route('/verify')
def verify():
    # img1 = Image.open('/tmp/1.jpg')
    # img2 = Image.open('/tmp/2.jpg')
    img1 = misc.imread('/tmp/1.jpg')
    img2 = misc.imread('/tmp/2.jpg')

    align_imgs = facenet_task.load_and_align_images((img1, img2))

    res = facenet_task.verify(align_imgs)

    return res
