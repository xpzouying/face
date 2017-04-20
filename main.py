# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import yaml

from ngface import caffe_model
from ngface import app


def yaml_config_example():
    """Return yaml config message

    """

    return """Please add *config.yml*, and config it like this,
    HTTP:
        host: localhost
        port: 8080

    Model_Path:
        detect: ./models/mtcnn_detect
        verify: ~/Models/pretrained_models/Facenet/20170216-091149
    """

def main():
    # read config file
    with open('config.yml') as f:
        conf = yaml.load(f.read())

    # read related config
    det_dir = conf.get('Model_Dir').get('detect')
    if det_dir is None:
        print(yaml_config_example())

    host = conf.get('HTTP').get('host')
    if host is None:
        host = 'localhost'

    port = conf.get('HTTP').get('port')
    if port is None:
        port = "8080"


    # start message
    start_message = '[*] Listen {host}:{port}.' \
        '\nDetect model path={det_dir}\n'.format(
            host=host,
            port=port,
            det_dir=det_dir
        )
    print(start_message)

    # init models
    # init caffe model
    caffe_model.init_caffe_model(det_dir)

    app.run(host=host, port=port)


if __name__ == '__main__':
    main()
