# -*- coding: utf8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from ngface import caffe_model
from ngface import app


def main(args):
    # init models
    # init caffe model
    caffe_model.init_caffe_model(args.model_dir)

    app.run(host=args.host, port=args.port)


def parse_arguments(argv):
    """Init argvs and parse

    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--host', type=str, default="127.0.0.1",
                        help="Listen host. Default is 127.0.0.1")
    parser.add_argument('--port', type=int, default=8080,
                        help="Listen port. Default is 8080")
    parser.add_argument('--model_dir', type=str,
                        default='./models/mtcnn_detect',
                        help="Directory of models")

    return parser.parse_args(argv)


if __name__ == '__main__':
    arg_parser = parse_arguments(sys.argv[1:])

    main(arg_parser)
