#!/bin/bash


exec gunicorn -w 2 -b 0.0.0.0:8080 main:main -k gevent