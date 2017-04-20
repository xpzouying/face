# -*- coding: utf-8 -*-



from unittest import TestCase
import requests


host = 'localhost'
port = '8080'

detect_api = 'http://{host}:{port}/detect'.format(
    host=host,
    port=port
)

class TestFaceFunction(TestCase):

    def test_detect_one_face(self):

        data = {
            'img': open('one_face.jpg', 'rb')
        }

        resp = requests.post(detect_api, files=data)

        j = resp.json()
        faces = j.get('faces')

        self.assertEqual(len(faces), 1)

