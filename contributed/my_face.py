# coding=utf-8
"""Face Detection and Recognition"""
# MIT License
#
# Copyright (c) 2017 François Gervais
#
# This is the work of David Sandberg and shanren7 remodelled into a
# high level container. It's an attempt to simplify the use of such
# technology and provide an easy to use facial recognition package.
#
# https://github.com/davidsandberg/facenet
# https://github.com/shanren7/real_time_face_recognition
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pickle
import os

import cv2
import numpy as np
import tensorflow as tf
from scipy import misc

import align.detect_face
import facenet


gpu_memory_fraction = 0.3
facenet_model_checkpoint = os.path.dirname(__file__) + "/../models/facenet/20180303-142138"
# facenet_model_checkpoint = os.path.dirname(__file__) + "/../models/frozen/facenet-20170512-110547.pb"
# stored sklearn SVC model
# see also: classifier.py to train your own classifier
classifier_model = os.path.dirname(__file__) + "/../models/lfw_classifier.pkl"
debug = False


class Face:
    def __init__(self):
        self.name = None
        self.bounding_box = None
        self.image = None
        self.container_image = None
        self.embedding = None


class Recognition:
    def __init__(self):
        self.detect = Detection()
        self.encoder = Encoder()
        self.identifier = Identifier()

    # no usage found
    # 不使用classifier, 自己通过调用此函数指定人名
    def add_identity(self, image, person_name):
        faces = self.detect.find_faces(image)

        # 图片上只有一个人
        if len(faces) == 1:
            face = faces[0]
            # 指定人名(参数)
            face.name = person_name
            # 计算人脸图片的embedding
            face.embedding = self.encoder.generate_embedding(face)
            return faces

    def identify(self, image):
        """从frame上识别人脸, 人名和bounding box, frame上可能包括多个人脸"""
        faces = self.detect.find_faces(image)

        for i, face in enumerate(faces):
            if debug:
                cv2.imshow("Face: " + str(i), face.image)
            # 计算人脸图片的embedding
            face.embedding = self.encoder.generate_embedding(face)
            # 根据embedding寻找最符合的人名
            face.name = self.identifier.identify(face)

        return faces


class Identifier:
    """根据face的embedding, 找出最可能的人名, 数据库来自/../models/lfw_classifier.pkl"""
    def __init__(self):
        with open(classifier_model, 'rb') as infile:
            self.model, self.class_names = pickle.load(infile)

    def identify(self, face):
        if face.embedding is not None:
            # 使用训练好的classifier, 通过embedding预测人名(class)
            predictions = self.model.predict_proba([face.embedding])
            best_class_indices = np.argmax(predictions, axis=1)
            return self.class_names[best_class_indices[0]]


class Encoder:
    def __init__(self):
        self.sess = tf.Session()
        with self.sess.as_default():
            facenet.load_model(facenet_model_checkpoint)

    def generate_embedding(self, face):
        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        prewhiten_face = facenet.prewhiten(face.image)

        # Run forward pass to calculate embeddings
        feed_dict = {images_placeholder: [prewhiten_face], phase_train_placeholder: False}
        return self.sess.run(embeddings, feed_dict=feed_dict)[0]


class Detection:
    """使用mtcnn在图片中寻找人脸bounding box, 可以存在多个人脸"""

    # face detection parameters
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    def __init__(self, face_crop_size=160, face_crop_margin=32):
        self.pnet, self.rnet, self.onet = self._setup_mtcnn()
        self.face_crop_size = face_crop_size
        self.face_crop_margin = face_crop_margin

    def _setup_mtcnn(self):
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                return align.detect_face.create_mtcnn(sess, None)

    def find_faces(self, image):
        faces = []

        face = Face()
        face.container_image = image
        face.bounding_box = np.zeros(4, dtype=np.int32)

        # by default face_crop_size=160, face_crop_margin=32
        # img_size is for example (480, 640), i.e (height, width)
        img_size = np.asarray(image.shape)[0:2]

        my_width = 180
        my_height = 250
        face.bounding_box[0] = (img_size[1] - my_width) / 2  # x1
        face.bounding_box[1] = 50  # y1
        face.bounding_box[2] = face.bounding_box[0] + my_width  # x2
        face.bounding_box[3] = face.bounding_box[1] + my_height  # y2
        cropped = image[face.bounding_box[1]:face.bounding_box[3], face.bounding_box[0]:face.bounding_box[2], :]
        face.image = misc.imresize(cropped, (self.face_crop_size, self.face_crop_size), interp='bilinear')

        faces.append(face)

        return faces
