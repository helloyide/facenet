import csv
import math
import os
import pickle

import numpy as np
import tensorflow as tf
from scipy import misc

import facenet

vgg_num_image_per_folder = 1  # -1 means all
vgg_gender_meta_file_path = "../datasets/VGGFace2/meta/identity_meta.csv"
vgg_age_meta_file_path = "../datasets/VGGFace2/meta/test_agetemp_imglist.txt"
vgg_images_path = "../datasets/VGGFace2/train_mtcnnpy_182"

lfw_meta_female_file_path = "../datasets/lfw/gender/female_names.txt"
lfw_meta_male_file_path = "../datasets/lfw/gender/male_names.txt"
lfw_images_path = "../datasets/lfw/lfw_mtcnnpy_160"

data_save_path = "../datasets/VGGFace2/gender/saved_data"

# inception resnet v2 trained with vggface2, acc: 0.992, validation: 0.958
feature_extraction_model = "../models/facenet/20180324-080308"
batch_size = 100
image_size = 160

do_flip = True

image_paths_m = []
image_paths_f = []


def build_id2gender_map():
    result = {}
    # empty newline is to fix Windows bug
    with open(vgg_gender_meta_file_path, "r", newline="") as file:
        reader = csv.reader(file)
        # skip header
        next(reader)
        for line in reader:
            id, _, _, _, gender = line
            result[id] = gender.strip()
    return result


def load_data(img_paths, flip, img_size, do_prewhiten=True):
    m = len(img_paths)
    imgs = np.zeros((m, img_size, img_size, 3))
    for i in range(m):
        img = misc.imread(img_paths[i])
        if img.ndim == 2:
            img = facenet.to_rgb(img)
        if do_prewhiten:
            img = facenet.prewhiten(img)
        img = facenet.crop(img, False, image_size)
        if flip:
            img = np.fliplr(img)
        imgs[i, :, :, :] = img
    return imgs


id2gender = build_id2gender_map()

# each element is a map:
# image_path: image full path
# gender: 'm', 'f'
# age_young: True, False
data = []
image_paths = []
with open(vgg_age_meta_file_path, "r", newline="") as file:
    lines = file.readlines()
    id = None
    gender = None
    age_young = False
    for i, line in enumerate(lines):
        if i % 20 == 0:
            id = line.split("/")[0]
            gender = id2gender[id]
        if i % 10 == 0:
            age_young = not age_young
        image_path = os.path.abspath(os.path.join(vgg_images_path, line.strip()))
        image_paths.append(image_path)
        data.append({"image_path": image_path, "gender": gender, "age_young": age_young})

with tf.Graph().as_default():
    with tf.Session() as sess:
        # Load the model
        print('Loading feature extraction model')
        facenet.load_model(feature_extraction_model)

        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]

        # Run forward pass to calculate embeddings
        print('Calculating features for images')
        nrof_images = len(image_paths)
        nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / batch_size))
        emb_array = np.zeros((nrof_images, embedding_size))
        for i in range(nrof_batches_per_epoch):
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, nrof_images)
            paths_batch = image_paths[start_index:end_index]
            images = load_data(paths_batch, do_flip, image_size)
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)


for i, d in enumerate(data):
    d["embedding"] = emb_array[i]

# Saving data array
data_save_path_exp = os.path.expanduser(data_save_path)
if do_flip:
    data_save_path_exp += "_flip"
# data is a list with length 2000
# elements are {
#   'image_path': str
#   'gender': 'f'/'m'
#   'age_young': bool
#   'embedding': ndarray with shape (128,) dtype float64
# }
with open(data_save_path_exp, 'wb') as outfile:
    pickle.dump(data, outfile)
print('Saved data to file "%s"' % data_save_path_exp)
