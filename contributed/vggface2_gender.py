import math
import pickle

import h5py
import tensorflow as tf
import numpy as np
import csv
import os

from sklearn.svm import SVC

import facenet
from facenet import ImageClass

# arr = np.random.rand(2,3)
# print(arr)
# with h5py.File("test.hdf5", "w") as file:
#     file.create_dataset("arr", arr.shape, arr.dtype, arr)
#
#
# with h5py.File("test.hdf5", "r") as file:
#     print(file.keys())
#     print(file["arr"])
#     print(file["arr"][()])


# -1 means all
vgg_num_image_per_folder = 1
vgg_identity_meta_file_path = "../datasets/VGGFace2/meta/identity_meta.csv"
vgg_images_path = "../datasets/VGGFace2/train_mtcnnpy_182"

lfw_meta_female_file_path = "../datasets/lfw/gender/female_names.txt"
lfw_meta_male_file_path = "../datasets/lfw/gender/male_names.txt"
lfw_images_path = "../datasets/lfw/lfw_mtcnnpy_160"

classifier_output_path = "../datasets/VGGFace2/gender/classifier.pkl"
feature_extraction_model = "../models/facenet/20180322-001311"
batch_size = 100
image_size = 160
# mode = 'TRAIN'
mode = 'TEST'

image_paths_m = []
image_paths_f = []

if mode == 'TRAIN':
    with open(vgg_identity_meta_file_path, "r", newline="") as file:
        reader = csv.reader(file)
        # skip header
        next(reader)
        for line in reader:
            folder, name, _, _, gender = line
            folder_path = os.path.abspath(os.path.join(vgg_images_path, folder))
            num_image = 0
            for image_name in os.listdir(folder_path):
                if num_image < vgg_num_image_per_folder:
                    if gender.strip() == "m":
                        image_paths_m.append(os.path.join(folder_path, image_name))
                    else:
                        image_paths_f.append(os.path.join(folder_path, image_name))
                    num_image += 1
else:
    with open(lfw_meta_male_file_path, "r") as file:
        reader = file.read().splitlines()
    for line in reader:
        if len(line.strip()) == 0:
            continue
        folder = line[:-9]
        image_name = line
        folder_path = os.path.abspath(os.path.join(lfw_images_path, folder))
        image_paths_m.append(os.path.join(folder_path, image_name))

    with open(lfw_meta_female_file_path, "r") as file:
        reader = file.read().splitlines()
    for line in reader:
        if len(line.strip()) == 0:
            continue
        folder = line[:-9]
        image_name = line
        folder_path = os.path.abspath(os.path.join(lfw_images_path, folder))
        image_paths_f.append(os.path.join(folder_path, image_name))

print("number of images (m/f):", len(image_paths_m) + len(image_paths_f), len(image_paths_m), len(image_paths_f))

dataset = [ImageClass("m", image_paths_m), ImageClass("f", image_paths_f)]
image_paths, labels = facenet.get_image_paths_and_labels(dataset)

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
            images = facenet.load_data(paths_batch, False, False, image_size)
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)

        classifier_filename_exp = os.path.expanduser(classifier_output_path)

        if mode == 'TRAIN':
            print('Training classifier')
            model = SVC(kernel='linear', probability=True)
            model.fit(emb_array, labels)

            # Saving classifier model
            class_names = [cls.name for cls in dataset]
            with open(classifier_filename_exp, 'wb') as outfile:
                pickle.dump((model, class_names), outfile)
            print('Saved classifier model to file "%s"' % classifier_filename_exp)

        else:
            # Classify images
            print('Testing classifier')
            with open(classifier_filename_exp, 'rb') as infile:
                (model, class_names) = pickle.load(infile)

            print('Loaded classifier model from file "%s"' % classifier_filename_exp)

            predictions = model.predict_proba(emb_array)
            best_class_indices = np.argmax(predictions, axis=1)
            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

            # for i in range(len(best_class_indices)):
            #     print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))

            accuracy = np.mean(np.equal(best_class_indices, labels))
            print('Accuracy: %.3f' % accuracy)
