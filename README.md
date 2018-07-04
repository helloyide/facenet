# Face Recognition using Tensorflow 

This project is forked from [davidsandberg/facenet](https://github.com/davidsandberg/facenet), which is a TensorFlow implementation of the face recognizer described in the papers
["FaceNet: A Unified Embedding for Face Recognition and Clustering"](http://arxiv.org/abs/1503.03832), ["A Discriminative Feature Learning Approach for Deep Face Recognition"](http://ydwen.github.io/papers/WenECCV16.pdf), ["Deep Face Recognition"](http://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf).

## Changes

In comparison to the original project:
* more args are added to the train_softmax script to improve the training
* a new network implementation: SqueezeNet v1.1, which has 2.4 times less computation thant the original one.
* new pre-trained models for tensorflow and tensorflow.js
* new contributed scripts

## New arguments for training

Most of these changes are done in train_softmax.py:
* added "--continue_ckpt_dir" to allow continue training from one checkpoint
* added "--random_brightness" for random brightness data augmentation
* added "--snapshot_at_step" to make a checkpoint at the specific step
* added "--lfw_epoch_interval" to control how often to test against the lfw dataset
* added "--summary_iteration_interval" to control how often the summary is recorded
* made "--data_dir argument" support more train datasets (separated by comma)
* fixed a bug about "--nrof_preprocess_threads", it was not used and hardcoded to 4 before

The learning rate can be adjusted manually either by changing the schedule file on the fly (the file content is read in each epoch, train will be stopped if a 0 learning rate is found) or by changing the "--learning_rate" and continue the training with "--continue_ckpt_dir".


## SqueezeNet v1.1
The authors of the original paper improved the SqueezeNet, based on their web page, [SqueezeNet_v1.1](https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1) should be much faster than the original one.

The implementation is src/models/squeezenet_v1_1.py which is based on src/models/squeezenet.py with new network structure. It also uses He weights initialization instead of Xavier. 


## Pre-trained models
| Model name      | LFW accuracy | Training dataset | Architecture |
|-----------------|--------------|------------------|-------------|
| [20170511-185253](https://drive.google.com/file/d/0B5MzpY9kBtDVOTVnU3NIaUdySFE) | 0.987        | CASIA-WebFace    | [Inception ResNet v1](https://github.com/davidsandberg/facenet/blob/master/src/models/inception_resnet_v1.py) |
| [20170512-110547](https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk) | 0.992        | MS-Celeb-1M      | [Inception ResNet v1](https://github.com/davidsandberg/facenet/blob/master/src/models/inception_resnet_v1.py) |


## New contributed scripts and other changes 
* contributed/photo_face_recognition.py: draw face recognition boxes on a single photo
* contributed/vggface2_gender.py: use vggface2 embedding feature vectors to train a gender classifier
* src/validate_on_lfw.py: add Precision and Recall measure in lfw validation
