web url:
https://github.com/davidsandberg/facenet

don't forget to set the PYTHONPATH to src:
export PYTHONPATH=/home/cn1h/PycharmProjects/facenet/src

https://github.com/davidsandberg/facenet/wiki/Train-a-classifier-on-own-images
python src/classifier.py TRAIN E:/datasets/lfw/lfw_mtcnnpy_160 "C:/Users/a/Google Drive/AI/facenet/models/facenet/20170512-110547" "C:/Users/a/Google Drive/AI/facenet/models/lfw_classifier.pkl" --batch_size 1000 --min_nrof_images_per_class 40 --nrof_train_images_per_class 35 --use_split_dataset
python src/classifier.py CLASSIFY E:/datasets/lfw/lfw_mtcnnpy_160 "C:/Users/a/Google Drive/AI/facenet/models/facenet/20170512-110547" "C:/Users/a/Google Drive/AI/facenet/models/lfw_classifier.pkl" --batch_size 1000 --min_nrof_images_per_class 40 --nrof_train_images_per_class 35 --use_split_dataset

freeze graph
python src/freeze_graph.py "C:/Users/a/Google Drive/AI/facenet/models/facenet/20170512-110547" "C:/Users/a/Google Drive/AI/facenet/models/frozen/facenet-20170512-110547.pb"


