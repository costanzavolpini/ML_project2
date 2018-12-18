import sys

import _pickle as cPickle
import numpy as np
import csv
#from models import *
from models_full_pretrained_train import *
from feature_extraction import train_test_features, save
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
import os



#save("train_pos_full.txt", "train_neg_full.txt", True, -1, -1, 'embedding_hashtag_split/features_pretrained_full_no_old_clean.dat')
[X_train, y, X_test, max_features, W] = cPickle.load(open("embedding_hashtag_split/features_pretrained_full_no_old_clean.dat", "rb"))



model3 = build_model3(X_train, W, kernel_size=5, dropout_U=0, dropout_W=0, units=100, filters=32, pool_size=2)
model3.summary()


model3.fit(X_train, y, validation_split=0.1, nb_epoch=1, batch_size=256, verbose=1)
model3.save('embedding_hashtag_split/model3_full_dataset_1_epochs_no_old_clean.plk')


model3.fit(X_train, y, validation_split=0.1, nb_epoch=1, batch_size=256, verbose=1)
model3.save('embedding_hashtag_split/model3_full_dataset_2_epochs_no_old_clean.plk')


model3.fit(X_train, y, validation_split=0.1, nb_epoch=1, batch_size=256, verbose=1)
model3.save('embedding_hashtag_split/model3_full_dataset_3_epochs_no_old_clean.plk')

model3.fit(X_train, y, validation_split=0.1, nb_epoch=1, batch_size=256, verbose=1)
model3.save('embedding_hashtag_split/model3_full_dataset_4_epochs_no_old_clean.plk')
