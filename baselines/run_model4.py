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



[X_train, y, X_test, max_features, W] = cPickle.load(open("embedding_hashtag_split/features_pretrained_full_no_old_clean.dat", "rb"))



model4 = build_model4(X_train, W)
model4.summary()


model4.fit(X_train, y, validation_split=0.1, nb_epoch=1, batch_size=1000, verbose=1)
model4.save('embedding_hashtag_split/model4_full_dataset_1_epochs.plk')


model4.fit(X_train, y, validation_split=0.1, nb_epoch=1, batch_size=1000, verbose=1)
model4.save('embedding_hashtag_split/model4_full_dataset_2_epochs.plk')


model4.fit(X_train, y, validation_split=0.1, nb_epoch=1, batch_size=1000, verbose=1)
model4.save('embedding_hashtag_split/model4_full_dataset_3_epochs.plk')
