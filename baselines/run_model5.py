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



#[X_train, y, X_test, max_features, W] = cPickle.load(open("embedding_hashtag_split/features_pretrained_full.dat", "rb"))
[X_train, y, X_test, max_features, W] = cPickle.load(open("embedding_hashtag_split/features_pretrained_full_no_old_clean.dat", "rb"))



model5 = build_model5(X_train, W)
model5.summary()


model5.fit(X_train, y, validation_split=0.1, nb_epoch=1, batch_size=256, verbose=1)
model5.save('embedding_hashtag_split/model5_full_dataset_1_epochs_no_old_clean.plk')


model5.fit(X_train, y, validation_split=0.1, nb_epoch=1, batch_size=256, verbose=1)
model5.save('embedding_hashtag_split/model5_full_dataset_2_epochs_no_old_clean.plk')


model5.fit(X_train, y, validation_split=0.1, nb_epoch=1, batch_size=256, verbose=1)
model5.save('embedding_hashtag_split/model5_full_dataset_3_epochs_no_old_clean.plk')

model5.fit(X_train, y, validation_split=0.1, nb_epoch=1, batch_size=256, verbose=1)
model5.save('embedding_hashtag_split/model5_full_dataset_4_epochs_no_old_clean.plk')

model5.fit(X_train, y, validation_split=0.1, nb_epoch=1, batch_size=256, verbose=1)
model5.save('embedding_hashtag_split/model5_full_dataset_5_epochs_no_old_clean.plk')
