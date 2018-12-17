import sys

import _pickle as cPickle
import numpy as np
import csv
#from models import *
from models_small_pretrained_train import *
from feature_extraction import train_test_features, save
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
import os


save("train_pos_full.txt", "train_neg_full.txt", True, -1, 'embedding_hashtag_split/features_pretrained_full.dat')
