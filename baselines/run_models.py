import _pickle as cPickle
import numpy as np
import csv
from models import *
from feature_extraction import train_test_features, dumpFeatures


print("hi")




# dumpFeatures(False, True, True, None, 'features_pretrained_bigram_small.dat')
# [X_train, y, X_test, max_features, W] = cPickle.load(open("features_pretrained_bigram_small.dat", "rb"))
# model2 = build_model2(max_features+1, W.shape[1], X_train.shape[1], embeddings=W, trainable=True)
# model2.summary()
# model2.fit(X_train, y, validation_split=0.1, nb_epoch=1, batch_size=128, verbose=1)


print("dumping..")
dumpFeatures(True, False, True, None, 'features_pretrained_full.dat')
print("dumped!")
[X_train, y, X_test, max_features, W] = cPickle.load(open("features_pretrained_full.dat", "rb"))
#[X_train, y, X_test, max_features, W] = train_test_features(full=True, n_gram=False, pretrained=True, nb_words=None)

model2 = build_model2(max_features+1, W.shape[1], X_train.shape[1], embeddings=W, trainable=True)
model2.summary()
model2.fit(X_train, y, validation_split=0.1, nb_epoch=2, batch_size=128, verbose=1)
model2.save('features_pretrained_full_MODEL')
