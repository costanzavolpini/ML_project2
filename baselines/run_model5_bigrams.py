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
from keras.models import load_model


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
#             print(y_pred)
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})



#save('train_pos_full.txt', 'train_neg_full.txt', False, -1, 2, 'embedding_hashtag_split/features_bigrams_full_no_old_clean.dat')
[X_train, y, X_test, max_features, W] = cPickle.load(open("embedding_hashtag_split/features_bigrams_full_no_old_clean.dat", "rb"))




#model5 = build_model5(X_train, W, input_dim=max_features+1, output_dim=50)
#model5.summary()


#model5.fit(X_train, y, validation_split=0.1, nb_epoch=1, batch_size=256, verbose=1)
#model5.save('embedding_hashtag_split/model5_bigrams_full_dataset_1_epochs_no_old_clean.plk')
model5 = load_model('embedding_hashtag_split/model5_bigrams_full_dataset_1_epochs_no_old_clean.plk')
predicted_labels = model5.predict_proba(X_test)
predicted_labels[predicted_labels >= 0.5] = 1
predicted_labels[predicted_labels < 0.5] = -1
create_csv_submission(np.arange(1, X_test.shape[0]+1), predicted_labels, 'model5_bigrams_full_dataset_1_epochs_no_old_clean.csv')


model5.fit(X_train, y, validation_split=0.1, nb_epoch=1, batch_size=256, verbose=1)
model5.save('embedding_hashtag_split/model5_bigrams_full_dataset_2_epochs_no_old_clean.plk')
predicted_labels = model5.predict_proba(X_test)
predicted_labels[predicted_labels >= 0.5] = 1
predicted_labels[predicted_labels < 0.5] = -1
create_csv_submission(np.arange(1, X_test.shape[0]+1), predicted_labels, 'model5_bigrams_full_dataset_2_epochs_no_old_clean.csv')


model5.fit(X_train, y, validation_split=0.1, nb_epoch=1, batch_size=256, verbose=1)
model5.save('embedding_hashtag_split/model5_bigrams_full_dataset_3_epochs_no_old_clean.plk')
predicted_labels = model5.predict_proba(X_test)
predicted_labels[predicted_labels >= 0.5] = 1
predicted_labels[predicted_labels < 0.5] = -1
create_csv_submission(np.arange(1, X_test.shape[0]+1), predicted_labels, 'model5_bigrams_full_dataset_3_epochs_no_old_clean.csv')

model5.fit(X_train, y, validation_split=0.1, nb_epoch=1, batch_size=256, verbose=1)
model5.save('embedding_hashtag_split/model5_bigrams_full_dataset_4_epochs_no_old_clean.plk')
predicted_labels = model5.predict_proba(X_test)
predicted_labels[predicted_labels >= 0.5] = 1
predicted_labels[predicted_labels < 0.5] = -1
create_csv_submission(np.arange(1, X_test.shape[0]+1), predicted_labels, 'model5_bigrams_full_dataset_4_epochs_no_old_clean.csv')

model5.fit(X_train, y, validation_split=0.1, nb_epoch=1, batch_size=256, verbose=1)
model5.save('embedding_hashtag_split/model5_bigrams_full_dataset_5_epochs_no_old_clean.plk')
predicted_labels = model5.predict_proba(X_test)
predicted_labels[predicted_labels >= 0.5] = 1
predicted_labels[predicted_labels < 0.5] = -1
create_csv_submission(np.arange(1, X_test.shape[0]+1), predicted_labels, 'model5_bigrams_full_dataset_5_epochs_no_old_clean.csv')
