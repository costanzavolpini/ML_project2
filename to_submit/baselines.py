import sys

from preprocessing import applyPreProcessing
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import numpy as np
import _pickle as cPickle
from ngram import *
import pandas as pd
import numpy as np
import re, nltk
import csv, string
from nltk import ngrams
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def train_test_features(pos_filename, neg_filename):
    np.random.seed(0)

    posfile = open(pos_filename, 'rb')
    negfile = open(neg_filename, 'rb')
    train_size = len(posfile.readlines()) + len(negfile.readlines())
    posfile = open(pos_filename, 'rb')
    negfile = open(neg_filename, 'rb')
    
    train_tweets = []
    train_labels = []
    count = 0

    print('Processing of ' + str(train_size) + ' tweets for the training set...')
    for tweet in posfile.readlines():
        tweet = applyPreProcessing(tweet.decode('utf-8'))
        train_tweets.append(tweet)
        train_labels.append(1)
        count = count + 1
        if count%1000 == 0:
            print("Processing tweet " + str(count) + "/" + str(train_size))
    
    for tweet in negfile.readlines():
        tweet = applyPreProcessing(tweet.decode('utf-8'))
        train_tweets.append(tweet)
        train_labels.append(0)
        count = count + 1
        if count%1000 == 0:
            print("Processing tweet " + str(count) + "/" + str(train_size))
    
    posfile.close()
    negfile.close()

    
    test_tweets = []
    testfile = open('test_data.txt' ,'rb')
    test_size = len(testfile.readlines())
    testfile = open('test_data.txt' ,'rb')
    
    count = 0
    print('Processing of ' + str(test_size) + ' tweets for the testing set...')
    for tweet in testfile:
        tweet = tweet.decode('utf-8')
        tweet = ','.join(tweet.split(',')[1:])
        tweet = applyPreProcessing(tweet)

        test_tweets.append(tweet)
        count = count + 1
        if count%1000 == 0:
            print("Processing tweet " + str(count) + "/" + str(test_size))

    return train_tweets, train_labels, test_tweets

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
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})

train_tweets, train_labels, test_tweets = train_test_features("train_pos_full.txt", "train_neg_full.txt")

count_vectorizer = CountVectorizer(ngram_range=(1,2))
vectorized_data = count_vectorizer.fit_transform(train_tweets)
indexed_data = hstack((np.array(range(0,vectorized_data.shape[0]))[:,None], vectorized_data))

data_train, data_test, targets_train, targets_test = train_test_split(indexed_data, train_labels, test_size=0.2, random_state=0)
data_train_index = data_train[:,0]
data_train = data_train[:,1:]
data_test_index = data_test[:,0]
data_test = data_test[:,1:]

vectorized_data_test = count_vectorizer.transform(test_tweets)
indexed_data_test = hstack((np.array(range(0,vectorized_data_test.shape[0]))[:,None], vectorized_data_test))


# Logistic Regression
print("Logistic Regression")
lf = LogisticRegression(random_state=0, solver='sag', max_iter=30)
# train
lf.fit(data_train, targets_train)
# test
final_test = indexed_data_test.tocsr()[:,1:]

final_score = lf.predict(final_test)
final_score[final_score >= 0.5] = 1
final_score[final_score < 0.5] = -1

create_csv_submission(np.arange(1, final_test.shape[0]+1), final_score, 'logistic_full.csv')


# Random Forest
print("Random Forest")
rf = RandomForestClassifier(n_estimators=30, max_depth=100, random_state=0)
rf.fit(data_train, targets_train)

final_score = rf.predict(final_test)
final_score[final_score >= 0.5] = 1
final_score[final_score < 0.5] = -1

create_csv_submission(np.arange(1, final_test.shape[0]+1), final_score, 'randomforest_full.csv')