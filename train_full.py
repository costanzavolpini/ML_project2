import spacy

#import model as model_config
#from data_utils import load as load_data, extract_features
import numpy as np
import csv

import re
import numpy as np
import csv
import pandas as pd
import pickle
import string
from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
import re
import spacy


import spacy
import pickle
import pandas as pd
import numpy as np
import time

from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import LabelEncoder


twitter_data_neg_small = './twitter-datasets/train_neg.txt'
twitter_data_pos_small = './twitter-datasets/train_pos.txt'
twitter_data_small     = './twitter-datasets/train_small.txt'

twitter_data_neg_full  = './twitter-datasets/train_neg_full.txt'
twitter_data_pos_full  = './twitter-datasets/train_pos_full.txt'
twitter_data_full      = './twitter-datasets/train_full.txt'
nlp = spacy.load('en_core_web_lg')


# Use dictionary from http://luululu.com/tweet/typo-corpus-r1.txt
# http://people.eng.unimelb.edu.au/tbaldwin/etc/emnlp2012-lexnorm.tgz
# to handle abbreviations, mistakes...etc. (IN )

# LULU-CORPUS
# (1) INSERT (IN): a character is added to the original word.
# (2) REMOVE (RM): a character is removed from the original word.
# (3) REPLACE1 (R1): the order of character is different from the original word (the number of differences is one).
# (4) REPLACE2 (R2): a character is different from the original word

final_corpus = {}

def corpusReplace(corpus):
    for word in corpus:
        word = word.decode('utf8')
        word = word.split()
        final_corpus[word[0]] = word[1]    

corpus_lulu = open('corpus/lulu-corpus.txt', 'rb')
corpusReplace(corpus_lulu)
corpus_lulu.close()

corpus_emnlp = open('corpus/emnlp-corpus.txt', 'rb')
corpusReplace(corpus_emnlp)
corpus_emnlp.close()

def applyCorpus(tweet):
    new_tweet = ''
    for w in tweet.split(' '):
        if w in final_corpus.keys():
            #Replace with correct value
            new_word = final_corpus[w]
            new_tweet = new_tweet + ' ' + new_word
        else:
            new_tweet = new_tweet + ' ' + w
    return new_tweet

#raw_data_train['text'] = raw_data_train.text.apply(applyCorpus)   
         
def cleanTweet(tweet):
    tweet = re.sub('<url>','',tweet)
    tweet = re.sub('<user>','',tweet)
    tweet = re.sub(r'\$\w*', '', tweet)
    tweet = tweet.lower()
    tweet = re.sub(r'https?:\/\/.*\/\w*', '', tweet)
    tweet = re.sub(r'#\w*', '', tweet) #hashtag
    tweet = re.sub(r'[' + punctuation.replace('@', '') + ']+', ' ', tweet) # puntuaction
    tweet = re.sub(r'\s\s+', ' ', tweet)
    tweet = tweet.lstrip(' ') 
    tweet = ''.join(c for c in tweet if c <= '\uFFFF') 
    return tweet

#raw_data_train['text'] = raw_data_train.text.apply(cleanTweet)

def extract_features(docs, max_length):
    docs = list(docs)
    X = np.zeros((len(docs), max_length), dtype='int32')
    for i, doc in enumerate(docs):
        j = 0
        for token in doc:
            if token.has_vector and not token.is_punct and not token.is_space:
                X[i, j] = token.rank + 1
                j += 1
                if j >= max_length:
                    break
    return X

def load_twitter_data_small(from_cache=False):
    cached_data_path = twitter_data_small + '.cached.pkl'

    if from_cache:
        print('Loading data from cache...')
        with open(cached_data_path, 'rb') as f:
            return pickle.load(f)

    max_length = 100

    print('Loading and preparing data...')
    raw_data_neg = pd.read_csv(twitter_data_neg_small, header=None, sep="\n", encoding='latin1', names=['text'],
                               error_bad_lines=False, warn_bad_lines=False, quoting=csv.QUOTE_NONE).drop_duplicates()
    raw_data_neg['text'] = raw_data_neg['text'].apply(cleanTweet)
    raw_data_neg['text'] = raw_data_neg['text'].apply(applyCorpus)
    raw_data_neg['label'] = 0
    raw_data_neg = raw_data_neg

    raw_data_pos = pd.read_csv(twitter_data_pos_small, header=None, sep="\n", encoding='latin1', names=['text'],
                               error_bad_lines=False, warn_bad_lines=False, quoting=csv.QUOTE_NONE).drop_duplicates()
    raw_data_pos['text'] = raw_data_pos['text'].apply(cleanTweet)
    raw_data_pos['text'] = raw_data_pos['text'].apply(applyCorpus)
    raw_data_pos['label'] = 1
    raw_data_pos = raw_data_pos
    
    raw_data = pd.concat([raw_data_neg, raw_data_pos], ignore_index=True)
#     raw_data = raw_data[:10000]
    
    #print(raw_data)

    # Parse tweet texts
    docs = list(nlp.pipe(raw_data['text'], batch_size=1000, n_threads=8))
    print(max([len(x) for x in docs])) 
    
    y = raw_data['label'].values
    
    # Pull the raw_data into vectors
    X = extract_features(docs, max_length=max_length)
    
    # Split into train and test sets
    rs = ShuffleSplit(n_splits=2, random_state=42, test_size=0.2)
    train_indices, test_indices = next(rs.split(X))
    
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    
    docs = np.array(docs, dtype=object)
    docs_train = docs[train_indices]
    docs_test = docs[test_indices]
    
    numeric_data = X_train, y_train, X_test, y_test
    raw_data = docs_train, docs_test

    with open(cached_data_path, 'wb') as f:
        pickle.dump((numeric_data, raw_data), f)
    
    return numeric_data, raw_data

def load_twitter_data_full(from_cache=False):
    cached_data_path = twitter_data_full + '.cached.pkl'

    if from_cache:
        print('Loading data from cache...')
        with open(cached_data_path, 'rb') as f:
            return pickle.load(f)

    max_length = 100

    print('Loading and preparing data...')
    raw_data_neg = pd.read_csv(twitter_data_neg_full, header=None, sep="\n", encoding='latin1', names=['text'],
                               error_bad_lines=False, warn_bad_lines=False, quoting=csv.QUOTE_NONE).drop_duplicates()
    raw_data_neg['text'] = raw_data_neg['text'].apply(cleanTweet)
    raw_data_neg['text'] = raw_data_neg['text'].apply(applyCorpus)
    raw_data_neg['label'] = 0
    raw_data_neg = raw_data_neg

    raw_data_pos = pd.read_csv(twitter_data_pos_full, header=None, sep="\n", encoding='latin1', names=['text'],
                               error_bad_lines=False, warn_bad_lines=False, quoting=csv.QUOTE_NONE).drop_duplicates()
    raw_data_pos['text'] = raw_data_pos['text'].apply(cleanTweet)
    raw_data_pos['text'] = raw_data_pos['text'].apply(applyCorpus)
    raw_data_pos['label'] = 1
    raw_data_pos = raw_data_pos
    
    raw_data = pd.concat([raw_data_neg, raw_data_pos], ignore_index=True)
#     raw_data = raw_data[:10000]
    
    #print(raw_data)

    # Parse tweet texts
    docs = list(nlp.pipe(raw_data['text'], batch_size=5000, n_threads=8))
    print(max([len(x) for x in docs])) 
    
    y = raw_data['label'].values
    
    # Pull the raw_data into vectors
    X = extract_features(docs, max_length=max_length)
    
    # Split into train and test sets
    rs = ShuffleSplit(n_splits=2, random_state=42, test_size=0.2)
    train_indices, test_indices = next(rs.split(X))
    
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    
    docs = np.array(docs, dtype=object)
    docs_train = docs[train_indices]
    docs_test = docs[test_indices]
    
    numeric_data = X_train, y_train, X_test, y_test
    raw_data = docs_train, docs_test

    with open(cached_data_path, 'wb') as f:
        pickle.dump((numeric_data, raw_data), f)
    
    return numeric_data, raw_data

def load(data_name, *args, **kwargs):
    load_fn_map = {
        'twitter_data_small': load_twitter_data_small,
        'twitter_data_full': load_twitter_data_full,
    }
    return load_fn_map[data_name](*args, **kwargs)


# Load Twitter data full
(X_train, y_train, X_test, y_test), (docs_train, docs_test) = load('twitter_data_full', from_cache=True)

print('X_train:', X_train.shape)
print('y_train:', y_train.shape)
print('X_test:', X_test.shape)
print('y_test:', y_test.shape)

import spacy
import keras
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Flatten, Conv1D, MaxPooling1D, GlobalAveragePooling1D, LSTM, merge
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.constraints import maxnorm
from keras.regularizers import l2
from keras import backend as K

def get_embeddings(vocab):
    max_rank = max(lex.rank+1 for lex in vocab if lex.has_vector)
    vectors = np.ndarray((max_rank+1, vocab.vectors_length), dtype='float32')
    for lex in vocab:
        if lex.has_vector:
            vectors[lex.rank + 1] = lex.vector
    return vectors


vocab_nlp = spacy.load('en_core_web_lg', parser=False, tagger=False, entity=False)
print('Preparing embeddings...')
embeddings = get_embeddings(vocab_nlp.vocab)

def build_model_twitter4(max_length=100,
                nb_filters=64,
                kernel_size=3,
                pool_size=2,
                regularization=0.01,
                weight_constraint=2.,
                dropout_prob=0.4,
                clear_session=True):
    if clear_session:
        K.clear_session()

    model = Sequential()
    model.add(Embedding(
        embeddings.shape[0],
        embeddings.shape[1],
        input_length=max_length,
        trainable=False,
        weights=[embeddings]))

    model.add(LSTM(100, recurrent_dropout = 0.2, dropout = 0.2))
    model.add(Dense(1, activation = 'sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

    return model
    
def build_model_twitter5(max_length=100,
                nb_filters=64,
                kernel_size=3,
                pool_size=2,
                regularization=0.01,
                weight_constraint=2.,
                dropout_prob=0.4,
                clear_session=True):
    if clear_session:
        K.clear_session()

    model = Sequential()
    model.add(Embedding(
        embeddings.shape[0],
        embeddings.shape[1],
        input_length=max_length,
        trainable=False,
        weights=[embeddings]))

    model.add(Conv1D(padding = "same", kernel_size = 3, filters = 32, activation = "relu"))
    model.add(MaxPooling1D(pool_size = 2))
    model.add(LSTM(100))
    model.add(Dense(1, activation = 'sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

    return model

#Model 5
np.random.seed(42)


epochs = 12
batch_size = 160


# Take a look at the shapes
print('X_train:', X_train.shape)
print('y_train:', y_train.shape)
print('X_test:', X_test.shape)
print('y_test:', y_test.shape)

tb_callback = keras.callbacks.TensorBoard(
        histogram_freq=0, write_graph=True)

model = build_model_twitter5()
model.fit(X_train, y_train,
          validation_data=(X_test, y_test),
          batch_size=batch_size, epochs=epochs,
          callbacks=[tb_callback])