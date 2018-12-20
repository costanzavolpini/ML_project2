import sys


from preprocessing import applyPreProcessing
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import numpy as np
import _pickle as cPickle
from ngram import *


def train_test_features(pos_filename, neg_filename, pretrained=True, nb_words=-1, n_gram=-1):
    """
    Function that does the cleaning,the tokenization, add the n-grams, build the weight matrix(pretrained)
    depending on the arguments
    Arguments: pos_filename: the name of the file that contains the positive samples
               neg_filename: the name of the file that contains the negative samples 
               pretrained: defines whether the glove200 pretrained embedding should be used
               nb_words: defines the number of words that (None to take all the words otherwise takes an integer value N which will take the N most frequent words)
               n_gram: defines whether n-grams should be used ( anything below 2 means no n-gram is involved)
    Returns : train_sequences: a padded 2D array that contains the word indices for each tweet of the training set (1 row = 1 tweet)
              test_sequences: a padded 2D array that contains the word indices for each tweet of the test set
              labels: train labels
              vocab_size: Maximum word index in the list of list train_sequences
              embedding_matrix: if pretrained=True returns the embedding_matrix builded by the help of glove, None otherwise
              Inspired from keras examples: https://github.com/fchollet/keras/blob/master/examples/imdb_fasttext.py
              and https://github.com/fchollet/keras/blob/master/examples/pretrained_word_embeddings.py
    """
    np.random.seed(0)


    posfile = open(pos_filename, 'rb')
    negfile = open(neg_filename, 'rb')
    train_size = len(posfile.readlines()) + len(negfile.readlines())
    posfile = open(pos_filename, 'rb')
    negfile = open(neg_filename, 'rb')
    
    train_tweets = []
    count = 0

    print('Processing of ' + str(train_size) + ' tweets for the training set...')
    for tweet in posfile.readlines() + negfile.readlines():
        tweet = applyPreProcessing(tweet.decode('utf-8'))
        train_tweets.append(tweet)
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

    print('Tokenization of tweets...')
    if nb_words == -1:
        tokenizer = Tokenizer(filters='')
    else:
        tokenizer = Tokenizer(nb_words=nb_words, filters='')
    tokenizer.fit_on_texts(train_tweets)
    train_sequences = tokenizer.texts_to_sequences(train_tweets)
    test_sequences = tokenizer.texts_to_sequences(test_tweets)
    
    
    word_index = tokenizer.word_index
    vocab_size = len(word_index)
    print('Found %s unique tokens.' % vocab_size)

    
    maxlen = 30
    
    if n_gram > 1:
        maxlen = 60
        ngram_range = n_gram
        print('Adding {}-gram features'.format(ngram_range))
        # Create set of unique n-gram from the training set.
        ngram_set = set()
        for input_list in train_sequences:
            for i in range(2, ngram_range + 1):
                set_of_ngram = create_ngram_set(input_list, ngram_value=i)
                ngram_set.update(set_of_ngram)

        # Dictionary mapping n-gram token to a unique integer.
        # Integer values are greater than vocab_size in order
        # to avoid collision with existing features.
        start_index = vocab_size + 1
        token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
        indice_token = {token_indice[k]: k for k in token_indice}

        # vocab_size is the highest integer that could be found in the dataset.
        vocab_size = np.max(list(indice_token.keys())) + 1

        # Augmenting X_train and X_test with n-grams features
        train_sequences = add_ngram(train_sequences, token_indice, ngram_range)
        test_sequences = add_ngram(test_sequences, token_indice, ngram_range)
        print('Average train sequence length: {}'.format(np.mean(list(map(len, train_sequences)), dtype=int)))
        print('Average test sequence length: {}'.format(np.mean(list(map(len, test_sequences)), dtype=int)))
    
    
    
    print('Pad sequences')
    train_sequences = sequence.pad_sequences(train_sequences, maxlen=maxlen)
    test_sequences = sequence.pad_sequences(test_sequences, maxlen=maxlen)
    print('train_sequences shape:', train_sequences.shape)
    print('test_sequences shape:', test_sequences.shape)

    labels = np.array(int(train_size/2) * [0] + int(train_size/2) * [1])

    print('Shuffling the training set...')
    indices = np.arange(train_sequences.shape[0])
    np.random.shuffle(indices)
    train_sequences = train_sequences[indices]
    labels = labels[indices]
    
    embedding_matrix = None

    if pretrained:

        EMBEDDING_DIM = 200 # glove vectors have dimension = 200

        # first, build index mapping words in the embeddings set
        # to their embedding vector
        print('Indexing word vectors (from glove.twitter.27B.200d.txt)')
        embeddings_index = {}
        f = open('embeddings/glove.twitter.27B.200d.txt','rb')
        for line in f:
            values = line.split()
            word = values[0].decode('utf-8')
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

        print('Found %s word vectors.' % len(embeddings_index))

        # second, prepare text samples and their labels
        print('Processing text dataset')
        embedding_matrix = np.zeros((vocab_size + 1, EMBEDDING_DIM))
        for word, i in word_index.items():
            if i > vocab_size:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        print('Embedding matrix created.')

    return train_sequences, labels, test_sequences, vocab_size, embedding_matrix

def save(pos_filename, neg_filename, pretrained, nb_words, n_gram, output):
    trains, labels, tests, vocab, embedding = train_test_features(pos_filename, neg_filename, pretrained, nb_words, n_gram)
    cPickle.dump([trains, labels, tests, vocab, embedding],open(output, 'wb'))
    return
