import sys


from preprocess import clean
from preprocessing import applyPreProcessing
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import numpy as np
import _pickle as cPickle


# def train_test_features(full=True, n_gram=False, pretrained=True, nb_words=None):
def train_test_features(pos_filename, neg_filename, pretrained=True, nb_words=-1):
    """
    Function that does the cleaning,the tokenization, add the n-grams, build the weight matrix(pretrained)
    depending on the arguments
    Arguments: pos_filename: the name of the file that contains the positive samples
               neg_filename: the name of the file that contains the negative samples 
               pretrained (True to use the glove200 pretraining)
               nb_words (None to take all the words otherwise takes an integer value N which will take the N most frequent words)
    Returns : train_sequences (List of list of word indexes for each tweet for the training)
              test_sequences (List of list of word indexes for each tweet for the test)
              labels (Labels)
              vocab_size (Maximum word index in the list of list train_sequences)
              embedding_matrix (if pretrained=True returns the embedding_matrix builded by the help of glove)
              This code was inspired from : https://github.com/fchollet/keras/blob/master/examples/imdb_fasttext.py
              and from : https://github.com/fchollet/keras/blob/master/examples/pretrained_word_embeddings.py
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
#     for tweet in posfile:
    for tweet in posfile.readlines() + negfile.readlines():
#         tweet = tweet.decode('utf-8')
#         tweet = clean(tweet.decode('utf-8'))
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
#         tweet = tweet.decode('utf-8')
        tweet = clean(tweet.decode('utf-8'))
#         tweet = tweet[tweet.find(',')+1:]
        tweet = ','.join(tweet.split(',')[1:])
#         tweet = clean(tweet)
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
    print('Pad sequences')
    train_sequences = sequence.pad_sequences(train_sequences, maxlen=maxlen)
    test_sequences = sequence.pad_sequences(test_sequences, maxlen=maxlen)
    print('train_sequences shape:', train_sequences.shape)
    print('test_sequences shape:', test_sequences.shape)

    labels = np.array(int(train_size/2) * [0] + int(train_size/2) * [1])

    print('Shuffling of the training set...')
    indices = np.arange(train_sequences.shape[0])
    np.random.shuffle(indices)
    train_sequences = train_sequences[indices]
    labels = labels[indices]
    
    embedding_matrix = None

    if pretrained:

        EMBEDDING_DIM = 200
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

def save(pos_filename, neg_filename, pretrained, nb_words, output):
    trains, labels, tests, vocab, embedding = train_test_features(pos_filename, neg_filename, pretrained, nb_words)
    cPickle.dump([trains, labels, tests, vocab, embedding],open(output, 'wb'))
    return