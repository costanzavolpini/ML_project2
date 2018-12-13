import keras
from keras.models import Sequential
from keras.layers import Dense, Input, GlobalMaxPooling1D, GlobalAveragePooling1D, Conv1D, MaxPooling1D, Embedding
from keras.layers.embeddings import Embedding
import _pickle as cPickle


#[X_train, y, X_test, max_features, W] = cPickle.load(open("features_pretrained_small.dat", "rb"))

#embeddings = W
#trainable=True

#def build_model1(input_dim, output_dim, input_length, embeddings=None, trainable=False):
def build_model1(input_dim=96339, output_dim=200, input_length=30):
    [X_train, y, X_test, max_features, W] = cPickle.load(open("features_pretrained_small.dat", "rb"))

    embeddings = W
    trainable=True
    print("model 1")
    model = Sequential()
    if embeddings is None:
        model.add(Embedding(input_dim, output_dim, input_length=input_length))
    else:
        model.add(Embedding(input_dim, output_dim, input_length=input_length, trainable=trainable, weights=[embeddings]))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


#def build_model2(input_dim, output_dim, input_length, embeddings=None, trainable=False):
def build_model2(input_dim=96339, output_dim=200, input_length=30):
    [X_train, y, X_test, max_features, W] = cPickle.load(open("features_pretrained_small.dat", "rb"))

    embeddings = W
    trainable=True
    print("model 2")
    model = Sequential()
    if embeddings is None:
        model.add(Embedding(input_dim, output_dim, input_length=input_length))
    else:
        model.add(Embedding(input_dim, output_dim, input_length=input_length, trainable=trainable, weights=[embeddings]))

    model.add(Conv1D(128, 5, activation='relu'))
    model.add(MaxPooling1D(5))
    #model.add(Conv1D(128, 5, activation='relu'))
    #model.add(MaxPooling1D(5))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


    return model
