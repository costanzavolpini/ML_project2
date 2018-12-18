import keras
from keras.models import Sequential
from keras.layers import Dense, Input, GlobalMaxPooling1D, GlobalAveragePooling1D, Conv1D, MaxPooling1D, Embedding, Dropout, LSTM, Activation, BatchNormalization
from keras.layers.embeddings import Embedding
import _pickle as cPickle


#[X_train, y, X_test, max_features, W] = cPickle.load(open("features_pretrained_small.dat", "rb"))

#embeddings = W
#trainable=True

#def build_model1(input_dim, output_dim, input_length, embeddings=None, trainable=False):
def build_model1(X_train, W, dropout_rate=0.5):

    input_dim = W.shape[0]
    output_dim = W.shape[1]
    input_length = X_train.shape[1]

    embeddings = W
    trainable=True
    print("model 1")
    model = Sequential()
    if embeddings is None:
        model.add(Embedding(input_dim, output_dim, input_length=input_length))
    else:
        model.add(Embedding(input_dim, output_dim, input_length=input_length, trainable=trainable, weights=[embeddings]))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


#def build_model2(input_dim, output_dim, input_length, embeddings=None, trainable=False):
#def build_model2(input_dim=96339, output_dim=200, input_length=30, kernel_size=5, dropout_rate=0.5):
def build_model2(input_dim=96339, output_dim=200, input_length=30, kernel_size=5, dropout_rate=0.5):
    #[X_train, y, X_test, max_features, W] = cPickle.load(open("features_pretrained_small.dat", "rb"))
    [X_train, y, X_test, max_features, W] = cPickle.load(open("features_pretrained_small_newEMBEDDING.dat", "rb"))

    input_dim = W.shape[0]
    output_dim = W.shape[1]
    input_length = X_train.shape[1]

    embeddings = W
    trainable=True
    print("model 2")
    model = Sequential()
    if embeddings is None:
        model.add(Embedding(input_dim, output_dim, input_length=input_length))
    else:
        model.add(Embedding(input_dim, output_dim, input_length=input_length, trainable=trainable, weights=[embeddings]))

    model.add(Conv1D(128, kernel_size, activation='relu'))
    model.add(MaxPooling1D(5))
    model.add(Dropout(dropout_rate))
    #model.add(Conv1D(128, 5, activation='relu'))
    #model.add(MaxPooling1D(5))
    model.add(Conv1D(128, kernel_size, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


    return model


#def build_model3(input_dim=96339, output_dim=200, input_length=30, kernel_size=5, dropout_rate=0.5, units):
#def build_model3(input_dim=92891, output_dim=200, input_length=30, kernel_size=5, dropout_W=0, dropout_U=0, units=100, filters=32):
def build_model3(X_train, W, kernel_size=5, dropout_W=0, dropout_U=0, units=100, filters=32, pool_size=2):
    input_dim = W.shape[0]
    output_dim = W.shape[1]
    input_length = X_train.shape[1]

    embeddings = W
    trainable=True
    print("model 3")
    model = Sequential()
    if embeddings is None:
        model.add(Embedding(input_dim, output_dim, input_length=input_length))
    else:
        model.add(Embedding(input_dim, output_dim, input_length=input_length, trainable=trainable, weights=[embeddings]))


    model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation = "relu"))
    model.add(MaxPooling1D(pool_size = pool_size))
    model.add(LSTM(units, dropout_W=dropout_W, dropout_U=dropout_U))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation = 'sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

    return model


def build_model4(X_train, W, kernel_size=5, dropout_W=0, dropout_U=0, units=100, filters=32):
    input_dim = W.shape[0]
    output_dim = W.shape[1]
    input_length = X_train.shape[1]

    embeddings = W
    trainable=True
    print("model 4")
    model = Sequential()
    if embeddings is None:
        model.add(Embedding(input_dim, output_dim, input_length=input_length))
    else:
        model.add(Embedding(input_dim, output_dim, input_length=input_length, trainable=trainable, weights=[embeddings]))


    #model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation = "relu"))
    #model.add(MaxPooling1D(pool_size = 2))
    model.add(LSTM(units, dropout_W=dropout_W, dropout_U=dropout_U))
#     model.add(Dense(128, activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(64, activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(32, activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(1, activation = 'sigmoid'))
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation = 'sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

    return model


# like model3, but with batch normalization
def build_model5(X_train, W, kernel_size=5, dropout_W=0.2, dropout_U=0.2, units=100, filters=32):
    input_dim = W.shape[0]
    output_dim = W.shape[1]
    input_length = X_train.shape[1]

    embeddings = W
    trainable=True
    print("model 5")
    model = Sequential()
    if embeddings is None:
        model.add(Embedding(input_dim, output_dim, input_length=input_length))
    else:
        model.add(Embedding(input_dim, output_dim, input_length=input_length, trainable=trainable, weights=[embeddings]))


    model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation = "relu"))
    model.add(MaxPooling1D(pool_size = 2))
    model.add(LSTM(units, dropout_W=dropout_W, dropout_U=dropout_U))
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation = 'sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

    return model
