import pandas as pd
import numpy as np

from support_functions import *
from sklearn.ensemble import RandomForestClassifier
from keras.preprocessing.text import Tokenizer

import gensim
from gensim.models.keyedvectors import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers.embeddings import Embedding
from keras.layers import Convolution1D, MaxPooling1D
import numpy as np
from keras.utils import np_utils


#######################################
########## READ IN DATA ###############
#######################################

print('read and structure dates train file...')
train = pd.read_csv('dataset/train.csv')
train['timestamp'] = pd.to_datetime(train['timestamp'])

print('read and structure dates test file...')
test = pd.read_csv('dataset/test.csv')
test['timestamp'] = pd.to_datetime(test['timestamp'])

print 'train rows: %d' % train.shape[0]
print 'test rows: %d' % test.shape[0]

######################################
########## CREATE TARGETS ############
######################################
print('create targets...')


train.sort_values('timestamp', inplace = True) # sort data so that the targets make sense
targets = []
targets.append(np.nan)
for idx in range(1,train.shape[0]):
    prev_price = train.iloc[idx-1].loc['Price']
    curr_price = train.iloc[idx].loc['Price']
    if curr_price >= prev_price:
        curr_targ = 1
    else:
        curr_targ = 0
    targets.append(curr_targ)
train['targets'] = targets

######################################
########## CLEAN POSTS DATA ##########
######################################
print('clean text data')

train = train.dropna()

print('clean reviews...')
clean_train_reviews_raw, clean_test_reviews_raw = cleanwords(train, test)

######################################
############## RUN MODEL #############
######################################

# vectorize the text samples into a 2D integer tensor

google_embeddings = 'dataset/GoogleNews-vectors-negative300.bin'
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 300
maxlen = 50
nb_classes = 2
max_features = 20000


#################### WORD2VEC ##############
print('read in google trained word vector...')
word2vec = KeyedVectors.load_word2vec_format(google_embeddings, binary=True)

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(clean_train_reviews_raw)

sequences_train = tokenizer.texts_to_sequences(clean_train_reviews_raw)
sequences_test = tokenizer.texts_to_sequences(clean_test_reviews_raw)

X_train = sequence.pad_sequences(sequences_train, maxlen=maxlen)
X_test = sequence.pad_sequences(sequences_test, maxlen=maxlen)

Y_train = np_utils.to_categorical(train['targets'], nb_classes)

word_index = tokenizer.word_index

nb_words = min(MAX_NB_WORDS, len(word_index))+1


print('create word embeddings for model')
embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))

notinwordlist = []
for word, i in word_index.items():
    if word in word2vec.vocab:
        embedding_matrix[i] = word2vec.word_vec(word)
    else:
        notinwordlist.append(word)
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
print('Not Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) != 0))

n_symbols = len(embedding_matrix)
vocab_dim = len(embedding_matrix[0])
nb_epoch=2
batch_size=64


X_train_val, X_cv_val, y_train_val, y_cv_val = createtraintestsplit(X_train, Y_train)

model = Sequential()
model.add(Embedding(output_dim=vocab_dim, input_dim=n_symbols, weights=[embedding_matrix]))  # note you have to put embedding weights in a list by convention
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2)) 
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print('train model for cv score...')
model.fit(X_train_val, y_train_val, validation_data=(X_cv_val, y_cv_val), epochs=nb_epoch, batch_size=batch_size)

score_train, acc_train = model.evaluate(X_train_val, y_train_val,
                            batch_size=batch_size)

score_cv, acc_cv = model.evaluate(X_cv_val, y_cv_val,
                            batch_size=batch_size)

print('Train accuracy:', acc_train)
print('CV accuracy:', acc_cv)


model_final = Sequential()
model_final.add(Embedding(output_dim=vocab_dim, input_dim=n_symbols, weights=[embedding_matrix]))  # note you have to put embedding weights in a list by convention
model_final.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2)) 
model_final.add(Dense(nb_classes))
model_final.add(Activation('softmax'))
model_final.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print('retrain model on whole set of train for outputs....')
model_final.fit(X_train, Y_train, epochs=nb_epoch, batch_size=batch_size)


######################################
############# SAVE MODEL #############
######################################

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model_final.save_weights("model.h5")
print("Saved model to disk")

######################################
############## SAVE OUTPUTS ##########
######################################

ypreds = model_final.predict_classes(X_test, verbose=0)
# test['ypreds'] = ypreds
test['ypreds'] = ypreds


print('saving outputs to csv')
test.to_csv('testpreds_word2vecLSTM_final.csv', index = False)
