import pandas as pd
import numpy as np
from keras.models import load_model
from keras.models import model_from_json

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

# read in model and make predictions

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

MAX_NB_WORDS = 200000
EMBEDDING_DIM = 300
maxlen = 50
nb_classes = 2
max_features = 20000


#################### WORD2VEC ##############

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(clean_train_reviews_raw)
sequences_test = tokenizer.texts_to_sequences(clean_test_reviews_raw)
X_test = sequence.pad_sequences(sequences_test, maxlen=maxlen)

word_index = tokenizer.word_index

nb_words = min(MAX_NB_WORDS, len(word_index))+1

# load json and create model
json_file = open('model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model/model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data 
# Define X_test & Y_test data first
loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

ypreds = loaded_model.predict_classes(X_test, verbose=0)
# test['ypreds'] = ypreds
test['ypreds'] = ypreds


print('saving outputs to csv')
test.to_csv('output/testpreds_word2vecLSTM_final.csv', index = False)
