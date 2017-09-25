import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
import HTMLParser
from bs4 import BeautifulSoup
from markdown import markdown
import urllib2
from nltk.stem import PorterStemmer
from nltk import word_tokenize 
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from nltk.stem.wordnet import WordNetLemmatizer
from markdown import markdown
from nltk.corpus import stopwords
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, SimpleRNN, GRU

from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from collections import defaultdict
from keras import backend as K
K.set_image_dim_ordering('th')
from keras.layers.convolutional import Convolution1D

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



lmtzr = WordNetLemmatizer()
class WordNetLemmatizer(object):
    def __init__(self):
        self.lmtzr = lmtzr.lemmatize    
    def __call__(self, doc):
        return [self.lmtzr(t) for t in word_tokenize(doc)]

tfidfvect = TfidfVectorizer(
    ngram_range=(2,2),
    encoding = 'utf-8',
    tokenizer = WordNetLemmatizer(),
    # tokenizer = PorterTokenizer(),
    stop_words = stopwords.words('english'),
    lowercase = True
)

countvect = CountVectorizer(
    ngram_range=(2,2),
    encoding = 'utf-8',
    tokenizer = WordNetLemmatizer(),
    # tokenizer = PorterTokenizer(),
    stop_words = stopwords.words('english'),
    lowercase = True
)

def clean_text(example):
    example1 = example.replace("\\\'", "")
    example2 = example1.replace('"b','')
    example3 = example2.replace("'b","")
    example4 = example3.replace('U.S.', 'US')
    example5 = example4.replace('U.N.', 'UN')
    example6 = example5.replace('U.K.', 'UK')
    example7 = example6.replace('al queda', 'alqueda')
    return example7


def structure_text(example, featuretype, remove_stopwords=True):
    example1 = clean_text(example)
    example2 = BeautifulSoup(example1,'lxml')
    example3 = re.sub("[^a-zA-Z]", " ", example2.get_text())
    example4 = example3.lower() 
    example5 = example4.split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in example5 if not w in stops]  
        
    if featuretype == 'tfidf' or featuretype == 'count':
        return( " ".join( words )) 
    else: 
        return words

def cleanwords(train, test, featuretype='tfidf'):
    clean_train_reviews_raw = []
    clean_test_reviews_raw = []
    for i in range(train.shape[0]):
        clean_train_reviews_raw.append( structure_text( train['News'].iloc[i], featuretype ) )
    for i in range(test.shape[0]):
        clean_test_reviews_raw.append( structure_text( test['News'].iloc[i], featuretype ) )
    return clean_train_reviews_raw, clean_test_reviews_raw


def convertfeatures(train, test, featuretype, model=None, num_features='tfidf'):
    if featuretype == 'tfidf':
        train_data_features = tfidfvect.fit_transform(train)
        train_data_features = train_data_features.toarray()
        test_data_features = tfidfvect.transform(test)
        test_data_features = test_data_features.toarray()        
    elif featuretype == 'count':
        train_data_features = countvect.fit_transform(train)
        train_data_features = train_data_features.toarray()
        test_data_features = countvect.fit_transform(test)
        test_data_features = test_data_features.toarray()
    elif featuretype == 'word2vec':
        train_data_features = getAvgFeatureVecs( train, model, num_features )
        test_data_features = getAvgFeatureVecs( test, model, num_features )
    return train_data_features, test_data_features

def convertfeatures(train, test, featuretype, model=None, num_features='tfidf'):
    if featuretype == 'tfidf':
        train_data_features = tfidfvect.fit_transform(train)
        train_data_features = train_data_features.toarray()
        test_data_features = tfidfvect.transform(test)
        test_data_features = test_data_features.toarray()        
    elif featuretype == 'count':
        train_data_features = countvect.fit_transform(train)
        train_data_features = train_data_features.toarray()
        test_data_features = countvect.fit_transform(test)
        test_data_features = test_data_features.toarray()
    elif featuretype == 'word2vec':
        train_data_features = getAvgFeatureVecs( train, model, num_features )
        test_data_features = getAvgFeatureVecs( test, model, num_features )
    return train_data_features, test_data_features

def createtraintestsplit(X_train, y_train):
    X_train_val = X_train[:1483,:]
    X_cv_val = X_train[1483:,:]
    
    y_train_val = y_train[:1483,:]
    y_cv_val = y_train[1483:,:]
    return X_train_val, X_cv_val, y_train_val, y_cv_val


def makeFeatureVec(words, model, num_features):
    featureVec = np.zeros((num_features,),dtype="float32")
    nwords = 0.
    index2word_set = set(model.index2word)
    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    featureVec = np.divide(featureVec,nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    counter = 0
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    for review in reviews:
        print "Review %d of %d" % (counter, len(reviews))
        #print 'review', review
        reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
        counter = counter + 1
    return reviewFeatureVecs


def LSTMmodel(max_features = 10000, nb_classes = 2):
    model = Sequential()
    model.add(Embedding(max_features, 128, dropout=0.2))
    model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2)) 
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model