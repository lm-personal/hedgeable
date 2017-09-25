Hedgeable

# FINAL MODEL:
- My final model was using a LSTM using Word2Vec word embeddings.
- I have saved the model into the model folder
- To read in the model that I have created and generate outputs please run the evaluate_model.py script - first you will have to unzip the model.h5.zip and then can run the script.
- The whole script is DJIAI_reddit_model.py (however this uses the pre-trained Google word vector and this file is too large to upload to github.)
- the file testpreds_word2vecLSTM_final.csv will include the test output and the predictions.

# PROCESS
- To test over-fitting/learning I split up the data into a train and CV since I will treat the test set as the output as we do not have the labels for the test set. I tried to mimic the same time-frame by using the same range of dates that is on the test set just one year before.
- Train 2008-08-08 to 2014-06-30
- Validation 2014-07-01 to 2015-06-30
- Test 2015-07-01 to 2016-06-30

# INITAL ANALYSIS
- The model I initially used was just a simple Random Forest with 1-gram tfidf. I also played around with BOWs. I chose my baseline model to be Random Forest so I could analyze feature importance and get some understanding that the model made sense.
- The baseline score I got was 49% with bi-grams .

- Since this was below a random guess, I retried with bi-grams and received a score of 52% on the validation set.

# MODEL ANALYSIS
- I got two models one with a 97% train and a 56% CV. 
- However due to over fitting I tried to use less epochs and my predictions were created with a model with a 80% train and 54% CV --> seems to generalize much better.

# FOR THE FUTURE

1. Since the news is happening the same day of the stock there may be some lag - think about changing the date of prediction.
2. Could break up the titles to different rows, perform sentiment analysis and then use that as a feature to count how many positive titles and negative titles and see if there is a correlation.
5. Try CNNs.
6. Try hyper parameter tunning with Bayesian optimization or perhaps grid search.
7. Try topic modeling to get less noisy features.
