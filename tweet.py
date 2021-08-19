# 3
import numpy as np 
import pandas as pd 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Importing the dataset       
tweets = pd.read_csv("C:\\Users\\WIN10\\Desktop\\LEARNING\\NaiveBayesProblemStatement\\Disaster_tweets_NB.csv")
tweets.head()
tweets.info()
x_train = tweets["text"]
y_train = tweets["target"]
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
import re
import string
def process_tweet(tweet):

    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and  # remove stopwords
            word not in string.punctuation):  # remove punctuation
            # tweets_clean.append(word)
            stem_word = stemmer.stem(word)  # stemming word
            tweets_clean.append(stem_word)

    return tweets_clean

custom_tweet = "OMG house burned :( #bad #morning http://fire.com"
# print cleaned tweet
print(process_tweet(custom_tweet))
def lookup(freqs, word, label):
    n = 0  # freqs.get((word, label), 0)

    pair = (word, label)
    if (pair in freqs):
        n = freqs[pair]

    return n


def count_tweets(result, ntweets, ys):

    for y, tweet in zip(ys, ntweets):
        for word in process_tweet(tweet):
            # define the key, which is the word and label tuple
            pair = (word, y)

            # if the key exists in the dictionary, increment the count
            if pair in result:
                result[pair] += 1

            # else, if the key is new, add it to the dictionary and set the count to 1
            else:
                result[pair] = 1
    

    return result
freqs = count_tweets({}, x_train, y_train)

def train_naive_bayes(freqs, train_x, train_y):
    
    loglikelihood = {}
    logprior = 0

   
    # calculate V, the number of unique words in the vocabulary
    vocab = set([pair[0] for pair in freqs.keys()])
    V = len(vocab)

    # calculate N_real and N_notreal
    N_real = N_notreal = 0
    for pair in freqs.keys():
        # if the label is real (greater than zero)
        if pair[1] > 0:

            # Increment the number of real words by the count for this (word, label) pair
            N_real += freqs[pair]

        # else, the label is negative
        else:

            # increment the number of not real words by the count for this (word,label) pair
            N_notreal += freqs[pair]

    # Calculate D, the number of documents
    D = len(train_x)

    # Calculate D_pos, the number of real disaster
    D_real = sum(train_y==1)

    # Calculate D_notreal, the number of not real disaster 
    D_notreal = sum(train_y==0)

    # Calculate logprior
    logprior = np.log(D_real)- np.log(D_notreal)

    # For each word in the vocabulary...
    for word in vocab:
        # get the real and not real frequency of the word
        freq_real = lookup(freqs,word,1)
        freq_notreal = lookup(freqs,word,0)

        # calculate the probability that each word is real and not real
        p_w_real = (freq_real+1)/(N_real+V)
        p_w_notreal = (freq_notreal+1)/(N_notreal+V)

        # calculate the log likelihood of the word
        loglikelihood[word] = np.log(p_w_real)-np.log(p_w_notreal)


    return logprior, loglikelihood

logprior, loglikelihood = train_naive_bayes(freqs, x_train, y_train)
print(logprior)
print(len(loglikelihood))

def naive_bayes_predict(tweet, logprior, loglikelihood):
   
    # process the tweet to get a list of words
    word_l = process_tweet(tweet)

    # initialize probability to zero
    p = 0

    # add the logprior
    p += logprior

    for word in word_l:

        # check if the word exists in the loglikelihood dictionary
        if word in loglikelihood:
            # add the log likelihood of that word to the probability
            p += loglikelihood[word]

    return p

my_tweet = "@sakuma_en If you pretend to feel a certain way the feeling can become genuine all by accident. -Hei..."
p = naive_bayes_predict(my_tweet, logprior, loglikelihood)
print('The expected output is', p)

def test_naive_bayes(test_x, test_y, logprior, loglikelihood):

    accuracy = 0  # return this properly

    y_hats = []
    for tweet in test_x:
        # if the prediction is > 0
        if naive_bayes_predict(tweet, logprior, loglikelihood) > 0:
            # the predicted class is 1
            y_hat_i = 1
        else:
            # otherwise the predicted class is 0
            y_hat_i = 0

        # append the predicted class to the list y_hats
        y_hats.append(y_hat_i)

    # error is the average of the absolute values of the differences between y_hats and test_y
    error = np.mean(np.absolute(y_hats-test_y))

    # Accuracy is 1 minus the error
    accuracy = 1-error

    return accuracy

test_data = pd.read_csv("C:\\Users\\WIN10\\Desktop\\LEARNING\\NaiveBayesProblemStatement\\Disaster_tweets_NB.csv")
y_predicted = []

for data in test_data["text"]:
    result = naive_bayes_predict(data, logprior, loglikelihood)
    if(result>=0):
        y_predicted.append(1)
    if(result<0):
        y_predicted.append(0)
        
y_predicted
iyd = test_data["id"]
trgt = y_predicted
dict = {'id': iyd, 'target': trgt} 
df = pd.DataFrame(dict) 
df.head()
df.to_csv("submission.csv", index=False)

