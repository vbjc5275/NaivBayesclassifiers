# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 08:31:01 2020

@author: Jerry
"""
import re
import random
from math import log,exp
from sklearn.metrics import confusion_matrix
from collections import defaultdict
from sklearn.naive_bayes import BernoulliNB

def read_file(filepath="SMSSpamCollection.txt"):
    """
    :return mails: 訊息列表
    """
    mails = []
    with open(filepath,encoding="utf-8") as file:
        for line in file.readlines():
            is_spam,message = line.split("\t")
            is_spam = 1 if is_spam=="spam" else 0
            message = message.strip()
            mails.append((message,is_spam))
    return mails



    
def count_words(training_set):
    """training set consists of pairs (message, is_spam)"""
    counts = defaultdict(lambda :[0,0])
    for message, is_spam in training_set:
        for word in tokenize(message):
            counts[word][0 if is_spam else 1] += 1
    return counts

def word_probabilities(word_count,total_spams, total_non_spams,k=0.5):
    word_prob = {}
    for word,count in word_count.items():
       prob_if_spam =  (1+count[0])/(2*k+total_spams)
       prob_if_not_spam = (1+count[1])/(2*k+total_non_spams)
       word_prob[word] = (prob_if_spam,prob_if_not_spam)
    return word_prob

def spam_probability(word_probs, message):
    message_words = tokenize(message)
    log_prob_if_spam = log_prob_if_not_spam = 0.0

    for word, (prob_if_spam, prob_if_not_spam) in word_probs.items():

        # for each word in the message,
        # add the log probability of seeing it
        if word in message_words:
            log_prob_if_spam += log(prob_if_spam)
            log_prob_if_not_spam += log(prob_if_not_spam)

        # for each word that's not in the message
        # add the log probability of _not_ seeing it
        else:
            log_prob_if_spam += log(1.0 - prob_if_spam)
            log_prob_if_not_spam += log(1.0 - prob_if_not_spam)

    prob_if_spam = exp(log_prob_if_spam)
    prob_if_not_spam = exp(log_prob_if_not_spam)
    return prob_if_spam / (prob_if_spam + prob_if_not_spam)
  
word_count = count_words(mails)

total_spams = len([is_spam for _,is_spam in mails if is_spam])
total_non_spams = len([is_spam for _,is_spam in mails if not is_spam])
word_probs = word_probabilities(word_count,total_spams, total_non_spams)

#spam_probability(word_probs, mails[8][0])

y_proba = []
y_true = []
for message,actual in mails:
    y_proba.append(spam_probability(word_probs, message))
    y_true.append(actual)
    
y_pred  = [1 if proba>0.5 else 0 for proba in y_proba ]
tn, fp, fn, tp = confusion_matrix(y_true,y_pred).ravel()
(tn+tp)/(fp+fn+tn+tp)



class NaiveBayesClassifier():
    def __init__(self, k = 1):
        self.k = k  # smoothing factor
        self.class_count_ = [0,0] 
        self.word_count_ = defaultdict(lambda :[0,0])
        self.word_prob_ = defaultdict(lambda :[0,0])
        
    def tokenize(self,message):
        message = message.lower()                      
        all_words = re.findall("[a-z]+", message) 
        return set(all_words)
    
    def train(self,messages,spam_or_not):
        #1-計算單詞次數
        for message,is_spam in zip(messages,spam_or_not):
            for word in self.tokenize(message):
                self.word_count_[word][is_spam] += 1
            self.class_count_[is_spam] += 1 
            
        #2-計算機率
        #每種類別數量
        for word,count in self.word_count_.items():
            word_prob_if_spam = (self.k+count[1])/(2*self.k+self.class_count_[1])
            word_prob_if_non_spam = (self.k+count[0])/(2*self.k+self.class_count_[0])
            
            self.word_prob_[word][1] = word_prob_if_spam
            self.word_prob_[word][0] = word_prob_if_non_spam

    
    def predict_proba(self,message):
        log_prob_if_spam = 0
        log_prob_if_non_spam = 0
        
        all_words = self.tokenize(message)
        for word,word_prob in self.word_prob_.items():
            if word in all_words:
                log_prob_if_spam += log(word_prob[1])
                log_prob_if_non_spam += log(word_prob[0])
                
            else:
                log_prob_if_spam += log(1-word_prob[1])
                log_prob_if_non_spam += log(1-word_prob[0])
        
        prob_if_spam = exp(log_prob_if_spam)
        prob_if_non_spam = exp(log_prob_if_non_spam )
        return prob_if_spam/(prob_if_spam+prob_if_non_spam)
    
    def predict(self,message):
        return 1 if self.predict_proba(message)>0.5 else 0
    
             
def split_data(X,y):
    for i in range(len(X)):
        X[i].append()
    data = X
    random.shuffle(data)
    train_data = data[:50]
    test_data = data[50:]  
    
mails = read_file()
random.shuffle(mails)
train_num  = round(0.8*len(mails))
train_X = [ mail[0] for mail in mails[:train_num]]
train_y = [ mail[1] for mail in mails[:train_num]]

test_X = [ mail[0] for mail in mails[train_num:]]
test_y = [ mail[1] for mail in mails[train_num:]]

nb = NaiveBayesClassifier()
nb.train(train_X,train_y)

y_pred = []
for x in test_X:
    y_pred.append(nb.predict(x))

from sklearn.metrics import plot_confusion_matrix,confusion_matrix
plot_confusion_matrix(test_y,y_pred,labels=["spam","not spam"])  # doctest: +SKIP
    
tn, fp, fn, tp = confusion_matrix(y_true=test_y,y_pred,["spam","not spam"]).ravel()
(tn+tp)/(fp+fn+tn+tp)


