# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 15:38:15 2022

@author: siddharth rawal
"""

import pandas as pd
import re
import nltk
from timeit import default_timer as time
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer 
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics  import accuracy_score
from sklearn.ensemble import RandomForestClassifier




## Cleaning the text and removing the special characters 
def clean_review(text):
    
    text= re.sub("[^-9A-Za-z ]", "" , text)
    return text

## Function to do the lemmatization here for the next step so that we can clean each review
def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(w) for w in text]

## Function to preprocess data(removing emply rows), removing useless words and joining back the text
def preporocess_data(data):
    
    
    data.dropna(axis=0,how='any',thresh=None,subset=None,inplace=True)
    data['text'] = data['text'].apply(clean_review)
    data["text"] = data["text"].apply(nltk.word_tokenize)

    stop_words = stopwords.words('english')

    data["text"] = data["text"].apply(lambda words: [word for word in words if word not in stop_words])

    data['text'].apply(lemmatize_text)
    data['text'] = [' '.join(x) for x in data['text']]
    
    return data

## Function to apply machine lerning algorithm on cleaned train data and getting the accuracy on test data
def MultinomialNBalgo(train_dt, test_dt):
    
    X_train = data_train['text'].values
    y_train = data_train['label'].values
    X_test = data_test['text'].values
    y_test = data_test['label'].values

    vectorizer = TfidfVectorizer()
    train_vectors = vectorizer.fit_transform(X_train)
    test_vectors = vectorizer.transform(X_test)

    clf = MultinomialNB().fit(train_vectors, y_train)
    predicted = clf.predict(test_vectors)
    print("Accuracy: ", (accuracy_score(y_test,predicted)*100))
    
def RandomForestClass(train_dt, test_dt):
    
     X_train = data_train['text'].values
     y_train = data_train['label'].values
     X_test = data_test['text'].values
     y_test = data_test['label'].values
     
     vectorizer = TfidfVectorizer()
     train_vectors = vectorizer.fit_transform(X_train)
     test_vectors = vectorizer.transform(X_test)
     
     clf = RandomForestClassifier(n_estimators = 10, criterion = "entropy")
     clf.fit(train_vectors, y_train)
     predicted = clf.predict(test_vectors)
     print("Accuracy: ", (accuracy_score(y_test,predicted)*100))
     
## Main function 
if __name__ == '__main__':
    start_time = time()
    data_train = pd.read_csv('train.csv')
    data_train = data_train.drop(data_train.columns[[1, 2]], axis=1)
    preporocess_data(data_train)   
    data_test = pd.read_csv('test.csv')
    data_test = data_test.drop(data_test.columns[[1, 2]], axis=1)
    preporocess_data(data_test)   
    data_testlabel = pd.read_csv('labels.csv')
    data_test = data_test.merge(data_testlabel, on ="id", how = 'inner') 

    #MultinomialNBalgo(data_train, data_test)
    #print("Elapsed Time: ",time() - start_time, "seconds")
    
    RandomForestClass(data_train, data_test)
    print("Elapsed Time: ",time() - start_time, "seconds")
    









