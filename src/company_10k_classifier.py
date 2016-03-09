# -*- coding: utf-8 -*-
"""
Created on Mon Mar 07 22:45:13 2016

@author: Team 6
10-K Classifier
"""
#Importing required modules
import os
import re
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from nltk import stem
from sklearn.feature_selection import SelectPercentile, f_classif
from textblob import TextBlob
from sklearn.ensemble import RandomForestClassifier


#Creates the master dataframe from the text files and assigns them labels
def get_dataset(path):
    dataset=[]
    try:
        os.chdir(path)
    except:
        print "Incorrect path name!"
    
    for filename in os.listdir(path):
        f=open(filename,'r')
        if re.search("POS",filename):
            dataset.append([f.read(),"pos"])
        else:
            dataset.append([f.read(),"neg"])
    dataset=pd.DataFrame(dataset)
    dataset.columns = ['MDA_Text','Sentiment']
    return dataset

#Splitting into training and testing set
def split(df,test_ratio):
    return train_test_split(df, test_size = test_ratio, stratify = df['Sentiment'])
    
#Function to stem words a string    
def stemming(x):
    stemmer = stem.SnowballStemmer("english")
    words=x.split()
    doc=[]
    for word in words:
        word=stemmer.stem(word)
        doc.append(word)
    return " ".join(doc)

#Function to remove all non-characters from MD&As
def preprop(dataset):
    dataset['MDA_Text']=dataset['MDA_Text'].str.replace("[^a-zA-Z]", ' ')
    return dataset

#Function to create features of total positive and total negative words based on Loughran McDonald Dictionary
def count_fin_words(lmd,dataset):
    #Modifying the Dictionary  
    lmd=lmd[['Word','Positive','Negative']]
    lmd['Sum']=lmd['Positive']+lmd['Negative']
    lmd=lmd[lmd.Sum != 0]
    lmd=lmd.drop(['Sum'],axis=1)
    lmd.loc[lmd['Positive']>0, 'Positive'] = 1
    lmd.loc[lmd['Negative']>0, 'Negative'] = -1
    lmd['Word']=lmd['Word'].str.lower()
    #Counting the words in the MDA
    tf = CountVectorizer(analyzer='word', min_df = 0, stop_words = 'english')
    tfidf_matrix =  tf.fit_transform(dataset['MDA_Text'].values)
    feature_names = tf.get_feature_names() 
    tfidf_array = tfidf_matrix.toarray()
    tfidf_df = pd.DataFrame(tfidf_array)
    tfidf_df.columns = [i.lower() for i in feature_names] 
    tfidf_df = tfidf_df.T 
    tfidf_df['Word']=tfidf_df.index
    #Merging the results
    result_df = pd.merge(tfidf_df, lmd, how='inner',left_on='Word',right_on='Word')
    col_list=list(result_df)
    result_df_pos=result_df[result_df.Positive==1]
    result_df_neg=result_df[result_df.Negative==-1]
    result_df[col_list[0:len(dataset)]].sum(axis=0)
    #Counting the positive and negative words in a financial context per document
    pos_words_sum=result_df_pos[col_list[0:len(dataset)]].sum(axis=0)
    neg_words_sum=result_df_neg[col_list[0:len(dataset)]].sum(axis=0)
    #Adding new features to the master dataframe
    dataset['Tot_pos']=pos_words_sum.values
    dataset['Tot_neg']=neg_words_sum.values
    return dataset

#Function to create polarity score feature
def polarity_score(dataset):
    polarity=[]
    polarity_score=[]
    for mda,sentiment in zip(dataset['MDA_Text'].values,dataset['Sentiment'].values):
        blob=TextBlob(mda)
        score = blob.sentiment.polarity
        polarity.append([score,sentiment])
        polarity_score.append(score)
    dataset['Polarity']=polarity_score
    return dataset

#Function to add features to the train and test set based on vectorizer
def vect_features(vectorizer,train,test):
    features_train_transformed = vectorizer.fit_transform(train['MDA_Text'].values)
    feature_names = vectorizer.get_feature_names()
    features_train_transformed = features_train_transformed.toarray()
    train_data = pd.DataFrame(features_train_transformed)
    train_data.columns = feature_names
    train=pd.concat([train,train_data],axis=1)
    features_test_transformed = vectorizer.transform(test['MDA_Text'].values)
    features_test_transformed = features_test_transformed.toarray()
    test_data = pd.DataFrame(features_test_transformed)
    test_data.columns = feature_names
    test=pd.concat([test,test_data],axis=1)
    return train,test

#Function to create Classification Report   
def report(test,predictions):
   pd.crosstab(test['Sentiment'], predictions, rownames=['Actual'], colnames=['Predicted'], margins=True)
   a=accuracy_score(test['Sentiment'],predictions)
   p=precision_score(test['Sentiment'],predictions, pos_label = "pos")
   r=recall_score(test['Sentiment'].values,predictions, pos_label = "pos")
   f=f1_score(test['Sentiment'].values,predictions, pos_label = "pos")
   print "Accuracy = ",a,"\nPrecision =",p,"\nRecall = ",r,"\nF-Score = ",f 

#Function to create models and print accuracies
def model(classifier,train,test,column):
    targets = train['Sentiment'].values
    train_data=train.values
    predictors = train_data[0:,column:]
    classifier.fit(predictors,targets)
    test_data=test.values
    predictions=classifier.predict(test_data[0:,column:])
    report(test_1,predictions)
    return predictions

#Reading the Loughran McDonald Dictionary    
os.chdir("D:\Joel\UC Berkeley\Courses\IEOR 242 - Applications of Data Analysis\Project\Homework 7")
lmd = pd.read_excel("LoughranMcDonald_MasterDictionary_2014.xlsx")      

#Defining the path    
path = "D:\Joel\UC Berkeley\Courses\IEOR 242 - Applications of Data Analysis\Project\Homework 7\MDAs"

#Creating the master dataframe
dataset=get_dataset(path)

#Preprocessing the master dataframe
dataset=preprop(dataset)

#Adding total positive and total negative words based on Loughran McDonald Dictionary to the master dataframe
dataset=count_fin_words(lmd,dataset)

#Creating polarity score feature
dataset=polarity_score(dataset)

#Stemming the MD&A Text
stemmer = stem.SnowballStemmer("english")
dataset['MDA_Text']=dataset['MDA_Text'].apply(stemming)

#Splitting to training and testing
train, test = split(dataset,0.25)
train=train.reset_index(drop=True)
test=test.reset_index(drop=True)

#Model 1 - Baseline Model
#Algorithm: Bernoulli Naive Bayes 
#Features: Contains all words
vectorizer_1 = CountVectorizer(stop_words='english')
train_1,test_1 = vect_features(vectorizer_1,train,test)
classifier = BernoulliNB(fit_prior=False)
predictions = model(classifier,train_1,test_1,5)

#Model 2 
#Algorithm: Bernoulli Naive Bayes 
#Features: Contains top 50 words
vectorizer_2 = CountVectorizer(stop_words='english',max_features=50)
train_2,test_2 = vect_features(vectorizer_2,train,test)
classifier = BernoulliNB(fit_prior=False)
predictions = model(classifier,train_1,test_1,5)

#Model 3 
#Algorithm: Multinomial Naive Bayes 
#Features: CountVectorizer of all words
vectorizer_3 = CountVectorizer(stop_words='english')
train_3,test_3 = vect_features(vectorizer_3,train,test)
classifier = MultinomialNB(fit_prior=False)
predictions = model(classifier,train_3,test_3,5)

#Model 4 
#Algorithm: Multinomial Naive Bayes
#Features: CountVectorizer of only top 50 words and 2-grams
vectorizer_4 = CountVectorizer(stop_words='english',max_features=50,ngram_range=(1,2),min_df=5,max_df=0.8)
train_4,test_4 = vect_features(vectorizer_4,train,test)
classifier = MultinomialNB(fit_prior=False)
predictions = model(classifier,train_4,test_4,5)


#Model 5
#Algorithm: Multinomial Naive Bayes
#Features: TfidfVectorizer of only top 50 words and 2-grams
vectorizer_5 = TfidfVectorizer(sublinear_tf=True,stop_words='english',max_features=50,ngram_range=(1,2),min_df=5,max_df=0.8)
train_5,test_5 = vect_features(vectorizer_5,train,test)
classifier = MultinomialNB(fit_prior=False)
predictions = model(classifier,train_5,test_5,5)


#Model 6
#Algorithm: Gaussian Naive Bayes
#Features: TfidfVectorizer of only top 50 words and 2-grams,otal positive words, total negative words, polarity score 
vectorizer_6 = TfidfVectorizer(sublinear_tf=True,stop_words='english',max_features=50,ngram_range=(1,2),min_df=5,max_df=0.8)
train_6,test_6 = vect_features(vectorizer_6,train,test)
classifier = GaussianNB()
predictions = model(classifier,train_6,test_6,2)

#Model 7 
#Algorithm: Uncalibrated Random Forest
#Features: TfidfVectorizer of only top 50 words and 2-grams, total positive words, total negative words, polarity score 
vectorizer_7 = TfidfVectorizer(sublinear_tf=True,stop_words='english',max_features=50,ngram_range=(1,2),min_df=5,max_df=0.8)
train_7,test_7 = vect_features(vectorizer_5,train,test)
classifier = RandomForestClassifier(n_estimators=1000)
predictions = model(classifier,train_7,test_7,2)

    
#Code to ingore terms of sparse matrix (Not Used)
#selector = SelectPercentile(f_classif,percentile=10)
#selector.fit(features_train_transformed,train['Sentiment'].values)
#features_train_transformed = selector.transform(features_train_transformed).toarray()
#features_test_transformed =  selector.transform(features_test_transformed).toarray()
  


