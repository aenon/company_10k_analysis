# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 16:03:01 2016

@author: Team 6
Naive Bayes Classifier
"""
import os
import re
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from nltk import stem



#Creates the dataset from the text files
def get_dataset(path):
    dataset=[]
    try:
        os.chdir(path)
    except:
        print "Incorrect path name!"
    
    for filename in os.listdir("."):
        f=open(filename,'r')
        if re.search("_POS",filename):
            dataset.append([f.read(),"pos"])
        else:
            dataset.append([f.read(),"neg"])
            
    #return dataset
    dataset=pd.DataFrame(dataset)
    dataset.columns = ['MD&A_Text','Sentiment']
    return dataset

#Splitting into training and testing set
def split(df,test_ratio):
    return train_test_split(df, test_size = test_ratio)
    

def df_to_list(df):
    return df.values.tolist()

#Function to stem words a string    
def stemming(x):
    words=x.split()
    doc=[]
    for word in words:
        word=stemmer.stem(word)
        doc.append(word)
    return " ".join(doc)    


path = "D:\Joel\UC Berkeley\Courses\IEOR 242 - Applications of Data Analysis\Project\Homework 6\MDNAs"
dataset=get_dataset(path)
train, test = split(dataset,0.25)
train_set = df_to_list(train)
test_set = df_to_list(test)

#Pre-processing Training and Test sets   
train['MD&A_Text'] = train['MD&A_Text'].str.replace("[^a-zA-Z]", ' ');
test['MD&A_Text'] = test['MD&A_Text'].str.replace("[^a-zA-Z]", ' ');
stemmer = stem.SnowballStemmer("english")
train['MD&A_Text']=train['MD&A_Text'].apply(stemming)
test['MD&A_Text']=test['MD&A_Text'].apply(stemming)
vectorizer = CountVectorizer(stop_words="english",ngram_range=(1,2))
counts = vectorizer.fit_transform(train['MD&A_Text'].values)

#Creating Pipeline
pipeline = Pipeline([
    ('vectorizer',  CountVectorizer(stop_words="english",ngram_range=(1,2),max_features=50)),
    ('tfidf_transformer',  TfidfTransformer()),
    ('classifier',  MultinomialNB(fit_prior=False)) ])

#Top word counts in training corpus
data=counts.toarray()
vocab=vectorizer.get_feature_names()
np.clip(data, 0, 1, out=data)
dist = np.sum(data, axis=0)
dic=[]
for tag, count in zip(vocab, dist):
    dic.append([tag,count])
topwords=pd.DataFrame(dic)	
topwords.columns = ['features','count'] 
topwords = topwords.sort('count', ascending=False)
topwords.head(20)

#Top word weights in training corpus
tf = TfidfVectorizer(analyzer='word', min_df = 0, stop_words = 'english', ngram_range=(1,2))
counts =  tf.fit_transform(train['MD&A_Text'].values)
data=counts.toarray()
vocab=tf.get_feature_names()
np.clip(data, 0, 1, out=data)
dist = np.sum(data, axis=0)
dic=[]
for tag, count in zip(vocab, dist):
    dic.append([tag,count])
topweights=pd.DataFrame(dic)
topweights.columns = ['features','count'] 
topweights = topweights.sort('count', ascending=False)
topwords.head(20)


#Training the classifier
pipeline.fit(train['MD&A_Text'].values,train['Sentiment'].values)
#Making predictions
predictions = pipeline.predict(test['MD&A_Text'].values)


#Accuracy Metrics
pd.crosstab(test['Sentiment'], predictions, rownames=['Actual'], colnames=['Predicted'], margins=True)
a=accuracy_score(test['Sentiment'],predictions)
p=precision_score(test['Sentiment'],predictions, pos_label = "pos")
r=recall_score(test['Sentiment'].values,predictions, pos_label = "pos")
f=f1_score(test['Sentiment'].values,predictions, pos_label = "pos")
print "Accuracy = ",a,"\nPrecision =",p,"\nRecall = ",r,"\nF-Score = ",f



