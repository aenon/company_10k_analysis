#!/bin/env python
# analysing10ks.py
# IEOR242 Applications in Data Analytics
# Jan 25 2016
# names...


# Import required libraries
import requests
# import bs4
from bs4 import BeautifulSoup

# Specify the urls to parse

Colgate_Palmolive2012 = 'http://www.sec.gov/Archives/edgar/data/21665/000144530513000275/cl-12312012x10k.htm'
Colgate_Palmolive2013 = 'http://www.sec.gov/Archives/edgar/data/21665/000154554714000003/cl-12312013x10k.htm'
Colgate_Palmolive2014 = 'http://www.sec.gov/Archives/edgar/data/21665/000162828015000846/cl-12312014x10k.htm'

Procter_Gamble2012 = 'http://www.sec.gov/Archives/edgar/data/80424/000008042412000063/fy2012financialstatementsf.htm'
Procter_Gamble2013 = 'http://www.sec.gov/Archives/edgar/data/80424/000008042413000063/fy201310kannualreport.htm'
Procter_Gamble2014 = 'https://www.sec.gov/Archives/edgar/data/80424/000008042414000057/fy201410kannualreport.htm'

Johnson_Johnson2012 = 'http://www.sec.gov/Archives/edgar/data/200406/000020040613000038/a2012123010-k.htm'
Johnson_Johnson2013 = 'http://www.investor.jnj.com/secfiling.cfm?filingID=200406-14-33&CIK=200406'
Johnson_Johnson2014 = 'http://www.sec.gov/Archives/edgar/data/200406/000020040615000004/a2014122810-k.htm'

url =  # CP2012

# Set up http connection
response = requests.get(url)

# read response into a markup
html = response.content

# make soup!
soup = BeautifulSoup(html, "html.parser")

# find text
soup.findAll('a')


# NLP
import nltk
from nltk.corpus import stopwords
print stopwords.words("english")

import re
# Use regular expressions to do a find-and-replace
letters_only = re.sub("[^a-zA-Z]",           # The pattern to search for
                      " ",                   # The pattern to replace it with
                      example1.get_text())  # The text to search
print letters_only
lower_case = letters_only.lower()        # Convert to lower case
words = lower_case.split()
words = [w for w in words if not w in stopwords.words("english")]
print words
# Split into words


# import the finance dictionary

# Let's use pandas

import pandas as pd
Library = pd.DataFrame(pd.read_csv(
    "LoughranMcDonald_MasterDictionary_2014.csv"))

import csv
# route of the file has to be changed
with open('LoughranMcDonald_MasterDictionary_2014.csv', mode='r') as infile:
    reader = csv.reader(infile, delimiter=',')
    with open('LoughranMcDonald_MasterDictionary_2014_new.csv', mode='w') as outfile:
        writer = csv.writer(outfile)
        mydict = {rows[0]: rows[1] for rows in reader}

# Word Count
from collections import Counter
cnt = Counter(Texts)

##########################################################################
# Table Count


def table_count(url):
    response = requests.get(url)
    html = response.content
    soup = BeautifulSoup(html, "html.parser")
    # Mihir Comments: Seems to find all words in the text which are table not
    table = soup.find_all("table")
    return len(table)

# Image Count


def img_count(url):
    response = requests.get(url)
    html = response.content
    soup = BeautifulSoup(html, "html.parser")
    img = soup.find_all("img")
    return len(img)

# Page Count


def pages_count(url):
    response = requests.get(url)
    html = response.content
    soup = BeautifulSoup(html, "html.parser")
    lines = soup.find_all("hr")
    return len(lines) - 1
##########################################################################

# Code to find number of characters in the document
for script in soup(["script", "style"]):
    script.extract()    # rip it out
# get text
text = soup.get_text()
# break into lines and remove leading and trailing space on each
lines = (line.strip() for line in text.splitlines())
# break multi-headlines into a line each
chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
# drop blank lines
text = '\n'.join(chunk for chunk in chunks if chunk)
print len(text)

# Text stemming
# Remove stopwords
from nltk.corpus import stopwords
pattern = re.compile(
    r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
noStopText = pattern.sub('', originalText)

##########################################################################
# Code for summarizing stats for the 10-K file
import requests
import re
from bs4 import BeautifulSoup
from collections import Counter
from nltk.corpus import stopwords
url = 'http://www.sec.gov/Archives/edgar/data/21665/000144530513000275/cl-12312012x10k.htm'
response = requests.get(url)
html = response.content
soup = BeautifulSoup(html, "html.parser")
text1 = soup.get_text()
b = len(text1.split(" "))
print b
for script in soup(["script", "style"]):
    script.extract()
text = soup.get_text()
lines = (line.strip() for line in text.splitlines())
chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
text = '\n'.join(chunk for chunk in chunks if chunk)
#scores = soup.find_all(text=re.compile('Management'))
a = len(text.split(" "))
print a
c = len(text)
print c
table = soup.find_all("table")
print len(table)
img = soup.find_all("img")
print len(img)
pg = soup.find_all("hr")
print len(pg) - 1
cnt = Counter(text1.split())
# print cnt
pattern = re.compile(
    r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
noStopText = pattern.sub('', text)
cnt1 = Counter(noStopText.split())
print cnt1


##########################################################################
# Code with Stemming

# Import Libraries
import requests
import bs4
import nltk
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk import stem
stemmer = stem.PorterStemmer()
response = requests.get(url)

html = response.content
soup = BeautifulSoup(html, "html.parser")
text = soup.get_text()
table = soup.find_all("table")
print "\nNumber of Tables", len(table)
img = soup.find_all("img")
print "Number of Images", len(img)
pg = soup.find_all("hr")
print "Number of Pages", len(pg) - 1

letters_only = re.sub("[^a-zA-Z]", " ", text)
lower_case = letters_only.lower()
words = lower_case.split()
words = [w for w in words if not w in stopwords.words("english")]
words = [stemmer.stem(word) for word in words]
word_count = Counter(words)

for key, value in sorted(word_count.iteritems(), key=lambda (k, v): (v, k)):
    print "%s: %s" % (key, value)
##########################################################################
# Rcode for BarPLot and WordClouds
mydata < - read.csv("IE242.csv")
library(tm)
library(SnowballC)
library(wordcloud)
library(RColorBrewer)
wordcloud(words=mydata$Word, freq=mydata$Frequency, min.freq=1,
          max.words=200, random.order=FALSE, rot.per=0.35,
          colors=brewer.pal(8, "Dark2"))
barplot(mydata[1:16, ]$Frequency, las=2, names.arg=mydata[1:16, ]$Word,
        col="lightblue", main="Most frequent words",
        ylab="Word frequencies")
