#!/home/s_sn_giraffe/anaconda2/bin/python
# coding: utf-8

# In[28]:

from urllib import urlopen
import re
import bs4
from bs4 import BeautifulStoneSoup
import os
from os import path
import unicodedata
import shutil


# In[40]:

def get_MDNA(html_file):
    webpage = urlopen(html_file).read()
    print "Making soup with", html_file
    try:
        soup = bs4.BeautifulSoup(webpage, "html5lib")
    except:
        print "Error while making soup"
    else:
        print "Soup is done"
    soup_encoded = unicode(soup.get_text()).encode('ascii', 'replace').replace('\n', '').replace('\t', '').lower()
    soup_encoded_soup = bs4.BeautifulSoup(soup_encoded, "html5lib")
    print "soup is encoded"
    scores = soup_encoded_soup.find_all(text=re.compile('discussion and analysis',re.IGNORECASE))
    print len(scores), "scores found for file", html_file
    try:
        divs = [score.parent.parent.parent.parent.parent.parent.parent.parent for score in scores]
    except:
        divs = scores[0]
        print "divs = scores[0]"
    else:
        print "divs get"

    try:
        all_text = divs[0].get_text()
    except:
        all_text = divs
        print "all_text = divs"
    else:
        print "all_text get"

    try:
        splitspace = all_text.split('item 7')
        splitqm = all_text.split('item?7')
        if len(splitspace) > len(splitqm):
            splittext = splitspace
            print "Using splitspace"
        else:
            splittext = splitqm
            print "Using splitqm"
    except:
        print "Error getting split"
    else:
        print "Split got"

    try:
        MDNA_text = splittext[-2] + splittext[-1]
        MDNA_text = re.sub('<[^>]*>', '', MDNA_text)
    except:
        print "Error getting MDNA_text"
    else:
        print "Got MDNA_text"

    try:
        with open("MDNA_" + html_file + ".txt", "w") as text_file:
            text_file.write("{}".format(MDNA_text))
    except:
        print "Error saving file"
    else:
        print "MDNA_" + html_file + ".txt saved"


# In[41]:

html_files = [html_file for html_file in os.listdir('.') if html_file.endswith(".html")]


# In[ ]:

for i in range(len(html_files)):
    print " "
    print i, "opening file", html_files[i]
    try:
        get_MDNA(html_files[i])
    except:
        print "Error getting MDNA"
        #try:
        #    shutil.move('SP500_10Ks/' + html_file, '10Ks_errors/' + html_file)
        #except:
        #    print "Error moving file"
        #else:
        #    print "File moved"
    else:
        print "Got MDNA"
        #try:
        #    shutil.move('SP500_10Ks/' + html_file, '10Ks_finished/' + html_file)
        #except:
        #    print "Error moving file"
        #else:
        #    print "File moved"
