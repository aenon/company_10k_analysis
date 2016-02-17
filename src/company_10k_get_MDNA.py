#!/bin/env python
# coding: utf-8
# company_10K_get_MDNA.py
# IEOR242 Applications in Data Analytics Group 06
# Feb 14 2016

from urllib import urlopen
import re
import bs4


def get_MDNA(html_file):
    webpage = urlopen(html_file).read().decode('utf-8)')
    soup = bs4.BeautifulSoup(webpage, "html.parser")
    scores = soup.find_all(text=re.compile('DISCUSSION AND ANALYSIS'))
    divs = [score.parent.parent.parent.parent.parent.parent.parent.parent for score in scores]
    all_text = divs[0].getText()
    MDNA_text = all_text[:all_text.find('FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA')]
    return MDNA_text

def main():
    company_years = ['CP2012', 'CP2013', 'CP2014', 'PG2012', 'PG2013']
    # Paths to the MDNA parts from the scrapped company 10K files
    html_files = ["../data/company10k/" + company_years[i] +
                  ".html" for i in range(len(company_years))]
    MDNA_texts = [get_MDNA(html_files[i])
                   for i in range(len(MDNA_files))]
    print MDNA_texts

if __name__ == '__main__':
    main()
