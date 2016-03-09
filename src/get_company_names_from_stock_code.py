#!/usr/bin/python

import pandas as pd

df = pd.read_csv('secwiki_tickers.csv')
dp = pd.read_csv('portfolio.lst',names=['pTicker'])

pTickers = dp.pTicker.values  # converts into a list

tmpTickers = []

for i in range(len(pTickers)):
  test = df[df.Ticker==pTickers[i]]
  if not (test.empty):
    print("%-10s%s" % (pTickers[i], list(test.Name.values)[0]))
