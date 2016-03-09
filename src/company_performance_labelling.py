# -*- coding: utf-8 -*-
"""
Created on Mon Mar 07 09:16:39 2016
Week 6: Document Labeling based on stock prices
@author: mihir.tamhankar
"""
import pandas as pd
from datetime import datetime
import csv

RAI=pd.read_csv("C:/Downloads/Staples/WMT.csv")
#SP500=pd.read_csv("C:/Downloads/Staples/SP500.csv")

def getreturns(dataframe):
    date=dataframe['Date']
    price=dataframe['Adj Close']
    month=[]
    year=[]
    marchprice=[]
    junprice=[]
    q2ret=[]
    finyear=[]
    for d in date:
        dt=datetime.strptime(d, '%d-%m-%Y')
        month.append(dt.month)
        year.append(dt.year)
    for i in range(0,len(date)):
       if month[i]==3:
          marchprice.append(price[i])
          finyear.append(year[i])
       elif month[i]==6:
          junprice.append(price[i])
       else:
          continue
    for j in range(0,len(marchprice)):
       q2ret.append((junprice[j]-marchprice[j])/marchprice[j])
       
    rows=zip(finyear,q2ret)
    with open('q2ret.csv','wb') as f:
        writer=csv.writer(f)
        for r in rows:
            writer.writerow(r)

getreturns(RAI)