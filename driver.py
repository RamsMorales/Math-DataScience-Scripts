#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 14:34:34 2024

@author: ramsonmunoz
"""
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt


pd.set_option("display.max_columns",None)
# Define of list of strings that represent the tickers I want to look at:
companies = ['AMZN','TSLA','MRNA','RGR','CALM','JPM','CENX']
#companies = ['JPM', 'VZ', 'XOM', 'DIS', 'PFE', 'CSCO', 'HD']
#companies = ['SPOT', 'DIS', 'ADI', 'FTNT', 'DLTR', 'WBA', 'SMPL']

start = '2024-09-06'
end = datetime.now().strftime('%Y-%m-%d')
resolution = '1d'

companyDailyReturns = []

for company in companies:
    ticker = yf.Ticker(company) #creates an instance of the ticker for a given ticker in my list of companies
    tickerHistory = ticker.history(start=start, end=end, interval=resolution) # pulls the history for that ticker starting and ending at given points with the given period, in our case 1d, and 20 days worth of data

    #print(tickerHistory.info())
    #priceOpen = tickerHistory[['Open']] # selects only open price data from all of the provided data. Useful if I want to compute intraday returns
    priceClose = tickerHistory[['Close']].to_numpy()  # selects only closing price data from all of the provided data. passes to numpy array

    dailyReturnVector = np.zeros(priceClose.size - 1)

    for i in range(1,priceClose.size):
        dailyReturnVector[i-1] = (priceClose[i].item() - priceClose[i -1].item()) / priceClose[i].item() # passes daily return for each company into a vector

    companyDailyReturns.append(dailyReturnVector)


companyDailyReturns = np.stack(companyDailyReturns).T #gets us a matrix of returns where each column is a stock and the rows are the daily returns for the 20 days.

'''
Now we need to mean-center the columns so that we can compute the covariance. Here we take advantage of broadcasting.
'''

#print(companyDailyReturns)

for i in range(companyDailyReturns.shape[1]):
    companyDailyReturns[:,i] = companyDailyReturns[:,i] - np.mean(companyDailyReturns[:,i]) # we subtract each column by its mean

#print(companyDailyReturns)

'''
now we implement the covariace formula of X.T @ X which in our case returns a 7x7 matrix which, once scaled, should be covariance 
'''

covariaceMatrix = (companyDailyReturns.T @ companyDailyReturns) * ((companyDailyReturns.shape[1] - 1) ** - 1) # we are doing 1/n-1 * X.T @ X here

#creating a figure to plot covariance matrix in matplotlib
tickers = companies
plt.figure(0)
cax = plt.matshow(covariaceMatrix)
cbar = plt.colorbar(cax)
# Title and labels
plt.xticks(ticks=np.arange(len(tickers)), labels=tickers, rotation=45)
plt.yticks(ticks=np.arange(len(tickers)), labels=tickers)

# Title and labels
plt.title('Covariance of 7 stocks for previous 20 days of daily returns')
#plt.tight_layout()
#plt.xlabel('Stocks')
#plt.ylabel('Stock Tickers')



'''
In order to generate the corrrelation matrix, we simply must generate R = SCS where C is the computed covariance matrix above, R is the correlation matrix and 
S is a diagonal matrix where the diagonal elements are the reciprocals of the standard deviations of each respective stock. 

Luckily, the diagonal elemenents of the covariance matrix are precisely the variances of each stock. So what we need to do is pull the diagonal elements. take the reciprocal of each and then the square root of each.
Then, we can mulitply using the formula above to get the correlation matrix. We should expect that the correlation matrix is 
'''

reciprocalStdMatrix = np.diag(np.power(np.sqrt(np.diag(covariaceMatrix)),-1)) # generates S
#print(reciprocalStdMatrix)

correlationMatrix = (reciprocalStdMatrix @ covariaceMatrix) @ reciprocalStdMatrix

print("The covariance matrix for last 20 daily returns of our portfolio:\n")
print(covariaceMatrix,end='\n\n')

print("The correlation matrix for last 20 daily returns of our portfolio:\n")
print(correlationMatrix,end='\n\n')

#creating a figure to plot covariance matrix in matplotlib
tickers = companies
plt.figure(1)
cax = plt.matshow(correlationMatrix)
cbar = plt.colorbar(cax)
# Title and labels
plt.xticks(ticks=np.arange(len(tickers)), labels=tickers, rotation=45)
plt.yticks(ticks=np.arange(len(tickers)), labels=tickers)

# Title and labels
plt.title('Correlation of 7 stocks for previous 20 days of daily returns')
#plt.tight_layout()
#plt.xlabel('Stocks')
#plt.ylabel('Stock Tickers')

#plt.show()


#plt.show()


'''
To evaluate protfolio performace I am going to build a small simulation of the portfolio accross the time period of 20 days.

We will assume that we start with $70 distrbuted evenly accross all stocks.

We will assume no taxes, or fees for trading, and we are allowed to buy partial stocks.

We also assume that the portfolio is alive for only the 20 day period with the initial influx of cash on day 1. 
'''

portfolio = np.zeros([21,7]) # we initialize an empty portfolio.
portfolio[0,:] += 100 # we purchace $100 worth of each stock at the end of the 0th day.
totalPortfolioValueDaily = np.zeros([21,1])
totalPortfolioValueDaily[0,0] = sum(portfolio[0,:])
#print(totalPortfolioValueDaily)
#print(portfolio)

for i in range(0,companyDailyReturns.shape[0]): # looping through the days
    portfolio[i+1,:] += (1+companyDailyReturns[i,:]) * portfolio[i,:] #allows us to track the performace of some initial ammount invested in each stock across 20 days. Utilizes broadcasting to compute todaysValue = (1+%return) * yesterdays value
    totalPortfolioValueDaily[i+1, 0] = sum(portfolio[i+1, :])
#print(portfolio)
#print(totalPortfolioValueDaily)

indices = np.arange(len(totalPortfolioValueDaily))
#print(indices)

plt.figure(3)

plt.plot(indices,totalPortfolioValueDaily)
# Title and labels
plt.title('Value of total initial investment over 20 day period')
plt.tight_layout()
plt.xlabel('Day')
plt.ylabel('Value in $')
plt.xticks(indices)



plt.figure(4)

for i in range(len(tickers)):
    plt.plot(indices,portfolio[:,i],label=tickers[i])


# Title and labels
plt.title('Value of $100/stock initial investment over 20 day period')
plt.tight_layout()
plt.xlabel('Day')
plt.ylabel('Value in $')
plt.xticks(indices)
#legend
plt.legend(title='Stock Tickers')


plt.show()


