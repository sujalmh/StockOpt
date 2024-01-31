import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
from bs4 import BeautifulSoup
import requests

def retrieve_data(tickers):

    link = f'https://www.screener.in/screens/1159293/top-25/'

    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Max-Age': '3600',
        'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0'
    }

    req = requests.get(link, headers)
    soup = BeautifulSoup(req.content, 'html.parser')

    table_html = soup.find("tbody")
    i = 0
    
    stock_name={}

    for tr in soup.find_all("tr"):

        if i != 0:
            td_html = tr.find_all('td')
            try:
                a_html = td_html[1].find('a', href=True)
                href = a_html['href']
                ticker = ""
                c = 9
                print(href)
                while (href[c] != "/"):
                    ticker += href[c]
                    c += 1
                tickers.append(ticker)
                stock_name[ticker]=name
                name=a_html.contents
            except:
                continue
        i += 1
    return tickers

def add_NS(tickers):
    for count in range(len(tickers)):
        try:
            int(tickers)
            tickers[count] = "^" + tickers[count]
        except:
            tickers[count] = tickers[count] + ".NS" #NSE index
    return tickers

def moving_average(series, window_size):
    return series.rolling(window=window_size).mean()

income=50000
expenditure=30000
savings = income-expenditure # Modified to get input from the form

savings_percent = (savings / income) * 100

tickers_raw=[]
tickers_raw = retrieve_data(tickers_raw)
tickers = add_NS(tickers_raw)

# Assuming ohlc is a DataFrame with the adjusted close prices
ohlc = yf.download(tickers, period="max")
ohlc = ohlc.dropna(axis=1, how='all')
prices = ohlc["Adj Close"].dropna(how="all")
prices.replace(np.nan, 0, inplace=True)  # Replace NaN values with 0

normalized_prices = prices.div(prices.max(axis=0))

# Calculate moving average with a window size of 10
window_size = 100
smoothed_prices = normalized_prices.apply(lambda x: moving_average(x, window_size))

# Plotting
fig, ax = plt.subplots(figsize=(12, 6))

# Plot smoothed prices
for column in smoothed_prices.columns:
    ax.plot(smoothed_prices.index, smoothed_prices[column], label=column)

ax.set_xlabel('Date')
ax.set_ylabel('Smoothed Normalized Price')
ax.legend()

plt.title(f'Smoothed Normalized Adjusted Close Prices Over Time (Moving Average Window = {window_size})')
plt.show()
