from flask import Flask, render_template, jsonify, request, redirect, url_for
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from pypfopt import risk_models, expected_returns, EfficientFrontier, BlackLittermanModel, DiscreteAllocation, objective_functions, black_litterman, risk_models
import requests
from bs4 import BeautifulSoup
import io
import base64
import pandas as pd
import openai

openai.api_key = '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'



app = Flask(__name__)
stock_data = {}  
stock_list = []

themes = {
    'cyberpunk': {'css_file': 'css/cyberpunk.css'},
    'minimalistic': {'css_file': 'css/minimalistic.css'},
    'artistic': {'css_file': 'css/artistic.css'},
    'space':{'css_file':'css/space.css'},
}

def generate_plot(amount, name):
    sorted_data = sorted(zip(amount, name), key=lambda x: x[0], reverse=True)
    sorted_amount, sorted_name = zip(*sorted_data)

    # Plot for the main pie chart
    plt.figure(figsize=(10, 10))

    labels = [f"{sorted_name[i]}" for i in range(len(sorted_name))]

    def func(pct):
        return f"{pct:.1f}%"

    num_colors = len(sorted_name)
    colors = plt.cm.get_cmap('tab20', num_colors)

    plt.pie(sorted_amount, labels=labels, autopct=lambda pct: func(pct),
            startangle=140, pctdistance=0.85, colors=colors(np.arange(num_colors)))  

    img_buffer_plot = io.BytesIO()
    plt.savefig(img_buffer_plot, transparent=True, bbox_inches='tight', format='png')
    img_base64_plot = base64.b64encode(img_buffer_plot.getvalue()).decode()

    # Clear the figure to avoid saving the same plot again
    plt.clf()

    plt.figure(figsize=(6, 6)) # Adjust the size based on your needs
    ax = plt.gca()
    legend_labels = [f"{sorted_name[i]}: ₹{sorted_amount[i]:,.2f}" for i in range(len(sorted_name))]

    custom_legend = [plt.Line2D([0], [0], marker='o', color='w', label=label,
                                markerfacecolor=colors(i), markersize=10) for i, label in enumerate(legend_labels)]
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()
    plt.legend(handles=custom_legend, loc='center right', bbox_to_anchor=(1.25, 0.5), ncol=1)

    img_buffer_legend = io.BytesIO()
    plt.savefig(img_buffer_legend, transparent=True, format='png', bbox_inches='tight') # Add bbox_inches='tight'
    img_base64_legend = base64.b64encode(img_buffer_legend.getvalue()).decode()

    # Clear the figure to avoid saving the same plot again
    plt.clf()

    return img_base64_plot, img_base64_legend

def retrieve_data(risk_appetite,tickers):
    if risk_appetite=="low":
        link = f'https://www.screener.in/screens/1150285/large_cap/'

    elif risk_appetite=="medium":
        link = f'https://www.screener.in/screens/1151198/mid_cap/'

    else:
        link = f'https://www.screener.in/screens/973/small-cap-high-roce/'

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
                name=a_html.text
                name = name.strip()
                stock_name[ticker]=name
            except:
                continue
        i += 1
    return tickers, stock_name

def add_NS(tickers):
    for count in range(len(tickers)):
        try:
            int(tickers)
            tickers[count] = "^" + tickers[count]
        except:
            tickers[count] = tickers[count] + ".NS" #NSE index
    return tickers

def optimize_risk_return(ef,prices,savings):
    ef.max_sharpe()
    weights = ef.clean_weights()

    ef.portfolio_performance(verbose=True)
    latest_prices = prices.iloc[-2]
    print(latest_prices)

    da = DiscreteAllocation(weights, latest_prices,
                            total_portfolio_value=savings, short_ratio=0.3)
    alloc, leftover = da.greedy_portfolio()
    print(alloc)
    
    return prices,alloc, weights, latest_prices,leftover
def optimize_risk_only(ef,prices,savings):
    ef.min_volatility() 
    weights = ef.clean_weights()
    ef.portfolio_performance(verbose=True) 
    latest_prices = prices.iloc[-2]
    print(latest_prices)
    da = DiscreteAllocation(weights, latest_prices,
                            total_portfolio_value=savings, short_ratio=0.3)
    alloc, leftover = da.greedy_portfolio() 
    return prices,alloc, weights, latest_prices, leftover

def bl_optimize_risk_return(ef,prices,total_portfolio_value):
    ef.max_sharpe()
    weights = ef.clean_weights()
    weight=dict(weights)
    amounts=list(weight.values())
    name=list(weight.keys())
    amount=[a*total_portfolio_value for a in amounts]
    da = DiscreteAllocation(weights, prices.iloc[-2], total_portfolio_value=total_portfolio_value)
    alloc, leftover = da.greedy_portfolio()
    return alloc, leftover, amount, name
def bl_optimize_risk_only(ef,prices,total_portfolio_value):
    ef.min_volatility()
    weights = ef.clean_weights()
    weight=dict(weights)
    amounts=list(weight.values())
    name=list(weight.keys())
    amount=[a*total_portfolio_value for a in amounts]
    da = DiscreteAllocation(weights, prices.iloc[-2], total_portfolio_value=total_portfolio_value)
    alloc, leftover = da.greedy_portfolio()
    return alloc, leftover, amount, name


def returns_plot(ohlc, alloc, stock_name):
    tickers = list(alloc.keys())

    ohlc = ohlc.dropna(axis=1, how='all')
    prices = ohlc["Adj Close"].dropna(how="all")
    prices.replace(np.nan, 0, inplace=True)

    log_returns = np.log(prices[tickers] / prices[tickers].shift(1))
    log_returns.replace([np.inf, -np.inf], np.nan, inplace=True)
    log_returns.dropna(inplace=True)

    mean_returns = log_returns.mean()

    plt.figure(figsize=(10, 6))
    mean_returns.plot(kind='bar', color='#ff964f')
    plt.xlabel("Stocks", fontsize=14)
    plt.xticks(range(len(list(alloc.keys()))), [stock_name[ticker] for ticker, value in alloc.items()], rotation=45, ha='right')
    plt.ylabel("Mean Log Returns", fontsize=14)
    plt.subplots_adjust(top=0.9)
    img_buffer_plot = io.BytesIO()
    plt.savefig(img_buffer_plot, bbox_inches='tight', format='png')
    img_base64_plot = base64.b64encode(img_buffer_plot.getvalue()).decode()

    plt.clf()

    return img_base64_plot
def price_plot(prices, alloc, stock_name):
    tickers = list(alloc.keys())

    normalized_prices = prices[tickers].div(prices[tickers].max(axis=0))

    # Calculate moving average with a window size of 10
    window_size = 100
    def moving_average(series, window_size):
        return series.rolling(window=window_size).mean()

    smoothed_prices = normalized_prices.apply(lambda x: moving_average(x, window_size))

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot smoothed prices
    for column in smoothed_prices.columns:
        stock_name_value = stock_name.get(column, column)  # Use stock name from the dictionary or the ticker if not found
        ax.plot(smoothed_prices.index, smoothed_prices[column], label=stock_name_value)

    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Normalized Price', fontsize=14)
    ax.legend()

    img_buffer_plot = io.BytesIO()
    plt.savefig(img_buffer_plot, bbox_inches='tight', format='png')
    img_base64_plot = base64.b64encode(img_buffer_plot.getvalue()).decode()

    # Clear the figure to avoid saving the same plot again
    plt.clf()

    return img_base64_plot

def bl_returns_plot(ohlc,alloc): 
    tickers = list(alloc.keys())

    ohlc = ohlc.dropna(axis=1, how='all')
    prices = ohlc["Adj Close"].dropna(how="all")
    prices.replace(np.nan, 0, inplace=True)  

    log_returns = np.log(prices[tickers] / prices[tickers].shift(1))

    log_returns.replace([np.inf, -np.inf], np.nan, inplace=True)

    log_returns.dropna(inplace=True)

    mean_returns = log_returns.mean()

    plt.figure(figsize=(10, 6))
    mean_returns.plot(kind='bar', color='#ff964f')
    plt.xlabel("Stocks")
    plt.ylabel("Mean Log Returns")
    img_buffer_plot = io.BytesIO()
    plt.savefig(img_buffer_plot, transparent=True, bbox_inches='tight', format='png')
    img_base64_plot = base64.b64encode(img_buffer_plot.getvalue()).decode()

    plt.clf()
    return img_base64_plot
def bl_price_plot(prices, alloc):
    tickers = list(alloc.keys())

    normalized_prices = prices[tickers].div(prices[tickers].max(axis=0))

    window_size = 100
    def moving_average(series, window_size):
        return series.rolling(window=window_size).mean()

    smoothed_prices = normalized_prices.apply(lambda x: moving_average(x, window_size))

    fig, ax = plt.subplots(figsize=(12, 6))

    for column in smoothed_prices.columns:
        ax.plot(smoothed_prices.index, smoothed_prices[column], label=column)

    ax.set_xlabel('Date')
    ax.set_ylabel('Normalized Price')
    ax.legend()

    img_buffer_plot = io.BytesIO()
    plt.savefig(img_buffer_plot, transparent=True, bbox_inches='tight', format='png')
    img_base64_plot = base64.b64encode(img_buffer_plot.getvalue()).decode()

    plt.clf()

    return img_base64_plot

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_portfolio')
def generate_portfolio():
    return render_template('generate_portfolio.html')

@app.route('/optimize_portfolio')
def optimize_portfolio():
    return render_template('optimize_portfolio.html',stock_data=stock_data)

@app.route('/add_stock', methods=['POST'])
def add_stock():
    
    global stock_data
    stock = request.form.get('stock_ticker')
    view = float(request.form.get('view'))
    confidence = int(request.form.get('confidence'))
    if stock not in stock_data:
        stock_data[stock] = {
            'view': view,
            'confidence': int(confidence)/100
        }
    else:
        stock_data[stock]['view'] = view
        stock_data[stock]['confidence'] = int(confidence)/100


    return redirect(url_for('optimize_portfolio'))

@app.route('/remove_stock/<stock>', methods=['GET'])
def remove_stock(stock):
    global stock_data
    if stock in stock_data:
        del stock_data[stock]
    return redirect(url_for('optimize_portfolio'))

@app.route('/generate_portfolio_result', methods=['POST'])
def generate_portfolio_result():
    global result, plot_image, all_stocks
    table_data=[]
    tickers_raw = []
    stock_list = []

    income=int(request.form.get("income"))
    expenditure=int(request.form.get("expenditure"))
    savings = income-expenditure # Modified to get input from the form
    risk_appetite = request.form.get("riskAppetite")
    method = request.form.get("method")
    savings_percent = (savings / income) * 100
    
    if savings_percent < 25:
        risk_suggestion = "low"
        available_investment = savings * 0.75
        fixed_deposit = savings - available_investment

    elif 25 <= savings_percent <= 35:
        risk_suggestion = "medium"
        available_investment = savings * 0.65
        fixed_deposit = savings - available_investment
    
    else:
        risk_suggestion = "high"
        available_investment = savings * 0.25
        fixed_deposit = savings - available_investment

    table_data = [
        ("Available Investment Amount", f"₹ {available_investment:.2f}"),
        ("Amount for Fixed Deposit", f"₹ {fixed_deposit:.2f}")
    ]

    tickers_raw , stock_name= retrieve_data(risk_appetite,tickers_raw)
    
    tickers = add_NS(tickers_raw)
    stock_name = dict(zip(tickers, stock_name.values()))
    yf.set_tz_cache_location("custom/cache/location")

    data = yf.download(tickers, period="max")
    data=data.dropna(axis=1, how='all') #Data preprocessing
    prices = data["Adj Close"]

    S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()   #Data analysis
    mu = expected_returns.capm_return(prices)

    ef = EfficientFrontier(mu, S) #Portfolio Optimization

    if method == "risk_only":
        prices, alloc, weights, latest_prices, leftover = optimize_risk_only(ef,prices,savings)
    else:
        prices, alloc, weights, latest_prices,leftover = optimize_risk_return(ef,prices,savings)

    weight_stocks_dict  = {ticker: name for ticker, name in stock_name.items() if ticker in list(weights.keys())}
    names = list(weight_stocks_dict.values())
    ratio = list(weights.values())
    number = []
    amount = []

    for n in range(len(latest_prices)):
        amount.append(savings * ratio[n])
        number.append(amount[n] / latest_prices[n])
    all_stocks = [(str(ticker)[:-3], names[tickers.index(ticker)], number,np.float64(number)*latest_prices[ticker]) for ticker, (ticker,number) in zip(tickers, alloc.items())]
    all_stocks.sort(key=lambda stock: stock[2], reverse=True)

    plot_image,legend_image = generate_plot([stock[2] for stock in all_stocks], [stock[1] for stock in all_stocks])

    result = {
        "Initial Investment": f"₹ {savings:.2f}",
        "Optimized Allocation": {names[tickers.index(ticker)]: f"₹ {np.float64(number) * latest_prices[ticker]:,.2f}" for
                                 ticker, (ticker, number) in zip(tickers, alloc.items())},
        "Allocated": f"₹ {savings-leftover:.2f}",
        "Leftover": f"₹ {leftover:.2f}",
        "Expenditure": f"₹ {expenditure:.2f}",
        "Risk Appetite": risk_appetite,
        "Savings Percent": f"{savings_percent:.2f}%",
        "Suggested Risk": risk_suggestion,
    }

    data_2_years=data.loc[data.index >= pd.to_datetime("today") - pd.DateOffset(years=1)]
    returns_img=returns_plot(data_2_years,alloc,stock_name)
    price_img=price_plot(prices, alloc,stock_name)
    alloc_tickers=[s[0] for s in all_stocks]
    tickers=" ".join(alloc_tickers)
    print(tickers)

    prompt = "provided all stocks seperated by spaces, reply with two sentences why each given stock is good, without any introductions, Return the output where ticker, stock name and description seperated by * and each stock is seperated by @."

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt+tickers},
            {"role": "assistant", "content": '\n\n{TICKER}*{STOCK_NAME}*{DESCRIPTION}@{TICKER1}*{STOCK_NAME1}*{DESCRIPTION1}@{TICKER2}*{STOCK_NAME2}*{DESCRIPTION2}'},
        ]
    )
    complete=completion.choices[0].message.content
   
    print(complete)

    stocks_complete=complete.split("\n")
    desc_list=[]
    for stock_complete in stocks_complete:
        abcd=stock_complete.split("*")
        desc_list.append(abcd)
    print(len(desc_list),desc_list)
    for stock in range(len(all_stocks)):
        stock_dict={}
        stock_dict["ticker"]=all_stocks[stock][0]
        stock_dict["price"]="₹"+str(round((all_stocks[stock][3]/all_stocks[stock][2]),2))
        stock_dict["image_url"]=all_stocks[stock][0]+".png"
        stock_dict["stock_name"]=desc_list[stock][1]
        stock_dict["description"]=desc_list[stock][2]
        stock_dict["buy_info"]={'label': 'Number to Buy', 'value': all_stocks[stock][2]}
        stock_dict["amount"]={'label': 'Amount to Buy', 'value': "₹"+str(round(all_stocks[stock][3],2))}
        stock_list.append(stock_dict)

    print(stock_list)
    return render_template("generate_portfolio_result.html", result=result, plot_image=plot_image, legend_image=legend_image, returns_img=returns_img, price_img=price_img, all_stocks=all_stocks, table_data=table_data)

@app.route('/optimize_portfolio_result', methods=['POST'])
def optimize_portfolio_result():
    global total_portfolio_value
    all_stocks=[]
    total_portfolio_value = int(request.form.get('total_portfolio_value'))
    method = method = request.form.get("method")
    tickers_raw=list(stock_data.keys())

    tickers=[]
    for ticker in tickers_raw:
        tickers.append(ticker.upper()+".NS")

    ohlc = yf.download(tickers, period="max")
    print(ohlc.head())
    prices = ohlc["Adj Close"]

    market_prices = yf.download("^NSEI", period="max")["Adj Close"]
        
    mcaps = {}
    for t in tickers:
        stock = yf.Ticker(t)
        mcaps[t] = stock.info["marketCap"]

    S = risk_models.CovarianceShrinkage(prices).ledoit_wolf() 
    delta = black_litterman.market_implied_risk_aversion(market_prices)
    market_prior = black_litterman.market_implied_prior_returns(mcaps, delta, S)

    views=[stock_data[item]['view'] for item in tickers_raw] 
    viewdict=dict(zip(tickers, views))


    bl = BlackLittermanModel(S, pi=market_prior, absolute_views=viewdict)
    confidences=[stock_data[item]['confidence'] for item in tickers_raw] 

    bl = BlackLittermanModel(S, pi=market_prior, absolute_views=viewdict, omega="idzorek", view_confidences=confidences)

    np.diag(bl.omega)
    ret_bl = bl.bl_returns()
    S_bl = bl.bl_cov()

    ef = EfficientFrontier(ret_bl, S_bl)
    ef.add_objective(objective_functions.L2_reg)
    
    if method == "risk_only":
        alloc, leftover, amount, name = bl_optimize_risk_only(ef, prices, total_portfolio_value)
    else:
        alloc, leftover, amount, name = bl_optimize_risk_return(ef, prices, total_portfolio_value)
       
    plot_image,legend_image=generate_plot(amount,[stock for stock in name])    

    all_stocks = [(value1[:-3], value2) for index, (value1, value2) in enumerate(zip(list(alloc.keys()), list(alloc.values())))]
    
    price_img=bl_price_plot(prices, alloc)
    data_2_years=ohlc.loc[ohlc.index >= pd.to_datetime("today") - pd.DateOffset(years=1)]
    returns_img=bl_returns_plot(data_2_years,alloc)
    return render_template('optimize_portfolio_result.html', plot_image=plot_image, all_stocks=all_stocks, legend_image=legend_image, returns_img=returns_img, price_img=price_img,leftover=leftover,total_portfolio_value=total_portfolio_value)

@app.route('/run-again', methods=['GET'])
def run_again():
    global stock_data
    stock_data = {}
    return redirect(url_for('index'))

@app.route('/loading', methods=['POST'])
def loading():
    return render_template ("loading.html")

@app.route('/choose_theme')
def choose_theme():
    return render_template('theme.html')

@app.route('/<theme>')
def theme(theme):
    theme_data = themes.get(theme, {})
    return render_template('card.html', stock_data=stock_list, theme_data=theme_data)

if __name__ == '__main__':
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug=True)