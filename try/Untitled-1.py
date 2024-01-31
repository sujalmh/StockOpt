tickers=["LICI","COALINDIA","TCS", "HINDZINC", "INFY", "ITC", "HAL", "MPHASIS", "HCLTECH", "EICHERMOT", "DRREDDY", "BAJAJ-AUTO", "HEROMOTOCO", "TECHM", "WIPRO", "ZYDUSLIFE", "OBEROIRLTY", "ONGC", "BAJAJHLDNG" ,"HINDALCO", "INDUSTOWER", "RELIANCE" ,"GAIL", "AUROPHARMA" , "NHPC"] #list of tickers
for count in range(len(tickers)):
    try:
        int(tickers)
        tickers[count] = "^" + tickers[count]
    except:
        tickers[count] = tickers[count] + ".NS" #NSE index'
print(tickers)