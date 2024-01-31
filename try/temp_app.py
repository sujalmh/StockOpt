#Libraries
import matplotlib
import numpy as np

from flask import Flask, render_template, request, redirect, url_for

matplotlib.use("Agg")

app = Flask(__name__)

investor_choice = ""

@app.route("/", methods=["GET", "POST"])
def select_investor():
    global investor_choice
    
    if request.method == "POST":
        investor_choice = request.form.get("investor_choice")
        return redirect(url_for("add_info"))
    
    return render_template("select_investor.html")

@app.route("/add_info", methods=["GET", "POST"])
def add_info():
    global investor_choice, result, plot_image

    
    if investor_choice == "":
        return redirect(url_for("select_investor"))
    
    # Declare result and plot_image locally within the function
    
    plot_image = None
    
    table_data=[]
    all_stocks=[]
    result = dict()
    if request.method == "POST":
        income = float(request.form.get("income"))
        expenditure = float(request.form.get("expenditure"))
        risk_appetite = request.form.get("risk_appetite")
        
        risk_suggestion, result,savings=fetch(income,expenditure,risk_appetite)
        ef,names,latest_prices = optimise(result,savings)

        ef.max_sharpe()
        weights = ef.clean_weights()
        ratio = weights.values()
        number = []
        amount = []
        ratio = list(ratio)
        
        for n in range(len(latest_prices)):
            amount.append(savings*ratio[n])
            number.append(amount[n]/latest_prices[n])
            result = {
        "Income": f"₹ {income:.2f}",
        "Expenditure": f"₹ {expenditure:.2f}",
        "Risk Appetite": risk_appetite,
        "Savings Percent": f"{savings_percent:.2f}%",
        "Suggested Risk": risk_suggestion,
        "Leftover": f"₹ {leftover:.2f}"
    }
            
        

        if request.method == "GET":
            result = None
            plot_image = None
        
        all_stocks = [(str(ticker)[:-3], names[tickers.index(ticker)], number,np.float64(number)*latest_prices[ticker]) for ticker, (ticker,number) in zip(tickers, alloc.items())]
        all_stocks.sort(key=lambda stock: stock[2], reverse=True)
        plot_image = generate_plot([stock[3] for stock in all_stocks], [stock[1] for stock in all_stocks])


    return render_template("add_info.html", result=result, plot_image=plot_image, all_stocks=all_stocks, table_data=table_data)




if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")

