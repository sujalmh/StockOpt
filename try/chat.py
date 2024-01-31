completion = "ABCAPITAL*ABCAPITAL LICI*The company has a strong balance sheet, with low debt levels and a high dividend yield.@GICRE*Power Grid Corporation of India Ltd.*The company has a strong portfolio of long-term contracts and is well-positioned to benefit from the government's focus on infrastructure.@INDIANB*Coal India Ltd.*The company benefits from low cost of production and is the largest coal producer in the world.@JSL*Reliance Industries Ltd.*The company has a strong presence in the oil and gas industry, with a diversified portfolio of products and services.@OIL*Steel Authority of India Ltd.*The company has a strong presence in the steel industry and has a well-diversified customer base.@SAIL*Axis Bank Ltd.*The company is one of the largest private sector banks in India and is well-capitalized to take advantage of growth opportunities in the banking sector.@CANBK*Canara Bank Ltd.*The company is a leading public sector bank in India and is well-positioned to benefit from government initiatives to promote financial inclusion."
stocks_complete=completion.split("@")
print(stocks_complete)
desc_list=[]
for stock_complete in stocks_complete:
    abcd=stock_complete.split("*")
    desc_list.append(abcd)
print(desc_list)
