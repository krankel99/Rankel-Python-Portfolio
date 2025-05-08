# Portfolio Analyzer

## Project Overview
Portfolio Analyzer helps investors and analysts quickly assess the health and performance of their equity portfolios. By entering or uploading a list of tickers, quantities, and purchase prices, you get real‑time valuations, key metrics (total invested, current value, P/L %, Sharpe ratio), sector and position breakdowns, and beta calculations against the S&P 500. Additionally, you can paste in earnings‑call transcripts or news articles to gauge sentiment using a transformer‑based model, all in one interactive Streamlit app.

## Setup and Run Instructions
- Do `pip install -r requirements.txt`

- Then `streamlit run My_Streamlit_App.py`
- 
- View deployed app: https://rankel-python-portfolio-diro957ngmif7w7zq6c8kr.streamlit.app/
## App Features
###  Homepage
- Upload a CSV or type in your portfolio
###  Portfolio Overview
- **Holdings Table**: Shows cost basis, current price/value, P/L in dollars & %
- **Metrics Panel**: Invested, Current Value, P/L%, Sharpe Ratio, with in-app definitions
- **Portfolio Composition**: Pie chart of position weights
- **Sector Breakdown**: Pie chart of sector allocation
- **Beta Metrics**: Individual stock and portfolio beta (5-year monthly)
###  Stock Charts
- Select any ticker and time range (1 mo, 3 mo, 6 mo, 1 yr) to display the stock's history
###  Portfolio vs S&P 500
- Compare portfolio performance to the S&P 500 over the last 6 months
###  Sentiment Analysis
- **Transformer-Based**: Hugging Face pipeline label & confidence score (0-1)

## References and Resources
- yfinance: [https://pypi.org/project/yfinance/
](url)
- Investopedia: [https://www.investopedia.com/articles/07/sharpe_ratio.asp#:~:text=The%20Sharpe%20ratio%20describes%20how,of%20risk%20of%20the%20investment.](url)
## Visual Examples
![image](https://github.com/user-attachments/assets/df95424d-c15e-4af2-9c6b-a746a895f710)
![image](https://github.com/user-attachments/assets/640d1f01-00c6-4fd5-ab97-c90f51c82df7)
![image](https://github.com/user-attachments/assets/315dcefc-3453-47bb-9059-1fbdbcc29f31)
![image](https://github.com/user-attachments/assets/5ff236ff-c151-4cd8-ac03-d3d2ed04dc9e)
![image](https://github.com/user-attachments/assets/b74dd710-ac0a-4dcd-8ce7-9ee76bd65eef)




