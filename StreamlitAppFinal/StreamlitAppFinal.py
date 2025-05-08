import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import time

# Initialize transformer pipeline
@st.cache_resource(show_spinner=False)
def load_transformer():
    # explicitly name the model to avoid “defaulted to …” warnings
    return pipeline(
        "sentiment-analysis",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        framework="tf",      # force TensorFlow backend
        device=-1            # CPU only
    )

# at the top of your script, after imports:
t_sent = load_transformer()
# Streamlit configuration

st.set_page_config(page_title="Portfolio Analyzer", layout="wide")
st.markdown(
    """
    <style>
    /* Increase size of all markdown paragraph text */
    .streamlit-expanderHeader, .stMarkdown p, .stMarkdown div {
        font-size: 20px !important;
    }
    /* Increase header sizes */
    .stMarkdown h1 { font-size: 36px !important; }
    .stMarkdown h2 { font-size: 32px !important; }
    .stMarkdown h3 { font-size: 28px !important; }
    </style>
    """, unsafe_allow_html=True)
# Helper to fetch current prices and cache
@st.cache_data(ttl=900)
def fetch_prices(tickers):
    prices = {}
    for t in tickers:
        for attempt in range(3):
            try:
                hist = yf.Ticker(t).history(period="1d", timeout=30)
                prices[t] = hist["Close"].iloc[-1]
                break
            except Exception:
                if attempt == 2:
                    prices[t] = np.nan
                else:
                    time.sleep(1)
    return prices

# Setting up default portfolio
if 'portfolio_df' not in st.session_state:
    df0 = pd.DataFrame({
        'Ticker': ['AAPL', 'MSFT', 'GOOG'],
        'Quantity': [10, 5, 10],
        'Purchase Price': [210, 390, 160]
    })
    prices0 = fetch_prices(df0['Ticker'].tolist())
    df0['Current Price'] = df0['Ticker'].map(prices0)
    df0['Current Value'] = df0['Quantity'] * df0['Current Price']
    df0['Cost Basis'] = df0['Quantity'] * df0['Purchase Price']
    df0['Profit/Loss ($)'] = df0['Current Value'] - df0['Cost Basis']
    df0['Profit/Loss (%)'] = df0['Profit/Loss ($)'] / df0['Cost Basis'] * 100
    st.session_state['portfolio_df'] = df0.dropna(subset=['Current Price'])


# Sidebar
pages = [
    "Homepage",
    "Portfolio Overview",
    "Stock Charts",
    "Portfolio vs S&P 500",
    "Sentiment Analysis"
]
page = st.sidebar.radio("Navigate to", pages)


# Homepage
if page == "Homepage":
    st.header("Homepage")
    st.markdown("Welcome to Portfolio Analyzer, your all‑in‑one financial dashboard! " \
    "Here you can easily upload or type in your stock portfolio to see real‑time valuations, performance metrics (like total invested, current value, P/L %, and Sharpe ratio), "
    "and visual breakdowns by position and sector. Dive deeper with interactive price charts for individual holdings or compare your portfolio’s cumulative return against the S&P 500. " \
    "Finally, switch to Sentiment Analysis to gauge market mood—enter recent earnings call transcripts or news articles and get quick transformer‑based sentiment scores. " \
    "Whether you’re tracking your long‑term investments or testing a new trading idea, Portfolio Analyzer gives you the insights you need—all in one simple app.")
    st.subheader('Upload or Enter Your Portfolio')
    method = st.selectbox("Input method", ["Upload CSV", "Type Manually"])
    df = None
    if method == "Upload CSV":
        f = st.file_uploader("CSV file", type="csv")
        if f:
            df = pd.read_csv(f)
    else:
        txt = st.text_area("Lines as: Ticker,Quantity,Purchase Price")
        if txt:
            rows = [ln.split(',') for ln in txt.splitlines() if ln.strip()]
            df = pd.DataFrame(rows, columns=["Ticker", "Quantity", "Purchase Price"])
            df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
            df['Purchase Price'] = pd.to_numeric(df['Purchase Price'], errors='coerce')
    if df is not None:
        if {'Ticker', 'Quantity', 'Purchase Price'}.issubset(df.columns):
            prices = fetch_prices(df['Ticker'].dropna().tolist())
            df['Current Price'] = df['Ticker'].map(prices)
            df['Current Value'] = df['Quantity'] * df['Current Price']
            df['Cost Basis'] = df['Quantity'] * df['Purchase Price']
            df['Profit/Loss ($)'] = df['Current Value'] - df['Cost Basis']
            df['Profit/Loss (%)'] = df['Profit/Loss ($)'] / df['Cost Basis'] * 100
            st.session_state['portfolio_df'] = df.dropna(subset=['Current Price'])
            st.success("Portfolio loaded.")
        else:
            st.error("Columns must be Ticker, Quantity, Purchase Price.")


# Portfolio Overview Page
elif page == "Portfolio Overview":
    st.header("Portfolio Overview")
    df = st.session_state['portfolio_df']
    # Layout: holdings table and metrics
    table_col, metrics_col = st.columns([3,1])
    with table_col:
        st.subheader("Holdings")
        st.dataframe(df.style.format({
            'Purchase Price': '${:,.2f}',
            'Current Price': '${:,.2f}',
            'Current Value': '${:,.2f}',
            'Cost Basis': '${:,.2f}',
            'Profit/Loss ($)': '${:,.2f}',
            'Profit/Loss (%)': '{:+.2f}%'
        }))
    with metrics_col:
        st.subheader("Metrics & Definitions")
        # Definitions
        defs = [
            ("Invested", "Total cost basis of your portfolio."),
            ("Value", "Current market value of your holdings."),
            ("P/L %", "Percentage gain or loss from your investment."),
            ("Sharpe Ratio", "How much return you are getting relative to the risk. 1 is considered acceptable, but you want higher.")
        ]
        # Calculate values
        invested = df['Cost Basis'].sum()
        current_val = df['Current Value'].sum()
        pnl_pct = (current_val - invested) / invested * 100 if invested else 0
        hist = pd.DataFrame({t: yf.Ticker(t).history(period='1y')['Close'] for t in df['Ticker']})
        ret = hist.pct_change().dropna()
        weights = df['Current Value'] / current_val
        port_ret = (ret * weights.values).sum(axis=1)
        sharpe = (port_ret.mean() / port_ret.std()) * np.sqrt(252)
        vals = [invested, current_val, pnl_pct, sharpe]

        # Display definitions and values
        for (label, desc), val in zip(defs, vals):
            st.markdown(f"<p style='font-size:16px; margin:2px'><strong>{label}</strong>: {desc}</p>", unsafe_allow_html=True)
            suffix = "%" if label == "P/L %" else ""
            display_val = f"{val:,.2f}{suffix}"
            st.markdown(f"<p style='font-size:24px; margin-bottom:12px'><strong>{display_val}</strong></p>", unsafe_allow_html=True)

    # Charts
    comp_col, sec_col = st.columns(2)
    with comp_col:
        st.subheader("Portfolio Composition")
        st.markdown("Each slice shows weight by current value.")
        fig, ax = plt.subplots(figsize=(3,3))
        ax.pie(df['Current Value'], labels=df['Ticker'], autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)

    with sec_col:
        st.subheader("Sector Breakdown")
        st.markdown("Diversification across sectors.")
        sectors = {t: yf.Ticker(t).info.get('sector', 'Unknown') for t in df['Ticker']}
        sec_vals = {}
        for t, s in sectors.items():
            sec_vals[s] = sec_vals.get(s, 0) + df.loc[df['Ticker'] == t, 'Current Value'].iloc[0]
        sec_df = pd.DataFrame.from_dict(sec_vals, orient='index', columns=['Value'])
        sec_df['%'] = sec_df['Value'] / sec_df['Value'].sum()
        colors = plt.cm.tab20.colors[:len(sec_df)]
        fig2, ax2 = plt.subplots(figsize=(3,3))
        ax2.pie(sec_df['Value'], labels=sec_df.index, autopct='%1.1f%%', startangle=90, colors=colors)
        ax2.axis('equal')
        st.pyplot(fig2)

    # Beta Metrics
    st.subheader("Beta Metrics (5y monthly)")
    st.markdown("Beta measures sensitivity of a stock to the S&P 500 (1 is the market volatility).")
    market = yf.Ticker('^GSPC').history(period='5y', interval='1mo')['Close']
    mret = market.pct_change().dropna()
    betas = {}
    for i, t in enumerate(df['Ticker']):
        sp = yf.Ticker(t).history(period='5y', interval='1mo')['Close']
        sret = sp.pct_change().dropna()
        betas[t] = round(sret.cov(mret) / mret.var(), 2)

    port_beta = round(sum(betas[t] * weights.iloc[i] for i, t in enumerate(df['Ticker'])), 2)
    beta_df = pd.DataFrame.from_dict(betas, orient='index', columns=['Beta'])
    beta_df.loc['Portfolio'] = port_beta
    st.dataframe(beta_df)

# Stock Charts Page
elif page == "Stock Charts":
    st.header("Stock Charts")
    df = st.session_state['portfolio_df']
    ticker = st.selectbox("Select Ticker", df['Ticker'].tolist())
    period = st.selectbox("Select Time Range", ["1mo","3mo","6mo","1y"])
    hist = yf.Ticker(ticker).history(period=period)
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(hist.index, hist['Close'], linewidth=2)
    ax.set_title(f"{ticker} Price Chart ({period})", fontsize=18)
    ax.set_xlabel("Date", fontsize=14)
    ax.set_ylabel("Close Price ($)", fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)


# Portfolio vs S&P 500 Page
elif page == "Portfolio vs S&P 500":
    st.header("Portfolio vs S&P 500")
    st.markdown("This graph shows your portfolio's returns in comparison to the S&P 500's returns.")
    df = st.session_state['portfolio_df']
    weights = df['Current Value'] / df['Current Value'].sum()
    comb = pd.DataFrame({t: yf.Ticker(t).history(period='6mo')['Close'] for t in df['Ticker']})
    port = (comb.pct_change() * weights.values).sum(axis=1).add(1).cumprod()
    sp = yf.Ticker('^GSPC').history(period='6mo')['Close']
    sp_val = sp.pct_change().add(1).cumprod()
    fig, ax = plt.subplots()
    ax.plot(port, label='Portfolio')
    ax.plot(sp_val, label='S&P 500')
    ax.legend()
    st.pyplot(fig)



# Sentiment Analysis Page
elif page == "Sentiment Analysis":
    st.header("Sentiment Analysis")
    default_text = """
Apple (AAPL) is slated to report fiscal second-quarter results after the market closes Thursday, with investors likely to be watching for clues on how shifting trade policies could impact the iPhone maker.

Bank of America analysts recently reiterated a “buy” rating, but lowered their price target to $240 from $250, citing "uncertainty in the economic outlook."

The analysts said they anticipate Apple could see some near-term upside "driven by some pull forward of demand" driven by tariff worries, but warned that demand could weaken as concerns about the economy weigh on consumers.

Several analysts have also warned the company could be hurt by a trade war with China, where Apple manufactures an estimated 90% of its products. President Donald Trump has said he expects tariffs on China “will come down substantially” but not drop to zero. To help mitigate some risk, Apple is reportedly planning to move the assembly of all iPhones it sells in the U.S. to India by the end of 2026.

Of the 15 analysts covering Apple tracked by Visible Alpha, nine have a “buy” or equivalent rating for the stock, with four “hold” ratings, and two “sell” ratings. Their consensus price target just under $235 would suggest 13% upside from Monday's intraday price of about $208. Shares of Apple have dropped 17% since the start of the year."""
    txt = st.text_area("Please input text from a recent earnings call transcript or other articles, such as news about the stock or company.", value=default_text, height=300)
    content = txt



    # Transformer Analysis
    st.subheader("Transformer-Based Sentiment Analysis")
    st.markdown("""

In this section, we use a transformer‑based model for [sentiment analysis](https://huggingface.co/docs/transformers/v4.50.0/en/task_summary#natural-language-processing) via Hugging Face’s `pipeline`. [See more on sentiment analysis with Hugging Face.](https://huggingface.co/blog/sentiment-analysis-python)

- This approach leverages pre‑trained models that can capture nuanced language patterns, context, and idiomatic expressions.  
- The pipeline returns:
  - a **label** (e.g., **POSITIVE** or **NEGATIVE**), and  
  - a **confidence score** between **0** and **1**, representing the model’s estimated probability that the input belongs to the predicted class.

> **Note:** Transformer‑based methods often demand more memory and compute than simpler rule‑based or lexicon‑driven approaches, so performance may vary depending on your environment and text length.  
""")
    # Use preinitialized pipeline t_sent
tr = t_sent(content, truncation=True)[0]
score = tr["score"]
label = tr["label"]
if label == "POSITIVE":
    st.success(f"{label} ({score:.2f})")
else:
    st.error(f"{label} ({score:.2f})")