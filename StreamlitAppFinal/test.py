import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline

# Streamlit configuration and CSS
st.set_page_config(page_title="Portfolio Analyzer", layout="wide")
st.markdown(
    """
    <style>
    .streamlit-expanderHeader, .stMarkdown p, .stMarkdown div { font-size:20px !important; }
    .stMarkdown h1 { font-size:36px !important; }
    .stMarkdown h2 { font-size:32px !important; }
    .stMarkdown h3 { font-size:28px !important; }
    .dataframe tbody td { font-size:24px !important; }
    .dataframe thead th { font-size:26px !important; }
    </style>
    """, unsafe_allow_html=True)

# Helper function to fetch and cache prices
def fetch_prices(tickers):
    prices = {}
    for t in tickers:
        try:
            prices[t] = yf.Ticker(t).history(period='1d')['Close'].iloc[0]
        except:
            prices[t] = np.nan
    return prices

# Initialize default portfolio
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

# Page navigation
dpages = ["Homepage", "Portfolio Overview", "Stock Charts", "Portfolio vs S&P 500", "Sentiment Analysis"]
page = st.sidebar.radio("Navigate to", dpages)

# --- Homepage ---
if page == "Homepage":
    st.header("Portfolio Analyzer")
    st.write("**Your all-in-one financial dashboard**")
    st.markdown("Upload a CSV or type your portfolio to get started.")
    st.subheader("Upload or Enter Your Portfolio")
    method = st.selectbox("Input method", ["Upload CSV", "Type Manually"])
    df = None
    if method == "Upload CSV":
        f = st.file_uploader("Choose a CSV file", type="csv")
        if f:
            df = pd.read_csv(f)
    else:
        txt = st.text_area("Enter lines as `Ticker,Quantity,Purchase Price`")
        if txt:
            rows = [row.split(',') for row in txt.splitlines() if row.strip()]
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
            st.success("Portfolio loaded successfully.")
        else:
            st.error("DataFrame must include columns: Ticker, Quantity, Purchase Price.")

# --- Portfolio Overview ---
elif page == "Portfolio Overview":
    st.header("Portfolio Overview")
    df = st.session_state['portfolio_df']
    # Calculate metrics
    total_cost = df['Cost Basis'].sum()
    total_value = df['Current Value'].sum()
    total_pnl = total_value - total_cost
    pnl_pct = (total_pnl / total_cost) * 100 if total_cost else 0
    # Annualized Sharpe ratio
    prices_hist = pd.DataFrame({t: yf.Ticker(t).history(period='1y')['Close'] for t in df['Ticker']})
    returns = prices_hist.pct_change().dropna()
    weights = df['Current Value'] / total_value
    port_returns = (returns * weights.values).sum(axis=1)
    sharpe = (port_returns.mean() / port_returns.std()) * np.sqrt(252)

    # Layout holdings and metrics
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("Holdings")
        st.dataframe(df.style.format({
            'Purchase Price':'${:,.2f}',
            'Current Price':'${:,.2f}',
            'Current Value':'${:,.2f}',
            'Cost Basis':'${:,.2f}',
            'Profit/Loss ($)':'${:,.2f}',
            'Profit/Loss (%)':'{:+.2f}%'
        }))
    with col2:
        st.subheader("Key Metrics")
        st.markdown(f"**Invested:** ${total_cost:,.2f}")
        st.markdown(f"**Current Value:** ${total_value:,.2f}")
        st.markdown(f"**P/L %:** {pnl_pct:+.2f}%")
        st.markdown(f"**Sharpe Ratio:** {sharpe:,.2f}")

    # Composition
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Portfolio Composition")
        fig, ax = plt.subplots(figsize=(3,3))
        ax.pie(df['Current Value'], labels=df['Ticker'], autopct='%1.1f%%'); ax.axis('equal')
        st.pyplot(fig)
    with c2:
        st.subheader("Sector Breakdown")
        sectors = {t: yf.Ticker(t).info.get('sector','Unknown') for t in df['Ticker']}
        sec_vals = {}
        for t, s in sectors.items():
            sec_vals[s] = sec_vals.get(s, 0) + df.loc[df['Ticker']==t, 'Current Value'].iloc[0]
        sec_df = pd.DataFrame.from_dict(sec_vals, orient='index', columns=['Value'])
        sec_df['%'] = sec_df['Value'] / sec_df['Value'].sum()
        fig2, ax2 = plt.subplots(figsize=(3,3))
        ax2.pie(sec_df['Value'], labels=sec_df.index, autopct='%1.1f%%'); ax2.axis('equal')
        st.pyplot(fig2)

    # Beta Metrics
    st.subheader("Beta (5y monthly)")
    market_ret = yf.Ticker('^GSPC').history(period='5y', interval='1mo')['Close'].pct_change().dropna()
    betas = {}
    for t in df['Ticker']:
        ret = yf.Ticker(t).history(period='5y', interval='1mo')['Close'].pct_change().dropna()
        betas[t] = round(ret.cov(market_ret) / market_ret.var(), 2)
    port_beta = round(sum(betas[t] * weights.iloc[i] for i, t in enumerate(df['Ticker'])), 2)
    beta_df = pd.DataFrame.from_dict(betas, orient='index', columns=['Beta'])
    beta_df.loc['Portfolio'] = port_beta
    st.dataframe(beta_df)

# --- Stock Charts ---
elif page == "Stock Charts":
    st.header("Stock Charts")
    df = st.session_state['portfolio_df']
    ticker = st.selectbox("Select Ticker", df['Ticker'])
    period = st.selectbox("Select Time Range", ["1mo", "3mo", "6mo", "1y"])
    hist = yf.Ticker(ticker).history(period=period)
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(hist.index, hist['Close'], linewidth=2)
    ax.set_title(f"{ticker} Price Chart ({period})")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

# --- Portfolio vs S&P 500 ---
elif page == "Portfolio vs S&P 500":
    st.header("Portfolio vs S&P 500")
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

# --- Sentiment Analysis ---
elif page == "Sentiment Analysis":
    st.header("Sentiment Analysis")
    default_text = """
Apple (AAPL) is slated to report fiscal second-quarter results after the market closes Thursday..."""
    st.markdown("Please input text from recent earnings call transcripts or news articles about the stock/company.")
    content = st.text_area("Text to analyze", value=default_text, height=300)

    # Rule-based Sentiment
    if 'custom_lex' not in st.session_state:
        st.session_state['custom_lex'] = {'good':1,'great':2,'excellent':3,'bad':-2,'poor':-3,'terrible':-5}
    st.subheader("Rule-based Analysis")
    st.markdown("**Current Rules:**")
    if st.session_state['custom_lex']:
        for w,s in st.session_state['custom_lex'].items(): st.write(f"- {w}: {s}")
    else:
        st.write("(No custom rules)")
    c1,c2 = st.columns(2)
    new_w = c1.text_input("Add word to lexicon")
    new_s = c2.number_input("Score for word", value=0)
    if st.button("Add Rule") and new_w:
        st.session_state['custom_lex'][new_w.lower()] = new_s
        st.success(f"Added rule: {new_w.lower()} = {new_s}")
    if st.button("Clear All Rules"):
        st.session_state['custom_lex'].clear()
        st.warning("Cleared all custom rules.")
    rule_score = sum(st.session_state['custom_lex'].get(w,0) for w in content.lower().split())
    if rule_score>0: st.success(f"Rule-based Score: {rule_score}")
    elif rule_score<0: st.error(f"Rule-based Score: {rule_score}")
    else: st.info(f"Rule-based Score: {rule_score}")

    # VADER Sentiment
    st.subheader("VADER Analysis")
    sid = SentimentIntensityAnalyzer()
    vs = sid.polarity_scores(content)
    c = vs['compound']
    if c>=0.05: st.success(f"VADER Compound: {c:.2f}")
    elif c<=-0.05: st.error(f"VADER Compound: {c:.2f}")
    else: st.info(f"VADER Compound: {c:.2f}")

    # Transformer Sentiment
    st.subheader("Transformer-Based Analysis")
    try:
        t_sent = pipeline('sentiment-analysis', device=-1)
        tr = t_sent(content, truncation=True)[0]
        lbl, sc = tr['label'], tr['score']
        if lbl == 'POSITIVE': st.success(f"{lbl} ({sc:.2f})")
        else: st.error(f"{lbl} ({sc:.2f})")
    except Exception as e:
        st.warning("Transformer model unavailable. " + str(e))
