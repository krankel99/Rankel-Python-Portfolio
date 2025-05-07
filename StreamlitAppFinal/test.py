import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from transformers import pipeline

# Streamlit UI configuration and CSS adjustments
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

# Helper to fetch current prices
def fetch_prices(tickers):
    prices = {}
    for t in tickers:
        try:
            prices[t] = yf.Ticker(t).history(period='1d')['Close'].iloc[0]
        except:
            prices[t] = np.nan
    return prices

# Default portfolio initialization
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

# Sidebar navigation
pages = [
    "Homepage",
    "Portfolio Overview",
    "Stock Charts",
    "Portfolio vs S&P 500",
    "Sentiment Analysis"
]
page = st.sidebar.radio("Navigate to", pages)

# --- Homepage ---
if page == "Homepage":
    st.header("Portfolio Analyzer")
    st.write("**Your all-in-one financial dashboard**")
    st.markdown("Upload a CSV or type your portfolio to get started.")
    method = st.selectbox("Input method", ["Upload CSV", "Type Manually"])
    df = None
    if method == "Upload CSV":
        uploaded = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded:
            df = pd.read_csv(uploaded)
    else:
        text = st.text_area("Enter lines as `Ticker,Quantity,Purchase Price`")
        if text:
            rows = [row.split(',') for row in text.splitlines() if row.strip()]
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
    # Compute key metrics
    total_cost = df['Cost Basis'].sum()
    total_value = df['Current Value'].sum()
    pnl_pct = ((total_value - total_cost) / total_cost) * 100 if total_cost else 0
    # Sharpe Ratio
    price_hist = pd.DataFrame({t: yf.Ticker(t).history(period='1y')['Close'] for t in df['Ticker']})
    returns = price_hist.pct_change().dropna()
    weights = df['Current Value'] / total_value
    port_returns = (returns * weights.values).sum(axis=1)
    sharpe = (port_returns.mean() / port_returns.std()) * np.sqrt(252)

    # Layout: holdings table and metrics
    tcol, mcol = st.columns([3,1])
    with tcol:
        st.subheader("Holdings")
        st.dataframe(df.style.format({
            'Purchase Price':'${:,.2f}',
            'Current Price':'${:,.2f}',
            'Current Value':'${:,.2f}',
            'Cost Basis':'${:,.2f}',
            'Profit/Loss ($)':'${:,.2f}',
            'Profit/Loss (%)':'{:+.2f}%'
        }))
    with mcol:
        st.subheader("Key Metrics")
        st.markdown(f"**Invested:** ${total_cost:,.2f}")
        st.markdown(f"**Current Value:** ${total_value:,.2f}")
        st.markdown(f"**P/L %:** {pnl_pct:+.2f}%")
        st.markdown(f"**Sharpe Ratio:** {sharpe:,.2f}")

    # Portfolio composition & sector breakdown
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Composition")
        fig, ax = plt.subplots(figsize=(3,3))
        ax.pie(df['Current Value'], labels=df['Ticker'], autopct='%1.1f%%'); ax.axis('equal')
        st.pyplot(fig)
    with c2:
        st.subheader("Sector Breakdown")
        sectors = {t: yf.Ticker(t).info.get('sector','Unknown') for t in df['Ticker']}
        sec_vals = {s: 0 for s in set(sectors.values())}
        for t,s in sectors.items(): sec_vals[s] += df.loc[df['Ticker']==t, 'Current Value'].iloc[0]
        sec_df = pd.DataFrame.from_dict(sec_vals, orient='index', columns=['Value'])
        sec_df['%'] = sec_df['Value'] / sec_df['Value'].sum()
        fig2, ax2 = plt.subplots(figsize=(3,3))
        ax2.pie(sec_df['Value'], labels=sec_df.index, autopct='%1.1f%%'); ax2.axis('equal')
        st.pyplot(fig2)

# --- Stock Charts ---
elif page == "Stock Charts":
    st.header("Stock Charts")
    df = st.session_state['portfolio_df']
    ticker = st.selectbox("Select Ticker", df['Ticker'])
    period = st.selectbox("Select Time Range", ["1mo","3mo","6mo","1y"])
    hist = yf.Ticker(ticker).history(period=period)
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(hist.index, hist['Close'], linewidth=2)
    ax.set_title(f"{ticker} Price Chart ({period})")
    plt.xticks(rotation=45); plt.tight_layout()
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
    content = st.text_area("Text to analyze", value=default_text, height=300)

    # Transformer-based sentiment only
    st.subheader("Transformer-Based Analysis")
    try:
        t_pipe = pipeline(
            'sentiment-analysis',
            model='distilbert-base-uncased-finetuned-sst-2-english',
            tokenizer='distilbert-base-uncased-finetuned-sst-2-english',
            device=-1
        )
        result = t_pipe(content, truncation=True)[0]
        label, score = result['label'], result['score']
        if label == 'POSITIVE':
            st.success(f"{label} ({score:.2f})")
        else:
            st.error(f"{label} ({score:.2f})")
    except Exception as e:
        st.warning("Transformer analysis unavailable. " + str(e))
