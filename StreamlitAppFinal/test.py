import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Attempt to import and initialize transformer pipeline
has_transformer = False
try:
    from transformers import pipeline
    t_sent = pipeline('sentiment-analysis')
    has_transformer = True
except Exception:
    has_transformer = False

# Streamlit configuration
st.set_page_config(page_title="Portfolio Analyzer", layout="wide")
st.markdown(
    """
    <style>
    /* Increase markdown text size */
    .streamlit-expanderHeader, .stMarkdown p, .stMarkdown div { font-size:20px !important; }
    .stMarkdown h1 { font-size:36px !important; }
    .stMarkdown h2 { font-size:32px !important; }
    .stMarkdown h3 { font-size:28px !important; }
    /* Increase table font */
    .dataframe tbody td { font-size:24px !important; }
    .dataframe thead th { font-size:26px !important; }
    </style>
    """, unsafe_allow_html=True)

# Helper: fetch and cache current prices\@st.cache_data
def fetch_prices(tickers):
    prices = {}
    for t in tickers:
        try:
            prices[t] = yf.Ticker(t).history(period='1d')['Close'].iloc[0]
        except Exception:
            prices[t] = np.nan
    return prices

# Default portfolio setup
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
    st.markdown(
        "Welcome to Portfolio Analyzer, your all‑in‑one financial dashboard!"
    )
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

# --- Portfolio Overview ---
elif page == "Portfolio Overview":
    st.header("Portfolio Overview")
    df = st.session_state['portfolio_df']
    # Layout: table & metrics
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
        st.subheader("Metrics & Definitions")
        defs = [
            ("Invested","Total cost basis of your portfolio."),
            ("Value","Current market value of your holdings."),
            ("P/L %","Percentage gain or loss from your investment."),
            ("Sharpe Ratio","Annualized risk-adjusted return (0% risk-free).")
        ]
        invested = df['Cost Basis'].sum()
        current_val = df['Current Value'].sum()
        pnl_pct = (current_val-invested)/invested*100 if invested else 0
        hist = pd.DataFrame({t:yf.Ticker(t).history(period='1y')['Close'] for t in df['Ticker']})
        ret = hist.pct_change().dropna(); w = df['Current Value']/current_val
        port_ret = (ret*w.values).sum(axis=1)
        sharpe = (port_ret.mean()/port_ret.std())*np.sqrt(252)
        vals = [invested, current_val, pnl_pct, sharpe]
        for (lbl,desc),val in zip(defs,vals):
            st.markdown(f"<p style='font-size:16px'><strong>{lbl}</strong>: {desc}</p>", unsafe_allow_html=True)
            suf = "%" if lbl=="P/L %" else ""
            st.markdown(f"<p style='font-size:24px'><strong>{val:,.2f}{suf}</strong></p>", unsafe_allow_html=True)
    # Charts
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Composition")
        fig,ax=plt.subplots(figsize=(3,3))
        ax.pie(df['Current Value'], labels=df['Ticker'], autopct='%1.1f%%', startangle=90);ax.axis('equal')
        st.pyplot(fig)
    with c2:
        st.subheader("Sector Breakdown")
        sectors={t:yf.Ticker(t).info.get('sector','Unknown') for t in df['Ticker']}
        sv={}
        for t,s in sectors.items(): sv[s]=sv.get(s,0)+df.loc[df['Ticker']==t,'Current Value'].iloc[0]
        sdf=pd.DataFrame.from_dict(sv,orient='index',columns=['Value']); sdf['%']=sdf['Value']/sdf['Value'].sum()
        fig2,ax2=plt.subplots(figsize=(3,3))
        ax2.pie(sdf['Value'], labels=sdf.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.tab20.colors)
        ax2.axis('equal');st.pyplot(fig2)
    # Beta
    st.subheader("Beta (5y monthly)")
    mr=yf.Ticker('^GSPC').history(period='5y',interval='1mo')['Close'].pct_change().dropna()
    betas={t:round(yf.Ticker(t).history(period='5y',interval='1mo')['Close'].pct_change().dropna().cov(mr)/mr.var(),2) for t in df['Ticker']}
    port_b=sum(b* w.iloc[i] for i,(t,b) in enumerate(betas.items()))
    bdf=pd.DataFrame.from_dict(betas,orient='index',columns=['Beta']); bdf.loc['Portfolio']=round(port_b,2)
    st.dataframe(bdf)

# --- Stock Charts ---
elif page=="Stock Charts":
    st.header("Stock Charts")
    df=st.session_state['portfolio_df']
    t=st.selectbox("Ticker",df['Ticker']); p=st.selectbox("Range",["1mo","3mo","6mo","1y"])
    h=yf.Ticker(t).history(period=p)
    fig,ax=plt.subplots(figsize=(10,4))
    ax.plot(h.index,h['Close'],linewidth=2)
    ax.set_title(f"{t} Price ({p})");plt.xticks(rotation=45);plt.tight_layout();st.pyplot(fig)

# --- Portfolio vs S&P 500 ---
elif page=="Portfolio vs S&P 500":
    st.header("Portfolio vs S&P 500")
    df=st.session_state['portfolio_df']; w=df['Current Value']/df['Current Value'].sum()
    comb=pd.DataFrame({t:yf.Ticker(t).history(period='6mo')['Close'] for t in df['Ticker']})
    port=(comb.pct_change()*w.values).sum(axis=1).add(1).cumprod()
    sp=yf.Ticker('^GSPC').history(period='6mo')['Close'].pct_change().add(1).cumprod()
    fig,ax=plt.subplots(); ax.plot(port,label='Portfolio'); ax.plot(sp,label='S&P'); ax.legend(); st.pyplot(fig)

# --- Sentiment Analysis ---
elif page=="Sentiment Analysis":
    st.header("Sentiment Analysis")
    default_text="""
Apple (AAPL) is slated to report...
"""
    st.markdown("Please input text from recent earnings calls or news articles about the stock/company.")
    content=st.text_area("Text", value=default_text, height=300)
    # Transformer
    st.subheader("Transformer Analysis")
    if has_transformer:
        tr=t_sent(content, truncation=True)[0]; lab,sc=tr['label'],tr['score']
        st.success(f"{lab} ({sc:.2f})") if lab=='POSITIVE' else st.error(f"{lab} ({sc:.2f})")
    else:
        st.warning("Transformer model unavailable (no backend).")
