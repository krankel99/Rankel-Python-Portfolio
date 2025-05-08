import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from transformers import pipeline

# Initialize transformer pipeline once (cached)
@st.cache_resource(show_spinner=False)
def load_transformer():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        framework="pt",    # force PyTorch backend to avoid TF/Keras issues
        device=-1            # CPU only
    )

# Load pipeline
try:
    t_sent = load_transformer()
except Exception as e:
    t_sent = None

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
                    continue
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

# Sidebar navigation
pages = ["Homepage", "Portfolio Overview", "Stock Charts", "Portfolio vs S&P 500", "Sentiment Analysis"]
page = st.sidebar.radio("Navigate to", pages)

# ... rest of code remains unchanged, including Beta Metrics calculation

# --- Sentiment Analysis ---
elif page == "Sentiment Analysis":
    st.header("Sentiment Analysis")
    default_text = """
    Paste recent earnings call transcript or news about the company.
    """
    txt = st.text_area("Input text:", value=default_text, height=300)
    content = txt

    st.subheader("Transformer-Based Sentiment Analysis")
    st.markdown(
        """
        We use a transformer‑based model for sentiment analysis via Hugging Face's pipeline.
        The pipeline returns a **label** and a **confidence score** (0–1).
        """, unsafe_allow_html=True)
    if t_sent:
        try:
            tr = t_sent(content, truncation=True)[0]
            if tr['label'] == 'POSITIVE':
                st.success(f"{tr['label']} ({tr['score']:.2f})")
            else:
                st.error(f"{tr['label']} ({tr['score']:.2f})")
        except Exception:
            st.error("Transformer analysis failed. Please try again later.")
    else:
        st.warning("Transformer model unavailable.")
