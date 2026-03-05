import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json

# --- NEW MODERN SDK IMPORTS ---
from google import genai
from google.genai import types

# ==========================================
# 1. CONFIGURATION & PAGE SETUP
# ==========================================
st.set_page_config(page_title="Quantamental Stock Bot", layout="wide")

st.markdown("""
    <style>
    .metric-card {background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin: 10px 0;}
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. THE SENSES (Data Ingestion Layer)
# ==========================================
class DataIngestion:
    
    @staticmethod
    def get_market_data(ticker, period="2y"):
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            return df, stock.info
        except Exception as e:
            st.error(f"Error fetching market data: {e}")
            return None, None

# ==========================================
# 3. THE BRAIN (Feature Extraction Layer)
# ==========================================
class FeatureEngine:
    
    @staticmethod
    def calculate_technicals(df):
        df = df.copy()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        df['ATR'] = ranges.max(axis=1).rolling(14).mean()
        
        return df

    @staticmethod
    def extract_fundamentals(info):
        return {
            'pe_ratio': info.get('trailingPE', 0),
            'forward_pe': info.get('forwardPE', 0),
            'peg_ratio': info.get('pegRatio', 0),
            'roe': info.get('returnOnEquity', 0),
            'profit_margin': info.get('profitMargins', 0)
        }

    @staticmethod
    def extract_sentiment_llm(ticker, api_key):
        if not api_key:
            return 0.0, "No Gemini API Key provided. Skipping NLP."

        try:
            # Modern Client Initialization
            client = genai.Client(api_key=api_key)
            
            prompt = f"""
            You are a Wall Street quantitative analyst. Use your Google Search tool to find the most recent news, articles, and overall market sentiment for the stock ticker {ticker} from the past 24 to 48 hours.
            
            Determine the overall financial sentiment. Return ONLY a raw JSON object with no markdown formatting or backticks. It must have exactly two keys:
            "score": A float between -1.0 (extremely negative/bearish) and 1.0 (extremely positive/bullish).
            "reason": A single concise sentence explaining why you gave this score based on the specific news you found on the web.
            """
            
            # The modern way to call models with configuration and tools
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())]
                )
            )
            
            result_text = response.text.strip().replace('```json', '').replace('```', '').strip()
            result_dict = json.loads(result_text)
            
            score = float(result_dict.get('score', 0.0))
            reason = result_dict.get('reason', 'Analysis complete.')
            
            return score, reason
            
        except Exception as e:
            return 0.0, f"LLM API Error: {e}"

# ==========================================
# 4. THE JUDGMENT (Valuation & Decision Layer)
# ==========================================
class DecisionEngine:
    
    @staticmethod
    def calculate_fair_value(info, current_price):
        target_price = info.get('targetMeanPrice', current_price)
        peg = info.get('pegRatio', 1)
        if peg is None: peg = 1
        
        relative_value_implied = current_price * (1 + (1 - peg)) if 0 < peg < 2 else current_price
        fair_value = (0.7 * target_price) + (0.3 * relative_value_implied)
        return fair_value

    @staticmethod
    def evaluate(df, metrics, fair_value, current_price, sentiment_score):
        last_row = df.iloc[-1]
        
        is_high_quality = metrics['roe'] > 0.15 and metrics['profit_margin'] > 0.10
        margin_of_safety = 0.15
        buy_price_limit = fair_value * (1 - margin_of_safety)
        is_undervalued = current_price < buy_price_limit
        
        is_uptrend = last_row['Close'] > last_row['EMA_50']
        is_oversold = last_row['RSI'] < 35
        is_panic = sentiment_score < -0.6
        
        if is_high_quality and is_undervalued and is_panic and is_oversold:
            return "STRONG BUY (CONTRARIAN)", "High Quality + Undervalued + Peak Market Fear (Sentiment Crash)", is_high_quality, is_undervalued
        elif is_high_quality and is_undervalued and (is_uptrend or is_oversold):
            return "BUY", "High Quality + Undervalued + Technical Setup", is_high_quality, is_undervalued
        elif current_price >= fair_value:
            return "SELL / TRIM", "Price reached Fair Value target.", is_high_quality, is_undervalued
        elif last_row['Close'] < (last_row['Close'] - (2 * last_row['ATR'])):
             return "STOP LOSS", "Volatility Break (2x ATR)", is_high_quality, is_undervalued
        else:
            return "HOLD", "Currently within expected range. No actionable triggers.", is_high_quality, is_undervalued

# ==========================================
# 5. UI & EXECUTION
# ==========================================
st.title("🤖 Quantamental Research Architect")
st.markdown("A modular bot blending Technicals (The Brain), Fundamentals (The Senses), and LLM Live Web Sentiment.")

with st.sidebar:
    st.header("Control Panel")
    ticker_input = st.text_input("Enter Ticker Symbol", value="AAPL").upper()
    
    st.markdown("---")
    st.header("LLM Settings")
    gemini_key = st.text_input("Gemini API Key", type="password", help="Get a free key from Google AI Studio")
    if not gemini_key:
        st.warning("Enter your Gemini API key to enable live web search sentiment analysis.")

if ticker_input:
    with st.spinner('Ingesting Market Data...'):
        market_data, info = DataIngestion.get_market_data(ticker_input)
    
    if market_data is not None and not market_data.empty:
        current_price = market_data['Close'].iloc[-1]
        
        with st.spinner('Calculating Features & Gemini Browsing the Web...'):
            df_processed = FeatureEngine.calculate_technicals(market_data)
            fund_metrics = FeatureEngine.extract_fundamentals(info)
            
            # The LLM handles its own news retrieval using the new SDK
            sentiment_score, sentiment_reason = FeatureEngine.extract_sentiment_llm(ticker_input, gemini_key)
            
            fair_value = DecisionEngine.calculate_fair_value(info, current_price)
            signal, reason, quality_pass, value_pass = DecisionEngine.evaluate(
                df_processed, fund_metrics, fair_value, current_price, sentiment_score
            )
        
        # --- TOP METRICS ---
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current Price", f"${current_price:.2f}")
        c2.metric("Fair Value (Est)", f"${fair_value:.2f}", delta=f"{current_price - fair_value:.2f}")
        c3.metric("RSI (14)", f"{df_processed['RSI'].iloc[-1]:.2f}")
        
        roe_val = fund_metrics['roe']
        roe_display = f"{roe_val*100:.2f}%" if roe_val is not None else "N/A"
        c4.metric("Quality Score (ROE)", roe_display)
        
        # --- ACTIONABLE PRICE LEVELS ---
        st.markdown("### 🎯 Actionable Price Levels")
        max_entry_price = fair_value * 0.85 
        target_exit_price = fair_value      
        current_atr = df_processed['ATR'].iloc[-1]
        stop_loss_price = current_price - (2 * current_atr) 
        
        lvl1, lvl2, lvl3 = st.columns(3)
        lvl1.metric("🟢 Max Entry Price", f"${max_entry_price:.2f}")
        lvl2.metric("🎯 Target Exit", f"${target_exit_price:.2f}")
        lvl3.metric("🛑 Stop Loss", f"${stop_loss_price:.2f}")

        # --- LLM SENTIMENT ---
        st.markdown("### 🧠 LLM Live Web Sentiment")
        sentiment_color = "green" if sentiment_score > 0.1 else ("red" if sentiment_score < -0.1 else "gray")
        st.markdown(f"**Quant Score:** :{sentiment_color}[{sentiment_score:.2f}] (Scale: -1 to +1)")
        st.info(f"**AI Reasoning:** {sentiment_reason}")
        st.caption("⚡ Powered by Gemini Search Grounding (Live Web Browsing)")

        # --- SIGNAL BANNER ---
        st.markdown("---")
        if "BUY" in signal:
            st.success(f"### 🚀 SIGNAL: {signal}")
        elif "SELL" in signal or "STOP" in signal:
            st.error(f"### 📉 SIGNAL: {signal}")
        else:
            st.warning(f"### ⏸ SIGNAL: {signal}")
        st.write(f"**Logic:** {reason}")
        
        # --- DEEP DIVE & CHART ---
        st.subheader("Deep Dive Analysis")
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df_processed.index,
                            open=df_processed['Open'], high=df_processed['High'],
                            low=df_processed['Low'], close=df_processed['Close'], name='Price'))
            fig.add_trace(go.Scatter(x=df_processed.index, y=df_processed['EMA_50'], line=dict(color='orange', width=1), name='EMA 50'))
            
            fig.add_hline(y=max_entry_price, line_dash="dash", line_color="green", annotation_text="Max Entry")
            fig.add_hline(y=target_exit_price, line_dash="dash", line_color="blue", annotation_text="Target Exit")
            fig.add_hline(y=stop_loss_price, line_dash="dash", line_color="red", annotation_text="Stop Loss")
            
            fig.update_layout(title=f"{ticker_input} Price Action & Levels", xaxis_title="Date", yaxis_title="Price", height=500)
            st.plotly_chart(fig, use_container_width=True)
            
        with col_right:
            st.markdown("### The Brain (Features)")
            st.markdown(f"- **Trend:** {'Bullish' if df_processed['Close'].iloc[-1] > df_processed['EMA_50'].iloc[-1] else 'Bearish'}")
            st.markdown(f"- **Momentum (RSI):** {df_processed['RSI'].iloc[-1]:.2f}")
            st.markdown(f"- **Volatility (ATR):** {current_atr:.2f}")
            
            st.markdown("### The Judgment (Filters)")
            st.checkbox("High Quality Business (>15% ROE)", value=bool(quality_pass), disabled=True)
            st.checkbox("Undervalued (Price < Fair Value)", value=bool(value_pass), disabled=True)
            st.checkbox("Technical/Sentiment Setup", value=("BUY" in signal), disabled=True)
            
    else:
        st.error("Could not fetch data. Please check the ticker symbol.")