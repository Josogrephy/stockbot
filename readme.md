# 🤖 Quantamental Stock Research Bot

A modular, open-source stock research dashboard built with Streamlit. This bot blends **Technical Analysis**, **Fundamental Quality**, and **Live LLM Sentiment** (via Gemini 2.5 Flash and Google Search Grounding) to generate actionable entry and exit signals based on a "Contrarian Value" investing strategy.

## ✨ Features
This bot operates on a 3-layer unidirectional pipeline:

1. **The Senses (Data Ingestion):** Fetches real-time market data, price history, and fundamental metrics using `yfinance`.
2. **The Brain (Feature Extraction):** * Calculates technical indicators (EMA 50/200, RSI, ATR).
   * Extracts Quality Scores (ROE, Profit Margins).
   * **Live Web Browsing:** Uses the modern Google GenAI SDK to browse the live internet for recent news and scores market sentiment from `-1.0` (Extreme Panic) to `+1.0` (Euphoria).
3. **The Judgment (Decision Engine):** Calculates a weighted Fair Value and provides hard Actionable Price Levels (Max Entry, Target Exit, Stop Loss) based on a mathematical confluence of all data points.

## 🚀 How to Run Locally

### 1. Prerequisites
You will need Python installed on your machine and a **free Gemini API Key**.
* Get your free API key here: [Google AI Studio](https://aistudio.google.com/)

### 2. Installation
Clone this repository and navigate into the folder:
```bash
git clone [https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPOSITORY_NAME.git)
cd YOUR_REPOSITORY_NAME

pip install -r requirements.txt

streamlit run app.py