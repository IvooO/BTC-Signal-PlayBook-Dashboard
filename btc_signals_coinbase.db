import os
import time
import json
import smtplib
import sqlite3
from email.mime.text import MIMEText
from datetime import datetime, timezone
from typing import List, Tuple, Dict, Any

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from sqlalchemy import create_engine, text

# -----------------------
# CONFIG - edit or use env vars
# -----------------------
# Data source
COINBASE_PRODUCT = os.getenv("COINBASE_PRODUCT", "BTC-USD")
# Targets zijn 1 dag, 1 uur en 15 minuten.
TARGET_GRANULARITIES = [86400, 3600, 900] # 1d, 1h en 15m
CANDLE_LIMIT = int(os.getenv("CANDLE_LIMIT", "500"))

# Database
DB_PATH = os.getenv("BTC_SIGNALS_DB", "./btc_signals_coinbase.db")

# Email (Gmail)
GMAIL_SENDER = os.getenv("GMAIL_SENDER", "ivovanoverstraeten@gmail.com")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "mrhs sefm qykl ytj")
ALERT_EMAIL_TO = os.getenv("ALERT_EMAIL_TO", GMAIL_SENDER)

# Optional push
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
PUSHOVER_API_TOKEN = os.getenv("PUSHOVER_API_TOKEN", "")
PUSHOVER_USER_KEY = os.getenv("PUSHOVER_USER_KEY", "")

# Tuning
MC_GREEN_DOT_MIN_WAVE_STRENGTH = float(os.getenv("MC_GREEN_DOT_MIN_WAVE_STRENGTH", "0.0"))
MC_RED_DOT_MAX_WAVE_STRENGTH = float(os.getenv("MC_RED_DOT_MAX_WAVE_STRENGTH", "0.0"))
RSI_OVERSOLD = int(os.getenv("RSI_OVERSOLD", "30"))
RSI_OVERBOUGHT = int(os.getenv("RSI_OVERBOUGHT", "70"))
BREAKOUT_LOOKBACK = int(os.getenv("BREAKOUT_LOOKBACK", "50"))

# Granularity to Timeframe Display Name
GRANULARITY_MAP = {
    86400: {"name": "1 DAG (1d)", "short": "1D"},
    3600: {"name": "1 UUR (1h)", "short": "1H"},
    900: {"name": "15 MIN (15m)", "short": "15M"},
    300: {"name": "5 MIN", "short": "5M"}
}
GRANULARITY_NAMES = {k: v["name"] for k, v in GRANULARITY_MAP.items()}
GRANULARITY_SHORTS = {k: v["short"] for k, v in GRANULARITY_MAP.items()}


# -----------------------
# Helpers & init
# -----------------------
st.set_page_config(page_title="BTC PlayBook Signals", layout="wide")

# Custom CSS for Gemini-like look (Light theme, focus on color accents and clean typography)
st.markdown("""
<style>
/* Streamlit default theme is often light, we enhance typography and spacing */
.main-header {
    font-size: 2.5em;
    font-weight: 600;
    color: #1a73e8; /* Blue accent */
    margin-bottom: 0.5em;
}
.stMetric > div[data-testid="stMetricLabel"] {
    font-size: 0.9em;
    font-weight: 500;
    color: #5f6368; /* Google gray */
}
.stMetric > div[data-testid="stMetricValue"] {
    font-size: 1.8em;
    font-weight: 700;
}
.signal-box {
    padding: 10px;
    border-radius: 8px;
    margin-bottom: 10px;
    font-size: 0.9em;
}
.signal-bullish { background-color: #e6ffed; border: 1px solid #34a853; color: #34a853; }
.signal-bearish { background-color: #feebe8; border: 1px solid #ea4335; color: #ea4335; }
.signal-warning { background-color: #fffbe6; border: 1px solid #fbbc04; color: #fbbc04; }
.signal-inactive { background-color: #f8f9fa; border: 1px solid #adb5bd; color: #adb5bd; }

/* Dataframe styling for the overview table */
.stDataFrame {
    font-size: 0.9em;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">₿ BTC Signal PlayBook Dashboard</p>', unsafe_allow_html=True)
st.markdown("##### Real-time Multi-Timeframe Analyse (Coinbase Data)") 
st.markdown("---")

# SQLAlchemy engine for sqlite (simple)
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})

def init_db():
    with engine.begin() as conn:
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_utc TEXT,
            symbol TEXT,
            granularity INTEGER,
            price REAL,
            signal_type TEXT,
            indicators_json TEXT,
            message TEXT
        );
        """))
init_db()

# -----------------------
# Coinbase fetcher
# -----------------------
COINBASE_API_BASE = "https://api.exchange.coinbase.com" 

@st.cache_data(ttl=60)
def fetch_coinbase_ohlc(product: str, granularity: int, limit: int=500) -> pd.DataFrame:
    """
    Fetch OHLC candles from Coinbase Exchange public REST API.
    """
    url = f"{COINBASE_API_BASE}/products/{product}/candles"
    params = {"granularity": granularity, "limit": limit}
    headers = {"User-Agent": "Streamlit/Crypto-Trader-App"}
    
    r = requests.get(url, params=params, headers=headers, timeout=10)
    r.raise_for_status()
    data = r.json()
    
    if isinstance(data, dict) and 'message' in data:
        raise requests.exceptions.RequestException(f"Coinbase API Error: {data['message']}")

    # Formaat Coinbase: [ time, low, high, open, close, volume ]
    df = pd.DataFrame(data, columns=["time","low","high","open","close","volume"])
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time").sort_index()
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    return df

# -----------------------
# Indicator & Signal Logic
# -----------------------
SIGNAL_DEFINITIONS = [
    {"type": "MC_GREEN_DOT", "message": "MC-stijl Groene Stip (Bullish omkering)", "color": "green", "category": "Bullish"},
    {"type": "EMA_CROSS_UP", "message": "EMA8 kruist boven EMA21 (Trend up)", "color": "green", "category": "Bullish"},
    {"type": "RSI_RECOVERY", "message": f"RSI14 verlaat oversold ({RSI_OVERSOLD})", "color": "green", "category": "Bullish"},
    {"type": "STOCH_BULL", "message": "Stochastic %K kruist boven %D (Bullish)", "color": "green", "category": "Bullish"},
    {"type": "BREAKOUT", "message": "Uitbraak boven recente weerstand", "color": "green", "category": "Bullish"},

    {"type": "MC_RED_DOT", "message": "MC-stijl Rode Stip (Bearish omkering)", "color": "red", "category": "Bearish"},
    {"type": "EMA_CROSS_DOWN", "message": "EMA8 kruist onder EMA21 (Trend down)", "color": "red", "category": "Bearish"},
    {"type": "RSI_DIP", "message": f"RSI14 verlaat overbought ({RSI_OVERBOUGHT})", "color": "red", "category": "Bearish"},
    {"type": "STOCH_BEAR", "message": "Stochastic %K kruist onder %D (Bearish)", "color": "red", "category": "Bearish"},
    {"type": "BREAKDOWN", "message": "Uitbraak onder recente steun", "color": "red", "category": "Bearish"},

    {"type": "RSI_OVERBOUGHT", "message": f"RSI14 is boven overbought ({RSI_OVERBOUGHT})", "color": "orange", "category": "Neutral/Warning"},
    {"type": "RSI_OVERSOLD", "message": f"RSI14 is onder oversold ({RSI_OVERSOLD})", "color": "orange", "category": "Neutral/Warning"},
]

def compute_mc_style(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema8"] = EMAIndicator(df["close"], window=8).ema_indicator()
    df["ema21"] = EMAIndicator(df["close"], window=21).ema_indicator()
    df["wave_price"] = df["ema8"] - df["ema21"]
    df["rsi14"] = RSIIndicator(df["close"], window=14).rsi()
    df["rsi_ema8"] = EMAIndicator(df["rsi14"].fillna(50), window=8).ema_indicator()
    df["rsi_ema21"]= EMAIndicator(df["rsi14"].fillna(50), window=21).ema_indicator()
    df["mc_wave"] = 0.7 * ((df["ema8"] - df["ema21"]) / (df["close"] + 1e-9)) + 0.3 * ((df["rsi_ema8"] - df["rsi_ema21"])/100.0)
    df["mc_wave_s"] = EMAIndicator(df["mc_wave"].fillna(0), window=5).ema_indicator()
    df["mc_wave_mom"] = df["mc_wave_s"] - df["mc_wave_s"].shift(1)
    df["price_delta"] = df["close"] - df["close"].shift(1)
    df["mf_raw"] = df["volume"] * df["price_delta"]
    df["mf_pos_s"] = df["mf_raw"].where(df["mf_raw"]>0, 0.0).rolling(14, min_periods=1).sum()
    df["mf_neg_s"] = (-df["mf_raw"]).where(df["mf_raw"]<0, 0.0).rolling(14, min_periods=1).sum()
    df["mf_ratio"] = df["mf_pos_s"] / (df["mf_neg_s"] + 1e-9)
    df["mfi_like"] = 100 - (100 / (1 + df["mf_ratio"]))
    df["mc_green_dot"] = ((df["mc_wave_s"].shift(1) <= MC_GREEN_DOT_MIN_WAVE_STRENGTH) & (df["mc_wave_s"] > MC_GREEN_DOT_MIN_WAVE_STRENGTH) & (df["mc_wave_mom"] > 0) & (df["mfi_like"] > 45))
    df["mc_red_dot"] = ((df["mc_wave_s"].shift(1) >= MC_RED_DOT_MAX_WAVE_STRENGTH) & (df["mc_wave_s"] < MC_RED_DOT_MAX_WAVE_STRENGTH) & (df["mc_wave_mom"] < 0) & (df["mfi_like"] < 55))
    stoch = StochasticOscillator(df["high"], df["low"], df["close"], window=14, smooth_window=3)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()
    return df

def detect_signals(df: pd.DataFrame) -> List[Dict[str,str]]:
    if len(df) < BREAKOUT_LOOKBACK + 2: return []
    last, prev = df.iloc[-1], df.iloc[-2]
    signals = []
    
    # Bullish
    if bool(last.get("mc_green_dot", False)): signals.append({"type": "MC_GREEN_DOT", "message": "MC-stijl Groene Stip gedetecteerd", "color": "green"})
    if (prev["ema8"] < prev["ema21"]) and (last["ema8"] > last["ema21"]): signals.append({"type": "EMA_CROSS_UP", "message": "EMA8 kruist boven EMA21 (trend up)", "color": "green"})
    if (prev["rsi14"] < RSI_OVERSOLD) and (last["rsi14"] >= RSI_OVERSOLD): signals.append({"type": "RSI_RECOVERY", "message": f"RSI14 verlaat oversold ({RSI_OVERSOLD})", "color": "green"})
    if (df["stoch_k"].iloc[-2] < df["stoch_d"].iloc[-2]) and (df["stoch_k"].iloc[-1] > df["stoch_d"].iloc[-1]): signals.append({"type": "STOCH_BULL", "message": "Stochastic %K kruist boven %D (bullish)", "color": "green"})
    lookback = min(len(df)-1, BREAKOUT_LOOKBACK)
    recent_high = df["close"].iloc[-(lookback+1):-1].max()
    if last["close"] > recent_high and last["close"] > prev["close"]: signals.append({"type": "BREAKOUT", "message": f"Uitbraak boven recente {lookback}-candle high", "color": "green"})

    # Bearish
    if bool(last.get("mc_red_dot", False)): signals.append({"type": "MC_RED_DOT", "message": "MC-stijl Rode Stip gedetecteerd", "color": "red"})
    if (prev["ema8"] > prev["ema21"]) and (last["ema8"] < last["ema21"]): signals.append({"type": "EMA_CROSS_DOWN", "message": "EMA8 kruist onder EMA21 (trend down)", "color": "red"})
    if (prev["rsi14"] > RSI_OVERBOUGHT) and (last["rsi14"] <= RSI_OVERBOUGHT): signals.append({"type": "RSI_DIP", "message": f"RSI14 verlaat overbought ({RSI_OVERBOUGHT})", "color": "red"})
    if (df["stoch_k"].iloc[-2] > df["stoch_d"].iloc[-2]) and (df["stoch_k"].iloc[-1] < df["stoch_d"].iloc[-1]): signals.append({"type": "STOCH_BEAR", "message": "Stochastic %K kruist onder %D (bearish)", "color": "red"})
    recent_low = df["close"].iloc[-(lookback+1):-1].min()
    if last["close"] < recent_low and last["close"] < prev["close"]: signals.append({"type": "BREAKDOWN", "message": "Uitbraak onder recente steun", "color": "red"})

    # Warning
    if last["rsi14"] > RSI_OVERBOUGHT and not any(s['type'] == 'RSI_DIP' for s in signals): signals.append({"type": "RSI_OVERBOUGHT", "message": f"RSI14 is > {RSI_OVERBOUGHT} (Potentiële Top)", "color": "orange"})
    if last["rsi14"] < RSI_OVERSOLD and not any(s['type'] == 'RSI_RECOVERY' for s in signals): signals.append({"type": "RSI_OVERSOLD", "message": f"RSI14 is < {RSI_OVERSOLD} (Potentiële Bodem)", "color": "orange"})
    
    return signals

def log_signal(*args, **kwargs):
    # Logging blijft hetzelfde
    indicators_json = json.dumps(kwargs['indicators'])
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO signals (ts_utc, symbol, granularity, price, signal_type, indicators_json, message)
            VALUES (:ts_utc, :symbol, :granularity, :price, :signal_type, :indicators_json, :message)
        """), {
            "ts_utc": kwargs['ts_utc'], "symbol": kwargs['symbol'], "granularity": kwargs['granularity'],
            "price": kwargs['price'], "signal_type": kwargs['signal_type'],
            "indicators_json": indicators_json, "message": kwargs['message']
        })

def send_email_via_gmail(sender: str, app_password: str, recipient: str, subject: str, body: str) -> Tuple[bool,str]:
    if not sender or not app_password or not recipient: return False, "email_not_configured"
    try:
        msg = MIMEText(body)
        msg["Subject"], msg["From"], msg["To"] = subject, sender, recipient
        server = smtplib.SMTP("smtp.gmail.com", 587, timeout=20)
        server.ehlo(); server.starttls(); server.login(sender, GMAIL_APP_PASSWORD)
        server.sendmail(sender, [recipient], msg.as_string()); server.quit()
        return True, "sent"
    except Exception as e: return False, str(e)

def send_telegram(text: str) -> Tuple[bool,str]:
    if not (TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID): return False, "telegram_not_configured"
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        r = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=10)
        return (r.status_code == 200), r.text
    except Exception as e: return False, str(e)

def send_pushover(title: str, message: str) -> Tuple[bool,str]:
    if not (PUSHOVER_API_TOKEN and PUSHOVER_USER_KEY): return False, "pushover_not_configured"
    url = "https://api.pushover.net/1/messages.json"
    try:
        r = requests.post(url, data={"token": PUSHOVER_API_TOKEN, "user": PUSHOVER_USER_KEY, "title": title, "message": message}, timeout=10)
        return (r.status_code == 200), r.text
    except Exception as e: return False, str(e)


def get_data_and_signals(granularity: int) -> Tuple[pd.DataFrame, List[Dict[str, str]]]:
    """Haalt data op en detecteert signalen voor een gegeven granulariteit."""
    df = fetch_coinbase_ohlc(COINBASE_PRODUCT, granularity, limit=CANDLE_LIMIT) 
    df = compute_mc_style(df)
    signals = detect_signals(df)
    return df, signals

def create_candlestick_chart(df: pd.DataFrame, granularity: int) -> go.Figure:
    """Maakt een Candlestick-grafiek met EMA's en Signaalpunten."""
    fig = go.Figure()
    
    # Candlestick kleuren aanpassen voor een lichte achtergrond
    bull_color = '#34a853' # Green
    bear_color = '#ea4335' # Red
    
    fig.add_trace(go.Candlestick(x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"], 
                                 increasing_line_color=bull_color, decreasing_line_color=bear_color, 
                                 name="Prijs"))
    
    fig.add_trace(go.Scatter(x=df.index, y=df["ema8"], mode="lines", name="EMA8", line=dict(color='#1a73e8', width=1.5))) # Blue
    fig.add_trace(go.Scatter(x=df.index, y=df["ema21"], mode="lines", name="EMA21", line=dict(color='#fbbc04', width=1.5))) # Yellow/Orange

    df_signals = df.iloc[-50:].copy() 
    
    bull_dots = df_signals[df_signals["mc_green_dot"] == True]
    if not bull_dots.empty:
        fig.add_trace(go.Scatter(x=bull_dots.index, y=bull_dots["low"] * 0.99, mode="markers", 
                                 marker=dict(symbol='circle', size=10, color=bull_color, line=dict(width=1, color='DarkGreen')),
                                 name="Bullish Dot", hovertext="MC Groene Stip"))
    
    bear_dots = df_signals[df_signals["mc_red_dot"] == True]
    if not bear_dots.empty:
        fig.add_trace(go.Scatter(x=bear_dots.index, y=bear_dots["high"] * 1.01, mode="markers", 
                                 marker=dict(symbol='circle', size=10, color=bear_color, line=dict(width=1, color='DarkRed')),
                                 name="Bearish Dot", hovertext="MC Rode Stip"))


    timeframe_name = GRANULARITY_NAMES.get(granularity, f"{granularity}s")
    fig.update_layout(
        title=f"**{timeframe_name} Candlestick Chart**",
        yaxis_title="Prijs (USD)", 
        xaxis_title="Tijd (UTC)",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_rangeslider_visible=False,
        template='plotly_white', # Use light theme
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def display_signal_status(col, df: pd.DataFrame, signals: List[Dict], granularity: int):
    """Toont de status van de signalen in een duidelijke lijst met indicatorwaarden."""
    last_candle = df.iloc[-1]
    
    with col:
        st.markdown(f"#### 🔎 Huidige Indicatorwaarden")

        # Use columns for a concise metrics view
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric(label="RSI (14)", value=f"{last_candle['rsi14']:.2f}")
        m_col2.metric(label="Stoch %K", value=f"{last_candle['stoch_k']:.2f}")
        m_col3.metric(label="MC Wave (5-EMA)", value=f"{last_candle['mc_wave_s']:.3f}")

        st.markdown("---")
        st.markdown("#### ✅ Actieve PlayBook Signalen")
        
        status_container = st.container()
        
        active_found = False
        for definition in SIGNAL_DEFINITIONS:
            sig_type = definition['type']
            is_active = any(s['type'] == sig_type for s in signals)
            
            if is_active:
                active_found = True
                color_class = "signal-" + definition['category'].lower().replace('/', '_') # bullish, bearish, neutral_warning
                icon = "🟢" if definition['color'] == 'green' else "🔴" if definition['color'] == 'red' else "🟡"
                
                status_container.markdown(
                    f'<div class="signal-box {color_class}">**{icon} {definition["message"]}** ({sig_type})</div>', 
                    unsafe_allow_html=True
                )
        
        if not active_found:
            status_container.info("Geen actieve signalen op dit moment. Wacht op een setup.")

def get_signal_status_table(timeframes_data: List[Tuple[Any, int, pd.DataFrame, List[Dict]]]) -> pd.DataFrame:
    """Genereert een gecombineerde DataFrame met de status van alle signalen over alle timeframes."""
    
    # 1. Maak een lijst van alle mogelijke signalen
    signal_defs = {d['type']: d for d in SIGNAL_DEFINITIONS}
    
    # 2. Initialiseer de hoofddata
    table_data = []
    
    # 3. Vul de data
    for sig_type, definition in signal_defs.items():
        row = {'Signaal': definition['type'], 'Boodschap': definition['message']}
        
        for _, granularity, _, signals in timeframes_data:
            timeframe_key = GRANULARITY_SHORTS.get(granularity)
            
            is_active = any(s['type'] == sig_type for s in signals)
            
            if is_active:
                color = definition['color']
                if color == 'green':
                    row[timeframe_key] = "✅ Bullish"
                elif color == 'red':
                    row[timeframe_key] = "❌ Bearish"
                else:
                    row[timeframe_key] = "⚠️ Waarschuwing"
            else:
                row[timeframe_key] = "⚪ Inactief"
        
        table_data.append(row)
    
    # 4. Creëer de DataFrame en order de kolommen
    df_overview = pd.DataFrame(table_data)
    
    # Definieer de gewenste kolomvolgorde
    timeframe_cols = [GRANULARITY_SHORTS.get(g[1]) for g in timeframes_data]
    ordered_cols = ['Signaal', 'Boodschap'] + timeframe_cols
    
    return df_overview[ordered_cols]

# -----------------------
# Streamlit UI & main loop (runs on page refresh)
# -----------------------

try:
    # Haal data en signalen op voor de timeframes (1d, 1h, 15m)
    df_1d, signals_1d = get_data_and_signals(86400)
    df_1h, signals_1h = get_data_and_signals(3600)
    df_15m, signals_15m = get_data_and_signals(900)

    # Lijst met data voor de loops en functies
    timeframes_data = [
        (None, 86400, df_1d, signals_1d), 
        (None, 3600, df_1h, signals_1h),
        (None, 900, df_15m, signals_15m)
    ]
    
    # --- 1. Centrale Header en Metrics ---
    
    last_price = df_1d["close"].iloc[-1]
    
    st.subheader("Current Market Snapshot")
    col_price, col_1d, col_1h, col_15m = st.columns(4)

    # Kolom 1: Huidige Prijs (Gebaseerd op 1D data, maar is de meest recente)
    col_price.metric(
        label=f"HUIDIGE SPOT PRIJS ({COINBASE_PRODUCT})",
        value=f"${last_price:,.2f}",
        delta=f"Vol. {df_1d['volume'].iloc[-1]:,.0f} (1D)",
        delta_color="off"
    )
    
    # Kolom 2: 1 DAG (1D) laatste candle
    last_1d_close = df_1d["close"].iloc[-1]
    col_1d.metric(
        label=f"LAATSTE {GRANULARITY_SHORTS.get(86400)} SLUITING",
        value=f"${last_1d_close:,.2f}",
        delta=df_1d.index[-1].strftime('%Y-%m-%d') + " UTC",
        delta_color="off" # Gebruik delta om de tijd te tonen
    )
    
    # Kolom 3: 1 UUR (1H) laatste candle
    last_1h_close = df_1h["close"].iloc[-1]
    col_1h.metric(
        label=f"LAATSTE {GRANULARITY_SHORTS.get(3600)} SLUITING",
        value=f"${last_1h_close:,.2f}",
        delta=df_1h.index[-1].strftime('%Y-%m-%d %H:%M') + " UTC",
        delta_color="off"
    )
    
    # Kolom 4: 15 MIN (15M) laatste candle
    last_15m_close = df_15m["close"].iloc[-1]
    col_15m.metric(
        label=f"LAATSTE {GRANULARITY_SHORTS.get(900)} SLUITING",
        value=f"${last_15m_close:,.2f}",
        delta=df_15m.index[-1].strftime('%Y-%m-%d %H:%M') + " UTC",
        delta_color="off"
    )
    
    st.markdown("---")

    # --- 2. Multi-Timeframe Tabbladen ---
    st.markdown("## 📊 Signaal- en Grafiekanalyse")

    # Voeg het nieuwe overzichtstabblad toe
    tab_overview, tab_1d, tab_1h, tab_15m = st.tabs(["⭐ SIGNALEN OVERZICHT", "1 DAG", "1 UUR", "15 MINUTEN"])

    # Vul het OVERZICHT tabblad
    with tab_overview:
        st.markdown("#### Gecombineerde Status van Alle PlayBook Signalen")
        df_status = get_signal_status_table(timeframes_data)
        
        # Verbeterde weergave van het overzicht
        st.dataframe(df_status, use_container_width=True, hide_index=True)


    # Update de timeframes_data met de tab-variabelen voor de loop
    timeframes_data_with_tabs = [
        (tab_1d, 86400, df_1d, signals_1d), 
        (tab_1h, 3600, df_1h, signals_1h),
        (tab_15m, 900, df_15m, signals_15m)
    ]
    
    for tab, granularity, df, signals in timeframes_data_with_tabs:
        with tab:
            timeframe_name = GRANULARITY_NAMES.get(granularity, f"{granularity}s")
            
            st.markdown(f"### {timeframe_name} Analyse")
            
            col_chart, col_status = st.columns([3, 2])
            
            with col_chart:
                fig = create_candlestick_chart(df, granularity)
                st.plotly_chart(fig, use_container_width=True)
            
            with col_status:
                display_signal_status(st.container(), df, signals, granularity) 


    # --- 3. Logging en Alerting Sectie ---
    st.markdown("---")
    with st.expander("🔔 Alert- en Logging Status", expanded=False): 
        
        st.markdown("### 🔔 Huidige Alert Status")
        alert_col, log_col = st.columns([1,2]) 

        with log_col:
            st.markdown("##### Logging Details")
            st.caption("_Alleen Groen/Rood alerts op **1 DAG** worden verstuurd_")

        # Gebruik 1 DAG (86400) als de primaire alert-trigger
        all_signals = [(86400, df_1d, signals_1d), (3600, df_1h, signals_1h), (900, df_15m, signals_15m)]
        logged_count = 0

        for granularity, df, signals in all_signals:
            timeframe_name = GRANULARITY_NAMES.get(granularity, f"{granularity}s")
            
            if signals:
                for sig in signals:
                    sig_type = sig['type']
                    sig_color = sig['color']
                    ts_utc = df.index[-1].to_pydatetime().replace(tzinfo=timezone.utc).isoformat()
                    price = float(df["close"].iloc[-1])

                    # Logging logica
                    # Binnen de lus in de sectie 'Alert- en Logging Status':
                    with engine.connect() as conn:
                        # Haal de COUNT(*) op. Dit is de eerste (en enige) kolom, dus index 0.
                        cnt = conn.execute(text("SELECT COUNT(*) AS cnt FROM signals WHERE ts_utc = :ts AND signal_type = :st AND granularity = :g"),
                                           {"ts": ts_utc, "st": sig_type, "g": granularity}).fetchone()[0]
                        # ^ Aangepast van ["cnt"] naar [0]
                    if cnt == 0:
                        # Nieuw signaal -> log en alert
                        indicators = { "price": price, "ema8": float(df["ema8"].iloc[-1]), "ema21": float(df["ema21"].iloc[-1]), "mc_wave_s": float(df["mc_wave_s"].iloc[-1]), "mfi_like": float(df["mfi_like"].iloc[-1]), "rsi14": float(df["rsi14"].iloc[-1]) }
                        log_signal(ts_utc=ts_utc, symbol=COINBASE_PRODUCT, granularity=granularity, price=price, signal_type=sig_type, indicators=indicators, message=sig['message'])
                        logged_count += 1
                        
                        # Alert (alleen 1 DAG, Groen/Rood)
                        if sig_color in ['green', 'red'] and granularity == 86400:
                            subj = f"[{timeframe_name} ALERT] {sig_type} | {COINBASE_PRODUCT} {price:.2f}"
                            body = (f"**{timeframe_name} ALERT: {sig_type}**\n{sig['message']}\n\n{COINBASE_PRODUCT} {price:.2f}\nTime (UTC): {ts_utc}")
                            
                            ok_mail, res_mail = send_email_via_gmail(GMAIL_SENDER, GMAIL_APP_PASSWORD, ALERT_EMAIL_TO, subj, body)
                            ok_tg, res_tg = send_telegram(body)
                            ok_pu, res_pu = send_pushover(subj, sig['message'])
                            
                            with alert_col:
                                st.markdown(f"**{timeframe_name} ALERT:**")
                                st.markdown(f"📧 **E-mail:** {'🟢 OK' if ok_mail else '🔴 Fout'}")
                                st.markdown(f"🤖 **Telegram:** {'🟢 OK' if ok_tg else '🔴 Fout'}")
                                st.markdown(f"🔔 **Pushover:** {'🟢 OK' if ok_pu else '🔴 Fout'}")
                                st.caption("Log details in the table below.")

                        with log_col:
                            if sig_color == 'green': st.success(f"✅ LOGGED: {sig_type} ({timeframe_name}) @ ${price:.2f}")
                            elif sig_color == 'red': st.error(f"❌ LOGGED: {sig_type} ({timeframe_name}) @ ${price:.2f}")
                            else: st.warning(f"⚠️ LOGGED: {sig_type} ({timeframe_name}) @ ${price:.2f}")
            
        if logged_count == 0:
            with log_col:
                st.info("Geen nieuwe signalen om te loggen in deze update.")
    
    st.markdown("---")


except requests.exceptions.RequestException as e:
    error_msg = str(e)
    st.error(f"❌ Fout bij het ophalen van Coinbase-gegevens: API-verbinding mislukt.")
    st.warning("Controleer of de Coinbase Exchange API (`api.exchange.coinbase.com`) bereikbaar is.")
    st.caption(f"Details van de fout: {error_msg}")
except Exception as e:
    st.error(f"⚠️ Er is een onverwachte fout opgetreden: {e}")

# --- 4. Logs en Settings ---
st.markdown("## 📚 Recente Logs & Instellingen")
col_logs, col_settings = st.columns([3, 1])

with col_logs:
    st.subheader("📚 Recente Database Logs")
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("SELECT id, ts_utc, granularity, signal_type, price FROM signals ORDER BY id DESC LIMIT 20")).fetchall()
            if rows:
                df_logs = pd.DataFrame(rows, columns=["id","Tijd (UTC)","Granulariteit (s)","Signaal Type","Prijs"])
                df_logs["Timeframe"] = df_logs["Granulariteit (s)"].apply(lambda x: GRANULARITY_SHORTS.get(x, f"{x}s"))
                df_logs = df_logs[["Tijd (UTC)", "Timeframe", "Signaal Type", "Prijs"]]
                st.dataframe(df_logs, use_container_width=True, hide_index=True)
            else:
                st.info("Nog geen signalen gelogd.")
    except Exception as e:
        st.error(f"Fout bij het laden van de database logs: {e}")

with col_settings:
    st.subheader("⚙️ Systeem Instellingen")
    st.info(f"""
    **Product:** `{COINBASE_PRODUCT}`  
    **DB Pad:** `{DB_PATH}`  
    **RSI Drempels:** OB: **{RSI_OVERBOUGHT}** | OS: **{RSI_OVERSOLD}** **Email Alert:** {'✅ Config' if GMAIL_SENDER and GMAIL_APP_PASSWORD else '❌ Not Config'}  
    **Telegram Alert:** {'✅ Config' if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID else '❌ Not Config'}  
    **Pushover Alert:** {'✅ Config' if PUSHOVER_API_TOKEN and PUSHOVER_USER_KEY else '❌ Not Config'}
    """)
