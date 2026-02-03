
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import pytz

# --- 設定頁面 ---
st.set_page_config(
   page_title="Rainow 量化戰情室 Pro",
   page_icon="🧠",
   layout="wide",
   initial_sidebar_state="expanded"
)

# --- 核心邏輯區 (Rainow Brain) ---

@st.cache_data(ttl=60) # 60秒緩存，確保盤前盤後即時性
def get_stock_data(ticker):
   stock = yf.Ticker(ticker)
   
   # 1. 抓取日線 (計算指標用，較穩定)
   hist_daily = stock.history(period="3mo")
   
   # 2. 抓取即時價格 (含盤前盤後，prepost=True 是關鍵)
   rt_data = stock.history(period="5d", interval="1m", prepost=True)
   
   info = stock.info
   return hist_daily, rt_data, info

def calculate_technical_indicators(df):
   df = df.copy()
   
   # VWAP (10日)
   df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
   df['PV'] = df['TP'] * df['Volume']
   df['Rolling_VWAP_10D'] = df['PV'].rolling(window=10).sum() / df['Volume'].rolling(window=10).sum()
   
   # MFI (14日)
   typical_price = (df['High'] + df['Low'] + df['Close']) / 3
   money_flow = typical_price * df['Volume']
   
   positive_flow = [0] * len(df)
   negative_flow = [0] * len(df)
   
   for i in range(1, len(df)):
       if typical_price.iloc[i] > typical_price.iloc[i-1]:
           positive_flow[i] = money_flow.iloc[i]
       elif typical_price.iloc[i] < typical_price.iloc[i-1]:
           negative_flow[i] = money_flow.iloc[i]
           
   df['PosMF'] = pd.Series(positive_flow).rolling(window=14).sum()
   df['NegMF'] = pd.Series(negative_flow).rolling(window=14).sum()
   mfi_ratio = df['PosMF'] / df['NegMF']
   df['MFI'] = 100 - (100 / (1 + mfi_ratio))
   
   # K線與背離
   df['MFI_Divergence'] = df['MFI'] < 25 
   
   body = abs(df['Close'] - df['Open'])
   lower_shadow = np.minimum(df['Open'], df['Close']) - df['Low']
   upper_shadow = df['High'] - np.maximum(df['Open'], df['Close'])
   df['Is_Hammer'] = (lower_shadow >= body * 2) & (upper_shadow <= body * 0.5)
   
   df['Is_Engulfing'] = (df['Close'] > df['Open']) & \
                        (df['Open'] < df['Close'].shift(1)) & \
                        (df['Close'] > df['Open'].shift(1)) & \
                        (df['Close'].shift(1) < df['Open'].shift(1))
   
   return df

def rainow_brain(ticker, hist_daily, rt_data, info):
   # 決定當前價格
   if not rt_data.empty:
       current_price = rt_data['Close'].iloc[-1]
       last_trade_time = rt_data.index[-1]
       # 轉換時區顯示
       tz_ny = pytz.timezone('America/New_York')
       last_trade_time_ny = last_trade_time.astimezone(tz_ny)
       price_source = f"即時報價 ({last_trade_time_ny.strftime('%H:%M')} NY)"
   else:
       current_price = hist_daily['Close'].iloc[-1]
       price_source = "日線收盤價"

   # A. 估值
   target_low = info.get('targetLowPrice')
   target_high = info.get('targetHighPrice')
   val_source = "華爾街分析師"
   
   if target_low is None:
       ma50 = hist_daily['Close'].rolling(50).mean().iloc[-1]
       target_low = ma50 * 0.8
       target_high = ma50 * 1.2
       val_source = "50日均線推估"

   # B. 成長性
   eps_growth = info.get('earningsGrowth', None)
   if eps_growth is None:
       eps_growth = info.get('revenueGrowth', 0.05) 
   
   # C. 技術計算 (日線)
   df_daily = calculate_technical_indicators(hist_daily)
   latest_daily = df_daily.iloc[-1]
   vwap = latest_daily['Rolling_VWAP_10D']

   # --- 評分系統 ---
   score = 0
   reasons = []
   
   # 1. 業績否決
   if eps_growth is not None and eps_growth < -0.05:
       return {
           "verdict": "☠️ 絕對迴避 (Avoid)",
           "color": "inverse",
           "score": -99,
           "advice": "業績預期衰退，基本面惡化，屬於價值陷阱。",
           "reasons": [f"前瞻成長率為負 ({eps_growth:.1%})"],
           "data": {"price": current_price, "vwap": vwap, "val_low": target_low, "val_high": target_high, "eps": eps_growth, "val_source": val_source, "price_src": price_source}
       }

   # 2. 估值評分
   if current_price < target_low:
       score += 3
       reasons.append("✅ 價格低於安全邊際 (低估)")
   elif current_price > target_high:
       score -= 3
       reasons.append("❌ 價格高於合理區間 (高估)")
       
   # 3. 機構籌碼
   bias = (current_price - vwap) / vwap * 100
   
   if current_price > vwap:
       score += 1
       if latest_daily['Low'] <= vwap * 1.02 and current_price > vwap:
           score += 2
           reasons.append("🛡️ 機構在成本線護盤 (回踩有撐)")
       else:
           reasons.append(f"📈 股價位於機構成本線上 (+{bias:.1f}%)")
   else:
       score -= 2
       reasons.append(f"⚠️ 跌破機構成本線 ({bias:.1f}%)")
       
   # 4. 技術訊號
   if latest_daily['Is_Hammer'] or latest_daily['Is_Engulfing']:
       score += 2
       reasons.append("🕯️ 日線出現底部反轉訊號")
   if latest_daily['MFI_Divergence']:
       score += 2
       reasons.append("💰 MFI 進入超賣吸籌區")

   # 5. 結論
   if score >= 6:
       verdict = "💎 強力買入 (Strong Buy)"
       color = "green"
       advice = "完美風暴！估值便宜、機構護盤且有買訊。"
   elif 3 <= score <= 5 and current_price > vwap:
       verdict = "🚀 右側追擊 (Trend Buy)"
       color = "blue"
       advice = "趨勢強勢。資金動能強，適合順勢操作。"
   elif 0 <= score <= 2:
       verdict = "👀 觀望/等待 (Wait)"
       color = "gray"
       advice = "訊號不明。建議等待回落 VWAP 或更安全的價格。"
   else:
       verdict = "⚠️ 風險警示 (Warning)"
       color = "red"
       advice = "風險過高。可能買在山頂或接到刀子。"

   return {
       "verdict": verdict,
       "color": color,
       "score": score,
       "advice": advice,
       "reasons": reasons,
       "data": {
           "price": current_price, 
           "vwap": vwap, 
           "val_low": target_low, 
           "val_high": target_high, 
           "eps": eps_growth,
           "val_source": val_source,
           "mfi": latest_daily['MFI'],
           "price_src": price_source
       }
   }

# --- 介面呈現 (UI) ---

st.title("🧠 Rainow 量化戰情室 Pro")
st.caption("支援盤前、盤後、夜盤即時報價")
st.markdown("---")

with st.sidebar:
   st.header("🔍 標的搜尋")
   ticker_input = st.text_input("輸入美股代碼", value="NVDA").upper()
   st.caption("例如: TSLA, AAPL, PLTR")
   if st.button("🚀 啟動分析", type="primary"):
       st.session_state['analyze'] = True

if ticker_input:
   try:
       with st.spinner(f"正在連線即時報價系統分析 {ticker_input} ..."):
           hist_daily, rt_data, info = get_stock_data(ticker_input)
           
           if hist_daily.empty:
               st.error("❌ 找不到數據，請確認代碼。")
           else:
               result = rainow_brain(ticker_input, hist_daily, rt_data, info)
               data = result['data']

               st.header(result['verdict'])
               st.caption(f"報價來源: {data['price_src']}")

               color_map = {'green': st.success, 'blue': st.info, 'red': st.error, 'gray': st.warning, 'inverse': st.error}
               msg_func = color_map.get(result['color'], st.warning)
               if result['color'] == 'inverse':
                   msg_func(f"**操作建議：{result['advice']}**", icon="☠️")
               else:
                   msg_func(f"**操作建議：{result['advice']}**")

               col1, col2, col3, col4 = st.columns(4)
               with col1: st.metric("Rainow 綜合評分", f"{result['score']} 分")
               with col2: 
                   delta_color = "normal" if data['price'] > data['vwap'] else "inverse"
                   st.metric("現價 vs 機構成本", f"${data['price']:.2f}", f"VWAP ${data['vwap']:.2f}", delta_color=delta_color)
               with col3: st.metric("成長預期 (EPS Growth)", f"{data['eps']:.1%}" if data['eps'] else "N/A")
               with col4: st.metric("資金流向 (MFI)", f"{data['mfi']:.1f}", "低於25為超賣")

               c1, c2 = st.columns([1, 1])
               with c1:
                   st.subheader("💡 AI 決策邏輯")
                   if result['reasons']:
                       for r in result['reasons']: st.write(f"- {r}")
                   else: st.write("- 無顯著加分/扣分項目")
               
               with c2:
                   st.subheader("💰 估值狀態")
                   st.write(f"**資料來源：{data['val_source']}**")
                   current, low, high = data['price'], data['val_low'], data['val_high']
                   if current < low: st.progress(0.1, text="低估區")
                   elif current > high: st.progress(0.9, text="高估區")
                   else: st.progress(0.5, text="合理區")
                   st.text(f"安全價: ${low:.2f} | 風險價: ${high:.2f}")

   except Exception as e:
       st.error(f"發生錯誤: {str(e)}。可能為 API 連線問題。")
