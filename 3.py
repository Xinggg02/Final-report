import os
import numpy as np
import datetime
import pandas as pd
import streamlit as st 
import streamlit.components.v1 as stc 
import plotly.graph_objects as go
from plotly.subplots import make_subplots

###### (1) 開始設定 ######
html_temp = """
		<div style="background-color:#3872fb;padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;">金融資料視覺化呈現 (金融看板) </h1>
		<h2 style="color:white;text-align:center;">Financial Dashboard </h2>
		</div>
		"""
stc.html(html_temp)

## 读取Excel文件
df_original = pd.read_excel('(1215)2023_2024.xlsx')
df_original = df_original.drop('Unnamed: 0', axis=1)

##### 選擇資料區間
st.subheader("選擇開始與結束的日期, 區間:2023-01-03 至 2024-05-31")
start_date = st.text_input('選擇開始日期 (日期格式: 2023-01-03)', '2023-01-03')
end_date = st.text_input('選擇結束日期 (日期格式: 2024-05-31)', '2024-05-31')
start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
df = df_original[(df_original['time'] >= start_date) & (df_original['time'] <= end_date)]

###### (2) 轉化為字典 ######
KBar_dic = df.to_dict()

KBar_open_list = list(KBar_dic['open'].values())
KBar_dic['open'] = np.array(KBar_open_list)

KBar_dic['product'] = np.repeat('tsmc', KBar_dic['open'].size)

KBar_time_list = list(KBar_dic['time'].values())
KBar_time_list = [i.to_pydatetime() for i in KBar_time_list]  ## Timestamp to datetime
KBar_dic['time'] = np.array(KBar_time_list)

KBar_low_list = list(KBar_dic['low'].values())
KBar_dic['low'] = np.array(KBar_low_list)

KBar_high_list = list(KBar_dic['high'].values())
KBar_dic['high'] = np.array(KBar_high_list)

KBar_close_list = list(KBar_dic['close'].values())
KBar_dic['close'] = np.array(KBar_close_list)

KBar_volume_list = list(KBar_dic['volume'].values())
KBar_dic['volume'] = np.array(KBar_volume_list)

KBar_amount_list = list(KBar_dic['amount'].values())
KBar_dic['amount'] = np.array(KBar_amount_list)

###### (3) 改變 KBar 時間長度 (以下) ########

Date = start_date.strftime("%Y-%m-%d")

st.subheader("設定一根 K 棒的時間長度(天數)")
cycle_duration = st.number_input('輸入一根 K 棒的時間長度(單位:天)', value=1, key="KBar_duration")
cycle_duration = f'{int(cycle_duration)}D'

###### (4) 生成 K 棒 ######
def resample_kbars(df, cycle_duration):
    df.set_index('time', inplace=True)
    ohlc_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    df_resampled = df.resample(cycle_duration).apply(ohlc_dict).dropna()
    df_resampled.reset_index(inplace=True)
    return df_resampled

KBar_df = resample_kbars(df, cycle_duration)

###### (5) 將 Dataframe 欄位名稱轉換 ######
KBar_df.columns = [i[0].upper() + i[1:] for i in KBar_df.columns]

###### (6) 選擇技術指標 ######
st.subheader("選擇技術指標")
indicators = st.multiselect(
    '選擇一個或多個技術指標',
    ['移動平均線 (MA)', '相對強弱指標 (RSI)', 'MACD', '布林帶 (Bollinger Bands)']
)

##### 設定移動平均線參數 #####
if '移動平均線 (MA)' in indicators:
    st.subheader("設定計算長移動平均線(MA)的 K 棒數目(整數, 例如 10)")
    LongMAPeriod = st.slider('選擇一個整數', 0, 100, 10)
    st.subheader("設定計算短移動平均線(MA)的 K 棒數目(整數, 例如 2)")
    ShortMAPeriod = st.slider('選擇一個整數', 0, 100, 2)
    KBar_df['MA_long'] = KBar_df['Close'].rolling(window=LongMAPeriod).mean()
    KBar_df['MA_short'] = KBar_df['Close'].rolling(window=ShortMAPeriod).mean()
    if KBar_df['MA_long'].isna().any():
        last_nan_index_MA = KBar_df['MA_long'][::-1].index[KBar_df['MA_long'][::-1].apply(pd.isna)][0]
    else:
        last_nan_index_MA = -1  # 如果没有NaN值

##### 設定 RSI 參數 #####
if '相對強弱指標 (RSI)' in indicators:
    st.subheader("設定計算長RSI的 K 棒數目(整數, 例如 10)")
    LongRSIPeriod = st.slider('選擇一個整數', 0, 1000, 10)
    st.subheader("設定計算短RSI的 K 棒數目(整數, 例如 2)")
    ShortRSIPeriod = st.slider('選擇一個整數', 0, 1000, 2)
    
    def calculate_rsi(df, period=14):
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    KBar_df['RSI_long'] = calculate_rsi(KBar_df, LongRSIPeriod)
    KBar_df['RSI_short'] = calculate_rsi(KBar_df, ShortRSIPeriod)
    KBar_df['RSI_Middle'] = np.array([50] * len(KBar_df))
    if KBar_df['RSI_long'].isna().any():
        last_nan_index_RSI = KBar_df['RSI_long'][::-1].index[KBar_df['RSI_long'][::-1].apply(pd.isna)][0]
    else:
        last_nan_index_RSI = -1  # 如果没有NaN值

##### 設定 MACD 參數 #####
if 'MACD' in indicators:
    def calculate_macd(df, short_period=12, long_period=26, signal_period=9):
        short_ema = df['Close'].ewm(span=short_period, adjust=False).mean()
        long_ema = df['Close'].ewm(span=long_period, adjust=False).mean()
        macd = short_ema - long_ema
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        hist = macd - signal
        return macd, signal, hist
    
    KBar_df['MACD'], KBar_df['MACD_Signal'], KBar_df['MACD_Hist'] = calculate_macd(KBar_df)

##### 設定布林帶參數 #####
if '布林帶 (Bollinger Bands)' in indicators:
    st.subheader("設定計算布林帶的 K 棒數目(整數, 例如 20)")
    BollingerPeriod = st.slider('選擇一個整數', 0, 1000, 20)
    KBar_df['Bollinger_Mid'] = KBar_df['Close'].rolling(window=BollingerPeriod).mean()
    KBar_df['Bollinger_Std'] = KBar_df['Close'].rolling(window=BollingerPeriod).std()
    KBar_df['Bollinger_Upper'] = KBar_df['Bollinger_Mid'] + (KBar_df['Bollinger_Std'] * 2)
    KBar_df['Bollinger_Lower'] = KBar_df['Bollinger_Mid'] - (KBar_df['Bollinger_Std'] * 2)

###### (7) 畫圖 ######
st.subheader("畫圖")

##### K線圖, 移動平均線 MA #####
with st.expander("K線圖, 移動平均線"):
    fig1 = make_subplots(specs=[[{"secondary_y": True}]])
    
    #### include candlestick with rangeselector
    fig1.add_trace(go.Candlestick(x=KBar_df['Time'],
                                  open=KBar_df['Open'], high=KBar_df['High'],
                                  low=KBar_df['Low'], close=KBar_df['Close'], name='K線'),
                   secondary_y=True)  ## secondary_y=True 表示此圖形的y軸scale是在右邊而不是在左邊
    
    #### include a go.Bar trace for volumes
    fig1.add_trace(go.Bar(x=KBar_df['Time'], y=KBar_df['Volume'], name='成交量', marker=dict(color='black')),
                   secondary_y=False)  ## secondary_y=False 表示此圖形的y軸scale是在左邊而不是在右邊
    
    if '移動平均線 (MA)' in indicators:
        if last_nan_index_MA != -1:
            fig1.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_MA+1:], y=KBar_df['MA_long'][last_nan_index_MA+1:], mode='lines', line=dict(color='orange', width=2), name=f'{LongMAPeriod}-根 K棒 移動平均線'), 
                          secondary_y=True)
            fig1.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_MA+1:], y=KBar_df['MA_short'][last_nan_index_MA+1:], mode='lines', line=dict(color='pink', width=2), name=f'{ShortMAPeriod}-根 K棒 移動平均線'), 
                          secondary_y=True)
        else:
            fig1.add_trace(go.Scatter(x=KBar_df['Time'], y=KBar_df['MA_long'], mode='lines', line=dict(color='orange', width=2), name=f'{LongMAPeriod}-根 K棒 移動平均線'), 
                          secondary_y=True)
            fig1.add_trace(go.Scatter(x=KBar_df['Time'], y=KBar_df['MA_short'], mode='lines', line=dict(color='pink', width=2), name=f'{ShortMAPeriod}-根 K棒 移動平均線'), 
                          secondary_y=True)
    
    fig1.layout.yaxis2.showgrid=True
    st.plotly_chart(fig1, use_container_width=True)

##### K線圖, RSI #####
if '相對強弱指標 (RSI)' in indicators:
    with st.expander("K線圖, 長短 RSI"):
        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        
        #### include candlestick with rangeselector
        fig2.add_trace(go.Candlestick(x=KBar_df['Time'],
                                      open=KBar_df['Open'], high=KBar_df['High'],
                                      low=KBar_df['Low'], close=KBar_df['Close'], name='K線'),
                       secondary_y=True)  ## secondary_y=True 表示此圖形的y軸scale是在右邊而不是在左邊
        
        if last_nan_index_RSI != -1:
            fig2.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_RSI+1:], y=KBar_df['RSI_long'][last_nan_index_RSI+1:], mode='lines', line=dict(color='red', width=2), name=f'{LongRSIPeriod}-根 K棒 移動 RSI'), 
                          secondary_y=False)
            fig2.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_RSI+1:], y=KBar_df['RSI_short'][last_nan_index_RSI+1:], mode='lines', line=dict(color='blue', width=2), name=f'{ShortRSIPeriod}-根 K棒 移動 RSI'), 
                          secondary_y=False)
        else:
            fig2.add_trace(go.Scatter(x=KBar_df['Time'], y=KBar_df['RSI_long'], mode='lines', line=dict(color='red', width=2), name=f'{LongRSIPeriod}-根 K棒 移動 RSI'), 
                          secondary_y=False)
            fig2.add_trace(go.Scatter(x=KBar_df['Time'], y=KBar_df['RSI_short'], mode='lines', line=dict(color='blue', width=2), name=f'{ShortRSIPeriod}-根 K棒 移動 RSI'), 
                          secondary_y=False)
        
        fig2.layout.yaxis2.showgrid=True
        st.plotly_chart(fig2, use_container_width=True)

##### K線圖, MACD #####
if 'MACD' in indicators:
    with st.expander("K線圖, MACD"):
        fig3 = make_subplots(specs=[[{"secondary_y": True}]])
        
        #### include candlestick with rangeselector
        fig3.add_trace(go.Candlestick(x=KBar_df['Time'],
                                      open=KBar_df['Open'], high=KBar_df['High'],
                                      low=KBar_df['Low'], close=KBar_df['Close'], name='K線'),
                       secondary_y=True)  ## secondary_y=True 表示此圖形的y軸scale是在右邊而不是在左邊
        
        fig3.add_trace(go.Scatter(x=KBar_df['Time'], y=KBar_df['MACD'], mode='lines', line=dict(color='blue', width=2), name='MACD'), 
                      secondary_y=False)
        fig3.add_trace(go.Scatter(x=KBar_df['Time'], y=KBar_df['MACD_Signal'], mode='lines', line=dict(color='red', width=2), name='MACD 信號線'), 
                      secondary_y=False)
        fig3.add_trace(go.Bar(x=KBar_df['Time'], y=KBar_df['MACD_Hist'], name='MACD 柱狀圖'), secondary_y=False)
        
        fig3.layout.yaxis2.showgrid=True
        st.plotly_chart(fig3, use_container_width=True)

##### K線圖, 布林帶 #####
if '布林帶 (Bollinger Bands)' in indicators:
    with st.expander("K線圖, 布林帶"):
        fig4 = make_subplots(specs=[[{"secondary_y": True}]])
        
        #### include candlestick with rangeselector
        fig4.add_trace(go.Candlestick(x=KBar_df['Time'],
                                      open=KBar_df['Open'], high=KBar_df['High'],
                                      low=KBar_df['Low'], close=KBar_df['Close'], name='K線'),
                       secondary_y=True)  ## secondary_y=True 表示此圖形的y軸scale是在右邊而不是在左邊
        
        fig4.add_trace(go.Scatter(x=KBar_df['Time'], y=KBar_df['Bollinger_Mid'], mode='lines', line=dict(color='blue', width=2), name='布林帶中線'), 
                      secondary_y=True)
        fig4.add_trace(go.Scatter(x=KBar_df['Time'], y=KBar_df['Bollinger_Upper'], mode='lines', line=dict(color='red', width=2), name='布林帶上軌'), 
                      secondary_y=True)
        fig4.add_trace(go.Scatter(x=KBar_df['Time'], y=KBar_df['Bollinger_Lower'], mode='lines', line=dict(color='green', width=2), name='布林帶下軌'), 
                      secondary_y=True)
        
        fig4.layout.yaxis2.showgrid=True
        st.plotly_chart(fig4, use_container_width=True)
