import os
import numpy as np
import datetime
import pandas as pd
import streamlit as st 
import streamlit.components.v1 as stc 
import indicator_f_Lo2_short
import indicator_forKBar_short
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyoff
import twstock

###### (1) 開始設定 ######
html_temp = """
		<div style="background-color:#3872fb;padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;">股票資料呈現 </h1>
		<h2 style="color:white;text-align:center;">Final-report </h2>
		</div>
		"""
stc.html(html_temp)

# 定義一個函數來取得股票代碼和名稱
def load_stock_data(stock_ids):
    stock_dict = {}
    for stock_id in stock_ids:
        stock = twstock.Stock(stock_id)
        real = twstock.realtime.get(stock_id)
        name = real['info']['name'] if real['success'] else stock_id
        file_name = f"({stock_id})2019_2024.xlsx"
        if os.path.exists(file_name):
            stock_dict[name] = file_name
        else:
            st.warning(f"File not found: {file_name}")
    return stock_dict

# 股票代碼列表
stock_ids = ['0050', '00878', '006208', '1215', '1225', '2303', '2454', '2603', '2609', '2615']  
stock_dict = load_stock_data(stock_ids)

# 生成股票選擇列表
selected_stocks = st.multiselect("選擇股票", list(stock_dict.keys()), default=[list(stock_dict.keys())[0]])

if selected_stocks:
    for selected_stock in selected_stocks:
        if selected_stock in stock_dict:
            try:
                df_original = pd.read_excel(stock_dict[selected_stock])
                df_original = df_original.drop('Unnamed: 0', axis=1)

                df_original['time'] = pd.to_datetime(df_original['time'])
                
                ##### 選擇資料區間
                st.subheader(f"{selected_stock} - 選擇開始與結束的日期, 區間:2019-01-01 至 2024-05-31")
                start_date = st.text_input('選擇開始日期 (日期格式: 2019-01-01)', '2019-01-01')
                end_date = st.text_input('選擇結束日期 (日期格式: 2024-05-31)', '2024-05-31')
                start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
                end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
                df = df_original[(df_original['time'] >= start_date) & (df_original['time'] <= end_date)]

                ###### (2) 轉化為字典 ######:
                KBar_dic = df.to_dict()

                KBar_open_list = list(KBar_dic['open'].values())
                KBar_dic['open'] = np.array(KBar_open_list)

                KBar_dic['product'] = np.repeat(selected_stock, KBar_dic['open'].size)

                KBar_time_list = list(KBar_dic['time'].values())
                KBar_time_list = [i.to_pydatetime() for i in KBar_time_list]  # Timestamp to datetime
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

                ######  (3) 改變 KBar 時間長度 (以下)  ########

                Date = start_date.strftime("%Y-%m-%d")

                st.subheader("設定一根 K 棒的時間長度(天數)")
                cycle_duration = st.number_input('輸入一根 K 棒的時間長度(單位:天, 一日=1天)', value=1, key=f"KBar_duration_{selected_stock}")
                cycle_duration = int(cycle_duration)

                KBar = indicator_forKBar_short.KBar(Date, cycle_duration)  # 設定 cycle_duration 可以改成你想要的 KBar 週期

                for i in range(KBar_dic['time'].size):
                    time = KBar_dic['time'][i]
                    open_price = KBar_dic['open'][i]
                    close_price = KBar_dic['close'][i]
                    low_price = KBar_dic['low'][i]
                    high_price = KBar_dic['high'][i]
                    qty = KBar_dic['volume'][i]
                    amount = KBar_dic['amount'][i]

                    KBar.AddPrice(time, open_price, close_price, low_price, high_price, qty)

                KBar_dic = {}

                # 形成 KBar 字典 (新週期的):
                KBar_dic['time'] = KBar.TAKBar['time']
                KBar_dic['product'] = np.repeat(selected_stock, KBar_dic['time'].size)
                KBar_dic['open'] = KBar.TAKBar['open']
                KBar_dic['high'] = KBar.TAKBar['high']
                KBar_dic['low'] = KBar.TAKBar['low']
                KBar_dic['close'] = KBar.TAKBar['close']
                KBar_dic['volume'] = KBar.TAKBar['volume']

                KBar_df = pd.DataFrame(KBar_dic)

                #####  (i) 移動平均線策略   #####
                ####  設定長短移動平均線的 K棒 長度:
                st.subheader(f"{selected_stock} - 設定計算長移動平均線(MA)的 K 棒數目(整數, 例如 10)")
                LongMAPeriod = st.slider('選擇一個整數', 0, 100, 10, key=f"LongMAPeriod_{selected_stock}")
                st.subheader(f"{selected_stock} - 設定計算短移動平均線(MA)的 K 棒數目(整數, 例如 2)")
                ShortMAPeriod = st.slider('選擇一個整數', 0, 100, 2, key=f"ShortMAPeriod_{selected_stock}")

                #### 計算長短移動平均線
                KBar_df['MA_long'] = KBar_df['close'].rolling(window=LongMAPeriod).mean()
                KBar_df['MA_short'] = KBar_df['close'].rolling(window=ShortMAPeriod).mean()

                #### 尋找最後 NAN值的位置
                last_nan_index_MA = KBar_df['MA_long'][::-1].index[KBar_df['MA_long'][::-1].apply(pd.isna)][0]

                #####  (ii) RSI 策略   #####
                #### 順勢策略
                ### 設定長短 RSI 的 K棒 長度:
                st.subheader(f"{selected_stock} - 設定計算長RSI的 K 棒數目(整數, 例如 10)")
                LongRSIPeriod = st.slider('選擇一個整數', 0, 1000, 10, key=f"LongRSIPeriod_{selected_stock}")
                st.subheader(f"{selected_stock} - 設定計算短RSI的 K 棒數目(整數, 例如 2)")
                ShortRSIPeriod = st.slider('選擇一個整數', 0, 1000, 2, key=f"ShortRSIPeriod_{selected_stock}")

                ### 計算 RSI指標長短線, 以及定義中線
                def calculate_rsi(df, period=14):
                    delta = df['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                    return rsi

                KBar_df['RSI_long'] = calculate_rsi(KBar_df, LongRSIPeriod)
                KBar_df['RSI_short'] = calculate_rsi(KBar_df, ShortRSIPeriod)
                KBar_df['RSI_Middle'] = np.array([50] * len(KBar_dic['time']))

                ### 尋找最後 NAN值的位置
                last_nan_index_RSI = KBar_df['RSI_long'][::-1].index[KBar_df['RSI_long'][::-1].apply(pd.isna)][0]

                ###### (5) 將 Dataframe 欄位名稱轉換  ###### 
                KBar_df.columns = [i[0].upper() + i[1:] for i in KBar_df.columns]

                ###### (6) 增加Bollinger Bands ######
                st.subheader(f"{selected_stock} - 設定計算布林通道(Bollinger Bands)的 K 棒數目(整數, 例如 20)")
                BBPeriod = st.slider('選擇一個整數', 0, 100, 20, key=f"BBPeriod_{selected_stock}")
                KBar_df['MA'] = KBar_df['Close'].rolling(window=BBPeriod).mean()
                KBar_df['BB_upper'] = KBar_df['MA'] + 2 * KBar_df['Close'].rolling(window=BBPeriod).std()
                KBar_df['BB_lower'] = KBar_df['MA'] - 2 * KBar_df['Close'].rolling(window=BBPeriod).std()

                ###### (7) 畫圖 ######
                st.subheader("畫圖")

                ##### K線圖, 移動平均線 MA 和布林通道
                with st.expander(f"{selected_stock} - K線圖, 移動平均線和布林通道"):
                    fig1 = make_subplots(specs=[[{"secondary_y": True}]])

                    #### include candlestick with rangeselector
                    fig1.add_trace(go.Candlestick(x=KBar_df['Time'],
                                    open=KBar_df['Open'], high=KBar_df['High'],
                                    low=KBar_df['Low'], close=KBar_df['Close'], name='K線'),
                                   secondary_y=True)  # secondary_y=True 表示此圖形的y軸scale是在右邊而不是在左邊

                    #### include a go.Bar trace for volumes
                    fig1.add_trace(go.Bar(x=KBar_df['Time'], y=KBar_df['Volume'], name='成交量', marker=dict(color='black')), secondary_y=False)
                    fig1.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_MA+1:], y=KBar_df['MA_long'][last_nan_index_MA+1:], mode='lines', line=dict(color='orange', width=2), name=f'{LongMAPeriod}-根 K棒 移動平均線'), secondary_y=True)
                    fig1.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_MA+1:], y=KBar_df['MA_short'][last_nan_index_MA+1:], mode='lines', line=dict(color='pink', width=2), name=f'{ShortMAPeriod}-根 K棒 移動平均線'), secondary_y=True)
                    fig1.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_MA+1:], y=KBar_df['BB_upper'][last_nan_index_MA+1:], mode='lines', line=dict(color='blue', width=2), name='布林通道上軌'), secondary_y=True)
                    fig1.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_MA+1:], y=KBar_df['BB_lower'][last_nan_index_MA+1:], mode='lines', line=dict(color='blue', width=2), name='布林通道下軌'), secondary_y=True)

                    fig1.layout.yaxis2.showgrid = True
                    st.plotly_chart(fig1, use_container_width=True)

                ##### K線圖, RSI
                with st.expander(f"{selected_stock} - K線圖, 長短 RSI"):
                    fig2 = make_subplots(specs=[[{"secondary_y": True}]])

                    #### include candlestick with rangeselector
                    fig2.add_trace(go.Candlestick(x=KBar_df['Time'],
                                    open=KBar_df['Open'], high=KBar_df['High'],
                                    low=KBar_df['Low'], close=KBar_df['Close'], name='K線'),
                                   secondary_y=True)  # secondary_y=True 表示此圖形的y軸scale是在右邊而不是在左邊

                    fig2.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_RSI+1:], y=KBar_df['RSI_long'][last_nan_index_RSI+1:], mode='lines', line=dict(color='red', width=2), name=f'{LongRSIPeriod}-根 K棒 移動 RSI'), secondary_y=False)
                    fig2.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_RSI+1:], y=KBar_df['RSI_short'][last_nan_index_RSI+1:], mode='lines', line=dict(color='blue', width=2), name=f'{ShortRSIPeriod}-根 K棒 移動 RSI'), secondary_y=False)

                    fig2.layout.yaxis2.showgrid = True
                    st.plotly_chart(fig2, use_container_width=True)

                ##### 增加MACD圖表 #####
                with st.expander(f"{selected_stock} - MACD 圖表"):
                    st.subheader("MACD 計算參數")
                    macd_fast = st.slider('MACD 快線週期', 1, 50, 12, key=f"macd_fast_{selected_stock}")
                    macd_slow = st.slider('MACD 慢線週期', 1, 50, 26, key=f"macd_slow_{selected_stock}")
                    macd_signal = st.slider('MACD 信號線週期', 1, 50, 9, key=f"macd_signal_{selected_stock}")

                    KBar_df['EMA_fast'] = KBar_df['Close'].ewm(span=macd_fast, adjust=False).mean()
                    KBar_df['EMA_slow'] = KBar_df['Close'].ewm(span=macd_slow, adjust=False).mean()
                    KBar_df['MACD'] = KBar_df['EMA_fast'] - KBar_df['EMA_slow']
                    KBar_df['MACD_signal'] = KBar_df['MACD'].ewm(span=macd_signal, adjust=False).mean()
                    KBar_df['MACD_hist'] = KBar_df['MACD'] - KBar_df['MACD_signal']

                    fig3 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
                    fig3.add_trace(go.Scatter(x=KBar_df['Time'], y=KBar_df['MACD'], mode='lines', line=dict(color='blue', width=2), name='MACD'), row=1, col=1)
                    fig3.add_trace(go.Scatter(x=KBar_df['Time'], y=KBar_df['MACD_signal'], mode='lines', line=dict(color='red', width=2), name='MACD 信號線'), row=1, col=1)
                    fig3.add_trace(go.Bar(x=KBar_df['Time'], y=KBar_df['MACD_hist'], name='MACD 柱狀圖', marker_color='green'), row=2, col=1)

                    st.plotly_chart(fig3, use_container_width=True)

                ##### 基本信息展示 #####
                with st.expander(f"{selected_stock} - 股票基本信息"):
                    stock_info = twstock.codes[stock_id]
                    st.write(f"公司名稱: {stock_info.name}")
                    st.write(f"產業類別: {stock_info.industry}")
                    st.write(f"市場: {stock_info.market}")
                    st.write(f"上市日期: {stock_info.start}")

            except FileNotFoundError as e:
                st.error(f"Error: {e}")
else:
    st.error("Please select at least one stock.")
