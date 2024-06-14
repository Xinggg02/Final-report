import os
import numpy as np
import datetime
import pandas as pd
import streamlit as st
import indicator_f_Lo2_short
import indicator_forKBar_short
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import twstock

###### (1) 開始設定 ######
html_temp = """
<div style="background-color:#4CAF50;padding:15px;border-radius:15px">
    <h1 style="color:#FFFFFF;text-align:center;font-size:36px;">金融大數據期末APP-股票資料呈現</h1>
    <h2 style="color:#FFFFFF;text-align:center;font-size:28px;">Final-report</h2>
</div>
"""
st.markdown(html_temp, unsafe_allow_html=True)

# 定義一個函數來取得股票代碼和名稱
@st.cache_data
def load_stock_data(stock_ids):
    stock_dict = {}
    for stock_id in stock_ids:
        stock = twstock.Stock(stock_id)
        real = twstock.realtime.get(stock_id)
        name = real['info']['name'] if real['success'] else stock_id
        file_name = f"({stock_id})2019_2024.xlsx"
        if os.path.exists(file_name):
            stock_dict[name] = (file_name, stock_id)
        else:
            st.warning(f"找不到文件: {file_name}")
    return stock_dict

# 股票代碼列表
stock_ids = ['0050', '00878', '006208', '1215', '1225', '2303', '2454', '2603', '2609', '2615', '1216', '1210', '1201', '1303', '1301', '1102', '1101', '3443', '3055', '2451', '2891', '2890', '2881', '2880', '2882']
stock_dict = load_stock_data(stock_ids)

# 生成股票選擇列表
selected_stock = st.selectbox("選擇股票", list(stock_dict.keys()))

@st.cache_data
def load_excel_data(file_path):
    df = pd.read_excel(file_path)
    df = df.drop('Unnamed: 0', axis=1)
    df['time'] = pd.to_datetime(df['time'])
    return df

if selected_stock:
    if selected_stock in stock_dict:
        try:
            file_path, stock_id = stock_dict[selected_stock]
            df_original = load_excel_data(file_path)
            
            ##### 選擇資料區間 #####
            st.subheader(f"{selected_stock} - 選擇開始與結束的日期, 區間:2019-01-01 至 2024-05-31")
            start_date = st.date_input('選擇開始日期', datetime.date(2019, 1, 1), min_value=datetime.date(2019, 1, 1), max_value=datetime.date(2024, 5, 31), key='start_date')
            end_date = st.date_input('選擇結束日期', datetime.date(2024, 5, 31), min_value=datetime.date(2019, 1, 1), max_value=datetime.date(2024, 5, 31), key='end_date')
            start_date = datetime.datetime.combine(start_date, datetime.time.min)
            end_date = datetime.datetime.combine(end_date, datetime.time.min)
            df = df_original[(df_original['time'] >= start_date) & (df_original['time'] <= end_date)]

            ##### 選擇K棒時間範圍 #####
            st.subheader("選擇K棒的時間範圍")
            kbar_duration = st.selectbox("選擇K棒的時間範圍", ["日", "週", "月"], key='kbar_duration')

            if kbar_duration == "日":
                resample_period = 'D'
                short_ma_range = (5, 20)
                long_ma_range = (50, 200)
                short_rsi_range = (5, 14)
                long_rsi_range = (14, 30)
                bb_range = (20, 50)
                dc_range = (20, 50)
            elif kbar_duration == "週":
                resample_period = 'W'
                short_ma_range = (2, 8)
                long_ma_range = (10, 40)
                short_rsi_range = (2, 6)
                long_rsi_range = (6, 12)
                bb_range = (8, 16)
                dc_range = (8, 16)
            elif kbar_duration == "月":
                resample_period = 'M'
                short_ma_range = (1, 3)
                long_ma_range = (3, 12)
                short_rsi_range = (1, 3)
                long_rsi_range = (3, 6)
                bb_range = (3, 6)
                dc_range = (3, 6)

            df = df.resample(resample_period, on='time').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
                'amount': 'sum'
            }).dropna().reset_index()

            ###### (2) 轉化為字典 ######
            KBar_dic = df.to_dict()

            KBar_open_list = np.array(list(KBar_dic['open'].values()))
            KBar_dic['open'] = KBar_open_list

            KBar_dic['product'] = np.repeat(selected_stock, KBar_dic['open'].size)

            KBar_time_list = [i.to_pydatetime() for i in KBar_dic['time'].values()]
            KBar_dic['time'] = np.array(KBar_time_list)

            KBar_low_list = np.array(list(KBar_dic['low'].values()))
            KBar_dic['low'] = KBar_low_list

            KBar_high_list = np.array(list(KBar_dic['high'].values()))
            KBar_dic['high'] = KBar_high_list

            KBar_close_list = np.array(list(KBar_dic['close'].values()))
            KBar_dic['close'] = KBar_close_list

            KBar_volume_list = np.array(list(KBar_dic['volume'].values()))
            KBar_dic['volume'] = KBar_volume_list

            KBar_amount_list = np.array(list(KBar_dic['amount'].values()))
            KBar_dic['amount'] = KBar_amount_list

            ######  (3) 改變 KBar 時間長度  ########

            Date = start_date.strftime("%Y-%m-%d")

            st.subheader(f"設定一根 {kbar_duration} K 棒的時間長度(天數)")
            cycle_duration = st.number_input('輸入一根 K 棒的時間長度(單位:天, 一日=1天)', value=1, key='cycle_duration')
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

                # 添加调试信息
                if None in (time, open_price, close_price, low_price, high_price, qty):
                    st.error(f"数据缺失：time={time}, open={open_price}, close={close_price}, low={low_price}, high={high_price}, volume={qty}")
                    continue

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
            st.subheader(f"{selected_stock} - 設定計算長移動平均線(MA)的 K 棒數目")
            LongMAPeriod = st.slider('選擇一個整數', long_ma_range[0], long_ma_range[1], 10, key='LongMAPeriod')
            st.subheader(f"{selected_stock} - 設定計算短移動平均線(MA)的 K 棒數目")
            ShortMAPeriod = st.slider('選擇一個整數', short_ma_range[0], short_ma_range[1], 2, key='ShortMAPeriod')

            #### 計算長短移動平均線
            KBar_df['MA_long'] = KBar_df['close'].rolling(window=LongMAPeriod).mean()
            KBar_df['MA_short'] = KBar_df['close'].rolling(window=ShortMAPeriod).mean()

            #### 尋找最後 NAN值的位置
            last_nan_index_MA = KBar_df['MA_long'][::-1].index[KBar_df['MA_long'][::-1].apply(pd.isna)][0]

            #####  (ii) RSI 策略   #####
            #### 順勢策略
            ### 設定長短 RSI 的 K棒 長度:
            st.subheader(f"{selected_stock} - 設定計算長RSI的 K 棒數目")
            LongRSIPeriod = st.slider('選擇一個整數', long_rsi_range[0], long_rsi_range[1], 10, key='LongRSIPeriod')
            st.subheader(f"{selected_stock} - 設定計算短RSI的 K 棒數目")
            ShortRSIPeriod = st.slider('選擇一個整數', short_rsi_range[0], short_rsi_range[1], 2, key='ShortRSIPeriod')

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
            st.subheader(f"{selected_stock} - 設定計算布林通道(Bollinger Bands)的 K 棒數目")
            BBPeriod = st.slider('選擇一個整數', bb_range[0], bb_range[1], 20, key='BBPeriod')
            KBar_df['MA'] = KBar_df['Close'].rolling(window=BBPeriod).mean()
            KBar_df['BB_upper'] = KBar_df['MA'] + 2 * KBar_df['Close'].rolling(window=BBPeriod).std()
            KBar_df['BB_lower'] = KBar_df['MA'] - 2 * KBar_df['Close'].rolling(window=BBPeriod).std()

            ###### (7) 增加唐奇安通道 ######
            st.subheader(f"{selected_stock} - 設定計算唐奇安通道(Donchian Channels)的 K 棒數目")
            DCPPeriod = st.slider('選擇一個整數', dc_range[0], dc_range[1], 20, key='DCPPeriod')
            KBar_df['DC_upper'] = KBar_df['High'].rolling(window=DCPPeriod).max()
            KBar_df['DC_lower'] = KBar_df['Low'].rolling(window=DCPPeriod).min()

            ###### (8) 圖表 ######
            st.subheader("圖表")
            
            tabs = st.tabs(["K線圖和移動平均線", "K線圖和布林通道圖", "K線圖和唐奇安通道", "長短 RSI", "MACD 圖表"])

            ##### K線圖和移動平均線
            with tabs[0]:
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

                fig1.layout.yaxis2.showgrid = True
                st.plotly_chart(fig1, use_container_width=True)

            ##### 布林通道圖
            with tabs[1]:
                fig2 = make_subplots(specs=[[{"secondary_y": True}]])

                #### include candlestick with rangeselector
                fig2.add_trace(go.Candlestick(x=KBar_df['Time'],
                                open=KBar_df['Open'], high=KBar_df['High'],
                                low=KBar_df['Low'], close=KBar_df['Close'], name='K線'),
                               secondary_y=True)  # secondary_y=True 表示此圖形的y軸scale是在右邊而不是在左邊

                fig2.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_MA+1:], y=KBar_df['BB_upper'][last_nan_index_MA+1:], mode='lines', line=dict(color='blue', width=2), name='布林通道上軌'), secondary_y=True)
                fig2.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_MA+1:], y=KBar_df['BB_lower'][last_nan_index_MA+1:], mode='lines', line=dict(color='blue', width=2), name='布林通道下軌'), secondary_y=True)

                fig2.layout.yaxis2.showgrid = True
                st.plotly_chart(fig2, use_container_width=True)

            ##### K線圖, 唐奇安通道
            with tabs[2]:
                fig4 = make_subplots(specs=[[{"secondary_y":
