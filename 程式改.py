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
selected_stocks = st.multiselect("選擇股票", list(stock_dict.keys()), default=[list(stock_dict.keys())[0]])

@st.cache_data
def load_excel_data(file_path):
    df = pd.read_excel(file_path)
    df = df.drop('Unnamed: 0', axis=1)
    df['time'] = pd.to_datetime(df['time'])
    return df

if selected_stocks:
    for index, selected_stock in enumerate(selected_stocks):
        if selected_stock in stock_dict:
            try:
                file_path, stock_id = stock_dict[selected_stock]
                df_original = load_excel_data(file_path)
                
                ##### 選擇資料區間
                st.subheader(f"{selected_stock} - 選擇開始與結束的日期, 區間:2019-01-01 至 2024-05-31")
                start_date = st.text_input(f'選擇開始日期 (日期格式: 2019-01-01)', '2019-01-01', key=f"start_date_{index}")
                end_date = st.text_input(f'選擇結束日期 (日期格式: 2024-05-31)', '2024-05-31', key=f"end_date_{index}")
                start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
                end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
                df = df_original[(df_original['time'] >= start_date) & (df_original['time'] <= end_date)]

                # 重新采樣數據（例如，按周采樣）
                df = df.resample('W', on='time').agg({
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

                st.subheader("設定一根 K 棒的時間長度(天數)")
                cycle_duration = st.number_input('輸入一根 K 棒的時間長度(單位:天, 一日=1天)', value=1, key=f"KBar_duration_{index}")
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
                LongMAPeriod = st.slider('選擇一個整數', 0, 100, 10, key=f"LongMAPeriod_{index}")
                st.subheader(f"{selected_stock} - 設定計算短移動平均線(MA)的 K 棒數目(整數, 例如 2)")
                ShortMAPeriod = st.slider('選擇一個整數', 0, 100, 2, key=f"ShortMAPeriod_{index}")

                #### 計算長短移動平均線
                KBar_df['MA_long'] = KBar_df['close'].rolling(window=LongMAPeriod).mean()
                KBar_df['MA_short'] = KBar_df['close'].rolling(window=ShortMAPeriod).mean()

                #### 尋找最後 NAN值的位置
                last_nan_index_MA = KBar_df['MA_long'][::-1].index[KBar_df['MA_long'][::-1].apply(pd.isna)][0]

                #####  (ii) RSI 策略   #####
                #### 順勢策略
                ### 設定長短 RSI 的 K棒 長度:
                st.subheader(f"{selected_stock} - 設定計算長RSI的 K 棒數目(整數, 例如 10)")
                LongRSIPeriod = st.slider('選擇一個整數', 0, 1000, 10, key=f"LongRSIPeriod_{index}")
                st.subheader(f"{selected_stock} - 設定計算短RSI的 K 棒數目(整數, 例如 2)")
                ShortRSIPeriod = st.slider('選擇一個整數', 0, 1000, 2, key=f"ShortRSIPeriod_{index}")

                ### 計算 RSI指標長短線, 以及定義中線
                def calculate_rsi
