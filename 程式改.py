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
stock_ids = ['0050', '00878', '006208', '1215', '1225', '2303', '2454', '2603', '2609', '2615','1216','1210','1201','1303','1301','1102','1101','3443','3055','2451','2891','2890','2881','2880','2882']
stock_dict = load_stock_data(stock_ids)

# 生成股票選擇列表
selected_stocks = st.multiselect("選擇股票", list(stock_dict.keys()), default=[list(stock_dict.keys())[0]])

@st.cache_data
def load_excel_data(file_path):
    df = pd.read_excel(file_path)
    df = df.drop('Unnamed: 0', axis=1)
    df['time'] = pd.to_datetime(df['time'])
    return df

def calculate_rsi(df, period=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def backtest_strategy(KBar_df, LongMAPeriod, ShortMAPeriod, TradeVolume):
    KBar_df['signal'] = 0
    KBar_df['signal'][ShortMAPeriod:] = np.where(
        KBar_df['MA_short'][ShortMAPeriod:] > KBar_df['MA_long'][ShortMAPeriod:], 1, -1)
    KBar_df['position'] = KBar_df['signal'].shift()
    
    KBar_df['returns'] = KBar_df['Close'].pct_change()
    KBar_df['strategy'] = KBar_df['returns'] * KBar_df['position']
    
    KBar_df['trade_value'] = KBar_df['strategy'] * TradeVolume
    KBar_df['cumulative_returns'] = (KBar_df['trade_value'] + 1).cumprod() - 1
    KBar_df['cumulative_strategy'] = (KBar_df['strategy'] + 1).cumprod() - 1
    
    trade_results = {
        '交易總盈虧(元)': KBar_df['trade_value'].sum(),
        '平均每交易盈虧(元)': KBar_df['trade_value'].mean(),
        '平均投資報酬率': KBar_df['strategy'].mean(),
        '平均獲利(只看獲利的)(元)': KBar_df[KBar_df['trade_value'] > 0]['trade_value'].mean(),
        '平均虧損(只看虧損的)(元)': KBar_df[KBar_df['trade_value'] < 0]['trade_value'].mean(),
        '勝率': KBar_df[KBar_df['trade_value'] > 0].shape[0] / KBar_df[KBar_df['trade_value'] != 0].shape[0],
        '最大連續虧損(元)': KBar_df['trade_value'].min(),
        '最大盈虧回落(MDD)': (KBar_df['cumulative_returns'].max() - KBar_df['cumulative_returns'].min()),
        '績效風險比(交易總盈虧/最大盈虧回落(MDD))': KBar_df['trade_value'].sum() / (KBar_df['cumulative_returns'].max() - KBar_df['cumulative_returns'].min())
    }
    return trade_results

if selected_stocks:
    for index, selected_stock in enumerate(selected_stocks):
        if selected_stock in stock_dict:
            try:
                file_path, stock_id = stock_dict[selected_stock]
                df_original = load_excel_data(file_path)
                
                ##### 選擇資料區間 #####
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

                # 打印列名調試
                st.write("DataFrame columns:", KBar_df.columns)

                # 確保所有列名都是正確的
                if 'close' not in KBar_df.columns:
                    KBar_df.rename(columns={'Close': 'close'}, inplace=True)
                if 'open' not in KBar_df.columns:
                    KBar_df.rename(columns={'Open': 'open'}, inplace=True)
                if 'high' not in KBar_df.columns:
                    KBar_df.rename(columns={'High': 'high'}, inplace=True)
                if 'low' not in KBar_df.columns:
                    KBar_df.rename(columns={'Low': 'low'}, inplace=True)
                if 'volume' not in KBar_df.columns:
                    KBar_df.rename(columns={'Volume': 'volume'}, inplace=True)

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
                KBar_df['RSI_long'] = calculate_rsi(KBar_df, LongRSIPeriod)
                KBar_df['RSI_short'] = calculate_rsi(KBar_df, ShortRSIPeriod)
                KBar_df['RSI_Middle'] = np.array([50] * len(KBar_dic['time']))

                ### 尋找最後 NAN值的位置
                last_nan_index_RSI = KBar_df['RSI_long'][::-1].index[KBar_df['RSI_long'][::-1].apply(pd.isna)][0]

                ###### (5) 將 Dataframe 欄位名稱轉換  ######
                KBar_df.columns = [i[0].upper() + i[1:] for i in KBar_df.columns]

                ###### (6) 增加Bollinger Bands ######
                st.subheader(f"{selected_stock} - 設定計算布林通道(Bollinger Bands)的 K 棒數目(整數, 例如 20)")
                BBPeriod = st.slider('選擇一個整數', 0, 100, 20, key=f"BBPeriod_{index}")
                KBar_df['MA'] = KBar_df['Close'].rolling(window=BBPeriod).mean()
                KBar_df['BB_upper'] = KBar_df['MA'] + 2 * KBar_df['Close'].rolling(window=BBPeriod).std()
                KBar_df['BB_lower'] = KBar_df['MA'] - 2 * KBar_df['Close'].rolling(window=BBPeriod).std()

                ###### (7) 增加唐奇安通道 ######
                st.subheader(f"{selected_stock} - 設定計算唐奇安通道(Donchian Channels)的 K 棒數目(整數, 例如 20)")
                DCPeriod = st.slider('選擇一個整數', 0, 100, 20, key=f"DCPeriod_{index}")
                KBar_df['DC_upper'] = KBar_df['High'].rolling(window=DCPeriod).max()
                KBar_df['DC_lower'] = KBar_df['Low'].rolling(window=DCPeriod).min()

                ###### (8) 增加程式交易策略 ######
                st.subheader("程式交易:")
                strategy = st.selectbox("選擇交易策略", ["移動平均線黃金交叉做多，死亡交叉做空，<出場>結算平倉(期貨)，移動停損"])
                
                st.subheader(f"{selected_stock} - 設定策略參數")
                TradeVolume = st.slider('設置交易每次購買量', 1, 1000, 100, key=f"TradeVolume_{index}")

                # 計算策略結果
                trade_results = backtest_strategy(KBar_df, LongMAPeriod, ShortMAPeriod, TradeVolume)

                ###### (9) 畫圖 ######
                st.subheader("畫圖")

                ##### K線圖和移動平均線 #####
                with st.expander(f"{selected_stock} - K線圖和移動平均線"):
                    fig1 = make_subplots(specs=[[{"secondary_y": True}]])

                    #### include candlestick with rangeselector
                    fig1.add_trace(go.Candlestick(x=KBar_df['Time'],
                                    open=KBar_df['Open'], high=KBar_df['High'],
                                    low=KBar_df['Low'], close=KBar_df['Close'], name='K線'),
                                   secondary_y=True)

                    #### include a go.Bar trace for volumes
                    fig1.add_trace(go.Bar(x=KBar_df['Time'], y=KBar_df['Volume'], name='成交量', marker=dict(color='black')), secondary_y=False)
                    fig1.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_MA+1:], y=KBar_df['MA_long'][last_nan_index_MA+1:], mode='lines', line=dict(color='orange', width=2), name=f'{LongMAPeriod}-根 K棒 移動平均線'), secondary_y=True)
                    fig1.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_MA+1:], y=KBar_df['MA_short'][last_nan_index_MA+1:], mode='lines', line=dict(color='pink', width=2), name=f'{ShortMAPeriod}-根 K棒 移動平均線'), secondary_y=True)

                    fig1.layout.yaxis2.showgrid = True
                    st.plotly_chart(fig1, use_container_width=True)

                ##### K線圖和布林通道 #####
                with st.expander(f"{selected_stock} - K線圖和布林通道"):
                    fig2 = make_subplots(specs=[[{"secondary_y": True}]])

                    #### include candlestick with rangeselector
                    fig2.add_trace(go.Candlestick(x=KBar_df['Time'],
                                    open=KBar_df['Open'], high=KBar_df['High'],
                                    low=KBar_df['Low'], close=KBar_df['Close'], name='K線'),
                                   secondary_y=True)

                    fig2.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_MA+1:], y=KBar_df['BB_upper'][last_nan_index_MA+1:], mode='lines', line=dict(color='blue', width=2), name='布林通道上軌'), secondary_y=True)
                    fig2.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_MA+1:], y=KBar_df['BB_lower'][last_nan_index_MA+1:], mode='lines', line=dict(color='blue', width=2), name='布林通道下軌'), secondary_y=True)

                    fig2.layout.yaxis2.showgrid = True
                    st.plotly_chart(fig2, use_container_width=True)

                ##### K線圖和唐奇安通道 #####
                with st.expander(f"{selected_stock} - K線圖和唐奇安通道"):
                    fig3 = make_subplots(specs=[[{"secondary_y": True}]])

                    #### include candlestick with rangeselector
                    fig3.add_trace(go.Candlestick(x=KBar_df['Time'],
                                    open=KBar_df['Open'], high=KBar_df['High'],
                                    low=KBar_df['Low'], close=KBar_df['Close'], name='K線'),
                                   secondary_y=True)

                    fig3.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_MA+1:], y=KBar_df['DC_upper'][last_nan_index_MA+1:], mode='lines', line=dict(color='green', width=2), name='唐奇安通道上軌'), secondary_y=True)
                    fig3.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_MA+1:], y=KBar_df['DC_lower'][last_nan_index_MA+1:], mode='lines', line=dict(color='red', width=2), name='唐奇安通道下軌'), secondary_y=True)

                    fig3.layout.yaxis2.showgrid = True
                    st.plotly_chart(fig3, use_container_width=True)

                ##### 長短RSI #####
                with st.expander(f"{selected_stock} - 長短RSI"):
                    fig4 = make_subplots(specs=[[{"secondary_y": True}]])

                    fig4.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_RSI+1:], y=KBar_df['RSI_long'][last_nan_index_RSI+1:], mode='lines', line=dict(color='red', width=2), name=f'{LongRSIPeriod}-根 K棒 移動 RSI'), secondary_y=True)
                    fig4.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_RSI+1:], y=KBar_df['RSI_short'][last_nan_index_RSI+1:], mode='lines', line=dict(color='blue', width=2), name=f'{ShortRSIPeriod}-根 K棒 移動 RSI'), secondary_y=True)

                    fig4.layout.yaxis2.showgrid = True
                    st.plotly_chart(fig4, use_container_width=True)

                ##### 程式交易結果 #####
                with st.expander(f"{selected_stock} - 程式交易結果"):
                    trade_results_df = pd.DataFrame(list(trade_results.items()), columns=['項目', '數值'])
                    st.table(trade_results_df)

                ##### 基本信息展示 #####
                with st.expander(f"{selected_stock} - 股票基本信息"):
                   
