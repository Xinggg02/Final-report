import os
import numpy as np
import datetime
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import twstock

###### (1) 開始設定 ######
html_temp = """
<div style="background: linear-gradient(to right, #4CAF50, #81C784); padding: 20px; border-radius: 15px; box-shadow: 0px 4px 12px rgba(0,0,0,0.2); border: 2px solid #388E3C; animation: fadeIn 3s ease-in;">
    <style>
        @keyframes fadeIn {
            from {
                opacity: 0;
            }
            to {
                opacity: 1;
            }
        }
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        .title {
            font-family: 'Arial Black', Gadget, sans-serif;
            color: #fff;
            text-align: center;
            animation: fadeInUp 3s ease-in;
        }
        .icon {
            width: 50px;
            height: 50px;
            margin: 0 15px;
            transition: transform 0.3s, filter 0.3s;
        }
        .icon:hover {
            transform: scale(1.2);
            filter: brightness(1.2);
        }
        .subtext {
            color: #FFFFFF;
            text-align: center;
            font-size: 28px;
            font-family: 'Arial', Helvetica, sans-serif;
            animation: fadeInUp 3s ease-in;
        }
    </style>
    <div style="display: flex; align-items: center; justify-content: center;">
        <img src="https://img.icons8.com/doodle/48/000000/bank.png" class="icon">
        <h1 class="title">金融大數據期末APP-股票資料呈現</h1>
        <img src="https://img.icons8.com/fluency/48/total-sales-1.png" class="icon">
    </div>
    <h2 class="subtext">Final-report</h2>
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
st.subheader(f"股票各資訊")
selected_stocks = st.multiselect("選擇股票", list(stock_dict.keys()), default=[list(stock_dict.keys())[0]])

@st.cache_data
def load_excel_data(file_path):
    df = pd.read_excel(file_path)
    df = df.drop('Unnamed: 0', axis=1)
    df['time'] = pd.to_datetime(df['time'])
    return df

def detect_cross_signals(df, short_window, long_window):
    signals = pd.DataFrame(index=df.index)
    signals['signal'] = 0.0

    signals['Short_Mavg'] = df['close'].rolling(window=short_window, min_periods=1, center=False).mean()
    signals['Long_Mavg'] = df['close'].rolling(window=long_window, min_periods=1, center=False).mean()

    # Create signals
    signals['signal'][short_window:] = np.where(signals['Short_Mavg'][short_window:] > signals['Long_Mavg'][short_window:], 1.0, 0.0)
    
    # Generate trading orders
    signals['positions'] = signals['signal'].diff()

    return signals

def calculate_rsi(df, period=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def detect_rsi_signals(df, period=14, overbought=70, oversold=30):
    signals = pd.DataFrame(index=df.index)
    signals['RSI'] = calculate_rsi(df, period)

    signals['signal'] = 0.0
    signals['signal'] = np.where(signals['RSI'] > overbought, -1.0, np.where(signals['RSI'] < oversold, 1.0, 0.0))
    signals['positions'] = signals['signal'].diff()

    return signals

def calculate_kdj(df, period=9):
    low_min = df['low'].rolling(window=period, min_periods=1).min()
    high_max = df['high'].rolling(window=period, min_periods=1).max()
    rsv = (df['close'] - low_min) / (high_max - low_min) * 100

    df['K'] = rsv.ewm(com=2).mean()
    df['D'] = df['K'].ewm(com=2).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']

    return df

# 自定义KBar类
class KBar:
    def __init__(self, start_date, cycle_duration):
        self.start_date = start_date
        self.cycle_duration = cycle_duration
        self.TAKBar = {
            'time': [],
            'open': [],
            'high': [],
            'low': [],
            'close': [],
            'volume': [],
        }
    
    def AddPrice(self, time, open_price, close_price, low_price, high_price, volume):
        self.TAKBar['time'].append(time)
        self.TAKBar['open'].append(open_price)
        self.TAKBar['high'].append(high_price)
        self.TAKBar['low'].append(low_price)
        self.TAKBar['close'].append(close_price)
        self.TAKBar['volume'].append(volume)

if selected_stocks:
    for index, selected_stock in enumerate(selected_stocks):
        if selected_stock in stock_dict:
            try:
                file_path, stock_id = stock_dict[selected_stock]
                df_original = load_excel_data(file_path)
                
                ##### 選擇資料區間 #####
                with st.expander(f"{selected_stock} - 選擇開始與結束的日期, 區間:2019-01-01 至 2024-05-31"):
                    start_date = st.date_input(f'選擇開始日期', datetime.date(2019, 1, 1), min_value=datetime.date(2019, 1, 1), max_value=datetime.date(2024, 5, 31), key=f"start_date_{index}")
                    end_date = st.date_input(f'選擇結束日期', datetime.date(2024, 5, 31), min_value=datetime.date(2019, 1, 1), max_value=datetime.date(2024, 5, 31), key=f"end_date_{index}")
                start_date = datetime.datetime.combine(start_date, datetime.time.min)
                end_date = datetime.datetime.combine(end_date, datetime.time.min)
                df = df_original[(df_original['time'] >= start_date) & (df_original['time'] <= end_date)]

                ##### 選擇K棒時間範圍 #####
                with st.expander(f"{selected_stock} - 選擇K棒的時間範圍"):
                    kbar_duration = st.selectbox("選擇K棒的時間範圍", ["日", "週", "月"], key=f"kbar_duration_{index}")
                    cycle_duration = st.number_input('輸入一根 K 棒的時間長度(單位:日、週、月)', value=1, key=f"KBar_duration_{index}")

                if kbar_duration == "日":
                    resample_period = 'D'
                    long_ma_default = 50
                    short_ma_default = 20
                    long_rsi_default = 14
                    short_rsi_default = 7
                    bb_default = 20
                    dc_default = 20
                elif kbar_duration == "週":
                    resample_period = 'W'
                    long_ma_default = 10
                    short_ma_default = 4
                    long_rsi_default = 10
                    short_rsi_default = 3
                    bb_default = 4
                    dc_default = 4
                elif kbar_duration == "月":
                    resample_period = 'M'
                    long_ma_default = 3
                    short_ma_default = 1
                    long_rsi_default = 2
                    short_rsi_default = 1
                    bb_default = 1
                    dc_default = 1

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

                KBar_dic['product'] = np.repeat(selected_stock, len(KBar_dic['time']))

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

                cycle_duration = int(cycle_duration)

                KBar = KBar(Date, cycle_duration)  # 使用自定义KBar类

                for i in range(len(KBar_dic['time'])):
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
                KBar_dic['product'] = np.repeat(selected_stock, len(KBar.TAKBar['time']))
                KBar_dic['open'] = KBar.TAKBar['open']
                KBar_dic['high'] = KBar.TAKBar['high']
                KBar_dic['low'] = KBar.TAKBar['low']
                KBar_dic['close'] = KBar.TAKBar['close']
                KBar_dic['volume'] = KBar.TAKBar['volume']

                KBar_df = pd.DataFrame(KBar_dic)

                ##### 基本信息展示 #####
                with st.expander(f"{selected_stock} - 股票基本信息"):
                    stock_info = twstock.codes.get(stock_id, None)
                    if stock_info:
                        st.write(f"公司名稱: {stock_info.name}")
                        st.write(f"市場: {getattr(stock_info, 'market', 'N/A')}")
                        st.write(f"上市日期: {getattr(stock_info, 'start', 'N/A')}")
                        st.write(f"CFI: {getattr(stock_info, 'CFI', 'N/A')}")
                        st.write(f"ISIN: {getattr(stock_info, 'ISIN', 'N/A')}")
                        st.write(f"code: {getattr(stock_info, 'code', 'N/A')}")

                    else:
                        st.write("找不到該股票的詳細信息。")

                #####  (i) 移動平均線策略   #####
                ####  設定長短移動平均線的 K棒 長度:
                st.subheader(f"{selected_stock} - 設定計算長移動平均線(MA)的 K 棒數目")
                LongMAPeriod = st.slider('選擇一個整數', 10, 200, long_ma_default, key=f"LongMAPeriod_{index}")
                st.subheader(f"{selected_stock} - 設定計算短移動平均線(MA)的 K 棒數目")
                ShortMAPeriod = st.slider('選擇一個整數', 5, 50, short_ma_default, key=f"ShortMAPeriod_{index}")

                #### 計算長短移動平均線
                KBar_df['MA_long'] = KBar_df['close'].rolling(window=LongMAPeriod).mean()
                KBar_df['MA_short'] = KBar_df['close'].rolling(window=ShortMAPeriod).mean()

                #### 尋找最後 NAN值的位置
                last_nan_index_MA = KBar_df['MA_long'][::-1].index[KBar_df['MA_long'][::-1].apply(pd.isna)][0]

                #### 檢測交易信號
                ma_signals = detect_cross_signals(KBar_df, ShortMAPeriod, LongMAPeriod)
                KBar_df['MA_Signal'] = ma_signals['signal']
                KBar_df['MA_Positions'] = ma_signals['positions']

                #####  (ii) RSI 策略   #####
                #### 順勢策略
                ### 設定長短 RSI 的 K棒 長度:
                st.subheader(f"{selected_stock} - 設定計算長RSI的 K 棒數目")
                LongRSIPeriod = st.slider('選擇一個整數', 10, 20, long_rsi_default, key=f"LongRSIPeriod_{index}")
                st.subheader(f"{selected_stock} - 設定計算短RSI的 K 棒數目")
                ShortRSIPeriod = st.slider('選擇一個整數', 5, 14, short_rsi_default, key=f"ShortRSIPeriod_{index}")

                ### 計算 RSI指標長短線, 以及定義中線
                KBar_df['RSI_long'] = calculate_rsi(KBar_df, LongRSIPeriod)
                KBar_df['RSI_short'] = calculate_rsi(KBar_df, ShortRSIPeriod)
                KBar_df['RSI_Middle'] = np.array([50] * len(KBar_dic['time']))

                ### 尋找最後 NAN值的位置
                last_nan_index_RSI = KBar_df['RSI_long'][::-1].index[KBar_df['RSI_long'][::-1].apply(pd.isna)][0]

                rsi_signals = detect_rsi_signals(KBar_df, LongRSIPeriod)
                KBar_df['RSI_Signal'] = rsi_signals['signal']
                KBar_df['RSI_Positions'] = rsi_signals['positions']

                ###### (5) 將 Dataframe 欄位名稱轉換  ######
                KBar_df.columns = [i[0].upper() + i[1:] for i in KBar_df.columns]

                ###### (7) 增加唐奇安通道 ######
                st.subheader(f"{selected_stock} - 設定計算唐奇安通道(Donchian Channels)的 K 棒數目")
                DCPPeriod = st.slider('選擇一個整數', 10, 50, dc_default, key=f"DCPPeriod_{index}")
                KBar_df['DC_upper'] = KBar_df['High'].rolling(window=DCPPeriod).max()
                KBar_df['DC_lower'] = KBar_df['Low'].rolling(window=DCPPeriod).min()

                ###### 增加KDJ指標 ######
                st.subheader(f"{selected_stock} - 設定計算KDJ指標的 K 棒數目")
                KDJPeriod = st.slider('選擇一個整數', 5, 30, 9, key=f"KDJPeriod_{index}")
                KBar_df = calculate_kdj(KBar_df, KDJPeriod)

                ###### (8) 圖表 ######
                st.subheader("圖表")
                
                tabs = st.tabs(["K線圖和移動平均線", "K線圖和唐奇安通道", "長短 RSI", "MACD 圖表", "KDJ 圖表"])

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

                    # 添加交易信號
                    ma_buy_signals = KBar_df[KBar_df['MA_Positions'] == 1]
                    ma_sell_signals = KBar_df[KBar_df['MA_Positions'] == -1]
                    fig1.add_trace(go.Scatter(x=ma_buy_signals['Time'], y=ma_buy_signals['Close'], mode='markers', marker=dict(symbol='triangle-up', color='green', size=10), name='MA買入信號'))
                    fig1.add_trace(go.Scatter(x=ma_sell_signals['Time'], y=ma_sell_signals['Close'], mode='markers', marker=dict(symbol='triangle-down', color='red', size=10), name='MA賣出信號'))

                    fig1.layout.yaxis2.showgrid = True
                    st.plotly_chart(fig1, use_container_width=True)

                ##### K線圖, 唐奇安通道
                with tabs[1]:
                    fig4 = make_subplots(specs=[[{"secondary_y": True}]])

                    #### include candlestick with rangeselector
                    fig4.add_trace(go.Candlestick(x=KBar_df['Time'],
                                    open=KBar_df['Open'], high=KBar_df['High'],
                                    low=KBar_df['Low'], close=KBar_df['Close'], name='K線'),
                                   secondary_y=True)  # secondary_y=True 表示此圖形的y軸scale是在右邊而不是在左邊

                    #### include a go.Bar trace for volumes
                    fig4.add_trace(go.Bar(x=KBar_df['Time'], y=KBar_df['Volume'], name='成交量', marker=dict(color='black')), secondary_y=False)
                    fig4.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_MA+1:], y=KBar_df['DC_upper'][last_nan_index_MA+1:], mode='lines', line=dict(color='green', width=2), name='唐奇安通道上軌'), secondary_y=True)
                    fig4.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_MA+1:], y=KBar_df['DC_lower'][last_nan_index_MA+1:], mode='lines', line=dict(color='red', width=2), name='唐奇安通道下軌'), secondary_y=True)

                    fig4.layout.yaxis2.showgrid = True
                    st.plotly_chart(fig4, use_container_width=True)

                ##### 長短 RSI
                with tabs[2]:
                    fig3 = make_subplots(specs=[[{"secondary_y": True}]])

                    fig3.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_RSI+1:], y=KBar_df['RSI_long'][last_nan_index_RSI+1:], mode='lines', line=dict(color='red', width=2), name=f'{LongRSIPeriod}-根 K棒 移動 RSI'), secondary_y=True)
                    fig3.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_RSI+1:], y=KBar_df['RSI_short'][last_nan_index_RSI+1:], mode='lines', line=dict(color='blue', width=2), name=f'{ShortRSIPeriod}-根 K棒 移動 RSI'), secondary_y=True)

                    rsi_buy_signals = KBar_df[KBar_df['RSI_Positions'] == 1]
                    rsi_sell_signals = KBar_df[KBar_df['RSI_Positions'] == -1]
                    fig3.add_trace(go.Scatter(x=rsi_buy_signals['Time'], y=rsi_buy_signals['Close'], mode='markers', marker=dict(symbol='triangle-up', color='green', size=10), name='RSI買入信號'))
                    fig3.add_trace(go.Scatter(x=rsi_sell_signals['Time'], y=rsi_sell_signals['Close'], mode='markers', marker=dict(symbol='triangle-down', color='red', size=10), name='RSI賣出信號'))

                    fig3.layout.yaxis2.showgrid = True
                    st.plotly_chart(fig3, use_container_width=True)

                ##### 增加MACD圖表 #####
                with tabs[3]:
                    st.subheader("MACD 計算參數")
                    macd_fast = st.slider('MACD 快線週期', 1, 50, 12, key=f"macd_fast_{index}")
                    macd_slow = st.slider('MACD 慢線週期', 1, 50, 26, key=f"macd_slow_{index}")
                    macd_signal = st.slider('MACD 信號線週期', 1, 50, 9, key=f"macd_signal_{index}")

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

                ##### 增加KDJ圖表 #####
                with tabs[4]:
                    fig_kdj = make_subplots(specs=[[{"secondary_y": True}]])

                    fig_kdj.add_trace(go.Scatter(x=KBar_df['Time'], y=KBar_df['K'], mode='lines', line=dict(color='blue', width=2), name='K'), secondary_y=True)
                    fig_kdj.add_trace(go.Scatter(x=KBar_df['Time'], y=KBar_df['D'], mode='lines', line=dict(color='orange', width=2), name='D'), secondary_y=True)
                    fig_kdj.add_trace(go.Scatter(x=KBar_df['Time'], y=KBar_df['J'], mode='lines', line=dict(color='green', width=2), name='J'), secondary_y=True)

                    fig_kdj.layout.yaxis2.showgrid = True
                    st.plotly_chart(fig_kdj, use_container_width=True)


            except FileNotFoundError as e:
                st.error(f"Error: {e}")
else:
    st.error("請選擇至少一個股票。")

# 增加一些額外的統計數據
st.subheader("額外統計數據")
stat_option = st.selectbox("選擇要查看的統計數據", ["", "總成交量", "總成交額"])
selected_stat_stocks = st.multiselect("選擇要查看統計數據的股票", list(stock_dict.keys()))

if stat_option and selected_stat_stocks:
    for stock_name in selected_stat_stocks:
        if stock_name in stock_dict:
            file_path, stock_id = stock_dict[stock_name]
            df = load_excel_data(file_path)
            if stat_option == "總成交量":
                total_volume = df['volume'].sum()
                st.write(f"{stock_name} (代碼: {stock_id}) 總成交量: {total_volume}")
            elif stat_option == "總成交額":
                total_amount = df['amount'].sum()
                st.write(f"{stock_name} (代碼: {stock_id}) 總成交額: {total_amount}")
