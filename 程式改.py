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

def calculate_obv(df):
    obv = [0]  # 初始化 OBV 值
    for i in range(1, len(df)):
        if df['close'][i] > df['close'][i - 1]:
            obv.append(obv[-1] + df['volume'][i])
        elif df['close'][i] < df['close'][i - 1]:
            obv.append(obv[-1] - df['volume'][i])
        else:
            obv.append(obv[-1])
    df['obv'] = obv
    return df['obv']

def calculate_kdj(df, period=9, k_period=3, d_period=3):
    try:
        low_list = df['low'].rolling(window=period).min()
        high_list = df['high'].rolling(window=period).max()
        rsv = (df['close'] - low_list) / (high_list - low_list) * 100

        df['K'] = rsv.ewm(com=(k_period - 1), min_periods=1).mean()
        df['D'] = df['K'].ewm(com=(d_period - 1), min_periods=1).mean()
        df['J'] = 3 * df['K'] - 2 * df['D']

        return df[['K', 'D', 'J']]
    except Exception as e:
        st.error(f"計算 KDJ 指標時出錯: {e}")
        return pd.DataFrame()

def backtest_strategy(KBar_df, LongMAPeriod, ShortMAPeriod, TradeVolume):
    KBar_df['signal'] = 0
    KBar_df['signal'][ShortMAPeriod:] = np.where(
        KBar_df['MA_short'][ShortMAPeriod:] > KBar_df['MA_long'][ShortMAPeriod:], 1, -1)
    KBar_df['position'] = KBar_df['signal'].shift()
    
    KBar_df['returns'] = KBar_df['close'].pct_change()
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

                KBar_df = pd.DataFrame(KBar_dic)
                KBar_df['time'] = pd.to_datetime(KBar_df['time'])

                ###### (3) 計算策略 ######
                LongMAPeriod = 90
                ShortMAPeriod = 10
                TradeVolume = 1000

                KBar_df['MA_long'] = KBar_df['close'].rolling(window=LongMAPeriod).mean()
                KBar_df['MA_short'] = KBar_df['close'].rolling(window=ShortMAPeriod).mean()
                KBar_df['RSI'] = calculate_rsi(KBar_df)
                KBar_df['OBV'] = calculate_obv(KBar_df)
                KDJ_df = calculate_kdj(KBar_df)

                for column in KDJ_df.columns:
                    KBar_df[column] = KDJ_df[column]

                trade_results = backtest_strategy(KBar_df, LongMAPeriod, ShortMAPeriod, TradeVolume)

                st.subheader(f"{selected_stock} 交易績效")
                st.write(pd.DataFrame([trade_results]))

                ###### (4) 繪圖 ######
                fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                                    vertical_spacing=0.02, subplot_titles=(selected_stock, 'RSI', 'OBV', 'KDJ'),
                                    row_width=[0.2, 0.2, 0.2, 0.2])

                fig.add_trace(go.Candlestick(x=KBar_df['time'], open=KBar_df['open'], high=KBar_df['high'],
                                             low=KBar_df['low'], close=KBar_df['close'], name='Candlestick'), row=1, col=1)
                fig.add_trace(go.Scatter(x=KBar_df['time'], y=KBar_df['MA_long'], mode='lines', name='MA_long'), row=1, col=1)
                fig.add_trace(go.Scatter(x=KBar_df['time'], y=KBar_df['MA_short'], mode='lines', name='MA_short'), row=1, col=1)

                fig.add_trace(go.Scatter(x=KBar_df['time'], y=KBar_df['RSI'], mode='lines', name='RSI'), row=2, col=1)
                fig.add_trace(go.Bar(x=KBar_df['time'], y=KBar_df['volume'], name='Volume'), row=1, col=1)

                fig.add_trace(go.Scatter(x=KBar_df['time'], y=KBar_df['OBV'], mode='lines', name='OBV'), row=3, col=1)
                fig.add_trace(go.Scatter(x=KBar_df['time'], y=KBar_df['K'], mode='lines', name='K'), row=4, col=1)
                fig.add_trace(go.Scatter(x=KBar_df['time'], y=KBar_df['D'], mode='lines', name='D'), row=4, col=1)
                fig.add_trace(go.Scatter(x=KBar_df['time'], y=KBar_df['J'], mode='lines', name='J'), row=4, col=1)

                fig.update_layout(xaxis_rangeslider_visible=False)
                st.plotly_chart(fig)
            except Exception as e:
                st.error(f"處理 {selected_stock} 時出錯: {e}")
else:
    st.write("請選擇至少一支股票來顯示資料。")
