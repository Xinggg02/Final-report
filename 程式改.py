import streamlit as st
import st_pages

st.set_page_config(page_title="金融大數據期末APP", layout="wide")

st_pages.create_side_menu(
    menu_title="目錄",
    pages=[
        {"title": "首頁", "icon": "🏠", "file": "main.py"},
        {"title": "股票分析", "icon": "📈", "file": "stock_analysis.py"},
        {"title": "統計數據", "icon": "📊", "file": "statistics.py"}
    ]
)

###### (1) 開始設定 ######
html_temp = """
<div style="background-color:#4CAF50;padding:15px;border-radius:15px">
    <h1 style="color:#FFFFFF;text-align:center;font-size:36px;">金融大數據期末APP-股票資料呈現</h1>
    <h2 style="color:#FFFFFF;text-align:center;font-size:28px;">Final-report</h2>
</div>
"""
st.markdown(html_temp, unsafe_allow_html=True)

st.write("歡迎來到金融大數據期末APP，請選擇左側菜單進行操作。")

import streamlit as st
import pandas as pd
import twstock

st.set_page_config(page_title="統計數據", layout="wide")

@st.cache_data
def load_excel_data(file_path):
    df = pd.read_excel(file_path)
    df = df.drop('Unnamed: 0', axis=1)
    df['time'] = pd.to_datetime(df['time'])
    return df

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

stock_ids = ['0050', '00878', '006208', '1215', '1225', '2303', '2454', '2603', '2609', '2615', '1216', '1210', '1201', '1303', '1301', '1102', '1101', '3443', '3055', '2451', '2891', '2890', '2881', '2880', '2882']
stock_dict = load_stock_data(stock_ids)

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

