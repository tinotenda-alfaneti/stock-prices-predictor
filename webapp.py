import streamlit as st
# import pandas_datareader as web
import matplotlib.pyplot as plt
from plotly import graph_objs as go
from datetime import datetime
from model import predict_next_day

import yfinance as web

st.title('Atarist Stock Price Prediction')
stocks = ('AAPL.AX', 'GOOG.AX', 'MSFT.AX', 'AMZN.AX', 'TSLA.AX', 'JPM.AX', 'META.AX')
selected_stock = st.selectbox('SELECT STOCK', stocks)

@st.cache(persist=True)
def load_data(ticker):
    data = web.download(ticker, start='2010-01-01', end='2023-01-01')
    data.reset_index(inplace=True)
    return data

data = load_data(selected_stock)

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data['Date'], open=data['Open'], high=data["High"], low=data["Low"], close=data["Close"]))
    fig.layout.update(title_text='PAST DATA CANDLESTICK CHAT', xaxis_rangeslider_visible=True, template="plotly_dark")
    fig.update_yaxes(type="log")
    
    st.plotly_chart(fig)

plot_raw_data()

if st.button(f"PREDICT {selected_stock} NEXT DAY CLOSING PRICE"):
    st.write("Predicting...")
    
    prediction_price = predict_next_day(selected_stock, data)
    
    st.write(f"NEXT DAY'S PREDICTED CLOSING PRICE: {prediction_price}")



