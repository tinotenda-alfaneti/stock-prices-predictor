import streamlit as st
import pandas_datareader as web
import matplotlib.pyplot as plt
from plotly import graph_objs as go
from datetime import datetime
from model import *

st.title('Atarist Stock Price Prediction')
stocks = ('AAPL', 'GOOG', 'MSFT', 'AMZN')
selected_stock = st.selectbox('SELECT STOCK', stocks)

@st.cache(persist=True)
def load_data(ticker):
    data = web.DataReader(ticker, data_source='yahoo', start='2010-01-01', end='2020-01-01')
    data.reset_index(inplace=True)
    return data

data = load_data(selected_stock)

st.subheader('PAST DATA CHAT')

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data['Date'], open=data['Open'], high=data["High"], low=data["Low"], close=data["Close"]))
 
    fig.layout.update(title_text='CANDLESTICK CHAT', xaxis_rangeslider_visible=True, template="plotly_dark")
    fig.update_yaxes(type="log")
    
    st.plotly_chart(fig)

plot_raw_data()

#forecasting
#TODO: COnnect the model to the streamlit app