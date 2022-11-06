import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

#Loading Data
company = 'AAPL'

start = dt.datetime(2010,1,1)
end = dt.datetime(2020,1,1)

data = web.DataReader(company, 'yahoo', start, end)

#Prepare Data
scaler = MinMaxScaler(feature_range=(0,1)) #change all the values to be between 0 and 1
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1)) #reshape the data to be 2D
#only predict the closing price

prediction_days = 60 #how many days we want to use to predict the next day


#training data
x_train = []
y_train = []

#fill up the x_train and y_train
for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

#convert to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

#reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#Build the model
model = Sequential()

#add the first layer
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))

#add the second layer
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

#add the third layer
model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(25))
#add the output layer
model.add(Dense(units=1)) #prediction of the next closing value

#compile the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32)

#Testing the model
#Load the test data - how it would perfom on past data
test_start = dt.datetime(2020,1,1)
test_end = dt.datetime(2023,1,1)

#load the data
test_data = web.DataReader(company, 'yahoo', test_start, test_end)
actual_prices = test_data['Close'].values

#Get the predicted prices
total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

#model inputs
model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values

#reshape the data
model_inputs = model_inputs.reshape(-1,1)

#scale the data
model_inputs = scaler.transform(model_inputs)

#Predict Next Day
real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs+1), 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

#Get the predicted price
prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"Prediction: {prediction}")

#TODO: Predict the next 30 days


