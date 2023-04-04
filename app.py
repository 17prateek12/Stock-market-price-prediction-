import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st


st.title('Stock Prediction')


stock_symbol=st.text_input('Enter Stock Ticker','AMZN')
#last 5 years data with interval of 1 day
df = yf.download(tickers=stock_symbol,period='5y',interval='1d')
cls = df[['Close']]
ds = cls.values

#DataDescription
st.subheader('Data of last 5yrs')
st.write(df.tail())

#Visualisation
st.subheader('Closing price vs time chart')
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('closing price vs time chart 100mA')
ma100=df.Close.rolling(100).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100,label='100MA')
plt.plot(df.Close,label='Closing price')
plt.legend()

st.pyplot(fig)

st.subheader('closing price vs time chart with 100MA & 200MA')
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100,label='100MA')
plt.plot(ma200,label='200MA')
plt.plot(df.Close,label='Closing price')
plt.legend()
st.pyplot(fig)

from sklearn.preprocessing import MinMaxScaler
#Using MinMaxScaler for normalizing data between 0 & 1
normalizer = MinMaxScaler(feature_range=(0,1))
ds_scaled = normalizer.fit_transform(np.array(ds).reshape(-1,1))

#Defining test and train data sizes
train_size = int(len(ds_scaled)*0.70)
test_size = len(ds_scaled) - train_size

#Splitting data between train and test
ds_train, ds_test = ds_scaled[0:train_size,:], ds_scaled[train_size:len(ds_scaled),:1]

#creating dataset in time series for LSTM model 
#X[100,120,140,160,180] : Y[200]
def create_ds(dataset,step):
    Xtrain, Ytrain = [], []
    for i in range(len(dataset)-step-1):
        a = dataset[i:(i+step), 0]
        Xtrain.append(a)
        Ytrain.append(dataset[i + step, 0])
    return np.array(Xtrain), np.array(Ytrain)

#Taking 100 days price as one record for training
time_stamp = 100
X_train, y_train = create_ds(ds_train,time_stamp)
X_test, y_test = create_ds(ds_test,time_stamp)

#Reshaping data to fit into LSTM model
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

#loading model

model = load_model('LSTM_model.h5')

#Predicitng on train and test data
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

#Inverse transform to get actual value
train_predict = normalizer.inverse_transform(train_predict)
test_predict = normalizer.inverse_transform(test_predict)

#Combining predicted data and visualiszing
test = np.vstack((train_predict,test_predict))
fig2=plt.figure(figsize=(12,6))
plt.plot(normalizer.inverse_transform(ds_scaled),label='Actual')
plt.plot(test,label='Test')
plt.legend()
st.pyplot(fig2)


#Getting the last 100 days records
fut_inp = ds_test[(len(ds_test)-100):]
fut_inp = fut_inp.reshape(1,-1)
tmp_inp = list(fut_inp)

#Creating list of the last 100 data
tmp_inp = tmp_inp[0].tolist()

#Predicting next 30 days price suing the current data
#It will predict in sliding window manner (algorithm) with stride 1
lst_output=[]
n_steps=100
i=0
while(i<30):
    
    if(len(tmp_inp)>100):
        fut_inp = np.array(tmp_inp[1:])
        fut_inp=fut_inp.reshape(1,-1)
        fut_inp = fut_inp.reshape((1, n_steps, 1))
        yhat = model.predict(fut_inp, verbose=0)
        tmp_inp.extend(yhat[0].tolist())
        tmp_inp = tmp_inp[1:]
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        fut_inp = fut_inp.reshape((1, n_steps,1))
        yhat = model.predict(fut_inp, verbose=0)
        tmp_inp.extend(yhat[0].tolist())
        lst_output.extend(yhat.tolist())
        i=i+1
    

print(lst_output)

#Creating a dummy plane to plot graph one after another
plot_new=np.arange(1,101)
plot_pred=np.arange(101,131)


fig3=plt.figure(figsize=(12,6))
plt.plot(plot_new, normalizer.inverse_transform(ds_scaled[len(ds_scaled)-100:]),label='Trained data')
plt.plot(plot_pred, normalizer.inverse_transform(lst_output),label='Next 30 days')
plt.legend()
st.pyplot(fig3)

ds_new = ds_scaled.tolist()
#Entends helps us to fill the missing value with approx value
ds_new.extend(lst_output)
fig3=plt.figure(figsize=(12,6))
plt.plot(ds_new[1200:])
st.pyplot(fig3)

#Creating final data for plotting
final_graph = normalizer.inverse_transform(ds_new).tolist()

#Plotting final results with predicted value after 30 Days
fig3=plt.figure(figsize=(12,6))
plt.plot(final_graph,)
plt.ylabel("Price")
plt.xlabel("Time")
plt.title("{0} prediction of next month close".format(stock_symbol))
plt.axhline(y=final_graph[len(final_graph)-1], color = 'red', linestyle = ':', label = 'NEXT 30D: {0}'.format(round(float(*final_graph[len(final_graph)-1]),2)))
plt.legend()
st.pyplot(fig3)

day_input = st.text_input('Enter nth day to predict :',30)
print(int(day_input)+1)
print(len(ds_new))
print(len(final_graph)-30+int(day_input))
string1 = 'Predicted values after '+day_input+' days: '+str(final_graph[len(final_graph)-30+int(day_input)-1])
print(string1)
st.subheader(string1)
