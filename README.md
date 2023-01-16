# Stock-market-price-prediction-
The project consists of two different models: SVR and LSTM, which predict the prices of different companies. Objective of this project is find accuracy of both model on different span of data.

I have created the first SVR model that predicts the high, low, and adjclose parameters, and the second Lstm model which predict and calculate accuracy of adjclose and also predict the adjclose of next n number of days.

I have collected data from Tiingo, from 2018-2023, and from 2011-2023 of Tesla, and observed which model performed better on which data. 
#################RESULTS FROM SVR###################

AdjClose
The actual price: 122.4
The RBF SVR predicted : 111.54818098
The accuracy of the model is : 89.05228758169935

High
The RBF SVR predicted : 126.97796888
The actual price: 122.63
The accuracy of the model is : 97.25189594715812

Low
The RBF SVR predicted : 119.47112921
The actual price: 115.6
The accuracy of the model is : 97.05882352941175

NOTE# These accuracy is for data of TESLA from 2018-01-17 to 2023-01-13

AdjClose
The actual price: 122.4
The RBF SVR predicted : 137.36054329
The accuracy of the model is : 88.07189542483661

High
The RBF SVR predicted : 2.88604824
The actual price: 122.63
The accuracy of the model is : 1.6309222865530444

Low
The RBF SVR predicted : 6.15993094
The actual price: 115.6
The accuracy of the model is : 5.190311418685127

NOTE# These accuracy is for data of TESLA from 2011-01-18 to 2023-01-13

#################RESULTS FROM LSTM###################

The accuracy for AdjClose is : 80.14157881339638 for data of TESLA from 2018-01-17 to 2023-01-13

The accuracy for AdjClose is : 97.20555127257244 for data of TESLA from 2011-01-18 to 2023-01-13

For references and guidances search on google and youtube

