import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

train_data = pd.read_csv('StockMarket/SP500_train.csv')
test_data = pd.read_csv('StockMarket/SP500_test.csv')

training_target = train_data.iloc[:,5:6].values
test_target = test_data.iloc[:,5:6].values

minMaxScalar = MinMaxScaler(feature_range=(0,1))
training_target = minMaxScalar.fit_transform(training_target)
test_target = minMaxScalar.fit_transform(test_target)

x_train = []
y_train = []

for i in range(40,len(train_data)):
#    0 is the column index
#    we are using previous 20 prices in order to forecast next one
    x_train.append(training_target[i-40:i,0])
    y_train.append(training_target[i,0])
    
X_train = np.array(x_train)
y_train = np.array(y_train)

# (number of samples, num of features, 1)
X_train = np.reshape(X_train,(X_train.shape[0], X_train.shape[1],1))

model = Sequential()
model.add(LSTM(units=100, return_sequences = True, input_shape=(X_train.shape[1],1)))
model.add(Dropout(0.5))
model.add(LSTM(units=50, return_sequences = True))
model.add(Dropout(0.3))
model.add(LSTM(units = 50))
model.add(Dropout(0.3))
model.add(Dense(units=1))

model.compile(optimizer='SGD',loss='mean_squared_error')
model.fit(X_train,y_train,epochs=100,batch_size=32)

total_data = pd.concat((train_data['adj_close'],test_data['adj_close']),axis=0)
features = total_data[len(total_data)-len(test_data)-40:].values
features = features.reshape(-1,1)
features = MinMaxScaler.transform(features)

X_test = []
for i in range(40,60):
    X_test.append(features[i-40:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))

y_pred = model.predict(X_test)

y_pred = MinMaxScaler.inverse_transform(y_pred)


