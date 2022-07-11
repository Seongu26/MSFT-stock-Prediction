import numpy as np
import matplotlib.pyplot as plt
#plt.style.use('fivethirtyeight')
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional,Lambda
import math
from sklearn.metrics import mean_squared_error
from keras.optimizers import SGD
import warnings
warnings.filterwarnings('ignore') #Ignore warnings

raw_data = pd.read_csv('C:/Users/sungu/Desktop/MSFT.csv', index_col='Date', parse_dates=['Date'])
df = raw_data.copy()

print(df)

training_set = raw_data[:'2018'].iloc[:,1:2].values
test_set = raw_data['2019':].iloc[:,1:2].values

print(training_set.shape)
print(test_set.shape)

raw_data["High"][:'2018'].plot(figsize=(12,6),legend=True)
raw_data["High"]['2019':].plot(figsize=(12,6),legend=True)

plt.legend(['Training set (Before 2018)','Test set (2018 and beyond)'])
plt.title('MSFT stock price')
plt.show()

def plot_predictions(test,predicted):
    plt.plot(test, color='red',label='Real MicroSoft Stock Price')
    plt.plot(predicted, color='blue',label='Predicted MicroSoft Stock Price')
    plt.title('MicroSoft Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('MicroSoft Stock Price')
    plt.legend()
    plt.show()

def return_rmse(test,predicted):
    mse = mean_squared_error(test, predicted)
    print("The mean squared error is {}.".format(mse))

# Scale data
scaler = MinMaxScaler(feature_range=(0,1))
data_train = scaler.fit_transform(training_set)
data_test = scaler.transform(test_set)
print(data_train.shape)

# Build X and y
X_train = []
y_train = []
for i in range(60,8269):
    X_train.append(data_train[i-60:i,0])
    y_train.append(data_train[i,0])
X_train, y_train = np.array(X_train), np.array(y_train)


# LSTM model
model = Sequential()
model.add(LSTM(units=64, return_sequences = True,
                input_shape=(X_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units = 64))
model.add(Dropout(0.2))
model.add(Dense(units = 1))

#Compile model
model.compile(loss='mse', optimizer='adam')
model.summary()
model.fit(X_train,y_train,epochs=5,batch_size=32)

dataset_total = pd.concat((raw_data["High"][:'2018'],raw_data["High"]['2019':]),axis=0)
inputs = dataset_total[len(dataset_total)-len(data_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)
# Preparing X_test and predicting the prices
X_test = []
for i in range(60,874):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
predicted_stock_price = model.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

plot_predictions(test_set,predicted_stock_price)
return_rmse(test_set,predicted_stock_price)

#model 2 mixed, LSTM and GRU

model2 = Sequential()
# First LSTM layer with Dropout regularisation
model2.add(LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1],1)))
model2.add(Dropout(0.2))
model2.add(GRU(units=64, return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))
model2.add(Dropout(0.2))
model2.add(GRU(units=64, activation='tanh'))
model2.add(Dropout(0.2))
# The output layer
model2.add(Dense(units=1))
# Compiling the RNN
model2.compile(optimizer=SGD(lr=0.01, decay=1e-7, momentum=0.9, nesterov=False),loss='mse')
# Fitting to the training set
model2.fit(X_train,y_train,epochs=5,batch_size=150)

X_test1 = []
for i in range(60,874):
    X_test1.append(inputs[i-60:i,0])
X_test1 = np.array(X_test1)
X_test1 = np.reshape(X_test1, (X_test1.shape[0],X_test1.shape[1],1))
Mixed_predicted_stock_price = model2.predict(X_test1)
Mixed_predicted_stock_price = scaler.inverse_transform(Mixed_predicted_stock_price)

# Visualizing the results for Mixed model
plot_predictions(test_set,Mixed_predicted_stock_price)
return_rmse(test_set,Mixed_predicted_stock_price)

# LSTM is best for prediction.

