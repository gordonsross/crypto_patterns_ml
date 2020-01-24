
import csv
import logging
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from pandas.plotting import register_matplotlib_converters
from sklearn.preprocessing import MinMaxScaler

from crypto_patterns_ml.data import coinmetrics_bridge

log = logging.getLogger(__name__)

def load_data(assets, cm_metric, start_date, end_date):
    return coinmetrics_bridge.get_asset_prices(assets, 'PriceUSD', start_date, end_date)

def ml_analysis():
    pass    

def view_data(df):
    register_matplotlib_converters()
    # Range selection
    # df = df['2018-01':'2018-06']
    
    # MatplotLib
    df.plot()
    plt.show()
    

def main():

    register_matplotlib_converters()

    asset = 'eth'
    cm_metric = 'PriceUSD'
    start_date = '2014-01-10'
    end_date = '2020-01-15'
    df = load_data([asset], cm_metric, start_date, end_date)
    
    # Be aware, check the type of the datetime index values
    df.index = pd.to_datetime(df.index, errors='ignore', utc=True)        

    # view_data(df)

    # Split out training data set

    dataset = df.values

    train = dataset[0:750,:]
    valid = dataset[750:,:]

    # Normalize & convert the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    x_train_raw, y_train_raw = [], []
    
    for i in range(60, len(train)):
        x_train_raw.append(scaled_data[i - 60: i, 0])
        y_train_raw.append(scaled_data[i, 0])
    
    x_train, y_train = np.array(x_train_raw), np.array(y_train_raw)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

    # Predict 246 values, using past 60 from the train data
    inputs = df[len(df) - len(valid) - 60:].values
    inputs = inputs.reshape(-1, 1)
    inputs  = scaler.transform(inputs)

    x_test = []
    for i in range(60, inputs.shape[0]):
        x_test.append(inputs[i - 60:i, 0])
    
    x_test = np.array(X_test)

    x_test = np.reshape(X_test, (x_test.shape[0], x_test.shape[1], 1))
    closing_price = model.predict(x_test)
    closing_price = scaler.inverse_transform(closing_price)

    rms = np.sqrt(np.mean(np.power((valid - closing_price), 2)))

    print(rms)

    # View results
    train = df[:750]
    valid = df[750:]
    valid['Predictions'] = closing_price
    plt.plot(train[asset])
    plt.plot(valid[[asset, 'Predictions']])
    plt.legend()
    plt.show()


if __name__ == '__main__':    
    main()
    