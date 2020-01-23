
import csv
import logging
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler

from crypto_patterns_ml.data import coinmetrics_bridge

log = logging.getLogger(__name__)

def load_data(assets, cm_metric, start_date, end_date):
    return coinmetrics_bridge.get_asset_prices(assets, 'PriceUSD', start_date, end_date)

def ml_analysis():
    pass    

def main():
    assets = ['btc', 'ada', 'xrp']
    cm_metric = 'PriceUSD'
    start_date = '2019-01-01'
    end_date = '2020-01-22'
    df = load_data(assets, cm_metric, start_date, end_date)
    print(df.head())

if __name__ == '__main__':    
    main()
    