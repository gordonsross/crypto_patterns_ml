
import csv
import logging
import pandas as pd
import requests
import coinmetrics
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

from crypto_patterns_ml import coinmetrics_bridge


log = logging.getLogger(__name__)

def load_data_new():
    cm = coinmetrics.Community()
    supported_assets = cm.get_supported_assets()
    print("supported assets:\n", supported_assets)

    asset = "btc"
    available_data_types = cm.get_available_data_types_for_asset(asset)
    print("available data types:\n", available_data_types)

def load_df(assets):
    asset = 'btc'
    metric = 'PriceUSD,BlkCnt'
    start_date = '2019-01-01'
    end_date = '2020-01-01'

    cm = coinmetrics.Community()
    asset_data = cm.get_asset_data_for_time_range(asset, metric, start_date, end_date)
    return coinmetrics.cm_to_pandas(asset_data)    

def ml_analysis():
    df = load_df('btc')
    print(df.head())


def main():
    ml_analysis()    

if __name__ == '__main__':
    main()