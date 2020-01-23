"""
 Coinmetrics Bridge/Interface
    https://coinmetrics.io/
    https://coinmetrics.readthedocs.io/en/latest/index.html
    https://coinmetrics.readthedocs.io/en/latest/community.html#primary-methods

"""

import coinmetrics
import pandas as pd
import logging

log = logging.getLogger(__name__)
cm = coinmetrics.Community()

def get_supported_assets():
    return cm.get_supported_assets()

def get_asset_info(assets):
    asset_info = cm.get_asset_info(assets)
    return coinmetrics.cm_to_pandas(asset_info)

def get_asset_prices(assets, ccy_pair, start_date, end_date):
    
    results_df = None

    for asset in assets:
        asset_prices = cm.get_asset_data_for_time_range(asset, ccy_pair, start_date, end_date)        
        if isinstance(results_df, pd.DataFrame):
            tmp_df = coinmetrics.cm_to_pandas(asset_prices).rename(columns={'PriceUSD': asset})            
            results_df = pd.concat([tmp_df, results_df], axis=1, sort=True)                 
        else:
            results_df = coinmetrics.cm_to_pandas(asset_prices).rename(columns={'PriceUSD': asset})             

    return results_df

if __name__ == '__main__':    
    # print(get_supported_assets())
    # print(get_asset_info('btc,ada').head())
    print(get_asset_prices(['btc','ada','xrp'], 'PriceUSD', '2017-10-01', '2020-01-23'))
    
