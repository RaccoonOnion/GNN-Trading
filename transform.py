import pickle
import pandas as pd
import numpy as np



year = 2022
month_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

if __name__ == "__main__":
    # for month in month_list:
    #     with open(f'../factors/{year}/{month}.pkl', 'rb') as f:
    #         arr = pickle.load(f)
    #     with open(f'../data-batch/{year}/parts/stocks_{month}.pkl', 'rb') as f:
    #         stocks = pickle.load(f)
    #     with open(f'../data-batch/{year}/parts/dates_{month}.pkl', 'rb') as f:
    #         dates = pickle.load(f)
    #     fac_df = pd.DataFrame(arr, columns=stocks, index=dates)
    #     fac_df = fac_df.astype(np.float64)  # 回测框架需要数据类型为'npy_float64'
    #     fac_df.to_pickle(f'../factors/{year}/{month}_df.pkl')

    # combine factors
    for idx, month in enumerate(month_list):
        with open(f'../factors/{year}/{month}_df.pkl', 'rb') as f:
            df = pickle.load(f)
        if idx == 0:
            combo = df
        else:
            combo = pd.concat([combo, df], axis=0).fillna(np.nan)
    combo.to_pickle(f'../factors/{year}/combo.pkl')