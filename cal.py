import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
import pickle
from PyEMD import EMD


def emd_imf(signal): # building block, no need to change
    """
    This function is to calculate EMD of the time series.
    :params: signal: list

    :return: res_dict: a dict consists of the different imf list value
    """
    if isinstance(signal, list):
        signal = np.array(signal)
    assert isinstance(signal, np.ndarray)
    IMFs = EMD().emd(signal, np.arange(len(signal)))
    res_dict = {}
    for _ in range(IMFs.shape[0]):
        res_dict[f'imf_{_}'] = IMFs[_].tolist()
    return res_dict


def calculate_imf_features(n_zones, index_pair_for_one, zones_dict: dict, ticker_SMD_dict: dict,
                           n_imf_use=5) -> np.ndarray: # building block, no need to change
    """
    compute the correlation.

    important parameter: n_imf_use (default value is 5)

    :return: imf_features
    """
    assert isinstance(n_imf_use, int)
    imf_features = np.zeros((n_zones, n_zones, n_imf_use))
    ticker_A, ticker_B = None, None
    for pair in index_pair_for_one:
        if ticker_A != zones_dict[pair[0]]:
            ticker_A = zones_dict[pair[0]]
            ticker_A_SMD = ticker_SMD_dict[ticker_A]
        if ticker_B != zones_dict[pair[1]]:
            ticker_B = zones_dict[pair[1]]
            ticker_B_SMD = ticker_SMD_dict[ticker_B]

        ef = [0] * n_imf_use
        for n_imf in list(range(1, n_imf_use + 1)):  # n_imf_to_exact = n_imf_use
            if f'imf_{n_imf}' in ticker_A_SMD and f'imf_{n_imf}' in ticker_B_SMD:
                # to get both imf for both 2 tickers
                ef[n_imf - 1] = (np.corrcoef(ticker_A_SMD[f'imf_{n_imf}'],
                                             ticker_B_SMD[f'imf_{n_imf}'])[0][1]
                )
            else:  # exit the loop when there is no further imf correctlation
                break
        imf_features[pair[0]][pair[1]], imf_features[pair[1]][pair[0]] = np.array(ef), np.array(ef)

    return imf_features


def gen_graph(df, n_lookback_days, n_imf_use, lower_bound=0.8, upper_bound=0.9):

    '''
    the main algorithm to calculate the adjacency matrix and imf matrices
    '''

    if len(df) != n_lookback_days:
        warnings.warn(
            f'The number of lookforward days ({n_lookback_days}) is not equal to the length of dataframe ({len(df)}).')

    # adjacency matrix calculation
    correlation_matrix = np.abs(np.corrcoef(df.values, rowvar=False))
    correlation_matrix = np.where(correlation_matrix >= upper_bound, 0, correlation_matrix)
    adj_mat = np.where((correlation_matrix >= lower_bound) & (correlation_matrix < upper_bound), 1, 0)

    zones = df.columns.to_list()
    n_zones, zones_dict = len(zones), dict(zip(range(len(zones)), zones))

    # calculate imf matrices
    index_pair_for_one = np.argwhere(np.triu(adj_mat) == 1)  # get the index pair form upper triangle part of adj_mat
    ticker_SMD_dict = dict.fromkeys(zones)
    involde_index_idxs_np = np.unique(index_pair_for_one.flatten())
    for index_idx in tqdm(involde_index_idxs_np):
        ticker = zones_dict[index_idx]
        ticker_SMD_dict[ticker] = emd_imf(df[ticker].to_list())

    imf_features = calculate_imf_features(n_zones, index_pair_for_one, zones_dict, ticker_SMD_dict, n_imf_use)
    
    imf_matries_dict = {}
    for i in range(n_imf_use):
        imf_matries_dict[f"imf_{i+1}_matix"] = imf_features[:, :, i]

    return adj_mat, imf_matries_dict

if __name__ == "__main__":
    # read dataframe
    with open('specific_2015-01-05_2023-06-12.pkl', 'rb') as f:
        df = pickle.load(f)
    # calculate adjacency matrix and imf matrices
    adj_mat, imf_matries_dict = gen_graph(df[:150], 150, 5)





