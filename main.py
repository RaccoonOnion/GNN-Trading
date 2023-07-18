import argparse
import numpy as np
import time
import torch
from torch.utils.data import DataLoader
from utils import TSAT_parameter, loss_function, calculate_loss
from TSAT import make_TSAT_model
from dataset import construct_dataset
from train_test_interface import TrainTestInterface
import pickle

data_folder = "../data-batch"
factor_folder = "../factors"
num_workers = 8

if __name__ == '__main__':
    ## init args
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, help='gpu', default=0)
    parser.add_argument("--dataset", type=str, help='name of dataset', default='alpha-1')
    parser.add_argument("--back", type=int, help='back length', default=150)
    parser.add_argument("--forward", type=int, help='forward length', default=1)
    parser.add_argument("--year", type=int, help='testing year', default=2022)
    args = parser.parse_args()
    print(args)

    TSAT_parameters = TSAT_parameter(args.dataset)
    model_params, train_params, test_params = TSAT_parameters.parameters()

    ## Check GPU is available
    train_params['device'] = torch.device(f'cuda:{args.gpu}') if torch.cuda.is_available() else torch.device('cpu')
    print(f"device using is: {train_params['device']}")
    year = args.year
    month_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
    month_length_last = [20, 15, 23, 21, 18, 21, 22, 22, 20, 16, 22, 23]
    month_length = [23, 19, 16, 23, 19, 19, 21, 21, 23, 21, 16, 22, 22]

    node_features, imf_dic, adj = None, None, None
    train_len = 1
    for idx, month in enumerate(month_list):
        # read dataframe
        with open(f'{data_folder}/{year}/{train_len}/alpha_{year}_{month}.pkl', 'rb') as f:
            df = pickle.load(f)
        print(f"read original data finished.")
        # node_features
        node_features = np.multiply(df.to_numpy().transpose(), 1)
        # read graph
        with open(f'{data_folder}/{year}/G/imf_{month}.pkl', 'rb') as f:
            imf_dic = pickle.load(f)
        with open(f'{data_folder}/{year}/G/adj_{month}.pkl', 'rb') as f:
            adj = pickle.load(f)

        print(f"read graph finished.")
        model_params['n_nodes'] = len(node_features)
        model_params['n_output'] = len(node_features) * args.forward

        dataset = construct_dataset(node_features, args.back)
        # print(f"size of node_features: {node_features.shape}, size of dataset: {len(dataset)}")
        # exit()
        print(f"dataset construction finished.")
        TSAT_model = make_TSAT_model(**model_params)
        model_interface = TrainTestInterface(TSAT_model, model_params, train_params, test_params)
        print(f"train test construction finished.")
        model_interface.import_graph(imf_dic, adj)
        print(f"graph import finished.")
        model_interface.import_base_data(node_features[:,:args.back-1])
        print(f"base data import finished.")
        model_interface.import_dataset_train(dataset[:month_length[idx]], num_workers)
        print(f"train data import finished.")
        model_interface.import_dataset_test(dataset[month_length[idx]:], num_workers)
        print(f"test data import finished.")
        model_interface.train_model(args.back)
        model_interface.test_model(args.back, factor_folder, year, month)