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
import math

def train_M(n_graph: int, n_graph_freq: int, n_train: int, n_train_freq: int, G_id0: int, model_id0: int, df):
    ## init args
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--gpu", type=str, help='gpu', default=0)
    # parser.add_argument("--dataset", type=str, help='name of dataset', default='alpha-1')
    # parser.add_argument("--back", type=int, help='back length', default=150)
    # parser.add_argument("--forward", type=int, help='forward length', default=1)
    # parser.add_argument("--year", type=int, help='testing year', default=2022)
    # args = parser.parse_args()
    # print(args)

    TSAT_parameters = TSAT_parameter('specific')
    model_params, train_params, test_params = TSAT_parameters.parameters()
    forward = 1
    num_workers = 8

    ## Check GPU is available
    train_params['device'] = torch.device(f'cuda:{0}') if torch.cuda.is_available() else torch.device('cpu')
    print(f"device using is: {train_params['device']}")

    node_features, imf_dic, adj = None, None, None
    # node_features
    node_features = np.multiply(df.to_numpy().transpose(), 10) # scale with 10
    model_params['n_nodes'] = len(node_features)
    model_params['n_output'] = len(node_features) * forward
    TSAT_model = make_TSAT_model(**model_params)
    model_interface = TrainTestInterface(TSAT_model, model_params, train_params, test_params)
    print(f"train test construction finished.")
    # read graph
    G_id = G_id0

    N_graph = math.ceil((len(df)-n_graph)/n_graph_freq)
    for i in range(N_graph):
        with open(f'G/{G_id}/imf.pkl', 'rb') as f:
            imf_dic = pickle.load(f)
        with open(f'G/{G_id}/adj.pkl', 'rb') as f:
            adj = pickle.load(f)
        G_id += 1
        print(f"read graph finished.")
        node_features_sub = None
        if i + 1 == N_graph:
            node_features_sub = node_features[:,n_graph_freq*i:]
        else:
            node_features_sub = node_features[:,n_graph_freq*i:n_graph_freq*(i+1)+n_graph]
        dataset = construct_dataset(node_features_sub, n_graph)
        print(f"dataset construction finished.")
        model_interface.import_graph(imf_dic, adj)
        print(f"graph import finished.")
        model_interface.import_base_data(node_features_sub[:,:n_graph-1])
        print(f"base data import finished.")
        model_interface.import_dataset_train(dataset, num_workers)
        print(f"train data import finished.")
        # model_interface.import_dataset_test(dataset[month_length[idx]:], num_workers)
        # print(f"test data import finished.")
        model_interface.train_model(n_graph)
    return model_interface.TSAT_model
    # exit()
    # model_interface.test_model(n_graph, factor_folder, year, month)