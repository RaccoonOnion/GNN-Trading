from cal import gen_graph
import pickle
import os
import math
from train_model import train_M
import torch
# this is the function to cut the data into pieces we like
def split_data(n_graph: int, n_graph_freq: int, n_train: int, n_train_freq: int, data_df):
    # os.mkdir("./G/")
    # os.mkdir("./XY/")
#     os.mkdir("./Test/")
    assert len(data_df) > n_graph + n_train
    N_graph = 1 + math.floor((len(data_df)-n_graph) / n_graph_freq)
    N_train = 1 + math.floor((len(data_df)-n_graph-n_train) / n_train_freq)
    # print(N_graph, N_train) # 64 62
    # print(N_train)
    # for i in range(N_graph):
        # os.mkdir(f"./G/{i}")
        # df = data_df.iloc[i*n_graph_freq:i*n_graph_freq + n_graph,:]
#         print(len(df))
#         print(i*n_graph_freq, i*n_graph_freq + n_graph)
#         G = cal_graph(df)
#         save(G, f"./G/{i}")
        # adj_mat, imf_matries_dict = gen_graph(df, n_graph)
        # # 保存数组到文件
        # with open(f'./G/{i}/imf.pkl', 'wb') as f:
        #     pickle.dump(imf_matries_dict, f)
        # # 保存数组到文件
        # with open(f'./G/{i}/adj.pkl', 'wb') as f:
        #     pickle.dump(adj_mat, f)
    # exit()
    for j in range(N_train):
        # os.mkdir(f"./XY/{j}/")
        j = N_train - 1
        # os.mkdir(f"./M/{j}/")
        df = data_df.iloc[j*n_train_freq:n_graph+j*n_train_freq + n_train,:]
#         print(len(df))
#         print(n_graph+j*n_train_freq, n_graph+j*n_train_freq + n_train)
        # with open(f'./XY/{j}/df.pkl', 'wb') as f:
        #     pickle.dump(df, f)
        
        M = train_M(n_graph, n_graph_freq, n_train, n_train_freq, 0, 0, df)
        # torch.save(M.state_dict(), f'./M/{j}/model.pt')
        exit()
        # save(M, f"./M/{i}")

with open("specific_2015-01-05_2023-06-12.pkl", "rb") as f:
    data_df = pickle.load(f)
split_data(150, 30, 60, 30, data_df)