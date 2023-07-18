import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pickle
from utils import array_to_torch

class alphaDataset(Dataset):
    def __init__(self, xy_list):
        self.xy_list = np.array(xy_list)  # from list to numpy array

    def __len__(self):
        return len(self.xy_list)

    def __getitem__(self, key):
        if type(key) == slice:
            return alphaDataset(self.xy_list[key])
        return self.xy_list[key]


# def construct_dataset(data):  # back: back length default forward length is 1 TODO
#     len_tol = data.shape[1]
#     xy = []
#     for i in tqdm(range(len_tol - 1)):
#         xy.append((data[:, i:i + 1], data[:, i + 1:i + 2]))
#     return alphaDataset(xy)

def construct_dataset(data, back: int):  # back: back length default forward length is 1 TODO
    len_tol = data.shape[1]
    data = data[:, back - 1:]
    xy = []
    for i in tqdm(range(len_tol - back)):
        xy.append((data[:, i:i + 1], data[:, i + 1:i + 2]))
    return alphaDataset(xy)

def collate_func(batch):
    x_list, y_list = [], []
    for xy in batch:
        x_list.append(xy[0])
        y_list.append(xy[1])
    output_list = [array_to_torch(x_list), array_to_torch(y_list)]
    return output_list

if __name__ == "__main__":
    year = 2022
    month = "02"
    # 加载文件中的数组
    with open(f'../data-batch/{year}/alpha_{year}_{month}.pkl', 'rb') as f:
        df = pickle.load(f)
    node_features = df.to_numpy().transpose()
    dataset = construct_dataset(node_features, 150)
    print(len(dataset[0:1]))
    dl = DataLoader(dataset=dataset, batch_size=2)