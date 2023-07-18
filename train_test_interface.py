import argparse
from collections import defaultdict
from dataset import collate_func
import numpy as np
from sklearn.model_selection import train_test_split
from TSAT import make_TSAT_model
import torch
from torch.utils.data import DataLoader
from utils import TSAT_parameter, loss_function, calculate_loss, array_to_torch
import pickle


class TrainTestInterface():

    def __init__(self, TSAT_model, model_params: dict, train_params: dict, test_params: dict) -> None:
        self.TSAT_model = TSAT_model
        if torch.cuda.device_count() > 1:  # for parallel training using multiple GPUs, batch size shud >= #GPU
            print(f"There are {torch.cuda.device_count()} GPUs. We will run parallelly.")
            self.TSAT_model = torch.nn.DataParallel(self.TSAT_model)
        else:
            self.TSAT_model = self.TSAT_model.to(train_params['device'])  # send the model to GPU
        self.train_params = train_params
        self.model_params = model_params
        self.test_params = test_params
        self.criterion = loss_function(train_params['loss_function'])
        self.metric = loss_function(train_params["metric"])
        self.optimizer = torch.optim.Adam(self.TSAT_model.parameters())

    def import_dataset_train(self, dataset, num_workers) -> None:
        # Data Loader
        self.train_loader = DataLoader(dataset=dataset, batch_size=self.train_params['batch_size'],
                                       collate_fn=collate_func,
                                       shuffle=False,
                                       drop_last=False, num_workers=num_workers, pin_memory=False)

    def import_dataset_test(self, dataset, num_workers) -> None:
        # Data Loader
        self.test_loader = DataLoader(dataset=dataset, batch_size=self.test_params['batch_size'],
                                      collate_fn=collate_func,
                                      shuffle=False,
                                      drop_last=False, num_workers=num_workers, pin_memory=False)

    def import_graph(self, imf_dic, adj) -> None:
        self.imf_1_matrices = array_to_torch(imf_dic['imf_1_matix'])
        self.imf_2_matrices = array_to_torch(imf_dic['imf_2_matix'])
        self.imf_3_matrices = array_to_torch(imf_dic['imf_3_matix'])
        self.imf_4_matrices = array_to_torch(imf_dic['imf_4_matix'])
        self.imf_5_matrices = array_to_torch(imf_dic['imf_5_matix'])
        self.adjacency_matrix = array_to_torch(adj)

    def import_base_data(self, data):
        self.base = array_to_torch(data)

    def calculate_number_of_parameter(self) -> int:
        model_parameters = filter(lambda p: p.requires_grad, self.TSAT_model.parameters())
        return int(sum([np.prod(p.size()) for p in model_parameters]))

    def view_train_params(self):
        # view the training parameters
        return self.train_params

    def view_model_params(self):
        # view the model parameters
        return self.model_params

    def train_model(self, back) -> None:
        # training start
        self.TSAT_model.train()
        print(f'batches in train: {len(self.train_loader)}')
        for idx, batch in enumerate(self.train_loader):
            x, y = batch
            mask_x = torch.isnan(x)  # 创建一个布尔掩码，标记nan值的位置
            x = torch.where(mask_x, torch.zeros_like(x), x)  # 将nan值替换为0
            mask_y = torch.isnan(y)  # 创建一个布尔掩码，标记nan值的位置
            y = torch.where(mask_y, torch.zeros_like(y), y)  # 将nan值替换为0
            # print(x.size(), "x size")
            node_features = torch.cat((self.base[:, -back + 1:].unsqueeze(0).repeat(batch[0].size()[0], 1, 1), x),
                                          dim=2).float()
            # mask = torch.isnan(node_features)  # 创建一个布尔掩码，标记nan值的位置
            # node_features = torch.where(mask, torch.zeros_like(node_features), node_features)  # 将nan值替换为0
            self.base = torch.cat((self.base[:, -back + 1:], x[0]), dim=1)
            adjacency_matrix = self.adjacency_matrix.unsqueeze(0).repeat(batch[0].size()[0], 1, 1).to(
                self.train_params['device'])  # (batch, max_length, max_length)
            node_features = node_features.to(self.train_params['device'])  # (batch, max_length, d_node)
            imf_1_matrices = self.imf_1_matrices.unsqueeze(0).to(self.train_params['device'])  # (batch, max_length, max_length)
            imf_2_matrices = self.imf_2_matrices.unsqueeze(0).to(self.train_params['device'])  # (batch, max_length, max_length)
            imf_3_matrices = self.imf_3_matrices.unsqueeze(0).to(self.train_params['device'])  # (batch, max_length, max_length)
            imf_4_matrices = self.imf_4_matrices.unsqueeze(0).to(self.train_params['device'])  # (batch, max_length, max_length)
            imf_5_matrices = self.imf_5_matrices.unsqueeze(0).to(self.train_params['device'])  # (batch, max_length, max_length)
            y_true = y.flatten().to(self.train_params['device']).float()  # (batch, task_numbers)
            batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0  # (batch, max_length)
            # print(torch.any(torch.isnan(node_features)))
            # print(torch.any(torch.isnan(batch_mask)))
            # print(torch.any(torch.isnan(adjacency_matrix)))
            # print(torch.any(torch.isnan(imf_1_matrices)))
            # print(torch.any(torch.isnan(imf_2_matrices)))
            # print(torch.any(torch.isnan(imf_3_matrices)))
            # print(torch.any(torch.isnan(imf_4_matrices)))
            # print(torch.any(torch.isnan(imf_5_matrices)))

            # print(node_features)
            # print(batch_mask)
            # print(adjacency_matrix)
            # print(torch.all(adjacency_matrix == 0))
            # exit(torch.all(imf_1_matrices == 0))
            y_pred_normalization = self.TSAT_model(
                node_features, batch_mask, adjacency_matrix, imf_1_matrices, imf_2_matrices, imf_3_matrices,
                imf_4_matrices, imf_5_matrices
            )
            # print(torch.any(torch.isnan(y_pred_normalization)))
            # print(y_pred_normalization[0][0], y_true[0])
            # print(torch.abs(y_true).mean())
            # print(y_true[0])
            loss = calculate_loss(y_true, y_pred_normalization, self.train_params['loss_function'],
                                  self.criterion, self.train_params['device'])
            # print(torch.any(torch.isnan(loss)))
            # print(torch.any(torch.isnan(y_true)))
            # exit()
            # print(y_pred_normalization.flatten()[100], y_true.flatten()[100])
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if idx % 1 == 0:
                loss, current = loss.item(), (idx + 1)
                # print(f"loss: {loss:>7f}  [{current:>5d}/{len(self.train_loader):>5d}]")
            del adjacency_matrix
            del node_features
            del batch_mask
            del imf_1_matrices
            del imf_2_matrices
            del imf_3_matrices
            del imf_4_matrices
            del imf_5_matrices
            del y_true
            del y_pred_normalization
            torch.cuda.empty_cache()

    def test_model(self, back, factor_folder: str, year: str, month: str) -> None:
        # testing start
        num_batches = len(self.test_loader)
        self.TSAT_model.eval()
        test_loss = 0
        factor = None
        # load model and no_grad
        with torch.no_grad():
            for idx, batch in enumerate(self.test_loader):
                x, y = batch
                node_features = torch.cat((self.base[:, -back + 1:].unsqueeze(0).repeat(batch[0].size()[0], 1, 1), x),
                                          dim=2).float()
                self.base = torch.cat((self.base[:, -back + 1:], x[0]), dim=1)
                adjacency_matrix = self.adjacency_matrix.unsqueeze(0).repeat(batch[0].size()[0], 1, 1).to(
                    self.train_params['device'])  # (batch, max_length, max_length)
                node_features = node_features.to(self.train_params['device'])  # (batch, max_length, d_node)
                imf_1_matrices = self.imf_1_matrices.unsqueeze(0).to(self.train_params['device'])  # (batch, max_length, max_length)
                imf_2_matrices = self.imf_2_matrices.unsqueeze(0).to(self.train_params['device'])  # (batch, max_length, max_length)
                imf_3_matrices = self.imf_3_matrices.unsqueeze(0).to(self.train_params['device'])  # (batch, max_length, max_length)
                imf_4_matrices = self.imf_4_matrices.unsqueeze(0).to(self.train_params['device'])  # (batch, max_length, max_length)
                imf_5_matrices = self.imf_5_matrices.unsqueeze(0).to(self.train_params['device'])  # (batch, max_length, max_length)
                y_true = y.to(self.train_params['device']).float()  # (batch, task_numbers)
                batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0  # (batch, max_length)
                y_pred_normalization = self.TSAT_model(
                    node_features, batch_mask, adjacency_matrix, imf_1_matrices, imf_2_matrices, imf_3_matrices,
                    imf_4_matrices, imf_5_matrices
                )
                print(y_pred_normalization.flatten()[100], y_true.flatten()[100])
                # print(y_true.flatten()[-1])
                # print(node_features[0][0][-1])
                # save
                if idx == 0:
                    factor = y_pred_normalization.flatten().unsqueeze(0)
                else:
                    factor = torch.cat((factor, y_pred_normalization.flatten().unsqueeze(0)), dim=0)
                # torch.save(y_true_normalization,
                #            f"../result/{year}/{_train_params['loss_function']}/alpha_factor_{year}_tensor_{bat}_true.pt")
                # y_pred_normalization.cpu().detach().numpy()
                loss = calculate_loss(y_true, y_pred_normalization, self.train_params['metric'],
                                      self.metric, self.train_params['device'])
                test_loss += loss
                del adjacency_matrix
                del node_features
                del batch_mask
                del imf_1_matrices
                del imf_2_matrices
                del imf_3_matrices
                del imf_4_matrices
                del imf_5_matrices
                del y_true
                del y_pred_normalization
                torch.cuda.empty_cache()
            with open(f'{factor_folder}/{year}/{month}.pkl', 'wb') as f:
                pickle.dump(factor.cpu().detach().numpy(), f)
            test_loss /= num_batches
        print(f'num of batches: {num_batches}')
        print(f"Avg loss: {test_loss:>8f} \n")
