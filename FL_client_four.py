'''
用来建立与客户端相关的操作
'''
from enum import Enum
from typing import Tuple, List, Dict, Iterable, Any, Optional
#from tasks.graph_task_combine import DataFold
import torch.nn as nn
import torch
import numpy as np
import time
from tqdm import tqdm
import copy
from torch_geometric.data import DataLoader
from utils import cal_metrics, top_5, cal_early_stopping_metric, pretty_print_epoch_task_metrics
import torch.nn.functional as F
from utils import TaskFold, TaskIndexFold
from models import name_to_dataset_class, name_to_model_class


class SingleClient:
    def __init__(self,
                 client_id,
                 data: List[Any],
                 log_file,
                 task_index: TaskIndexFold,
                 task_id:TaskFold.VARMISUSE,
                 client_max_epochs: int,
                 client_batch_size: int,
                 make_task_input,
                 dataLoader_numworkers: int = 0,
                 device: str = "cuda"):
        self.data = data
        #self.epochs = epochs
        self.client_name = "client%d" % client_id
        self.log_file = log_file
        
        self.task_index = task_index
        self.task_id = task_id
        self.client_max_epochs = client_max_epochs
        self.client_batch_size = client_batch_size
        self.make_task_input = make_task_input
        self.device = device

        self.batch_iterator = DataLoader(self.data,
                                         batch_size=self.client_batch_size,
                                         shuffle=True,
                                         num_workers=dataLoader_numworkers)
        #num_workers=int(cpu_count()/2))

    def log_line(self, msg):
        with open(self.log_file, 'a') as log_fh:
            log_fh.write(msg + '\n')
        print(msg)

    def update_weights(
        self,
        model: nn.Module,
        optimizer: torch.optim,
        learning_rate: int,
        #batch_norm_dict: dict,
        quiet: bool = True,
    ):
        #self.model = model
        '''
        if batch_norm_dict is not None:
            # 如果batchNormdict不是空，则更新bn。这个bn为上一轮保存的
            for key in batch_norm_dict:
                self.model.state_dict()[key].data.copy_(batch_norm_dict[key])
        self.model.train()

        '''
        '''
        if batch_norm_dict is not None:
            # 如果batchNormdict不是空，则更新bn。这个bn为上一轮保存的
            for key in batch_norm_dict:
                model.state_dict()[key].data.copy_(batch_norm_dict[key])
        '''


        cur_optimizer = optimizer(model.parameters(), lr=learning_rate)

        train_loss_list = []
        train_task_metrics_list = []
        train_num_graphs_sum = 0
        train_num_batchs_sum = 0
        train_graphs_p_s, train_nodes_p_s, train_graphs, train_nodes = 0, 0, 0, 0
        for epoch in tqdm(range(1, self.client_max_epochs + 1)):
            #self.log_line("\t== %s Epoch %i" % (self.client_name, epoch))

            train_loss, train_task_metrics, train_num_graphs, train_num_batchs, train_graphs_p_s, train_nodes_p_s, train_graphs, train_nodes = self.__run_epoch(
                "%s epoch %i (training)" % (self.client_name, epoch), model, cur_optimizer, quiet=quiet)
            if not quiet:
                print("\r\x1b[K", end='')  #该函数意义将光标回到该行开头，并擦除整行。
                print("\033[A", end='')  #光标回到上一行。

            train_loss_list.append(train_loss)
            train_task_metrics_list.extend(train_task_metrics)
            train_num_graphs_sum += train_num_graphs
            train_num_batchs_sum += train_num_batchs

        self.log_line(
            "\t%s %s Train: loss: %.5f || %s || graphs/sec: %.2f | nodes/sec: %.0f | graphs: %.0f | nodes: %.0f " %
            (self.client_name, self.task_id, sum(train_loss_list) / len(train_loss_list),
             pretty_print_epoch_task_metrics(train_task_metrics_list, train_num_graphs_sum, train_num_batchs_sum),
             train_graphs_p_s, train_nodes_p_s, train_graphs, train_nodes))
        
        # get bn_params
        '''
        bn_params_dict = {}
        for key in model.state_dict().keys():
            if 'norm' in key:
                #print(key)
                bn_params_dict[key] = copy.deepcopy(model.state_dict()[key])
        '''
        
        cur_optimizer.zero_grad()
        return model.state_dict(), sum(train_loss_list) / len(train_loss_list), None#, bn_params_dict

    def __run_epoch(
        self,
        epoch_name: str,
        model,
        optimizer,
        quiet: Optional[bool] = False,
    ):
        """具体的每一轮训练。

        Args:
            epoch_name (str): 每一轮名称
            data (Iterable[Any]): 该轮的数据
            data_fold (DataFold): 是test还是train
            quiet (Optional[bool], optional): 当为真时，不显示任何信息。. Defaults to False.

        Returns:
            Tuple[float]: [description]
        """

        start_time = time.time()
        processed_graphs, processed_nodes, processed_batch = 0, 0, 0
        epoch_loss = 0.0
        task_metric_results = []

        step = 0
        for batch_data in self.batch_iterator:
            batch_data = batch_data.to(self.device)

            processed_graphs += batch_data.num_graphs
            processed_nodes += batch_data.num_nodes
            processed_batch += 1
            step += 1

            optimizer.zero_grad()
            model.train()

            task_batch_data = self.make_task_input(batch_data)
            logits = model(**task_batch_data, task_index=self.task_index)

            # TODO: 根据task进行任务的区分。
            if self.task_id == TaskFold.VARMISUSE:
                loss, metrics = self.criterion(logits, batch_data.label)
            elif self.task_id ==TaskFold.VARNAMING:
                loss, metrics = self.criterion(logits, batch_data.value_label)

            epoch_loss += loss.item()
            task_metric_results.append(metrics)

            loss.backward()
            optimizer.step()

            if not quiet:
                print("Runing %s, batch %i (has %i graphs). Loss so far: %.4f" %
                      (epoch_name, step, batch_data.num_graphs, epoch_loss / processed_batch),
                      end="\r")

        epoch_time = time.time() - start_time
        per_graph_loss = epoch_loss / processed_batch
        graphs_per_sec = processed_graphs / epoch_time
        nodes_per_sec = processed_nodes / epoch_time

        return per_graph_loss, task_metric_results, processed_graphs, processed_batch, graphs_per_sec, nodes_per_sec, processed_graphs, processed_nodes

    def criterion(self, y_score, y_true, criterion=torch.nn.CrossEntropyLoss()):
        loss = criterion(y_score, y_true)
        metrics = cal_metrics(F.softmax(y_score, dim=-1), y_true)
        return loss, metrics


def get_client_params(client_nums: int, data_nums: int, mode: str = "average") -> Dict:
    if mode.lower() == "iid":
        client_params = get_iid(client_nums, data_nums)
    if mode.lower() == "average":
        client_params = get_average(client_nums, data_nums)
    return client_params


def get_iid(client_nums: int, data_nums: int) -> Dict:
    # 这里根据id数量、data数量返回 data index
    data_params = dict()
    remain_data_set = list(range(data_nums))
    each_client_data_size = data_nums // client_nums  # 取整。
    for i in range(client_nums):
        data_params[i] = set(np.random.choice(remain_data_set, size=each_client_data_size, replace=False))
        remain_data_set = list(set(remain_data_set) - data_params[i])
    return data_params


def get_average(client_nums: int, data_nums: int) -> Dict:
    # 直接平均分配
    data_params = dict()
    remain_data_set = list(range(data_nums))
    each_client_data_size = data_nums // client_nums  # 取整。
    for i in range(client_nums):
        data_params[i] = list(range(i * each_client_data_size, (i + 1) * each_client_data_size))
    return data_params


def name_to_task_id(name: str):
    name = name.lower()
    if name in ["varnaming", "naming"]:
        return TaskFold.VARNAMING
    if name in ["varmisuse", "misuse"]:
        return TaskFold.VARMISUSE
    raise ValueError("Unkown task name '%s'" % name)


if __name__ == '__main__':
    print(100 // 7)
    print(get_client_params(9, 100))