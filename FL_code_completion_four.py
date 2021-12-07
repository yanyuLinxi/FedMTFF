from utils.task_utils import TaskFold, TaskIndexFold
from torch._C import device
from models import ResGAGN
import argparse
from enum import Enum
import time
import os.path as osp
from os import getpid
import json
from typing import Any, Dict, Optional, Tuple, List, Iterable, Union
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
#from dataProcessing import CGraphDataset, PYGraphDataset, LRGraphDataset
import numpy as np
from utils import cal_metrics, top_5, cal_early_stopping_metric, pretty_print_epoch_task_metrics, average_weights
from multiprocessing import cpu_count
from FL_client_four import SingleClient, get_client_params, name_to_task_id
from random import shuffle
import copy
from models import name_to_dataset_class, name_to_model_class


class DataFold(Enum):
    TRAIN = 0
    VALIDATION = 1
    TEST = 2


class FL_CodeCompletion_Four:
    @classmethod
    def default_args(cls):
        parser = argparse.ArgumentParser(description="Federated_DGAP")
        # 不包含naming的为varmisuse数据集。包含naming的为变量预测的数据集。
        parser.add_argument('--dataset_name_task1',
                            type=str,
                            default="csharpstatic",
                            help='the name of the dataset1. optional:[python, learning]')

        parser.add_argument('--dataset_name_task2',
                            type=str,
                            default="csharpStaticnaming",
                            help='the name of the dataset2. optional:[python, learning]')
        parser.add_argument('--dataset_name_task3',
                            type=str,
                            default="python",
                            help='the name of the dataset2. optional:[python, learning]')
        parser.add_argument('--dataset_name_task4',
                            type=str,
                            default="pythonNaming",
                            help='the name of the dataset2. optional:[python, learning]')

        parser.add_argument('--FL_model',
                            type=str,
                            default="fl_co_attention_four",
                            help='the fl model of features extract')

        parser.add_argument('--backbone_model',
                            type=str,
                            default="resgagn",
                            help='the backbone model of features extract')

        # 目前output model 参数没有用。
        parser.add_argument('--task_load_model_dir',
                            type=str,
                            default="checkpoint/",
                            help='')

        parser.add_argument('--task1', type=str, default="varmisuse", help='')
        parser.add_argument('--task2', type=str, default="varnaming", help='')
        parser.add_argument('--task3', type=str, default="varmisuse", help='')
        parser.add_argument('--task4', type=str, default="varnaming", help='')

        parser.add_argument('--optimizer', type=str, default="Adam", help='')
        parser.add_argument('--lr', type=float, default=0.001, help='')
        parser.add_argument('--lr_deduce_per_epoch', type=int, default=10, help='')
        parser.add_argument('--max_epochs', type=int, default=1500, help='')
        parser.add_argument('--cur_epoch', type=int, default=1, help='')
        parser.add_argument('--max_variable_candidates', type=int, default=5, help='')
        parser.add_argument('--batch_size', type=int, default=64, help='')
        parser.add_argument('--result_dir', type=str, default="trained_models/", help='')
        parser.add_argument('--dropout_rate', type=float, default=0., help='keep_prob = 1-dropout_rate')
        #parser.add_argument('--load_model_file', type=str, default=None, help='')
        #parser.add_argument('--load_model_file', type=str, default="trained_models/model_save/DGAP-Model_2021-06-06-10-57-33_2296_best_model.pt", help='')
        #parser.add_argument('--in_features', type=int, default=64, help='')  # in_features 为embedding中的graph_node_max_num_chars
        parser.add_argument('--h_features', type=int, default=64, help='')
        parser.add_argument('--out_features', type=int, default=64, help='')
        parser.add_argument('--graph_node_max_num_chars', type=int, default=19, help='')
        parser.add_argument('--max_node_per_graph', type=int, default=50, help='')
        parser.add_argument('--device', type=str, default="cuda", help='')
        parser.add_argument('--value_dict_dir',
                            type=str,
                            default="vocab_dict/python_terminal_dict_1k_value.json",
                            help='')
        parser.add_argument('--type_dict_dir',
                            type=str,
                            default="vocab_dict/python_terminal_dict_1k_type.json",
                            help='')
        parser.add_argument('--vocab_size', type=int, default=1001, help='')


        parser.add_argument('--dataset_train_data_dir_task1', type=str, default="data/csharp/train_data", help='')
        parser.add_argument('--dataset_validate_data_dir_task1', type=str, default="data/csharp/validate_data", help='')
        parser.add_argument('--dataset_train_data_dir_task2', type=str, default="data/csharp/train_data", help='')
        parser.add_argument('--dataset_validate_data_dir_task2', type=str, default="data/csharp/validate_data", help='')
        parser.add_argument('--dataset_train_data_dir_task3', type=str, default="data/python/train_data", help='')
        parser.add_argument('--dataset_validate_data_dir_task3', type=str, default="data/python/validate_data", help='')
        parser.add_argument('--dataset_train_data_dir_task4', type=str, default="data/python/train_data", help='')
        parser.add_argument('--dataset_validate_data_dir_task4', type=str, default="data/python/validate_data", help='')
        #--Federated setting
        parser.add_argument('--client_fraction',
                            type=float,
                            default=1,
                            help='Fraction of clients to be used for federated updates. Default is 0.1.')
        parser.add_argument('--client_nums', type=int, default=4, help='Number of clients. Default is 100.')
        parser.add_argument('--client_max_epochs', type=int, default=1, help='client epochs. Default is 10.')
        """
        parser.add_argument('--train_data_dir',
                            type=str,
                            default="data/learning/train_data",
                            help='')
        parser.add_argument('--validate_data_dir',
                            type=str,
                            default="data/learning/validate_data",
                            help='')
        """
        # args 获取命令行输入，和默认值。
        # 当需要读取参数时，在load中读取并设置。
        return parser

    @staticmethod
    def name() -> str:
        return "FedMTFF"

    def __init__(self, args):
        self.args = args
        self.run_id = "_".join([self.name(), self.args.backbone_model, time.strftime("%Y-%m-%d-%H-%M-%S"), str(getpid())])
        self._loaded_datasets_task1 = {}
        self._loaded_datasets_task2 = {}
        self._loaded_datasets_task3 = {}
        self._loaded_datasets_task4 = {}
        self.load_data()
        self.__make_model()
        self.__make_federated_client()

    @property
    def log_file(self):
        return osp.join(self.args.result_dir, "%s.log" % self.run_id)

    def log_line(self, msg):
        with open(self.log_file, 'a') as log_fh:
            log_fh.write(msg + '\n')
        print(msg)

    @property
    def best_model_file(self):
        return osp.join(self.args.result_dir, osp.join("model_save", "%s_best_model.pt" % self.run_id))

    def freeze_model_params(self, filter: Union[str, List[str]], reverse=False):
        """freeze model params by filter_str, the params with filter_str will set require_grad=False
            if reverse is True, then them will be set require_grad=True

        Args:
            filter ([type]): freeze filter, 
            reverse (bool, optional): see up. Defaults to False.
        """
        '''for key in self.model.parameters():
            print(key)'''
        '''
        for layer in self.model.modules():
            print(layer)
        '''
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad = not reverse
            for f in filter:
                if f.lower() in name.lower():
                    parameter.requires_grad = reverse

        self.log_line("freeze params in %s requires_grad be %s" % (filter, reverse))
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad == True:
                self.log_line(name)
                #print(parameter.shape)
                #print(parameter)

    def save_model(self, path: str) -> None:
        save_dict = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "params": vars(self.args),
        }
        torch.save(save_dict, path)

    def load_model(self, path) -> None:
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        self.args.cur_epoch = checkpoint['params']['cur_epoch']
        #self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def __make_model(self) -> None:
        # make model

        modelCls, appendix_args = name_to_model_class(self.args.backbone_model, self.args)
        self.log_line("model_args:" + json.dumps(appendix_args))

        num_edge_types_task1 = self.make_task_input_task1(None, get_num_edge_types=True)
        num_edge_types_task2 = self.make_task_input_task2(None, get_num_edge_types=True)
        num_edge_types_task3 = self.make_task_input_task3(None, get_num_edge_types=True)
        num_edge_types_task4 = self.make_task_input_task4(None, get_num_edge_types=True)
        assert num_edge_types_task1 == num_edge_types_task2, "task1 task2 num_edge not equal"

        task1_model = modelCls(num_edge_types=num_edge_types_task1,
                               in_features=self.args.graph_node_max_num_chars,
                               out_features=self.args.out_features,
                               embedding_out_features=self.args.h_features,
                               classifier_features=self.args.vocab_size,
                               embedding_num_classes=70,
                               dropout=self.args.dropout_rate,
                               max_variable_candidates=self.args.max_variable_candidates,
                               device=self.args.device,
                               **appendix_args)
        task2_model = modelCls(num_edge_types=num_edge_types_task1,
                               in_features=self.args.graph_node_max_num_chars,
                               out_features=self.args.out_features,
                               embedding_out_features=self.args.h_features,
                               classifier_features=self.args.vocab_size,
                               embedding_num_classes=70,
                               dropout=self.args.dropout_rate,
                               max_variable_candidates=self.args.max_variable_candidates,
                               device=self.args.device,
                               **appendix_args)
        task3_model = modelCls(num_edge_types=num_edge_types_task1,
                               in_features=self.args.graph_node_max_num_chars,
                               out_features=self.args.out_features,
                               embedding_out_features=self.args.h_features,
                               classifier_features=self.args.vocab_size,
                               embedding_num_classes=70,
                               dropout=self.args.dropout_rate,
                               max_variable_candidates=self.args.max_variable_candidates,
                               device=self.args.device,
                               **appendix_args)
        task4_model = modelCls(num_edge_types=num_edge_types_task1,
                               in_features=self.args.graph_node_max_num_chars,
                               out_features=self.args.out_features,
                               embedding_out_features=self.args.h_features,
                               classifier_features=self.args.vocab_size,
                               embedding_num_classes=70,
                               dropout=self.args.dropout_rate,
                               max_variable_candidates=self.args.max_variable_candidates,
                               device=self.args.device,
                               **appendix_args)

        task1_load_model_file = osp.join(self.args.task_load_model_dir, self.args.backbone_model.lower() + "_varmisuse_c#_best_model.pt")
        task2_load_model_file = osp.join(self.args.task_load_model_dir, self.args.backbone_model.lower() + "_codecompletion_c#_best_model.pt")
        task3_load_model_file = osp.join(self.args.task_load_model_dir, self.args.backbone_model.lower() + "_varmisuse_python_best_model.pt")
        task4_load_model_file = osp.join(self.args.task_load_model_dir, self.args.backbone_model.lower() + "_codecompletion_python_best_model.pt")


        checkpoint = torch.load(task1_load_model_file)
        task1_model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        #task1_model.eval()
        for name, parameter in task1_model.named_parameters():
            parameter.requires_grad = False


        checkpoint_task2 = torch.load(task2_load_model_file)
        task2_model.load_state_dict(checkpoint_task2['model_state_dict'], strict=False)
        task2_model.eval()
        for name, parameter in task2_model.named_parameters():
            parameter.requires_grad = False
        
        checkpoint_task3 = torch.load(task3_load_model_file, map_location=torch.device("cuda"))
        task3_model.load_state_dict(checkpoint_task3['model_state_dict'], strict=False)
        task3_model.eval()
        for name, parameter in task3_model.named_parameters():
            parameter.requires_grad = False
        
        checkpoint_task4 = torch.load(task4_load_model_file, map_location=torch.device("cuda"))
        task4_model.load_state_dict(checkpoint_task4['model_state_dict'], strict=False)
        task4_model.eval()
        for name, parameter in task4_model.named_parameters():
            parameter.requires_grad = False


        self.task_id_1 = name_to_task_id(self.args.task1)
        self.task_id_2 = name_to_task_id(self.args.task2)
        self.task_id_3 = name_to_task_id(self.args.task3)
        self.task_id_4 = name_to_task_id(self.args.task4)
        
        fl_model, appendix_args = name_to_model_class(self.args.FL_model, self.args)
        self.model = fl_model(
            task1_model,
            task2_model,
            task3_model,
            task4_model,
            out_features=self.args.h_features,
            classifier_features=self.args.vocab_size,
            max_variable_candidates=self.args.max_variable_candidates,
            task1=self.task_id_1,
            task2=self.task_id_2,
            task3=self.task_id_3,
            task4=self.task_id_4,
            **appendix_args,
        )
        #self.model = GraphConvolution(self.params['embedding_size'], self.params['embedding_size'], is_sparse_adjacency=False)

        self.model.to(self.args.device)
        #self.model.applsy(self.apply_weight_init)
        '''
        # TODO: FL load model.
        if self.args.load_model_file is not None:
            self.log_line("load model:" + self.args.load_model_file)
            self.load_model(self.args.load_model_file)
        '''
        #self.freeze_model_params(["varmisuse_output_layer", "varnaming_output_layer", "gat1", "pool1", "gat2", "pool2", "gat3", "pool3"], reverse=True)
        self.__make_train_step()

    def __make_train_step(self):

        # use optimizer
        lr = self.args.lr
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        #batchNumsPerEpoch = len(self._loaded_datasets[DataFold.TRAIN]) // self.args.batch_size + 1
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         int(self.args.lr_deduce_per_epoch / self.args.client_fraction),
                                                         gamma=0.6,
                                                         last_epoch=-1)

        self.args.num_model_params = sum(p.numel() for p in list(self.model.parameters()))  # numel()
        '''
        self.optimizer = torch.optim.Adam([{
            "params": value.parameters()
        } for key, value in self.all_model.items()],
                                          lr=lr)
        '''

    def __make_federated_client(self):

        client_nums = self.args.client_nums  # 2

        self.client = []

        # task1 client
        task1_client_nums = 1
        task2_client_nums = 1
        task3_client_nums = 1
        task4_client_nums = 1

        dataset_task1_data_nums = len(self._loaded_datasets_task1[DataFold.TRAIN])
        #self.task_id_1 = name_to_task_id(self.args.task1)
        client_data_params_task1 = get_client_params(task1_client_nums, dataset_task1_data_nums)

        for cid in range(task1_client_nums):
            client = SingleClient(
                client_id=len(self.client),
                data=self._loaded_datasets_task1[DataFold.TRAIN][client_data_params_task1[cid]],
                log_file=self.log_file,
                task_index=TaskIndexFold.Task1,
                task_id=self.task_id_1,
                client_max_epochs=self.args.client_max_epochs,
                client_batch_size=self.args.batch_size,
                make_task_input=self.make_task_input_task1,
                #dataLoader_numworkers=int(cpu_count() / 2),
                dataLoader_numworkers=0,
                device=self.args.device,
            )
            self.client.append(client)

        dataset_task2_data_nums = len(self._loaded_datasets_task2[DataFold.TRAIN])
        #self.task_id_2 = name_to_task_id(self.args.task2)
        client_data_params_task2 = get_client_params(task2_client_nums, dataset_task2_data_nums)

        for cid in range(task2_client_nums):
            client = SingleClient(
                client_id=len(self.client),
                data=self._loaded_datasets_task2[DataFold.TRAIN][client_data_params_task2[cid]],
                log_file=self.log_file,
                task_index=TaskIndexFold.Task2,
                task_id=self.task_id_2,
                client_max_epochs=self.args.client_max_epochs,
                client_batch_size=self.args.batch_size,
                make_task_input=self.make_task_input_task2,
                #dataLoader_numworkers=int(cpu_count() / 2),
                dataLoader_numworkers=0,
                device=self.args.device,
            )
            self.client.append(client)

        
        dataset_task3_data_nums = len(self._loaded_datasets_task3[DataFold.TRAIN])
        #self.task_id_3 = name_to_task_id(self.args.task3)
        client_data_params_task3 = get_client_params(task3_client_nums, dataset_task3_data_nums)

        for cid in range(task3_client_nums):
            client = SingleClient(
                client_id=len(self.client),
                data=self._loaded_datasets_task3[DataFold.TRAIN][client_data_params_task3[cid]],
                log_file=self.log_file,
                task_index=TaskIndexFold.Task3,
                task_id=self.task_id_3,
                client_max_epochs=self.args.client_max_epochs,
                client_batch_size=self.args.batch_size,
                make_task_input=self.make_task_input_task3,
                #dataLoader_numworkers=int(cpu_count() / 2),
                dataLoader_numworkers=0,
                device=self.args.device,
            )
            self.client.append(client)

        
        dataset_task4_data_nums = len(self._loaded_datasets_task4[DataFold.TRAIN])
        #self.task_id_4 = name_to_task_id(self.args.task4)
        client_data_params_task4 = get_client_params(task4_client_nums, dataset_task4_data_nums)

        for cid in range(task4_client_nums):
            client = SingleClient(
                client_id=len(self.client),
                data=self._loaded_datasets_task4[DataFold.TRAIN][client_data_params_task4[cid]],
                log_file=self.log_file,
                task_index=TaskIndexFold.Task4,
                task_id=self.task_id_4,
                client_max_epochs=self.args.client_max_epochs,
                client_batch_size=self.args.batch_size,
                make_task_input=self.make_task_input_task4,
                #dataLoader_numworkers=int(cpu_count() / 2),
                dataLoader_numworkers=0,
                device=self.args.device,
            )
            self.client.append(client)

    def load_data(self) -> None:
        """

        Raises:
            Exception: [description]
        """
        train_path_task1 = self.args.dataset_train_data_dir_task1
        validate_path_task1 = self.args.dataset_validate_data_dir_task1
        train_path_task2 = self.args.dataset_train_data_dir_task2
        validate_path_task2 = self.args.dataset_validate_data_dir_task2
        train_path_task3 = self.args.dataset_train_data_dir_task3
        validate_path_task3 = self.args.dataset_validate_data_dir_task3
        train_path_task4 = self.args.dataset_train_data_dir_task4
        validate_path_task4 = self.args.dataset_validate_data_dir_task4

        # if dataset dir not exists
        if not osp.exists(train_path_task1) or not osp.exists(validate_path_task1) or not osp.exists(
                train_path_task2) or not osp.exists(validate_path_task2):
            raise Exception("train data or validate data dir not exists error")

        if not osp.exists(self.args.value_dict_dir) or not osp.exists(self.args.type_dict_dir):
            raise Exception("vocab dict not exists error")

        with open(self.args.value_dict_dir, 'r') as f:
            value_dict = json.load(f)
        with open(self.args.type_dict_dir, 'r') as f:
            type_dict = json.load(f)



        datasetCls_task1, appendix_args_task1, self.make_task_input_task1 = name_to_dataset_class(
            self.args.dataset_name_task1, self.args)
        datasetCls_task2, appendix_args_task2, self.make_task_input_task2 = name_to_dataset_class(
            self.args.dataset_name_task2, self.args)
        datasetCls_task3, appendix_args_task3, self.make_task_input_task3 = name_to_dataset_class(
            self.args.dataset_name_task3, self.args)
        datasetCls_task4, appendix_args_task4, self.make_task_input_task4 = name_to_dataset_class(
            self.args.dataset_name_task4, self.args)

        self._loaded_datasets_task1[DataFold.TRAIN] = datasetCls_task1(
            train_path_task1,
            value_dict,
            type_dict,
            graph_node_max_num_chars=self.args.graph_node_max_num_chars,
            max_graph=20000,
            max_variable_candidates=self.args.max_variable_candidates,
            **appendix_args_task1)
        self._loaded_datasets_task1[DataFold.VALIDATION] = datasetCls_task1(
            validate_path_task1,
            value_dict,
            type_dict,
            self.args.graph_node_max_num_chars,
            max_graph=10000,
            max_variable_candidates=self.args.max_variable_candidates,
            **appendix_args_task1)

        self._loaded_datasets_task2[DataFold.TRAIN] = datasetCls_task2(
            train_path_task2,
            value_dict,
            type_dict,
            graph_node_max_num_chars=self.args.graph_node_max_num_chars,
            max_graph=20000,
            max_variable_candidates=self.args.max_variable_candidates,
            **appendix_args_task2)
        self._loaded_datasets_task2[DataFold.VALIDATION] = datasetCls_task2(
            validate_path_task2,
            value_dict,
            type_dict,
            self.args.graph_node_max_num_chars,
            max_graph=10000,
            max_variable_candidates=self.args.max_variable_candidates,
            **appendix_args_task2)
        
        self._loaded_datasets_task3[DataFold.TRAIN] = datasetCls_task3(
            train_path_task3,
            value_dict,
            type_dict,
            graph_node_max_num_chars=self.args.graph_node_max_num_chars,
            max_graph=20000,
            max_variable_candidates=self.args.max_variable_candidates,
            **appendix_args_task3)
        self._loaded_datasets_task3[DataFold.VALIDATION] = datasetCls_task3(
            validate_path_task3,
            value_dict,
            type_dict,
            self.args.graph_node_max_num_chars,
            max_graph=10000,
            max_variable_candidates=self.args.max_variable_candidates,
            **appendix_args_task3)
        
        self._loaded_datasets_task4[DataFold.TRAIN] = datasetCls_task4(
            train_path_task4,
            value_dict,
            type_dict,
            graph_node_max_num_chars=self.args.graph_node_max_num_chars,
            max_graph=20000,
            max_variable_candidates=self.args.max_variable_candidates,
            **appendix_args_task4)
        self._loaded_datasets_task4[DataFold.VALIDATION] = datasetCls_task4(
            validate_path_task4,
            value_dict,
            type_dict,
            self.args.graph_node_max_num_chars,
            max_graph=10000,
            max_variable_candidates=self.args.max_variable_candidates,
            **appendix_args_task4)

    def criterion(self, y_score, y_true, criterion=torch.nn.CrossEntropyLoss()):
        loss = criterion(y_score, y_true)
        metrics = cal_metrics(F.softmax(y_score, dim=-1), y_true)
        return loss, metrics

    def __run_epoch(
        self,
        epoch_name: str,
        data: Iterable[Any],
        data_fold: DataFold,
        batch_size: int,
        make_task_input: object,
        task_index: TaskIndexFold,
        task_id:TaskFold,
        quiet: Optional[bool] = False,
    ) -> Tuple[float]:
        """具体的每一轮训练。

        Args:
            epoch_name (str): 
            data (Iterable[Any]): 
            data_fold (DataFold): is test or train dataset
            quiet (Optional[bool], optional): . Defaults to False.

        Returns:
            Tuple[float]: [description]
        """
        batch_iterator = DataLoader(
            data,
            batch_size=batch_size,
            shuffle=True if data_fold == DataFold.TRAIN else False,
            #num_workers=int(cpu_count()/2))
            num_workers=0)

        start_time = time.time()
        processed_graphs, processed_nodes, processed_batch = 0, 0, 0
        epoch_loss = 0.0
        task_metric_results = []

        for batch_data in batch_iterator:
            batch_data = batch_data.to(self.args.device)

            processed_graphs += batch_data.num_graphs
            processed_nodes += batch_data.num_nodes
            processed_batch += 1

            # train in server only have validation
            if data_fold == DataFold.VALIDATION:
                self.model.eval()
                with torch.no_grad():
                    task_batch_data = make_task_input(batch_data)
                    logits = self.model(**task_batch_data, task_index=task_index)
                    if task_id == TaskFold.VARMISUSE:
                        loss, metrics = self.criterion(logits, batch_data.label)
                    if task_id == TaskFold.VARNAMING:
                        loss, metrics = self.criterion(logits, batch_data.value_label)
                    epoch_loss += loss.item()
                    task_metric_results.append(metrics)

            if not quiet:
                print("Runing %s, batch %i (has %i graphs). Loss so far: %.4f" %
                      (epoch_name, processed_batch, batch_data.num_graphs, epoch_loss / processed_batch),
                      end="\r")

        epoch_time = time.time() - start_time
        per_graph_loss = epoch_loss / processed_batch
        graphs_per_sec = processed_graphs / epoch_time
        nodes_per_sec = processed_nodes / epoch_time

        return per_graph_loss, task_metric_results, processed_graphs, processed_batch, graphs_per_sec, nodes_per_sec, processed_graphs, processed_nodes

    def train(self, quiet=False):
        """

        Args:
            quiet (bool, optional): [description]. Defaults to False.
        """
        self.log_line(json.dumps(vars(self.args), indent=4))
        total_time_start = time.time()

        clients_fraction = self.args.client_fraction
        clients_nums = self.args.client_nums
        clients_each_epoch = max(int(clients_fraction * clients_nums), 1)

        (best_valid_metric_task1, best_val_metric_epoch, best_val_metric_descr) = (float("+inf"), 0, "")
        (best_valid_metric_task2, best_val_metric_epoch, best_val_metric_descr) = (float("+inf"), 0, "")
        (best_valid_metric_task3, best_val_metric_epoch, best_val_metric_descr) = (float("+inf"), 0, "")
        (best_valid_metric_task4, best_val_metric_epoch, best_val_metric_descr) = (float("+inf"), 0, "")

        client_ids = list(range(len(self.client)))
        #self.bn_dict = {}
        for epoch in range(self.args.cur_epoch, self.args.max_epochs + 1):
            self.log_line("== Server Epoch %i" % epoch)

            shuffle(client_ids)
            cur_client_ids = client_ids[:clients_each_epoch]

            # Federated train
            client_weight_list = []
            client_loss_list = []
            #cur_bn_dict = {task: [] for task in TaskFold}  # 临时存放norm params
            for cur_client_id in cur_client_ids:
                cur_client = self.client[cur_client_id]
                client_weight, client_loss, bn_params = cur_client.update_weights(
                    copy.deepcopy(self.model), torch.optim.Adam,
                    self.scheduler.get_last_lr()[0])#, self.bn_dict.get(cur_client.task_id, None))
                client_weight_list.append(copy.deepcopy(client_weight))
                client_loss_list.append(copy.deepcopy(client_loss))

            global_weights = average_weights(client_weight_list)
            
            self.model.load_state_dict(global_weights)
            #self.scheduler.step()  
            self.log_line("==Server Epoch %i Train: loss: %.5f" %
                          (epoch, sum(client_loss_list) / len(client_loss_list)))

            # Fed Validate
            # --validate task1
            valid_loss, valid_task_metrics, valid_num_graphs, valid_num_batchs, valid_graphs_p_s, valid_nodes_p_s, test_graphs, test_nodes = self.__run_epoch(
                "epoch %i (validation)" % epoch,
                self._loaded_datasets_task1[DataFold.VALIDATION],
                DataFold.VALIDATION,
                self.args.batch_size,
                make_task_input=self.make_task_input_task1,
                task_index=TaskIndexFold.Task1,
                task_id=self.task_id_1,
                quiet=quiet)

            early_stopping_metric_task1 = cal_early_stopping_metric(valid_task_metrics)
            valid_metric_descr_task1 = pretty_print_epoch_task_metrics(valid_task_metrics, valid_num_graphs,
                                                                       valid_num_batchs)
            if not quiet:
                print("\r\x1b[K", end='')
            self.log_line(
                "%s %s valid: loss: %.5f || %s || graphs/sec: %.2f | nodes/sec: %.0f | graphs: %.0f | nodes: %.0f | lr: %0.6f"
                % (TaskIndexFold.Task1, self.task_id_1, valid_loss, valid_metric_descr_task1, valid_graphs_p_s, valid_nodes_p_s, test_graphs,
                   test_nodes, self.scheduler.get_last_lr()[0]))

            if early_stopping_metric_task1 < best_valid_metric_task1:
                self.args.cur_epoch = epoch + 1  
                self.save_model(self.best_model_file)
                self.log_line("  (Best epoch so far, target metric decreased to %.5f from %.5f. Saving to '%s')" %
                              (early_stopping_metric_task1, best_valid_metric_task1, self.best_model_file))
                best_valid_metric_task1 = early_stopping_metric_task1
                best_val_metric_epoch = epoch
                best_val_metric_descr = valid_metric_descr_task1

            # --validate task2
            valid_loss, valid_task_metrics, valid_num_graphs, valid_num_batchs, valid_graphs_p_s, valid_nodes_p_s, test_graphs, test_nodes = self.__run_epoch(
                "epoch %i (validation)" % epoch,
                self._loaded_datasets_task2[DataFold.VALIDATION],
                DataFold.VALIDATION,
                self.args.batch_size,
                make_task_input=self.make_task_input_task2,
                task_index=TaskIndexFold.Task2,
                task_id=self.task_id_2,
                quiet=quiet)

            early_stopping_metric_task2 = cal_early_stopping_metric(valid_task_metrics)
            valid_metric_descr_task2 = pretty_print_epoch_task_metrics(valid_task_metrics, valid_num_graphs,
                                                                       valid_num_batchs)
            if not quiet:
                print("\r\x1b[K", end='')
            self.log_line(
                "%s %s valid: loss: %.5f || %s || graphs/sec: %.2f | nodes/sec: %.0f | graphs: %.0f | nodes: %.0f | lr: %0.6f"
                % (TaskIndexFold.Task2, self.task_id_2, valid_loss, valid_metric_descr_task2, valid_graphs_p_s, valid_nodes_p_s, test_graphs,
                   test_nodes, self.scheduler.get_last_lr()[0]))

            if early_stopping_metric_task2 < best_valid_metric_task2:
                self.args.cur_epoch = epoch + 1 
                self.save_model(self.best_model_file)
                self.log_line("  (Best epoch so far, target metric decreased to %.5f from %.5f. Saving to '%s')" %
                              (early_stopping_metric_task2, best_valid_metric_task2, self.best_model_file))
                best_valid_metric_task2 = early_stopping_metric_task2
                best_val_metric_epoch = epoch
                best_val_metric_descr = valid_metric_descr_task2
            
            
            # --validate task3
            valid_loss, valid_task_metrics, valid_num_graphs, valid_num_batchs, valid_graphs_p_s, valid_nodes_p_s, test_graphs, test_nodes = self.__run_epoch(
                "epoch %i (validation)" % epoch,
                self._loaded_datasets_task3[DataFold.VALIDATION],
                DataFold.VALIDATION,
                self.args.batch_size,
                make_task_input=self.make_task_input_task3,
                task_index=TaskIndexFold.Task3,
                task_id=self.task_id_3,
                quiet=quiet)

            early_stopping_metric_task3 = cal_early_stopping_metric(valid_task_metrics)
            valid_metric_descr_task3 = pretty_print_epoch_task_metrics(valid_task_metrics, valid_num_graphs,
                                                                       valid_num_batchs)
            if not quiet:
                print("\r\x1b[K", end='')
            self.log_line(
                "%s %s valid: loss: %.5f || %s || graphs/sec: %.2f | nodes/sec: %.0f | graphs: %.0f | nodes: %.0f | lr: %0.6f"
                % (TaskIndexFold.Task3, self.task_id_3, valid_loss, valid_metric_descr_task3, valid_graphs_p_s, valid_nodes_p_s, test_graphs,
                   test_nodes, self.scheduler.get_last_lr()[0]))

            if early_stopping_metric_task3 < best_valid_metric_task3:
                self.args.cur_epoch = epoch + 1  
                self.save_model(self.best_model_file)
                self.log_line("  (Best epoch so far, target metric decreased to %.5f from %.5f. Saving to '%s')" %
                              (early_stopping_metric_task3, best_valid_metric_task3, self.best_model_file))
                best_valid_metric_task3 = early_stopping_metric_task3
                best_val_metric_epoch = epoch
                best_val_metric_descr = valid_metric_descr_task3
            

            
            # --validate task4
            valid_loss, valid_task_metrics, valid_num_graphs, valid_num_batchs, valid_graphs_p_s, valid_nodes_p_s, test_graphs, test_nodes = self.__run_epoch(
                "epoch %i (validation)" % epoch,
                self._loaded_datasets_task4[DataFold.VALIDATION],
                DataFold.VALIDATION,
                self.args.batch_size,
                make_task_input=self.make_task_input_task4,
                task_index=TaskIndexFold.Task4,
                task_id=self.task_id_4,
                quiet=quiet)

            early_stopping_metric_task4 = cal_early_stopping_metric(valid_task_metrics)
            valid_metric_descr_task4 = pretty_print_epoch_task_metrics(valid_task_metrics, valid_num_graphs,
                                                                       valid_num_batchs)
            if not quiet:
                print("\r\x1b[K", end='')
            self.log_line(
                "%s %s valid: loss: %.5f || %s || graphs/sec: %.2f | nodes/sec: %.0f | graphs: %.0f | nodes: %.0f | lr: %0.6f"
                % (TaskIndexFold.Task4, self.task_id_4, valid_loss, valid_metric_descr_task4, valid_graphs_p_s, valid_nodes_p_s, test_graphs,
                   test_nodes, self.scheduler.get_last_lr()[0]))

            if early_stopping_metric_task4 < best_valid_metric_task4:
                self.args.cur_epoch = epoch + 1 
                self.save_model(self.best_model_file)
                self.log_line("  (Best epoch so far, target metric decreased to %.5f from %.5f. Saving to '%s')" %
                              (early_stopping_metric_task4, best_valid_metric_task4, self.best_model_file))
                best_valid_metric_task4 = early_stopping_metric_task4
                best_val_metric_epoch = epoch
                best_val_metric_descr = valid_metric_descr_task4

    def test(self):
        pass
