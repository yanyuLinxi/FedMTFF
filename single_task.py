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
import numpy as np
from random import shuffle


class Single_Task:
    @classmethod
    def default_args(cls):
        parser = argparse.ArgumentParser(description="Single Task 模型预测")

        
        # 任务相关
        
        
        # 输出相关
        parser.add_argument('--result_dir', type=str, default="trained_models/", help='')

        # 读取模型、继续训练
        #parser.add_argument('--load_model_file', type=str, default="trained_models/model_save/DGAP-Model_2021-06-06-10-57-33_2296_best_model.pt", help='')

        # 模型训练
        parser.add_argument('--backbone_model',
                            type=str,
                            default="ggnn_feature_extract",
                            help='the backbone model of features extract')
        parser.add_argument('--optimizer', type=str, default="Adam", help='')
        parser.add_argument('--lr', type=float, default=0.001, help='')
        parser.add_argument('--lr_deduce_per_epoch', type=int, default=10, help='')
        parser.add_argument('--max_epochs', type=int, default=1500, help='')
        parser.add_argument('--cur_epoch', type=int, default=1, help='用做读取checkpoint再训练的参数，手动设置无效。')
        parser.add_argument('--batch_size', type=int, default=64, help='')
        parser.add_argument('--dropout_rate', type=float, default=0., help='keep_prob = 1-dropout_rate')
        parser.add_argument('--h_features', type=int, default=64, help='')
        parser.add_argument('--out_features', type=int, default=64, help='')
        parser.add_argument('--graph_node_max_num_chars', type=int, default=19, help='图中节点的初始特征维度')
        parser.add_argument('--max_node_per_graph', type=int, default=50, help='一个图最多的节点个数')
        parser.add_argument('--device', type=str, default="cuda", help='')




        return parser

    @staticmethod
    def name() -> str:
        return "Single-Task"

    def __init__(self, args):
        self.args = args
        self.run_id = "_".join([self.name(), time.strftime("%Y-%m-%d-%H-%M-%S"), str(getpid())])
        # self.load_data()

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
    
    
    def train(self, quiet=False):
        """对模型进行训练

        Args:
            quiet (bool, optional): _description_. Defaults to False.
        """
        
        # 在日志中打印当前的设置参数。
        self.log_line(json.dumps(vars(self.args), indent=4))



if __name__ == '__main__':
    cc_cls = Single_Task
    
    parser = cc_cls.default_args()
    args = parser.parse_args()
    cc = cc_cls(args)
    cc.train()
    print("Training Done!")