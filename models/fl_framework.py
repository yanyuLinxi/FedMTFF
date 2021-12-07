from torch_geometric.nn import GATConv, SAGPooling, GCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import GCNConv
from layers import EmbeddingLayer, MPLayer
import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import List, Tuple
#from ..fed_client import TaskFold
from utils import TaskFold, TaskIndexFold


class FL_FrameWork(MessagePassing):
    def __init__(
        self,
        model1,
        model2,
        out_features,
        classifier_features,
        max_variable_candidates=5,
        heads=8,
        task1:TaskFold = TaskFold.VARMISUSE,
        task2:TaskFold = TaskFold.VARNAMING,
        max_node_per_graph=50,
    ):
        super(FL_FrameWork, self).__init__()
        self.max_node_per_graph = max_node_per_graph
        self.max_variable_candidates = max_variable_candidates
        self.model1 = model1
        self.model2 = model2


        self.align_1_mmh = torch.nn.MultiheadAttention(out_features, heads)
        self.align_2_mmh = torch.nn.MultiheadAttention(out_features, heads)

        self.integrating_1_mmh = torch.nn.MultiheadAttention(out_features, heads)
        self.integrating_2_mmh = torch.nn.MultiheadAttention(out_features, heads)

        self.task1_mapping = nn.Sequential(nn.Linear(out_features, out_features), nn.ReLU(),
                                           nn.Linear(out_features, out_features))
        self.task2_mapping = nn.Sequential(nn.Linear(out_features, out_features), nn.ReLU(),
                                           nn.Linear(out_features, out_features))

        self.task1 = task1
        self.task2 = task2
        if task1 == TaskFold.VARMISUSE:
            self.task1_output_layer = nn.Linear(out_features * 2 + 1, 1)
        elif task1 == TaskFold.VARNAMING:
            self.task1_output_layer = nn.Linear(out_features, classifier_features)
        
        if task2 == TaskFold.VARMISUSE:
            self.task2_output_layer = nn.Linear(out_features * 2 + 1, 1)
        elif task2 == TaskFold.VARNAMING:
            self.task2_output_layer = nn.Linear(out_features, classifier_features)

    def forward(self,
                x,
                edge_list: List[torch.tensor],
                slot_id,
                candidate_ids,
                candidate_masks,
                batch_map: torch.Tensor,
                task_index: TaskIndexFold,):

        x_1 = self.model1(x, edge_list)
        x_2 = self.model2(x, edge_list)

        x_1 = x_1.view(self.max_node_per_graph, -1, x_1.shape[-1])
        x_2 = x_2.view(self.max_node_per_graph, -1, x_2.shape[-1])

        x_1_align, attention_wight = self.align_1_mmh(x_1, x_2, x_2)
        x_2_align, attention_wight = self.align_2_mmh(x_2, x_1, x_1)

        x_1_integrating, attention_wight = self.integrating_1_mmh(x_1_align, x_1_align, x_1_align)
        x_2_integrating, attention_wight = self.integrating_2_mmh(x_2_align, x_2_align, x_2_align)

        x_1_integrating_view = x_1_integrating.view(-1, x_1_integrating.shape[-1])
        x_2_integrating_view = x_2_integrating.view(-1, x_2_integrating.shape[-1])

        x_1_mapping = self.task1_mapping(x_1_integrating_view)
        x_2_mapping = self.task2_mapping(x_2_integrating_view)

        
        # 不对 output model需要传入task参数。
        if task_index == TaskIndexFold.Task1:
            # task1
            if self.task1 == TaskFold.VARMISUSE:
                return self.varmisuse_learning_output(x_1_mapping, slot_id, candidate_ids, candidate_masks, task_index)
            elif self.task1 == TaskFold.VARNAMING:
                return self.varnaming_learning_output(x_1_mapping, slot_id, task_index)
        elif task_index == TaskIndexFold.Task2:
            # task1
            if self.task2 == TaskFold.VARMISUSE:
                return self.varmisuse_learning_output(x_2_mapping, slot_id, candidate_ids, candidate_masks, task_index)
            elif self.task2 == TaskFold.VARNAMING:
                return self.varnaming_learning_output(x_2_mapping, slot_id, task_index)
        else:
            raise ValueError("Unkown task index '%d'" % task_index)
        '''
        if output_model.lower() == "python":
            return self.varmisuse_python_output(out, slot_id, candidate_ids, candidate_masks)
        if output_model.lower() == "learning":
            return self.varmisuse_learning_output(out, slot_id, candidate_ids, candidate_masks)
        '''

        '''
        if output_model.lower() == "python":
            return self.varnaming_python_output(out, slot_id)
        if output_model.lower() == "learning":
            return self.varnaming_learning_output(out, slot_id)
        '''

    def varmisuse_learning_output(self, out, slot_id, candidate_ids, candidate_masks, task_index):
        candidate_embedding = out[candidate_ids]  # shape: g*c, d
        slot_embedding = out[slot_id]  # shape: g, d

        candidate_embedding_reshape = candidate_embedding.view(-1, self.max_variable_candidates,
                                                               out.shape[-1])  # shape: g, c, d
        slot_inner_product = torch.einsum("cd,cvd->cv", slot_embedding, candidate_embedding_reshape)  #shape g, c

        slot_embedding_unsqueeze = torch.unsqueeze(slot_embedding, dim=1)  # shape: g,1,d
        slot_embedding_repeat = slot_embedding_unsqueeze.repeat(1, self.max_variable_candidates, 1)  # shape: g, c, d

        slot_cand_comb = torch.cat(
            [candidate_embedding_reshape, slot_embedding_repeat,
             torch.unsqueeze(slot_inner_product, dim=-1)], dim=2)  #shape: g, c, d*2+1

        if task_index == TaskIndexFold.Task1:
            logits = self.task1_output_layer(slot_cand_comb)  # shape: g, c, 1
        elif task_index == TaskIndexFold.Task2:
            logits = self.task2_output_layer(slot_cand_comb)  # shape: g, c, 1

        logits = torch.squeeze(logits, dim=-1)  # shape: g, c
        logits += (1.0 - candidate_masks.view(-1, self.max_variable_candidates)) * -1e7

        return logits


    def varnaming_learning_output(self, out, slot_id, task_index):
        slot_embedding = out[slot_id]  # shape: g, d
        if task_index == TaskIndexFold.Task1:
            return self.task1_output_layer(slot_embedding)
        elif task_index == TaskIndexFold.Task2:
            return self.task2_output_layer(slot_embedding)
            
