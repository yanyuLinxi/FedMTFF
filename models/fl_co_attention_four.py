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


class alignModule(nn.Module):
    def __init__(self, out_features, heads, max_node_per_graph):
        super(alignModule, self).__init__()
        self.max_node_per_graph = max_node_per_graph

        self.align_1_mmh = torch.nn.MultiheadAttention(out_features, heads)
        #self.align_1_mmh = torch.nn.MultiheadAttention(out_features, heads)

        self.FFN = nn.Sequential(nn.Linear(out_features * 2, out_features), nn.ReLU(),
                                 nn.Linear(out_features, out_features))

    def forward(self, x1, x2):
        x1_v = x1.view(self.max_node_per_graph, -1, x1.shape[-1])
        x2_v = x2.view(self.max_node_per_graph, -1, x2.shape[-1])

        out, _ = self.align_1_mmh(x1_v, x2_v, x2_v)
        return self.FFN(torch.cat([x1, out.view(-1, out.shape[-1])], dim=1))


class FL_Co_Attention_Four(MessagePassing):
    def __init__(
        self,
        model1,
        model2,
        model3,
        model4,
        out_features,
        classifier_features,
        max_variable_candidates=5,
        heads=8,
        task1: TaskFold = TaskFold.VARMISUSE,
        task2: TaskFold = TaskFold.VARNAMING,
        task3: TaskFold = TaskFold.VARMISUSE,
        task4: TaskFold = TaskFold.VARNAMING,
        max_node_per_graph=50,
    ):
        super(FL_Co_Attention_Four, self).__init__()
        self.max_node_per_graph = max_node_per_graph
        self.max_variable_candidates = max_variable_candidates
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.model4 = model4

        self.align1 = alignModule(out_features=out_features, heads=heads, max_node_per_graph=max_node_per_graph)
        self.align2 = alignModule(out_features=out_features, heads=heads, max_node_per_graph=max_node_per_graph)
        self.align3 = alignModule(out_features=out_features, heads=heads, max_node_per_graph=max_node_per_graph)
        self.align4 = alignModule(out_features=out_features, heads=heads, max_node_per_graph=max_node_per_graph)

        self.integrate1 = alignModule(out_features=out_features, heads=heads, max_node_per_graph=max_node_per_graph)
        self.integrate2 = alignModule(out_features=out_features, heads=heads, max_node_per_graph=max_node_per_graph)
        self.integrate3 = alignModule(out_features=out_features, heads=heads, max_node_per_graph=max_node_per_graph)
        self.integrate4 = alignModule(out_features=out_features, heads=heads, max_node_per_graph=max_node_per_graph)

        self.task1_mapping = nn.Sequential(nn.Linear(out_features, out_features), nn.ReLU(),
                                           nn.Linear(out_features, out_features))
        self.task2_mapping = nn.Sequential(nn.Linear(out_features, out_features), nn.ReLU(),
                                           nn.Linear(out_features, out_features))
        self.task3_mapping = nn.Sequential(nn.Linear(out_features, out_features), nn.ReLU(),
                                           nn.Linear(out_features, out_features))
        self.task4_mapping = nn.Sequential(nn.Linear(out_features, out_features), nn.ReLU(),
                                           nn.Linear(out_features, out_features))

        self.task1 = task1
        self.task2 = task2
        self.task3 = task3
        self.task4 = task4
        if task1 == TaskFold.VARMISUSE:
            self.task1_output_layer = nn.Linear(out_features * 2 + 1, 1)
        elif task1 == TaskFold.VARNAMING:
            self.task1_output_layer = nn.Linear(out_features * 2, classifier_features)

        if task2 == TaskFold.VARMISUSE:
            self.task2_output_layer = nn.Linear(out_features * 2 + 1, 1)
        elif task2 == TaskFold.VARNAMING:
            self.task2_output_layer = nn.Linear(out_features * 2, classifier_features)

        if task3 == TaskFold.VARMISUSE:
            self.task3_output_layer = nn.Linear(out_features * 2 + 1, 1)
        elif task3 == TaskFold.VARNAMING:
            self.task3_output_layer = nn.Linear(out_features * 2, classifier_features)

        if task4 == TaskFold.VARMISUSE:
            self.task4_output_layer = nn.Linear(out_features * 2 + 1, 1)
        elif task4 == TaskFold.VARNAMING:
            self.task4_output_layer = nn.Linear(out_features * 2, classifier_features)
        
        self.linear1 = nn.Linear(out_features*3, out_features)
        self.linear2 = nn.Linear(out_features*3, out_features)
        self.linear3 = nn.Linear(out_features*3, out_features)
        self.linear4 = nn.Linear(out_features*3, out_features)

    def forward(
        self,
        x,
        edge_list: List[torch.tensor],
        slot_id,
        candidate_ids,
        candidate_masks,
        batch_map: torch.Tensor,
        task_index: TaskIndexFold,
    ):

        x1 = self.model1(x, edge_list, batch_map)
        x2 = self.model2(x, edge_list, batch_map)
        x3 = self.model3(x, edge_list, batch_map)
        x4 = self.model4(x, edge_list, batch_map)

        '''
        x1_align = self.align1(x1, x2 - x1 + x3 - x1 + x4 - x1)
        x2_align = self.align2(x2, x1 - x2 + x3 - x2 + x4 - x2)
        x3_align = self.align3(x3, x1 - x3 + x2 - x3 + x4 - x3)
        x4_align = self.align4(x3, x1 - x4 + x2 - x4 + x3 - x4)
        '''
        x1_a = self.linear1(torch.cat([x2,x3,x4],dim=1))
        x2_a = self.linear1(torch.cat([x1,x3,x4],dim=1))
        x3_a = self.linear1(torch.cat([x1,x2,x4],dim=1))
        x4_a = self.linear1(torch.cat([x1,x2,x3],dim=1))
        x1_align = self.align1(x1, x1_a)
        x2_align = self.align2(x2, x2_a)
        x3_align = self.align3(x3, x3_a)
        x4_align = self.align4(x4, x4_a)
        


        x1_integrating = self.integrate1(x1_align, x1_align)
        x2_integrating = self.integrate2(x2_align, x2_align)
        x3_integrating = self.integrate3(x3_align, x3_align)
        x4_integrating = self.integrate4(x4_align, x4_align)

        x1_integrating_view = x1_integrating.view(-1, x1_integrating.shape[-1])
        x2_integrating_view = x2_integrating.view(-1, x2_integrating.shape[-1])
        x3_integrating_view = x3_integrating.view(-1, x3_integrating.shape[-1])
        x4_integrating_view = x4_integrating.view(-1, x4_integrating.shape[-1])

        x1_mapping = self.task1_mapping(x1_integrating_view)
        x2_mapping = self.task2_mapping(x2_integrating_view)
        x3_mapping = self.task3_mapping(x3_integrating_view)
        x4_mapping = self.task4_mapping(x4_integrating_view)

        # 不对 output model需要传入task参数。
        if task_index == TaskIndexFold.Task1:
            # task1
            if self.task1 == TaskFold.VARMISUSE:
                return self.varmisuse_learning_output(x1_mapping, slot_id, candidate_ids, candidate_masks, task_index)
            elif self.task1 == TaskFold.VARNAMING:
                return self.varnaming_learning_output(x1_mapping, slot_id, task_index)
        elif task_index == TaskIndexFold.Task2:
            # task1
            if self.task2 == TaskFold.VARMISUSE:
                return self.varmisuse_learning_output(x2_mapping, slot_id, candidate_ids, candidate_masks, task_index)
            elif self.task2 == TaskFold.VARNAMING:
                return self.varnaming_learning_output(x2_mapping, slot_id, task_index)
        elif task_index == TaskIndexFold.Task3:
            # task1
            if self.task3 == TaskFold.VARMISUSE:
                return self.varmisuse_learning_output(x3_mapping, slot_id, candidate_ids, candidate_masks, task_index)
            elif self.task3 == TaskFold.VARNAMING:
                return self.varnaming_learning_output(x3_mapping, slot_id, task_index)
        elif task_index == TaskIndexFold.Task4:
            # task1
            if self.task4 == TaskFold.VARMISUSE:
                return self.varmisuse_learning_output(x4_mapping, slot_id, candidate_ids, candidate_masks, task_index)
            elif self.task4 == TaskFold.VARNAMING:
                return self.varnaming_learning_output(x4_mapping, slot_id, task_index)
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
        elif task_index == TaskIndexFold.Task3:
            logits = self.task3_output_layer(slot_cand_comb)  # shape: g, c, 1
        elif task_index == TaskIndexFold.Task4:
            logits = self.task4_output_layer(slot_cand_comb)  # shape: g, c, 1

        logits = torch.squeeze(logits, dim=-1)  # shape: g, c
        logits += (1.0 - candidate_masks.view(-1, self.max_variable_candidates)) * -1e7

        return logits

    def varnaming_learning_output(self, out, slot_id, task_index):
        slot_embedding = out[slot_id]  # shape: g, d
        out = out.view(-1, 50, out.shape[-1])
        out = torch.sum(out, dim=1)
        '''
        if task_index == TaskIndexFold.Task1:
            return self.task1_output_layer(out)
        elif task_index == TaskIndexFold.Task2:
            return self.task2_output_layer(out)
        '''
        if task_index == TaskIndexFold.Task1:
            return self.task1_output_layer(torch.cat([slot_embedding, out], dim=1))
        elif task_index == TaskIndexFold.Task2:
            return self.task2_output_layer(torch.cat([slot_embedding, out], dim=1))
        elif task_index == TaskIndexFold.Task3:
            return self.task3_output_layer(torch.cat([slot_embedding, out], dim=1))
        elif task_index == TaskIndexFold.Task4:
            return self.task4_output_layer(torch.cat([slot_embedding, out], dim=1))
