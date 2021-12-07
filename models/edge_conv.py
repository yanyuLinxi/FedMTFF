from torch_geometric.nn import GATConv, SAGPooling, GCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import MessagePassing, FiLMConv, EdgeConv
from torch_geometric.nn.conv.gcn_conv import GCNConv
from layers import EmbeddingLayer, MPLayer
import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import List, Tuple


class BasicGAT(MessagePassing):
    def __init__(self, in_channels, out_channels, heads, concat, dropout, add_self_loops, bias, max_node_per_graph,
                 num_edge_types):
        super(BasicGAT, self).__init__()
        self.max_node_per_graph = max_node_per_graph
        self.gatconv = GATConv(in_channels=in_channels,
                               out_channels=out_channels,
                               heads=heads,
                               concat=concat,
                               dropout=dropout,
                               add_self_loops=add_self_loops,
                               bias=bias)

        #self.norm = nn.LayerNorm((max_node_per_graph, out_channels), elementwise_affine=False)
        #self.norm_w = nn.parameter.Parameter(torch.ones((max_node_per_graph, 1)))
        #self.norm_b = nn.parameter.Parameter(torch.zeros((max_node_per_graph, 1)))
        self.activation = nn.ReLU()

    def forward(self, input):
        x, edge_index = input
        out = self.gatconv(x, edge_index)
        #out = out.view(-1, self.max_node_per_graph, out.shape[-1])
        #out = self.norm(out)
        #out = torch.add(torch.mul(out, self.norm_w), self.norm_b)
        #return self.activation(out.view(-1, out.shape[-1]))
        return self.activation(out), edge_index


class Edge_Conv(MessagePassing):
    def __init__(self,
                 num_edge_types,
                 in_features,
                 out_features,
                 embedding_out_features,
                 classifier_features,
                 embedding_num_classes,
                 dropout=0,
                 max_variable_candidates=5,
                 model_epoch=2,
                 add_self_loops=False,
                 bias=True,
                 max_node_per_graph=50,
                 aggr="mean",
                 device="cpu",
                 output_model="learning"):
        super(Edge_Conv, self).__init__(aggr=aggr)
        # params set
        self.num_edge_types = num_edge_types
        self.device = device
        self.output_model = output_model.lower()
        self.dropout = dropout
        self.model_epoch = model_epoch
        self.max_variable_candidates = max_variable_candidates
        # 先对值进行embedding
        self.value_embeddingLayer = EmbeddingLayer(embedding_num_classes,
                                                   in_features,
                                                   embedding_out_features,
                                                   device=device)
        '''
        self.MessagePassingNN = nn.ModuleList([
            FiLMConv(in_channels=embedding_out_features, out_channels=out_features, num_relations=self.num_edge_types)
            for _ in range(self.model_epoch)
        ])
        '''
        self.MessagePassingNN = nn.ModuleList([
            nn.ModuleList([
                EdgeConv(
                   nn.Linear(out_features*2, out_features)) for _ in range(self.num_edge_types)
            ]) for j in range(self.model_epoch)
        ])

        if self.output_model == "learning":
            self.vocabLayer = nn.Linear(out_features * 2 + 1, 1)

    def forward(self, x, edge_list: List[torch.tensor], 
                batch_map: torch.Tensor):
        # 要把edge_list进行拼接，然后还要制作edge_type,
        # edge_type : size(E*edge_list_nums) 序号从0开始。
        edge_type_list = []
        for e_i in range(len(edge_list)):
            edge = edge_list[e_i]
            edge_type_list.append(torch.ones(edge.shape[1]) * e_i)

        #edge_type = torch.cat(edge_type_list, dim=0)
        #mask = edge_type == 0
        #edge = torch.cat(edge_list, dim=1)

        x_embedding = self.value_embeddingLayer(x)

        last_node_states = x_embedding
        for m_epoch in range(self.model_epoch):
            out_list = []
            for i in range(len(edge_list)):
                edge = edge_list[i]
                if edge.shape[0] != 0:
                    last_node_states = F.dropout(last_node_states, self.dropout, training=self.training)
                    out_list.append(self.MessagePassingNN[m_epoch][i](x=last_node_states, edge_index=edge))
            cur_node_states = torch.div(sum(out_list), len(edge_list))
            last_node_states = last_node_states + cur_node_states

        out = last_node_states
        return out
        #return self.learning_output(out, slot_id, candidate_ids, candidate_masks)

    def python_output(self, out, slot_id):
        slot_embedding = out[slot_id]  # shape: g, d

        return self.vocabLayer(slot_embedding)

    def learning_output(self, out, slot_id, candidate_ids, candidate_masks):
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
        logits = self.vocabLayer(slot_cand_comb)  # shape: g, c, 1
        logits = torch.squeeze(logits, dim=-1)  # shape: g, c
        logits += (1.0 - candidate_masks.view(-1, self.max_variable_candidates)) * -1e7

        return logits