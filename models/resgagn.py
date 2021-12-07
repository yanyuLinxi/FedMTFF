from torch_geometric.nn import GATConv, SAGPooling, GraphConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import GCNConv
from layers import EmbeddingLayer
import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import List, Tuple
from layers import GraphNorm
from utils import TaskFold


class BasicGAT(MessagePassing):
    def __init__(self, in_channels, out_channels, heads, concat, dropout, add_self_loops, bias, max_node_per_graph):
        super(BasicGAT, self).__init__()
        self.max_node_per_graph = max_node_per_graph
        self.gatconv = GATConv(in_channels=in_channels,
                               out_channels=out_channels,
                               heads=heads,
                               concat=concat,
                               dropout=dropout,
                               add_self_loops=add_self_loops,
                               bias=bias)
        self.lin1 = nn.Linear(heads*out_channels,out_channels)

        # 使用官方写的norm
        self.norm = GraphNorm(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x, edge, batch_map):
        out = self.gatconv(x, edge)
        out = F.relu(out)
        out = self.lin1(out)
        out = self.norm(out, batch_map)
        #return self.activation(out.view(-1, out.shape[-1]))
        return self.activation(out)


class ResGAGN(MessagePassing):
    """deep gat + graph attention pooling
    graph norm + mapping module
    
    deep gat:
       1. 8层gat，每一层一个graph norm。
       2. 每层的gat为1（待做实验确定最好的）
       3. 每层结束后进行concat。然后送入下一层。
    
    结束后的值 进行graph pooling + gcn

    Args:
        MessagePassing ([type]): [description]
    """
    def __init__(self,
                 num_edge_types,
                 in_features,
                 out_features,
                 embedding_out_features,
                 classifier_features,
                 embedding_num_classes,
                 heads=8,
                 concat=False,
                 dropout=0,
                 max_node_per_graph=50,
                 model_epoch=6,
                 max_variable_candidates=5,
                 add_self_loops=False,
                 bias=True,
                 aggr="mean",
                 device="cpu"):
        super(ResGAGN, self).__init__(aggr=aggr)
        # params set
        self.device = device
        self.max_variable_candidates = max_variable_candidates
        self.max_node_per_graph = max_node_per_graph
        self.model_epoch = model_epoch
        # 先对值进行embedding
        self.num_edge_types = num_edge_types
        # 先对值进行embedding
        self.value_embeddingLayer = EmbeddingLayer(embedding_num_classes,
                                                   in_features,
                                                   embedding_out_features,
                                                   device=device)
        self.norm = GraphNorm(out_features)  # embedding_norm

        # 然后进行gat
        # 输出值。
        self.MessagePassingNN = nn.ModuleList([
            nn.ModuleList([
                BasicGAT(in_channels=embedding_out_features,
                         out_channels=out_features,
                         heads=4,
                         concat=True,
                         dropout=dropout,
                         add_self_loops=add_self_loops,
                         bias=bias,
                         max_node_per_graph=max_node_per_graph)
                #nn.LayerNorm((self.max_node_per_graph, out_features),elementwise_affine=True),
                #nn.ReLU(),
                for _ in range(self.num_edge_types)
            ]) for j in range(self.model_epoch)
        ])
        self.lin = nn.ModuleList([
            nn.Linear(in_features=out_features*self.num_edge_types, out_features=out_features) for _ in range(self.model_epoch)
        ])


        self.output_linear = nn.Sequential(
            nn.Linear(out_features*self.model_epoch, out_features),
            nn.ReLU(),
        )
        self.varmisuse_output_layer = nn.Linear(out_features * 2 + 1, 1)


        
        self.varnaming_output_layer = nn.Linear(out_features, classifier_features)

    def forward(self,
                x,
                edge_list: List[torch.tensor],
                batch_map: torch.Tensor,):
        x_embedding = self.value_embeddingLayer(x)
        x_embedding_out = self.norm(x_embedding, batch_map)

        last_node_states = x_embedding_out
        out_concat = []
        for m_epoch in range(self.model_epoch):
            out_list = []
            for i in range(len(edge_list)):
                edge = edge_list[i]
                if edge.shape[0] != 0:
                    # 该种类型的边存在边
                    out_list.append(self.MessagePassingNN[m_epoch][i](last_node_states, edge, batch_map))

            node_states_cat = torch.cat(out_list, dim=1)
            cur_node_state = self.lin[m_epoch](node_states_cat)
            last_node_states = last_node_states + cur_node_state
            out_concat.append(last_node_states)
        out = torch.cat(out_concat, dim=1)

        return self.output_linear(out)


    def varmisuse_learning_output(self, out, slot_id, candidate_ids, candidate_masks):
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
        logits = self.varmisuse_output_layer(slot_cand_comb)  # shape: g, c, 1
        logits = torch.squeeze(logits, dim=-1)  # shape: g, c
        logits += (1.0 - candidate_masks.view(-1, self.max_variable_candidates)) * -1e7

        return logits

    def varnaming_learning_output(self, out, slot_id, global_embedding):
        slot_embedding = out[slot_id]  # shape: g, d

        return self.varnaming_output_layer(torch.cat([slot_embedding, global_embedding], dim=1))  # g, d*2 => g, classifier