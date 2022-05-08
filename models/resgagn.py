from torch_geometric.nn import GATConv, SAGPooling, GraphConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import MessagePassing
import torch.nn as nn
import torch
import torch.nn.functional as F
from .embedding_layer import EmbeddingLayer
from typing import List, Tuple
from .graph_norm import GraphNorm


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
                 embedding_num_classes,
                 dropout=0,
                 max_node_per_graph=50,
                 model_epoch=6,
                 add_self_loops=False,
                 bias=True,
                 aggr="mean",
                 device="cpu"):
        super(ResGAGN, self).__init__(aggr=aggr)
        # params set
        self.device = device
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
        '''
        self.type_embeddingLayer = EmbeddingLayer(embedding_num_classes,
                                                  in_features,
                                                  embedding_out_features,
                                                  device=device)
        '''
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
            #nn.Linear(out_features, out_features),
            #nn.ReLU(),
        )

        """
        self.conv1 = GCNConv(out_features*self.model_epoch, out_features, add_self_loops=add_self_loops)
        self.pool1 = SAGPooling(out_features, ratio=0.5)  # 0.5 pooling ratio

        self.conv2 = GCNConv(out_features, out_features, add_self_loops=add_self_loops)
        self.pool2 = SAGPooling(out_features, ratio=0.5)  # 0.5 pooling ratio

        self.conv3 = GCNConv(out_features, out_features, add_self_loops=add_self_loops)
        self.pool3 = SAGPooling(out_features, ratio=0.5)  # 0.5 pooling ratio
        """

    def forward(self,
                x,
                edge_list: List[torch.tensor],
                batch_map: torch.Tensor,
                **kwargs):
        x_embedding = self.value_embeddingLayer(x)
        x_embedding_out = self.norm(x_embedding, batch_map)
        #x_embedding_out = x_embedding_out.repeat(1, self.num_edge_types)  # shape: V D*E

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
            #node_states_cat = torch.div(sum(out_list), len(edge_list))
            last_node_states = last_node_states + cur_node_state
            out_concat.append(last_node_states)
        #out = last_node_states + x_embedding_out
        out = torch.cat(out_concat, dim=1)
        out = self.output_linear(out)
        return out

