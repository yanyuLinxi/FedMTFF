from torch_geometric.nn import GATConv, SAGPooling, GraphConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
# from torch_geometric.nn import MessagePassing
import torch.nn as nn
import torch
import torch.nn.functional as F
from .embedding_layer import EmbeddingLayer
from typing import List, Tuple
from .graph_norm import GraphNorm
from torch_geometric.nn.conv import MessagePassing

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_sparse import SparseTensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import softmax

class graph_transformer_network(MessagePassing):
    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        beta: bool = False,
        dropout: float = 0.,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        root_weight: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super(graph_transformer_network, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        self._alpha = None

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = Linear(in_channels[0], heads * out_channels)
        self.lin_query = Linear(in_channels[1], heads * out_channels)
        self.lin_value = Linear(in_channels[0], heads * out_channels)
        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = self.register_parameter('lin_edge', None)

        if concat:
            self.lin_skip = Linear(in_channels[1], heads * out_channels,
                                   bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        else:
            self.lin_skip = Linear(in_channels[1], out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.edge_dim:
            self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, return_attention_weights=None):
        # type: (Union[Tensor, PairTensor], Tensor, OptTensor, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, PairTensor], SparseTensor, OptTensor, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, PairTensor], Tensor, OptTensor, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, PairTensor], SparseTensor, OptTensor, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """

        H, C = self.heads, self.out_channels

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        query = self.lin_query(x[1]).view(-1, H, C)
        key = self.lin_key(x[0]).view(-1, H, C)
        value = self.lin_value(x[0]).view(-1, H, C)

        # propagate_type: (query: Tensor, key:Tensor, value: Tensor, edge_attr: OptTensor) # noqa
        out = self.propagate(edge_index, query=query, key=key, value=value,
                             edge_attr=edge_attr, size=None)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.root_weight:
            x_r = self.lin_skip(x[1])
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
                beta = beta.sigmoid()
                out = beta * x_r + (1 - beta) * out
            else:
                out += x_r

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads,
                                                      self.out_channels)
            key_j += edge_attr

        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = value_j
        if edge_attr is not None:
            out += edge_attr

        out *= alpha.view(-1, self.heads, 1)
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')



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


class residual_graph_attention(MessagePassing):
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
        super(residual_graph_attention, self).__init__(aggr=aggr)
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
        # self.norm = GraphNorm(out_features)  # embedding_norm
        '''
        self.type_embeddingLayer = EmbeddingLayer(embedding_num_classes,
                                                  in_features,
                                                  embedding_out_features,
                                                  device=device)
        '''
        # 然后进行gat
        # 输出值。
        self.MessagePassingNN =  nn.ModuleList([
                graph_transformer_network(out_features, out_features, heads=8, concat=False)
                #nn.LayerNorm((self.max_node_per_graph, out_features),elementwise_affine=True),
                #nn.ReLU(),
                for _ in range(self.num_edge_types)
            ])
        self.lin = nn.Linear(in_features=out_features*self.num_edge_types, out_features=out_features)

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
        #x_embedding_out = self.norm(x_embedding, batch_map)
        #x_embedding_out = x_embedding_out.repeat(1, self.num_edge_types)  # shape: V D*E

        last_node_states = x_embedding
        out_concat = []
        for i in range(len(edge_list)):
            edge = edge_list[i]
            if edge.shape[0] != 0:
                # 该种类型的边存在边
                out_concat.append(self.MessagePassingNN[i](last_node_states, edge))

        #out = last_node_states + x_embedding_out
        out = torch.cat(out_concat, dim=1)
        out = self.lin(out)
        return out

