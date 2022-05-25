from torch_geometric.nn import GATConv, SAGPooling, GCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import GCNConv
import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import List, Tuple

from utils.model_metrics import cal_metrics

# 本地库


class VarnamingOutputLayer(nn.Module):
    def __init__(
        self,
        out_features,
        classifier_nums,
        criterion=nn.CrossEntropyLoss(),
        metrics=cal_metrics,
        device="cpu",
    ):
        super(VarnamingOutputLayer, self).__init__()
        
        self.varnaming_linear = nn.Sequential(
            nn.Linear(out_features, out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features),
        )
        self.varmisuse_layer = nn.Linear(out_features, classifier_nums)
        self.criterion=criterion
        self.metrics = metrics

    def forward(self,
                output,
                slot_id,
                value_label=None, 
                **kwargs):
        output = self.varnaming_linear(output)
        
        slot_embedding = output[slot_id]  # shape: g, d
        
        logits = self.varmisuse_layer(slot_embedding)
        

        result = [logits]

        if value_label is not None:
            loss = self.criterion(logits, value_label)
            result.append(loss)
        
        if self.metrics:
            metrics = cal_metrics(F.softmax(logits, dim=-1), value_label)
            result.append(metrics)
            
        return result