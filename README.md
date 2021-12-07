# FedMTFF

This is an implementation of the Federated Learning based Multi-task Feature Fusion Framework (FedMTFF), which aims at integrating information on highly different tasks.

In this implementation, we discussed many module:

* **ResGAGN**: Residual connection Graph Attention network with Graph normalization. It deepens the network layers to extract more comprehensive semantic information. 
* **MTFF**: Multi-task feature fusion module, which uses the attention mechanism to fuse task-oriented features.
* **Federated Learning Strategy**: It uses a four-staged training strategy to separate feature extraction and feature fusion, which supports multiple clients running different tasks on different datasets for feature fusion.


# Running

To train a model, it suffices to run ```python FL_train_four.py --backbone_model resgagn```, for example as follows:
```python
$ python FL_train_four.py --backbone_model  resgagn
model_args:{}
{
    "dataset_name_task1": "csharpstatic",
    "dataset_name_task2": "csharpStaticnaming",
    "dataset_name_task3": "python",
    "dataset_name_task4": "pythonNaming",
    "FL_model": "fl_co_attention_four",
    "backbone_model": "resgagn",
    "task_load_model_dir": "checkpoint/",
    "task1": "varmisuse",
    "task2": "varnaming",
    "task3": "varmisuse",
    "task4": "varnaming",
    "optimizer": "Adam",
    "lr": 0.001,
    "lr_deduce_per_epoch": 10,
    "max_epochs": 1500,
    "cur_epoch": 1,
    "max_variable_candidates": 5,
    "batch_size": 64,
    "result_dir": "trained_models/",
    "dropout_rate": 0.0,
    "h_features": 64,
    "out_features": 64,
    "graph_node_max_num_chars": 19,
    "max_node_per_graph": 50,
    "device": "cuda",
    "value_dict_dir": "vocab_dict/python_terminal_dict_1k_value.json",
    "type_dict_dir": "vocab_dict/python_terminal_dict_1k_type.json",
    "vocab_size": 1001,
    "dataset_train_data_dir_task1": "data/csharp/train_data",
    "dataset_validate_data_dir_task1": "data/csharp/validate_data",
    "dataset_train_data_dir_task2": "data/csharp/train_data",
    "dataset_validate_data_dir_task2": "data/csharp/validate_data",
    "dataset_train_data_dir_task3": "data/python/train_data",
    "dataset_validate_data_dir_task3": "data/python/validate_data",
    "dataset_train_data_dir_task4": "data/python/train_data",
    "dataset_validate_data_dir_task4": "data/python/validate_data",
    "client_fraction": 1,
    "client_nums": 4,
    "client_max_epochs": 1,
    "num_model_params": 4640322
}
== Server Epoch 1
100%|███████████████████████| 1/1 [00:17<00:00, 17.14s/it] 
        client3 TaskFold.VARNAMING Train: loss: 2.59201 || Top1 0.709 Top2 0.735 Top3 0.741 Top4 0.746 Top5 0.750 || graphs/sec: 115.40 | nodes/sec: 5770 | graphs: 1978 | nodes: 98900
100%|███████████████████████| 1/1 [00:01<00:00,  1.75s/it]
        client2 TaskFold.VARMISUSE Train: loss: 1.22234 || Top1 0.417 Top2 0.674 Top3 0.852 Top4 0.945 Top5 1.000 || graphs/sec: 201.03 | nodes/sec: 10052 | graphs: 352 | nodes: 17600
100%|███████████████████████| 1/1 [00:16<00:00, 16.69s/it] 
        client0 TaskFold.VARMISUSE Train: loss: 0.66207 || Top1 0.705 Top2 0.906 Top3 0.974 Top4 0.991 Top5 1.000 || graphs/sec: 202.81 | nodes/sec: 10141 | graphs: 3385 | nodes: 169250
100%|███████████████████████| 1/1 [00:17<00:00, 17.71s/it] 
        client1 TaskFold.VARNAMING Train: loss: 2.17232 || Top1 0.462 Top2 0.805 Top3 0.830 Top4 0.832 Top5 0.835 || graphs/sec: 191.18 | nodes/sec: 9559 | graphs: 3385 | nodes: 169250
```

In this implementation, we fuse four clusters of clients where we regard running a task on a kind of language as a cluster of clients. 

Note that you can use ```--backbone_model``` to replace the implement model as follows:

* **resgagn**: The proposed network in this paper.
* **gnn_film** (Brockschmidt 2020): uses a feature-wiselinear modulation based GNN, where the passing mes-sage is conditioned on the target node features.
* **edge_conv** (Wang et al. 2019): proposes a GNN with theedge convolution module, which calculates edge featuresthat depended on the relationship between a node and itsneighbors.

# Datasets

Note that due to the limitations of CMT, we cannot upload all datasets. This will drastically change the results of the experiment. All our datasets come from public papers. For more accurate experiments, we strongly recommend that you download the datasets according to the following url and set them:

CSharp: https://aka.ms/iclr18-prog-graphs-dataset

Python: https://www.sri.inf.ethz.ch/research/plml

Download the two datasets and put the corresponding file into: ```data/$dataset name$/$train_data or validate_data$/raw ```.

Since the python dataset is too large compared to csharp, for more balanced training, we recommend that you limit the python dataset to 10,000 training data and 2,000 test data.

# Requirements

* torch >= 1.8.0
* torch-geometric  >= 1.7.0
* python >= 3.7
* numpy >= 1.19
* tqdm >= 4.56.1

We may have overlooked some other dependencies, please install the latest version of the package directly.

# SOTA
The sota log file is placed in ```sota``` folder. You can search the string "**Best epoch**"  from the last to the front to find the best validation results. 

In the log file:

* task1 => running VarMisuse on C#
* task2 => running Code Completion on C#
* task3 => running VarMisuse on python
* task4 => running Code Completion on python
