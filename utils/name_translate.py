from enum import Enum
import json
from multiprocessing import cpu_count

# 本地库
from models import GGNN
from dataProcessing import CSharpStaticGraphDatasetGenerator
from tasks import VarmisuseOutputLayer


class DataFold(Enum):
    TRAIN = 0
    VALIDATION = 1
    TEST = 2


concat_singal = "_- ,/\\"


def name_to_dataset(name: str, path: str, data_fold: DataFold, args, num_workers=int(cpu_count() / 2)):  # 返回类型为对象
    """将字符串转为数据预处理对象

    Args:
        name (str): 数据预处理的类名
        path (str): 数据目录地址
        args (_type_): 可能用到的参数

    Raises:
        ValueError: 类名不存在
    """
    name = name.lower()
    name = name.replace(concat_singal, "")
    if name in ["csharp"]:
        datasetCls = CSharpStaticGraphDatasetGenerator
    else:
        raise ValueError("Unkown dataset name '%s'" % name)

    # 导入字典
    with open(args.value_dict_dir, 'r') as f:
        value_dict = json.load(f)
    with open(args.type_dict_dir, 'r') as f:
        type_dict = json.load(f)

    # 返回数据库对象
    return datasetCls(
        path,
        value_dict,
        type_dict,
        num_edge_types=args.num_edge_types,
        batch_size=args.batch_size,
        shuffle=True if data_fold == DataFold.TRAIN else False,
        graph_node_max_num_chars=args.graph_node_max_num_chars,
        max_graph=20000,
        max_variable_candidates=args.max_variable_candidates,
        max_node_per_graph=args.max_node_per_graph,
        num_workers=num_workers,
        device=args.device,
    )


def name_to_model(name: str, args):
    """将字符串转为模型并返回。

    Args:
        name (str): 模型名称。推荐小写。
        args (_type_): 可能会用到的参数

    Raises:
        ValueError: 类名不存在
    """
    name = name.lower()
    name = name.replace(concat_singal, "")
    if name in ["ggnn", "graphgatedneuralnetwork"]:
        # TODO: 修改GGNN后修改参数。
        return GGNN(num_edge_types=args.num_edge_types,
                    in_features=args.graph_node_max_num_chars,
                    out_features=args.out_features,
                    embedding_out_features=args.h_features,
                    embedding_num_classes=70,
                    dropout=args.dropout_rate,
                    max_variable_candidates=args.max_variable_candidates,
                    device=args.device
                    )
    else:
        raise ValueError("Unkown model name '%s'" % name)


def name_to_output_model(name: str, args):
    """将字符串转为输出模型并返回。

    Args:
        name (str): 输出模型名称。推荐小写。
        args (_type_): 可能会用到的参数

    Raises:
        ValueError: 类名不存在
    """
    name = name.lower()
    name = name.replace(concat_singal, "")
    if name in ["vm", "varmisuse", "variablemisuse"]:
        return VarmisuseOutputLayer(out_features=args.out_features,
                                    max_variable_candidates=args.max_variable_candidates,
                                    device=args.device)
    elif name in ["cc", "codecompletion", "varnaming"]:
        # TODO: 待补充
        pass
    else:
        raise ValueError("Unkown output model name '%s'" % name)