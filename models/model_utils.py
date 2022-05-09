from dataProcessing import PYGraphDataset, PYMake_task_input
from dataProcessing import PYGraphDatasetNaming, PYNamingMake_task_input
from dataProcessing import CSharpStaticGraphDataset, CSharpStaticMake_task_input
from dataProcessing import CSharpStaticGraphDatasetNaming, CSharpStaticNamingMake_task_input


from models import ResGAGN, GNN_FiLM, Edge_Conv, GGNN
from models import FL_FrameWork
from models import FL_Co_Attention_Four

def name_to_dataset_class(name: str, args):
    name = name.lower()
    #return classname, appendix attribute, make_task_input_function,
    if name in ["python"]:
        return PYGraphDataset, {}, PYMake_task_input
    if name in ["pythonnaming"]:
        return PYGraphDatasetNaming, {}, PYNamingMake_task_input
    if name in ["csharpstatic"]:
        return CSharpStaticGraphDataset, {
            "max_node_per_graph": args.max_node_per_graph,
        }, CSharpStaticMake_task_input
    if name in ["csharpstaticnaming"]:
        return CSharpStaticGraphDatasetNaming, {"max_node_per_graph":args.max_node_per_graph,}, CSharpStaticNamingMake_task_input

    raise ValueError("Unkown dataset name '%s'" % name)


def name_to_model_class(name: str, args):
    name = name.lower()
    #return classname, appendix attribute
    if name in ["resgagn"]:
        return ResGAGN, {}
    if name in ["fl_framework"]:
        return FL_FrameWork, {}
    if name in ["fl_co_attention_four"]:
        return FL_Co_Attention_Four, {}
    if name in ["edge_conv"]:
        return Edge_Conv, {"max_node_per_graph":args.max_node_per_graph}
    if name in ["gnn_film"]:
        return GNN_FiLM, {}
    if name in ["ggnn"]:
        return GGNN, {"max_node_per_graph":args.max_node_per_graph}
    raise ValueError("Unkown model name '%s'" % name)