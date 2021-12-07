# encoding:utf-8
import os
import json
from collections import Counter
import gzip

# 统计所有变量（终结符）出现的次数，并创建频率字典。
# 根据频率字典排序取前N(指定)个。
def statistic_terminal_value(path: str,
                             terminal_dict_size,
                             suffix: str = ".json"):
    # 统计path下面的所有json文件。

    def count_single_json_terminal_dict(raw: str, value_terminal_dict: dict, type_terminal_dict):

        #candidateNodeList = json_dict['candidateNodeList']  # list
        for node in raw:
            node_value = node.get("value", "EMPTY")
            node_type = node.get("type")

            value_terminal_dict[node_value] = value_terminal_dict[node_value] + 1
            type_terminal_dict[node_type] = type_terminal_dict[node_type] + 1

    value_terminal_dict = Counter()
    type_terminal_dict = Counter()
    for root, dirs, files in os.walk(path):
        for file in files:
            if not file.endswith(suffix):
                continue
            absolute_file_path = os.path.join(root, file)
            with open(absolute_file_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    raw = json.loads(line)
                    count_single_json_terminal_dict(raw, value_terminal_dict, type_terminal_dict)

    value_terminal_dict_list_sorted = value_terminal_dict.most_common(terminal_dict_size)
    terminal_value_index_dict_sorted = {value_terminal_dict_list_sorted[i][0]: i for i in range(len(value_terminal_dict_list_sorted))}

    type_terminal_dict_list_sorted = type_terminal_dict.most_common(terminal_dict_size)
    terminal_type_index_dict_sorted = {type_terminal_dict_list_sorted[i][0]: i for i in range(len(type_terminal_dict_list_sorted))}
    return terminal_value_index_dict_sorted, value_terminal_dict_list_sorted, terminal_type_index_dict_sorted, type_terminal_dict_list_sorted


def statistic_learning_dataset_terminal_value(path: str,
                             terminal_dict_size,
                             suffix: str = ".jsonl.gz"):
    # 统计path下面的所有json文件。

    def count_single_json_terminal_dict(raw: str, value_terminal_dict: dict, type_terminal_dict):

        #candidateNodeList = json_dict['candidateNodeList']  # list
        ContextGraph = raw["ContextGraph"]
        SymbolCandidates = raw["SymbolCandidates"]
        NodeLabels = ContextGraph["NodeLabels"]
        NodeTypes = ContextGraph["NodeTypes"]

        correct_candidate_id = None
        for candidate in SymbolCandidates:
            if candidate["IsCorrect"]:
                correct_candidate_id = candidate['SymbolDummyNode']

        if correct_candidate_id is None:
            return

        node_value = NodeLabels[str(correct_candidate_id)]
        node_type = NodeTypes[str(correct_candidate_id)]

        value_terminal_dict[node_value] = value_terminal_dict[node_value] + 1
        type_terminal_dict[node_type] = type_terminal_dict[node_type] + 1

    value_terminal_dict = Counter()
    type_terminal_dict = Counter()
    for root, dirs, files in os.walk(path):
        for file in files:
            if not file.endswith(suffix):
                continue
            absolute_file_path = os.path.join(root, file)
            with gzip.open(absolute_file_path, "r") as f:
                raw_data = f.read()
                data_utf8 = raw_data.decode("utf-8")
                for raw_line in data_utf8.split("\n"):
                    if len(raw_line) == 0:  # 过滤最后一行的空行。
                        continue
                    raw_dict = json.loads(raw_line)
                    count_single_json_terminal_dict(raw_dict, value_terminal_dict, type_terminal_dict)

    value_terminal_dict_list_sorted = value_terminal_dict.most_common(terminal_dict_size)
    terminal_value_index_dict_sorted = {value_terminal_dict_list_sorted[i][0]: i for i in range(len(value_terminal_dict_list_sorted))}

    type_terminal_dict_list_sorted = type_terminal_dict.most_common(terminal_dict_size)
    terminal_type_index_dict_sorted = {type_terminal_dict_list_sorted[i][0]: i for i in range(len(type_terminal_dict_list_sorted))}
    return terminal_value_index_dict_sorted, value_terminal_dict_list_sorted, terminal_type_index_dict_sorted, type_terminal_dict_list_sorted



def terminal_dict_save(terminal_dict: dict, terminal_dict_list_sorted: dict, fileFolderPath: str = "dataProcessing/", fileName: str = "terminal_dict.json"):
    if not os.path.exists(fileFolderPath):
        os.makedirs(fileFolderPath)
    absolute_path = os.path.join(fileFolderPath, fileName)
    appear_freq_absolute_path = os.path.join(fileFolderPath, "appear_freq_"+fileName)
    with open(absolute_path, 'w') as f:
        json.dump(terminal_dict, f, indent=4)
    with open(appear_freq_absolute_path, 'w') as f:
        json.dump(terminal_dict_list_sorted, f, indent=4)


data_dir = r"W:\Scripts\Python\Vscode\tf-gnn-samples-master\data\learning"
terminal_dict_dst_dir = "vocab_dict/"
value_file_name = "learning_terminal_dict_1k_value.json"
type_file_name = "learning_terminal_dict_1k_type.json"
terminal_dict_size = 1000  # 1000 is EMPTY, 这个值是字典里的个数。在训练时考虑了空。这里不用考虑。
if __name__ == '__main__':
    terminal_value_index_dict_sorted, terminal_dict_list_sorted, terminal_type_index_dict_sorted, type_terminal_dict_list_sorted = statistic_learning_dataset_terminal_value(data_dir, terminal_dict_size)
    terminal_dict_save(terminal_value_index_dict_sorted, terminal_dict_list_sorted, terminal_dict_dst_dir, fileName=value_file_name)
    terminal_dict_save(terminal_type_index_dict_sorted, type_terminal_dict_list_sorted, terminal_dict_dst_dir, fileName=type_file_name)