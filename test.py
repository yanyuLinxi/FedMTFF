from dataProcessing import PYGraphDataset, LRStaticGraphDataset
from torch_geometric.data import DataLoader

if __name__ == '__main__':
    from torch_geometric.data import DataLoader
    
   
    gd = PYGraphDataset("data/pytemp/validate_data", value_dict=None, type_dict=None, max_node_per_graph=512)
    for idx, example in enumerate(gd[:2]):
        print("*** Example ***")
        print("idx: {}".format(idx))
        print("label: {}".format(example.label))
        #print("x: {}".format(example.x))
        print("x shape: {}".format(example.x.shape))
        print("slot_id: {}".format(example.slot_id))
        print("candidate_ids: {}".format(example.candidate_ids))
        print("candidate_ids len: {}".format(len(example.candidate_ids)))
        print("candidate_masks: {}".format(example.candidate_masks))
        #print("ast: {}".format(example.ast))
        print("ast shape: {}".format(example.ast_index.shape))
        #print("ncs: {}".format(example.ncs))
        print("ncs shape: {}".format(example.ncs_index.shape))
        #print("comesFrom: {}".format(example.comesFrom))
        print("comesFrom shape: {}".format(example.comesFrom_index.shape))
        #print("computedFrom: {}".format(example.computedFrom))
        print("computedFrom shape: {}".format(example.computedFrom_index.shape))
    dl = DataLoader(gd, batch_size=32)
    for d in dl:
        print("d.x.shape", d.x.shape)
        print("d.comesFrom_index.shape", d.comesFrom_index.shape)
        print("d.ast_index.shape", d.ast_index.shape)


    # gd = LRStaticGraphDataset("data/lrtemp/validate_data", value_dict=None, type_dict=None, max_node_per_graph=10000)
    # for idx, example in enumerate(gd[:10]):
    #     print("*** Example ***")
    #     print("idx: {}".format(idx))
    #     print("label: {}".format(example.label))
    #     #print("x: {}".format(example.x))
    #     print("x shape: {}".format(example.x.shape))
    #     print("slot_id: {}".format(example.slot_id))
    #     print("candidate_ids: {}".format(example.candidate_ids))
    #     print("candidate_ids len: {}".format(len(example.candidate_ids)))
    #     print("candidate_masks: {}".format(example.candidate_masks))
    #     #print("ast: {}".format(example.ast))
    #     print("Child_index shape: {}".format(example.Child_index.shape))
    #     #print("ncs: {}".format(example.ncs))
    #     print("NextToken_index shape: {}".format(example.NextToken_index.shape))
    #     #print("comesFrom: {}".format(example.comesFrom))
    #     print("LastUse_index shape: {}".format(example.LastUse_index.shape))
    #     #print("computedFrom: {}".format(example.computedFrom))
    #     print("LastWrite_index shape: {}".format(example.LastWrite_index.shape))
    # dl = DataLoader(gd, batch_size=32)
    # print("==========dataloader")
    # for d in dl:
    #     print("d.x.shape", d.x.shape)
    #     print("d.Child_index.shape", d.Child_index.shape)
    #     print("d.NextToken_index.shape", d.NextToken_index.shape)