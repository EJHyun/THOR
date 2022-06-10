import pickle
import copy
from torch.utils.data import DataLoader
import torch

dataset_list = ["ICEWS14","ICEWS0515","YAGO11k"]
for dataset in dataset_list:
    print("Preparing train & test quadruples of",dataset)
    print("\nRead dataset & Make Vocab/ID")
    train_data_directory = "./"+dataset+"/train.txt"
    valid_data_directory = "./"+dataset+"/valid.txt"
    test_data_directory = "./"+dataset+"/test.txt"
    train_quadruples = open(train_data_directory, 'r', encoding="UTF-8").read().lower().splitlines()
    valid_quadruples = open(valid_data_directory, 'r', encoding="UTF-8").read().lower().splitlines()
    test_quadruples = open(test_data_directory, 'r', encoding="UTF-8").read().lower().splitlines()
    train_quadruples = list(map(lambda x: x.split("\t"), train_quadruples))
    valid_quadruples = list(map(lambda x: x.split("\t"), valid_quadruples))
    test_quadruples = list(map(lambda x: x.split("\t"), test_quadruples))
    quadruples = []
    quadruples.extend(train_quadruples)
    quadruples.extend(valid_quadruples)
    quadruples.extend(test_quadruples)
    total_head_list, total_relation_list, total_tail_list, total_time_list = zip(*(quadruples))
    total_head_list = list(total_head_list)
    total_relation_list = list(total_relation_list)
    total_tail_list = list(total_tail_list)
    total_time_list = list(total_time_list)
    total_entity_list = copy.deepcopy(total_head_list)
    total_entity_list.extend(total_tail_list)
    total_entity_vocab = sorted(list(set(total_entity_list)))
    total_relation_vocab = sorted(list(set(total_relation_list)))
    total_time_vocab = sorted(list(set(total_time_list)))
    total_entity_id = list(range(len(total_entity_vocab)))
    total_relation_id = list(range(len(total_relation_vocab)))
    total_time_id = list(range(len(total_time_vocab)))
    total_relation_list_id = list(map(lambda x: total_relation_vocab.index(x), total_relation_list))
    total_head_list_id = list(map(lambda x: total_entity_vocab.index(x), total_head_list))
    total_tail_list_id = list(map(lambda x: total_entity_vocab.index(x), total_tail_list))
    total_time_list_id = list(map(lambda x: total_time_vocab.index(x), total_time_list))
    head_list, relation_list, tail_list, time_list = zip(*(train_quadruples))
    head_list = list(head_list)
    relation_list = list(relation_list)
    tail_list = list(tail_list)
    time_list = list(time_list)
    entity_list = copy.deepcopy(head_list)
    entity_list.extend(tail_list)
    entity_vocab = sorted(list(set(entity_list)))
    relation_vocab = sorted(list(set(relation_list)))
    time_vocab = sorted(list(set(time_list)))
    entity_id = list(range(len(entity_vocab)))
    relation_id = list(range(len(relation_vocab)))
    time_id = list(range(len(time_vocab)))
    relation_list_id = list(map(lambda x: total_relation_vocab.index(x), relation_list))
    head_list_id = list(map(lambda x: entity_vocab.index(x), head_list))
    tail_list_id = list(map(lambda x: entity_vocab.index(x), tail_list))
    time_list_id = list(map(lambda x: total_time_vocab.index(x), time_list))
    num_of_time = len(total_time_id)
    num_of_rel = len(total_relation_id)
    num_of_ent = len(total_entity_id)
    neighbor_batch_size = len(total_entity_vocab)
    trainset_batch_size = 100
    train_quadruples_ids = list(map(lambda x: (entity_vocab.index(x[0]),total_relation_vocab.index(x[1]), entity_vocab.index(x[2]),total_time_vocab.index(x[3])), train_quadruples))
    reciprocal_train_quadruples_ids = list(map(lambda x: (entity_vocab.index(x[2]), total_relation_vocab.index(x[1]) + num_of_rel, entity_vocab.index(x[0]),total_time_vocab.index(x[3])), train_quadruples))
    train_quadruples_ids.extend(reciprocal_train_quadruples_ids)
    train_dataloader = DataLoader(train_quadruples_ids, batch_size=trainset_batch_size, shuffle=True)
    test_quadruples_id = list(map(lambda x: (total_entity_vocab.index(x[0]), total_relation_vocab.index(x[1]), total_entity_vocab.index(x[2]),total_time_vocab.index(x[3])), test_quadruples))
    valid_quadruples_id = list(map(lambda x: (total_entity_vocab.index(x[0]), total_relation_vocab.index(x[1]), total_entity_vocab.index(x[2]),total_time_vocab.index(x[3])), valid_quadruples))
    train_dataloader_list = []
    for s, r, o, t in train_dataloader:
        train_dataloader_list.append([s.tolist(), r.tolist(), o.tolist(), t.tolist()])
    G_train_facts = set(train_quadruples_ids)
    filter_total = set(list(map(lambda x: (total_entity_vocab.index(x[0]), total_relation_vocab.index(x[1]), total_entity_vocab.index(x[2])), quadruples)))
    filter_time = {}
    for t__ in range(num_of_time):
        filter_time[t__] = set()
    for s, r, o, t, in quadruples:
        filter_time[total_time_vocab.index(t)].add((total_entity_vocab.index(s),total_relation_vocab.index(r),total_entity_vocab.index(o)))
    test_data = []
    entity_index = list(range(num_of_ent))
    for s,r,o,t in test_quadruples_id:
        o_filter = [entity_index.index(test_o) for test_o in entity_index if (s, r, test_o) in filter_total]
        s_filter = [entity_index.index(test_s) for test_s in entity_index if (test_s, r, o) in filter_total]
        o_filter_mask = torch.tensor([True] * num_of_ent)
        s_filter_mask = torch.tensor([True] * num_of_ent)
        o_filter_mask[o_filter] = torch.tensor([False])
        s_filter_mask[s_filter] = torch.tensor([False])
        o_time_filter = [entity_index.index(test_o) for test_o in entity_index if (s, r, test_o) in filter_time[t]]
        s_time_filter = [entity_index.index(test_s) for test_s in entity_index if (test_s, r, o) in filter_time[t]]
        o_time_filter_mask = torch.tensor([True] * num_of_ent)
        s_time_filter_mask = torch.tensor([True] * num_of_ent)
        o_time_filter_mask[o_time_filter] = torch.tensor([False])
        s_time_filter_mask[s_time_filter] = torch.tensor([False])
        test_data.append((s, r, o, t, o_filter_mask, s_filter_mask, o_time_filter_mask, s_time_filter_mask))

    valid_data = []
    for s,r,o,t in valid_quadruples_id:
        o_filter = [entity_index.index(test_o) for test_o in entity_index if (s, r, test_o) in filter_total]
        s_filter = [entity_index.index(test_s) for test_s in entity_index if (test_s, r, o) in filter_total]
        o_filter_mask = torch.tensor([True] * num_of_ent)
        s_filter_mask = torch.tensor([True] * num_of_ent)
        o_filter_mask[o_filter] = torch.tensor([False])
        s_filter_mask[s_filter] = torch.tensor([False])
        o_time_filter = [entity_index.index(test_o) for test_o in entity_index if (s, r, test_o) in filter_time[t]]
        s_time_filter = [entity_index.index(test_s) for test_s in entity_index if (test_s, r, o) in filter_time[t]]
        o_time_filter_mask = torch.tensor([True] * num_of_ent)
        s_time_filter_mask = torch.tensor([True] * num_of_ent)
        o_time_filter_mask[o_time_filter] = torch.tensor([False])
        s_time_filter_mask[s_time_filter] = torch.tensor([False])
        valid_data.append((s, r, o, t, o_filter_mask, s_filter_mask, o_time_filter_mask, s_time_filter_mask))
    num_of_time = len(total_time_id)
    num_of_rel = len(total_relation_id)
    num_of_ent = len(total_entity_id)
    num_of_train_ent = len(entity_id)
    neighbor_batch_size = len(total_entity_vocab)
    data = {
        'nums': [num_of_time, num_of_rel, num_of_ent, num_of_train_ent],
        'train_data': train_dataloader_list,
        'valid_data': valid_data,
        'test_data' : test_data,
        'train_facts': G_train_facts,
    }
    with open('data_'+dataset+'.pickle', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

