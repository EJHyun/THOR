import dgl
import torch
from dgl.data.utils import load_graphs

import numpy as np
import copy

from dgl.data.utils import save_graphs
import argparse

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser(description="Constructing Temporal Knowledge Graph")
parser.add_argument("--dataset", default="ICEWS14", choices=["ICEWS14", "ICEWS0515", "YAGO11k"], help="dataset folder name, which has train.txt, test.txt, valid.txt in it")
parser.add_argument("--window_size", default=8, type=int, help="window size to define relevant facts for each timestamps")
args = parser.parse_args()
dataset = args.dataset
half_window_size = args.window_size
neighbor_time_window_size = half_window_size*2 + 1

train_data_directory = "./"+dataset+"/train.txt"
valid_data_directory = "./"+dataset+"/valid.txt"
test_data_directory = "./"+dataset+"/test.txt"

train_quadruples = open(train_data_directory, 'r', encoding="UTF-8").read().lower().splitlines()
valid_quadruples = open(valid_data_directory, 'r', encoding="UTF-8").read().lower().splitlines()
test_quadruples = open(test_data_directory, 'r', encoding="UTF-8").read().lower().splitlines()
train_quadruples = list(map(lambda x: x.split("\t"), train_quadruples))
valid_quadruples = list(map(lambda x: x.split("\t"), valid_quadruples))
test_quadruples = list(map(lambda x: x.split("\t"), test_quadruples))

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

total_relation_list_id = list(map(lambda x: total_relation_vocab.index(x), relation_list))
total_head_list_id = list(map(lambda x: total_entity_vocab.index(x), head_list))
total_tail_list_id = list(map(lambda x: total_entity_vocab.index(x), tail_list))
total_time_list_id = list(map(lambda x: total_time_vocab.index(x), time_list))

relation_list_id = list(map(lambda x: total_relation_vocab.index(x), relation_list))
head_list_id = list(map(lambda x: entity_vocab.index(x), head_list))
tail_list_id = list(map(lambda x: entity_vocab.index(x), tail_list))
time_list_id = list(map(lambda x: total_time_vocab.index(x), time_list))

num_of_time = len(total_time_id)
num_of_rel = len(total_relation_id)
num_of_ent = len(total_entity_id)

Test_Global_Graph = dgl.DGLGraph(multigraph=True)
Test_Global_Graph.add_nodes(num_of_ent)
Test_Global_Graph.add_edges(total_head_list_id,total_tail_list_id)
Test_Global_Graph.add_edges(total_tail_list_id,total_head_list_id)
Test_Global_Graph.ndata['node_idx'] = torch.tensor(list(range(len(total_entity_id))))
Test_Global_Graph.edata['relation_idx'] = torch.cat([torch.tensor(total_relation_list_id), torch.tensor(total_relation_list_id) + torch.tensor(num_of_rel)]) # reciprocal
save_graphs("./"+dataset+"_Test_Global_Graph.bin", [Test_Global_Graph])


Test_time_split_Graph = dgl.DGLGraph(multigraph=True)
Test_time_split_Graph.add_nodes(num_of_ent * num_of_time)
rel_id_stack_list = []
for head_, time_, tail_, rel_ in zip(total_head_list_id, total_time_list_id, total_tail_list_id, total_relation_list_id):
    if time_ - half_window_size < 0 :
        rel_id_stack_list.extend([rel_] * (half_window_size + 1 + time_))
        Test_time_split_Graph.add_edges(np.array([num_of_time * head_ + time_] * (half_window_size + 1 + time_)),
                                       np.array(list(range(0, time_ + half_window_size+1))) + np.array([num_of_time * tail_]))
    elif time_ + half_window_size >= num_of_time :
        rel_id_stack_list.extend([rel_] * (num_of_time - time_ + half_window_size))
        Test_time_split_Graph.add_edges(np.array([num_of_time * head_ + time_] * (num_of_time - time_ + half_window_size)), 
                                       np.array(list(range(time_ - half_window_size, num_of_time))) + np.array([num_of_time * tail_]))
    else :
        rel_id_stack_list.extend([rel_] * neighbor_time_window_size)
        Test_time_split_Graph.add_edges(np.array([num_of_time * head_ + time_] * neighbor_time_window_size), 
                                       np.array(list(range(time_ - half_window_size, time_ + half_window_size+1))) + np.array([num_of_time * tail_]))
# Reciprocal edge
for head_, time_, tail_, rel_ in zip(total_head_list_id, total_time_list_id, total_tail_list_id, total_relation_list_id):
    if time_ - neighbor_time_window_size < -half_window_size-1:
        rel_id_stack_list.extend([rel_ + num_of_rel]*(half_window_size + 1 + time_))
        Test_time_split_Graph.add_edges(np.array(list(range(0, time_ + half_window_size+1))) + np.array([num_of_time * tail_]),
                                       np.array([num_of_time * head_ + time_] * (half_window_size + 1 + time_)))
    elif num_of_time - time_ - neighbor_time_window_size < -half_window_size:
        rel_id_stack_list.extend([rel_ + num_of_rel]*(num_of_time - time_ + half_window_size))
        Test_time_split_Graph.add_edges(np.array(list(range(time_ - half_window_size, num_of_time))) + np.array([num_of_time * tail_]), 
                                       np.array([num_of_time * head_ + time_] * (num_of_time - time_ + half_window_size)))
    else:
        rel_id_stack_list.extend([rel_ + num_of_rel]*neighbor_time_window_size)
        Test_time_split_Graph.add_edges(np.array(list(range(time_ - half_window_size, time_ + half_window_size+1))) + np.array([num_of_time * tail_]), 
                                       np.array([num_of_time * head_ + time_] * neighbor_time_window_size))

Test_time_split_Graph.ndata['node_idx'] = torch.tensor(list(range(len(total_entity_id) * num_of_time)))
Test_time_split_Graph.ndata['entity_idx'] = torch.tensor(list(range(len(total_entity_id)))).unsqueeze(1).repeat(1, num_of_time).view(-1)
Test_time_split_Graph.edata['relation_idx'] = torch.tensor(rel_id_stack_list)
save_graphs("./window"+str(half_window_size)+'_'+dataset+"_Test_time_split_Graph.bin", [Test_time_split_Graph])

Test_Line_Graph = dgl.transform.line_graph(Test_time_split_Graph, shared = True, backtracking = False)
Test_Relation_Graph_ = dgl.DGLGraph(multigraph=True)
Line_rel_num = len(set(Test_Line_Graph.ndata['relation_idx'].tolist()))
Test_Relation_Graph_.add_nodes(Line_rel_num)
Test_Relation_Graph_.add_edges(Test_Line_Graph.ndata['relation_idx'][Test_Line_Graph.edges()[0]], Test_Line_Graph.ndata['relation_idx'][Test_Line_Graph.edges()[1]])
edge_dict = {}
for u, v in zip(Test_Relation_Graph_.edges()[0].tolist(), Test_Relation_Graph_.edges()[1].tolist()):
    if (u,v) in edge_dict.keys():
        edge_dict[(u,v)] += 1
    else:
        edge_dict[(u,v)] = 1
Test_Relation_Graph = dgl.DGLGraph(multigraph=True)
Test_Relation_Graph.add_nodes(Line_rel_num)
for i in list(edge_dict.keys()):
    Test_Relation_Graph.add_edges(i[0], i[1])
Test_Relation_Graph.ndata['relation_idx'] = torch.tensor(list(range(Line_rel_num)))
save_graphs("./window"+str(half_window_size)+'_'+dataset+"_Test_Relation_Graph.bin", [Test_Relation_Graph])

Train_Global_Graph = dgl.DGLGraph(multigraph=True)
Train_Global_Graph.add_nodes(len(entity_id))
Train_Global_Graph.add_edges(head_list_id, tail_list_id)
Train_Global_Graph.add_edges(tail_list_id, head_list_id)
Train_Global_Graph.ndata['node_idx'] = torch.tensor(list(map(lambda x: total_entity_vocab.index(x), entity_vocab)))
Train_Global_Graph.edata['relation_idx'] = torch.cat([torch.tensor(relation_list_id), torch.tensor(relation_list_id) + torch.tensor(num_of_rel)])# 이 edge에 포함된 relation이 뭔지 id 부여
save_graphs("./"+dataset+"_Train_Global_Graph.bin", [Train_Global_Graph])

Train_time_split_Graph = dgl.DGLGraph(multigraph=True)
Train_time_split_Graph.add_nodes(len(entity_id) * num_of_time)
rel_id_stack_list = []
for head_, time_, tail_, rel_ in zip(head_list_id, time_list_id, tail_list_id, relation_list_id):
    if time_ - half_window_size < 0 :
        rel_id_stack_list.extend([rel_] * (half_window_size + 1 + time_))
        Train_time_split_Graph.add_edges(np.array([num_of_time * head_ + time_] * (half_window_size + 1 + time_)), 
                                 np.array(list(range(0, time_ + half_window_size+1))) + np.array([num_of_time * tail_]))
    elif time_ + half_window_size >= num_of_time :
        rel_id_stack_list.extend([rel_] * (num_of_time - time_ + half_window_size))
        Train_time_split_Graph.add_edges(np.array([num_of_time * head_ + time_] * (num_of_time - time_ + half_window_size)), 
                                 np.array(list(range(time_ - half_window_size, num_of_time))) + np.array([num_of_time * tail_]))
    else:
        rel_id_stack_list.extend([rel_] * neighbor_time_window_size)
        Train_time_split_Graph.add_edges(np.array([num_of_time * head_ + time_] * neighbor_time_window_size), 
                                 np.array(list(range(time_ - half_window_size, time_ + half_window_size+1))) + np.array([num_of_time * tail_]))
for head_, time_, tail_, rel_ in zip(head_list_id, time_list_id, tail_list_id, relation_list_id):
    if time_ - neighbor_time_window_size < -half_window_size-1:
        rel_id_stack_list.extend([rel_ + num_of_rel]*(half_window_size + 1 + time_))
        Train_time_split_Graph.add_edges(np.array(list(range(0, time_ + half_window_size+1))) + np.array([num_of_time * tail_]),
                                 np.array([num_of_time * head_ + time_] * (half_window_size + 1 + time_)))
    elif num_of_time - time_ - neighbor_time_window_size < -half_window_size:
        rel_id_stack_list.extend([rel_ + num_of_rel]*(num_of_time - time_ + half_window_size))
        Train_time_split_Graph.add_edges(np.array(list(range(time_ - half_window_size, num_of_time))) + np.array([num_of_time * tail_]),
                                 np.array([num_of_time * head_ + time_] * (num_of_time - time_ + half_window_size)))
    else:
        rel_id_stack_list.extend([rel_ + num_of_rel]*neighbor_time_window_size)
        Train_time_split_Graph.add_edges(np.array(list(range(time_ - half_window_size, time_ + half_window_size+1))) + np.array([num_of_time * tail_]),
                                 np.array([num_of_time * head_ + time_] * neighbor_time_window_size))
Train_time_split_Graph.ndata['node_idx'] = ((torch.tensor(list(map(lambda x: total_entity_vocab.index(x), entity_vocab))) * torch.tensor([num_of_time])).unsqueeze(1) + torch.tensor([list(range(num_of_time))])).view(-1) # 각 노드의 id부여
Train_time_split_Graph.ndata['entity_idx'] = torch.tensor(list(map(lambda x: total_entity_vocab.index(x), entity_vocab))).unsqueeze(1).repeat(1, num_of_time).view(-1)
Train_time_split_Graph.edata['relation_idx'] = torch.tensor(rel_id_stack_list)# 이 edge에 포함된 relation이 뭔지 id 부여
save_graphs("./window"+str(half_window_size)+'_'+dataset+"_Train_time_split_Graph.bin", [Train_time_split_Graph])
Train_Line_Graph = dgl.transform.line_graph(Train_time_split_Graph, shared = True, backtracking = False)
Train_Relation_Graph_ = dgl.DGLGraph(multigraph=True)
Line_rel_num = len(set(Train_Line_Graph.ndata['relation_idx'].tolist()))
Train_Relation_Graph_.add_nodes(Line_rel_num)
Train_Relation_Graph_.add_edges(Train_Line_Graph.ndata['relation_idx'][Train_Line_Graph.edges()[0]], Train_Line_Graph.ndata['relation_idx'][Train_Line_Graph.edges()[1]])
edge_dict = {}
for u, v in zip(Train_Relation_Graph_.edges()[0].tolist(), Train_Relation_Graph_.edges()[1].tolist()):
    if (u,v) in edge_dict.keys():
        edge_dict[(u,v)] += 1
    else:
        edge_dict[(u,v)] = 1
Train_Relation_Graph = dgl.DGLGraph(multigraph=True)
Train_Relation_Graph.add_nodes(Line_rel_num)
edge_freq = []
for i in list(edge_dict.keys()):
    Train_Relation_Graph.add_edges(i[0], i[1])
    edge_freq.extend([edge_dict[i]])
Train_Relation_Graph.ndata['relation_idx'] = torch.tensor(list(range(Line_rel_num)))
save_graphs("./window"+str(half_window_size)+'_'+dataset+"_Train_Relation_Graph.bin", [Train_Relation_Graph])

Train_Window_Graph = dgl.DGLGraph(multigraph=True)
Train_Window_Graph.add_nodes(len(entity_id) * num_of_time)
for i in range(len(entity_id) * num_of_time):
    if i % num_of_time - half_window_size < 0 :
        src = list(range(i//num_of_time * num_of_time, i + half_window_size + 1))
        src.remove(i)
        Train_Window_Graph.add_edges(src, [i] * (half_window_size + 1 + i % num_of_time -1))
    elif i % num_of_time + half_window_size >= num_of_time :
        src = list(range(i - half_window_size, (i//num_of_time + 1)*num_of_time))
        src.remove(i)
        Train_Window_Graph.add_edges(src, [i] * (num_of_time - (i % num_of_time) + half_window_size -1))
    else:
        src = list(range(i - half_window_size, i + half_window_size + 1))
        src.remove(i)
        Train_Window_Graph.add_edges(src, [i] * (neighbor_time_window_size -1))
save_graphs("./window"+str(half_window_size)+'_'+dataset+"_Train_Window_Graph.bin", [Train_Window_Graph])

Test_Window_Graph = dgl.DGLGraph(multigraph=True)
Test_Window_Graph.add_nodes(num_of_ent * num_of_time)
for i in range(num_of_ent * num_of_time):
    if i % num_of_time - half_window_size < 0 :
        src = list(range(i//num_of_time * num_of_time, i + half_window_size + 1))
        src.remove(i)
        Test_Window_Graph.add_edges(src, [i] * (half_window_size + 1 + i % num_of_time -1))
    elif i % num_of_time + half_window_size >= num_of_time :
        src = list(range(i - half_window_size, (i//num_of_time + 1)*num_of_time))
        src.remove(i)
        Test_Window_Graph.add_edges(src, [i] * (num_of_time - (i % num_of_time) + half_window_size -1))
    else:
        src = list(range(i - half_window_size, i + half_window_size + 1))
        src.remove(i)
        Test_Window_Graph.add_edges(src, [i] * (neighbor_time_window_size -1))
save_graphs("./window"+str(half_window_size)+'_'+dataset+"_Test_Window_Graph.bin", [Test_Window_Graph])