import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.data.utils import load_graphs
import pickle

import random
from tqdm import tqdm
import argparse

from utils import complex
from utils import distmult
from utils import transE
from utils import save_total_model
from utils import stabilized_NLL
from utils import self_supervised_loss
from utils import print_metrics
from utils import print_mrr
from utils import print_hms
from utils import rank
import model

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser(description="Constructing Temporal Knowledge Graph")
parser.add_argument("--dataset", default="ICEWS14", choices=["ICEWS14", "ICEWS0515", "YAGO11k"], help="dataset folder name, which has train.txt, test.txt, valid.txt in it")
parser.add_argument("--window_size", default="8", type=str, help="window size to read proper graph")
parser.add_argument("--device", default="cuda:0", choices=["cuda:0", "cuda:1", "cpu"], help="which gpu/cpu do you wanna use")
parser.add_argument("--aT_ratio", default=0.8, type=float, help="weighted sum ratio between TempGCN and aTempGCN")
parser.add_argument("--rel_ratio", default=0.2, type=float, help="ratio of RelGCN")
parser.add_argument("--SSL_ratio", default=1., type=float, help="ratio of Self Supervised Loss")
parser.add_argument("--score_function", default="distmult",choices=["complex","distmult","rotatE","transE"], help="choose score function")
parser.add_argument("--random_seed", default=1024, type=int, help="random_seed for random.random and torch")
parser.add_argument("--T", default="O", choices = ["O", "X"], help="relation_graph construct using T or not")
parser.add_argument("--epoch", default="5", help="epoch to read")
args = parser.parse_args()
random_seed = args.random_seed
random.seed(random_seed)
torch.manual_seed(random_seed)

data_name = args.dataset
window_size = args.window_size
device_0 = args.device
aT_ratio = args.aT_ratio
rel_ratio = args.rel_ratio
SSL_ratio = args.SSL_ratio
score_function = args.score_function
rel_T = args.T
epoch = args.epoch
likelihood = distmult

if score_function == "complex":
    likelihood = complex
elif score_function == "distmult":
    likelihood = distmult
elif score_function == "transE":
    likelihood = transE

print('Loading datas')
with open('./data/data_'+data_name+'.pickle', 'rb') as f:
    data = pickle.load(f)
num_of_time, num_of_rel, num_of_ent, num_of_train_ent = data['nums']
test_data = data['test_data']

print("Loading graphs")
data_path= "./data/"
Test_Global_Graph      = load_graphs(data_path+data_name+"_"+"Test_Global_Graph"+".bin")[0][0]
Test_time_split_Graph  = load_graphs(data_path+"window"+window_size+'_'+data_name+"_"+"Test_time_split_Graph"+".bin")[0][0]
Test_Window_Graph      = load_graphs(data_path+"window"+window_size+'_'+data_name+"_"+"Test_Window_Graph"+".bin")[0][0]
if rel_T == "O":
    Test_Relation_Graph    = load_graphs(data_path+"window"+window_size+'_'+data_name+"_"+"Test_Relation_Graph"+".bin")[0][0]
elif rel_T == "X":
    Test_Relation_Graph    = load_graphs(data_path+data_name+"_"+"TX_Test_Relation_Graph"+".bin")[0][0]
    
"""Main code"""
emb_dim = 100
negative_num = 500
temperature = 0.1

model = model.T_aT_R1_GCN_SSL(num_of_ent, num_of_time, num_of_rel * 2, emb_dim, temperature, device_0, aT_ratio, rel_ratio, random_seed)
model_name = "T_aT_R1_GCN_SSL"

print("Random seed for torch and random", random_seed)
print("Training with negative num      ", negative_num)
print("Using Device                    ", device_0)
print("Window size                     ", window_size)
print("Using Data                      ", data_name)
print("Using Model                     ", model_name)
print("Using Score function            ", score_function)

print("loading model")
checkpoint = torch.load(model_name+"__"+data_name+"__epoch_"+epoch+".pth")
model.load_state_dict(checkpoint['state_dict'])
print("model loaded")

model.eval()
with torch.no_grad(): 
    object_filtered_data_ranks = []
    subject_filtered_data_ranks = []
    r_object_filtered_data_ranks = []
    r_subject_filtered_data_ranks = []
    entity_index = list(range(num_of_ent))
    for s,r,o,t, o_filter_mask, s_filter_mask, _, __ in tqdm(test_data):
        entity_set = torch.tensor(entity_index)*num_of_time + torch.tensor([t])
        with Test_time_split_Graph.local_scope():
            with Test_Window_Graph.local_scope():
                with Test_Global_Graph.local_scope():
                    entity_embs, relation_emb = model(Test_time_split_Graph, Test_Window_Graph, Test_Global_Graph, Test_Relation_Graph, entity_set, torch.tensor([r, r + num_of_rel]), num_of_ent)
        score = likelihood(entity_embs[s], relation_emb[0], entity_embs[o]).item()
        reciprocal_score = likelihood(entity_embs[o], relation_emb[1], entity_embs[s]).item()
        objects_score = likelihood(entity_embs[s].repeat(num_of_ent, 1),
                                relation_emb[0].repeat(num_of_ent, 1), 
                                entity_embs)
        subjects_score = likelihood(entity_embs,
                                relation_emb[0].repeat(num_of_ent,1), 
                                entity_embs[o].repeat(num_of_ent, 1))
        filtered_objects_scores = objects_score[o_filter_mask].tolist()
        filtered_subjects_scores = subjects_score[s_filter_mask].tolist()
        object_filtered_rank = rank(sorted(filtered_objects_scores),score)
        subject_filtered_rank = rank(sorted(filtered_subjects_scores),score)
        object_filtered_data_ranks.append(object_filtered_rank)
        subject_filtered_data_ranks.append(subject_filtered_rank)

        r_object_filtered_rank = rank(sorted(filtered_objects_scores),reciprocal_score)
        r_subject_filtered_rank = rank(sorted(filtered_subjects_scores),reciprocal_score)
        r_object_filtered_data_ranks.append(r_object_filtered_rank)
        r_subject_filtered_data_ranks.append(r_subject_filtered_rank)
    MRR, h1, h3, h10 = print_metrics(r_object_filtered_data_ranks, r_subject_filtered_data_ranks)
    print(MRR, h1, h3, h10)




