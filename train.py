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
parser.add_argument("--score_function", default="distmult",choices=["complex","distmult","transE"], help="choose score function")
parser.add_argument("--random_seed", default=1024, type=int, help="random_seed for random.random and torch")
parser.add_argument("--T", default="O", choices = ["O", "X"], help="relation_graph construct using T or not")
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
train_dataloader_list = data['train_data']
valid_data = data['valid_data'] 
G_train_facts = data['train_facts']

print("Loading graphs")
data_path= "./data/"
Train_Global_Graph     = load_graphs(data_path+data_name+"_"+"Train_Global_Graph"+".bin")[0][0]
Test_Global_Graph      = load_graphs(data_path+data_name+"_"+"Test_Global_Graph"+".bin")[0][0]
Train_time_split_Graph = load_graphs(data_path+"window"+window_size+'_'+data_name+"_"+"Train_time_split_Graph"+".bin")[0][0]
Test_time_split_Graph  = load_graphs(data_path+"window"+window_size+'_'+data_name+"_"+"Test_time_split_Graph"+".bin")[0][0]
Train_Window_Graph     = load_graphs(data_path+"window"+window_size+'_'+data_name+"_"+"Train_Window_Graph"+".bin")[0][0]
Test_Window_Graph      = load_graphs(data_path+"window"+window_size+'_'+data_name+"_"+"Test_Window_Graph"+".bin")[0][0]
if rel_T == "O":
    Train_Relation_Graph   = load_graphs(data_path+"window"+window_size+'_'+data_name+"_"+"Train_Relation_Graph"+".bin")[0][0]
    Test_Relation_Graph    = load_graphs(data_path+"window"+window_size+'_'+data_name+"_"+"Test_Relation_Graph"+".bin")[0][0]
elif rel_T == "X":
    Train_Relation_Graph   = load_graphs(data_path+data_name+"_"+"TX_Train_Relation_Graph"+".bin")[0][0]
    Test_Relation_Graph    = load_graphs(data_path+data_name+"_"+"TX_Test_Relation_Graph"+".bin")[0][0]
"""
 The only difference between train_graphs and test_graphs is number of entities.
 Both of them use only train quadruples while construction.
 Except window graphs, which are virtual graphs used only to define window for each entity.
"""
"""Main code"""
emb_dim = 100
trainset_batch_size = 100
num_epochs = 10
negative_num = 500
temperature = 0.1
NLL_temperature = 1
SSL_neg = 500
patience = 5
patience_cnt = 0
prev_MRR = 0
best_MRR = 0
best_h1 = 0
best_h3 = 0
best_h10 = 0
best_epoch = 0


# 1+3+4+5
model = model.T_aT_R1_GCN_SSL(num_of_ent, num_of_time, num_of_rel * 2, emb_dim, temperature, device_0, aT_ratio, rel_ratio, random_seed)
model_name = "T_aT_R1_GCN_SSL"
optimizer = torch.optim.SparseAdam(model.parameters(), lr = 0.001)
start_epoch = 0

print("Random seed for torch and random", random_seed)
print("Trainset batch size             ", trainset_batch_size)
print("Training with negative num      ", negative_num)
print("Using Device                    ", device_0)
print("Window size                     ", window_size)
print("Using Data                      ", data_name)
print("Using Model                     ", model_name)
print("Using Score function            ", score_function)
print("NLL temperature                 ", NLL_temperature)
print("SSL_neg:                        ", SSL_neg)
print("SSL_ratio:                      ", SSL_ratio)
print("SSL_temperature:                ", temperature)
entity_box = list(range(num_of_train_ent))
for epoch in range(start_epoch, num_epochs):
    print("-epoch: ", epoch,"/ 0 ~",num_epochs-1,"processing")
    model.train()
    batch_loss = []
    for batch in tqdm(train_dataloader_list):
        head_batch, relation_batch, tail_batch, time_batch = batch
        positive_St = []
        positive_Ot = []
        negative_Ot = []
        negative_St = []
        relation_r = []
        negative_Ossl = []
        negative_Sssl = []
        bs = len(head_batch)
        for s,r,o,t in zip(head_batch, relation_batch, tail_batch, time_batch):
            random.shuffle(entity_box)
            object_negative_samples = []
            cnt = 0
            for neg_o in entity_box:
                if cnt < negative_num:
                    if (s,r,neg_o,t) not in G_train_facts:
                        object_negative_samples.append(neg_o)
                        cnt+=1
            random.shuffle(entity_box)
            subject_negative_samples = []
            cnt = 0
            for neg_s in entity_box:
                if cnt < negative_num:
                    if (neg_s,r,o,t) not in G_train_facts:
                        subject_negative_samples.append(neg_s)
                        cnt+=1
            random.shuffle(entity_box)
            object_ssl_negative = []
            subject_ssl_negative = []
            cnt = 0
            for neg_o in entity_box:
                if cnt < SSL_neg:
                    if (not (Train_Global_Graph.has_edge_between(s, neg_o)) and (neg_o is not s)):
                        object_ssl_negative.append(neg_o) 
                        cnt+=1
            random.shuffle(entity_box)
            cnt = 0
            for neg_s in entity_box:
                if cnt < SSL_neg:
                    if (not (Train_Global_Graph.has_edge_between(neg_s, o)) and (neg_s is not o)):
                        subject_ssl_negative.append(neg_s)
                        cnt+=1
            St_negative_samples = (torch.tensor(subject_negative_samples) * num_of_time + torch.tensor(t)).tolist()
            Ot_negative_samples = (torch.tensor(object_negative_samples) * num_of_time + torch.tensor(t)).tolist()
            Ot_ssl_negative = (torch.tensor(object_ssl_negative) * num_of_time + torch.tensor(t)).tolist()
            St_ssl_negative = (torch.tensor(subject_ssl_negative) * num_of_time + torch.tensor(t)).tolist()
            relation_r.extend([r])
            positive_St.extend([s * num_of_time + t])
            positive_Ot.extend([o * num_of_time + t])
            negative_Ot.extend(Ot_negative_samples)
            negative_St.extend(St_negative_samples)
            negative_Ossl.extend(Ot_ssl_negative)
            negative_Sssl.extend(St_ssl_negative)
        if window_size != '0':
            ssl_entity_set, ssl_entity_set_idx = torch.unique(torch.cat([torch.tensor(positive_St), 
                                                                         torch.tensor(positive_Ot), 
                                                                         torch.tensor(negative_Ossl), 
                                                                         torch.tensor(negative_Sssl)]), sorted=False, return_inverse = True)
            with Train_time_split_Graph.local_scope():
                with Train_Window_Graph.local_scope():
                    with Train_Global_Graph.local_scope():
                        Temp_entity_embs, Pos_sim= model.forward_SSL(Train_time_split_Graph, 
                                                                     Train_Window_Graph, 
                                                                     Train_Global_Graph, 
                                                                     ssl_entity_set, 
                                                                     torch.tensor(relation_r), 
                                                                     num_of_ent, 
                                                                     bs, 
                                                                     ssl_entity_set_idx)
            Pos_emb = (Temp_entity_embs[ssl_entity_set_idx[:bs * 2]]).unsqueeze(2)
            Neg_emb = Temp_entity_embs[ssl_entity_set_idx[bs * 2:]].view(bs*2, SSL_neg, 100)
            Neg_sim = torch.bmm(Neg_emb, Pos_emb).squeeze(2)
            SSL = self_supervised_loss(Pos_sim/temperature, Neg_sim/temperature)
        else:
            SSL = 0
        entity_set, entity_set_idx = torch.unique(torch.cat([torch.tensor(positive_St), 
                                                             torch.tensor(positive_Ot), 
                                                             torch.tensor(negative_St), 
                                                             torch.tensor(negative_Ot)]), sorted=False, return_inverse = True)
        with Train_time_split_Graph.local_scope():
            with Train_Window_Graph.local_scope():
                with Train_Global_Graph.local_scope():
                    entity_embs, relation_embs= model(Train_time_split_Graph, 
                                                      Train_Window_Graph, 
                                                      Train_Global_Graph, 
                                                      Train_Relation_Graph, 
                                                      entity_set, 
                                                      torch.tensor(relation_r), 
                                                      num_of_ent)
        Pos_score = likelihood(entity_embs[entity_set_idx[:bs]],
                               relation_embs, 
                               entity_embs[entity_set_idx[bs:bs * 2]])

        Neg_s_score = likelihood(entity_embs[entity_set_idx[bs*2:bs * (negative_num + 2)]],
                                 relation_embs.unsqueeze(1).repeat(1, negative_num, 1).view(-1, emb_dim),
                                 entity_embs[entity_set_idx[bs:bs * 2]].unsqueeze(1).repeat(1, negative_num, 1).view(-1, emb_dim))

        Neg_o_score = likelihood(entity_embs[entity_set_idx[:bs]].unsqueeze(1).repeat(1, negative_num, 1).view(-1, emb_dim),
                                 relation_embs.unsqueeze(1).repeat(1,negative_num, 1).view(-1, emb_dim),
                                 entity_embs[entity_set_idx[bs * (negative_num+2):]])
        NLL = stabilized_NLL(Pos_score, Neg_o_score.view(bs,-1), Neg_s_score.view(bs,-1))

        loss = NLL + (SSL * SSL_ratio)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss = loss.detach().item()
        batch_loss.append(loss)
    save_total_model(epoch, model, optimizer, model_name+"__"+data_name+"__epoch_"+str(epoch)+".pth")
    print("validation start")
    model.eval()
    with torch.no_grad(): 
        object_filtered_data_ranks = []
        subject_filtered_data_ranks = []
        r_object_filtered_data_ranks = []
        r_subject_filtered_data_ranks = []
        entity_index = list(range(num_of_ent))
        for s,r,o,t, o_filter_mask, s_filter_mask, _, __ in tqdm(valid_data):
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
        if best_MRR > MRR:
            patience_cnt += 1
            print("p_count: ",patience_cnt)
        else:
            best_MRR = MRR
            best_h1 = h1
            best_h3 = h3
            best_h10 = h10
            best_epoch = epoch
            patience_cnt = 0
        prev_MRR = MRR
        if patience_cnt == patience:
            print("breaks at epoch",epoch)
            print("Best epoch:",best_epoch)
            print("Best result\nMRR:",best_MRR,"\nHits@1:",best_h1,"\nHits@3:",best_h3,"\nHits@10:",best_h10)
            break
        print("best result",best_MRR,best_h1,best_h3,best_h10,"at",best_epoch,"with w=",window_size,"aT=", aT_ratio,"r=", rel_ratio,'\n', score_function, model_name,"device=", device_0, data_name, "rel_T: ",rel_T)