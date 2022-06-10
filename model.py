import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import binary_search

class T_aT_R1_GCN_SSL(nn.Module):
    def __init__(self, entity_num, time_num, relation_num, emb_dim, temperature, device__, aT_ratio, rel_ratio, random_seed):
        super(T_aT_R1_GCN_SSL, self).__init__()
        self.device_0 = torch.device(device__ if torch.cuda.is_available() else "cpu")
        torch.manual_seed(random_seed)
        rgcn_ratio = rel_ratio
        static_ratio = aT_ratio
        self.lambda_ = torch.tensor([static_ratio]).to(self.device_0)
        self.lambda_2 = torch.tensor([rgcn_ratio]).to(self.device_0)
        self.temperature = temperature
        self.time_num = time_num
        self.global_node_embedding_layer = nn.Embedding(entity_num, emb_dim, sparse = True).to(self.device_0)
        self.node_embedding_layer = nn.Embedding(entity_num * time_num, emb_dim, sparse = True).to(self.device_0)
        self.edge_embedding_layer = nn.Embedding(relation_num, emb_dim, sparse = True).to(self.device_0)
        nn.init.xavier_normal_(self.global_node_embedding_layer.weight.data)
        nn.init.xavier_normal_(self.node_embedding_layer.weight.data)
        nn.init.xavier_normal_(self.edge_embedding_layer.weight.data)

    def forward(self, g, SA_g, Glob_g, Rel_g, seed_nodes, relation_batch, neighbor_batch_size):
        g.readonly(True)
        Glob_g.readonly(True)
        seed_node_batch_size = len(seed_nodes)
        original_nodes = seed_nodes // torch.tensor(self.time_num)
        Global_seed, Global_seed_idx = torch.unique(original_nodes, sorted = False, return_inverse = True)
        Global_batch = dgl.contrib.sampling.NeighborSampler(Glob_g, batch_size = seed_node_batch_size,
                                                expand_factor = neighbor_batch_size,
                                                neighbor_type ='in',
                                                shuffle = False,
                                                num_hops = 2,
                                                seed_nodes = Global_seed,
                                                add_self_loop = False)
        for Global_flow in Global_batch:
            break
        Global_flow.copy_from_parent()
        Global_node_unique, Global_node_index = torch.unique(torch.cat([Global_flow.layers[0].data['node_idx'], 
                                                                        Global_flow.layers[1].data['node_idx'], 
                                                                        Global_flow.layers[2].data['node_idx']]), sorted = False, return_inverse = True)
        Glob_node_emb = self.global_node_embedding_layer(Global_node_unique.to(self.device_0))
        Glob_len0 = len(Global_flow.layers[0].data['node_idx'])
        Glob_len1 = len(Global_flow.layers[1].data['node_idx'])
        Glob_rel_unique, Glob_rel_index = torch.unique(torch.cat([Global_flow.blocks[0].data['relation_idx'],
                                                                  Global_flow.blocks[1].data['relation_idx']]), sorted = False, return_inverse = True)
        Glob_len2 = len(Global_flow.blocks[0].data['relation_idx'])
        self.Glob_rel_emb = self.edge_embedding_layer(Glob_rel_unique.to(self.device_0))

        Global_flow.layers[0].data['node_emb'] = Glob_node_emb[Global_node_index[:Glob_len0]]
        Global_flow.blocks[0].data['unique_idx'] = Glob_rel_index[:Glob_len2]
        Global_flow.layers[0].data['out_degree_sqrt'] = torch.sqrt(Global_flow.layer_out_degree(0).type(torch.FloatTensor)).unsqueeze(1).to(self.device_0) # degree normalization
        Global_flow.layers[1].data['in_degree_sqrt'] = torch.sqrt(Global_flow.layer_in_degree(1).type(torch.FloatTensor)).unsqueeze(1).to(self.device_0) # degree normalization
        Global_flow.block_compute(block_id = 0, message_func=self.msg_Global, reduce_func=self.reduce_GCN)
        Global_flow.layers[1].data['node_emb'] = Global_flow.layers[1].data['reduced'] + Glob_node_emb[Global_node_index[Glob_len0:Glob_len0+Glob_len1]]
        Global_flow.blocks[1].data['unique_idx'] = Glob_rel_index[Glob_len2:]
        Global_flow.layers[1].data['out_degree_sqrt'] = torch.sqrt(Global_flow.layer_out_degree(1).type(torch.FloatTensor)).unsqueeze(1).to(self.device_0) # degree normalization
        Global_flow.layers[2].data['in_degree_sqrt'] = torch.sqrt(Global_flow.layer_in_degree(2).type(torch.FloatTensor)).unsqueeze(1).to(self.device_0) # degree normalization
        Global_flow.block_compute(block_id = 1, message_func=self.msg_Global, reduce_func=self.reduce_GCN)
        Glob_n = Global_flow.layers[2].data['reduced'] + Glob_node_emb[Global_node_index[Glob_len0+Glob_len1:]]
        GCN_batch = dgl.contrib.sampling.NeighborSampler(g, 
                                                         batch_size = seed_node_batch_size,
                                                         expand_factor = neighbor_batch_size,
                                                         neighbor_type ='in',
                                                         shuffle = False,
                                                         num_hops = 2,
                                                         seed_nodes = seed_nodes,
                                                         add_self_loop = False)
        for node_flow in GCN_batch:
            break
        node_flow.copy_from_parent()
        node_unique, node_index = torch.unique(torch.cat([node_flow.layers[0].data['node_idx'], 
                                                          node_flow.layers[1].data['node_idx'], 
                                                          node_flow.layers[2].data['node_idx']]), sorted = False, return_inverse = True)
        node_emb = self.node_embedding_layer(node_unique.to(self.device_0))
        len0 = len(node_flow.layers[0].data['node_idx'])
        len1 = len(node_flow.layers[1].data['node_idx'])
        rel_unique, rel_index = torch.unique(torch.cat([node_flow.blocks[0].data['relation_idx'],
                                                        node_flow.blocks[1].data['relation_idx'],
                                                        relation_batch]), sorted = False, return_inverse = True)
        len2 = len(node_flow.blocks[0].data['relation_idx'])
        len3 = len(node_flow.blocks[1].data['relation_idx'])
        self.rel_emb = self.edge_embedding_layer(rel_unique.to(self.device_0))
        node_flow.layers[0].data['node_emb'] = node_emb[node_index[:len0]]
        node_flow.blocks[0].data['unique_idx'] = rel_index[:len2]
        node_flow.layers[0].data['out_degree_sqrt'] = torch.sqrt(node_flow.layer_out_degree(0).type(torch.FloatTensor)).unsqueeze(1).to(self.device_0) # degree normalization
        node_flow.layers[1].data['in_degree_sqrt'] = torch.sqrt(node_flow.layer_in_degree(1).type(torch.FloatTensor)).unsqueeze(1).to(self.device_0) # degree normalization
        node_flow.block_compute(block_id = 0, message_func=self.msg_GCN, reduce_func=self.reduce_GCN)
        node_flow.layers[1].data['node_emb'] = node_flow.layers[1].data['reduced'] + node_emb[node_index[len0:len0+len1]]
        node_flow.blocks[1].data['unique_idx'] = rel_index[len2:len2+len3]
        node_flow.layers[1].data['out_degree_sqrt'] = torch.sqrt(node_flow.layer_out_degree(1).type(torch.FloatTensor)).unsqueeze(1).to(self.device_0) # degree normalization
        node_flow.layers[2].data['in_degree_sqrt'] = torch.sqrt(node_flow.layer_in_degree(2).type(torch.FloatTensor)).unsqueeze(1).to(self.device_0) # degree normalization
        node_flow.block_compute(block_id = 1, message_func=self.msg_GCN, reduce_func=self.reduce_GCN)
        n = node_flow.layers[2].data['reduced'] + node_emb[node_index[len0+len1:]]
        n = Glob_n[Global_seed_idx] * self.lambda_ + n * (1 - self.lambda_)
        rel_seed, rel_seed_idx = torch.unique(relation_batch, sorted = False, return_inverse = True)
        rel_batch = dgl.contrib.sampling.NeighborSampler(Rel_g, 
                                                         batch_size = seed_node_batch_size,
                                                         expand_factor = neighbor_batch_size,
                                                         neighbor_type ='in',
                                                         shuffle = False,
                                                         num_hops = 1,
                                                         seed_nodes = rel_seed,
                                                         add_self_loop = False)
        for rel_flow in rel_batch:
            break
        rel_flow.copy_from_parent()
        edge_unique, edge_index = torch.unique(torch.cat([rel_flow.layers[0].data['relation_idx'], 
                                                          rel_flow.layers[1].data['relation_idx']]), sorted = False, return_inverse = True)
        edge_emb = self.edge_embedding_layer(edge_unique.to(self.device_0))
        len0 = len(rel_flow.layers[0].data['relation_idx'])
        rel_flow.layers[0].data['rel_emb'] = edge_emb[edge_index[:len0]]
        rel_flow.layers[0].data['out_degree_sqrt'] = torch.sqrt(rel_flow.layer_out_degree(0).type(torch.FloatTensor)).unsqueeze(1).to(self.device_0) # degree normalization
        rel_flow.layers[1].data['in_degree_sqrt'] = torch.sqrt(rel_flow.layer_in_degree(1).type(torch.FloatTensor)).unsqueeze(1).to(self.device_0)   # degree normalization
        rel_flow.block_compute(block_id = 0, message_func=self.msg_rel, reduce_func=self.reduce_rel)
        e = rel_flow.layers[1].data['reduced'] * self.lambda_2 + edge_emb[edge_index[len0:]] * (1-self.lambda_2)
        return n, e[rel_seed_idx]

    def forward_SSL(self, g, SSL_g, Glob_g, seed_nodes, relation_batch, neighbor_batch_size, bs, seed_idx):
        g.readonly(True)
        SSL_g.readonly(True)
        seed_node_batch_size = len(seed_nodes)
        SSL_seed, SSL_seed_idx = torch.unique(seed_nodes[seed_idx[:bs*2]], sorted=False, return_inverse = True)
        SSL_batch = dgl.contrib.sampling.NeighborSampler(SSL_g, 
                                                         batch_size = seed_node_batch_size,
                                                         expand_factor = neighbor_batch_size,
                                                         neighbor_type ='in',
                                                         shuffle = False,
                                                         num_hops = 1,
                                                         seed_nodes = SSL_seed,
                                                         add_self_loop = False)
        for SSL_flow in SSL_batch:
            break
        GCN_seed, GCN_seed_idx = torch.unique(torch.cat([seed_nodes, 
                                                         SSL_flow.layer_parent_nid(1), 
                                                         SSL_flow.layer_parent_nid(0)]), sorted = False, return_inverse = True)
        GCN_seed_batch_size = len(GCN_seed)
        GCN_batch = dgl.contrib.sampling.NeighborSampler(g, 
                                                         batch_size = GCN_seed_batch_size,
                                                         expand_factor = neighbor_batch_size,
                                                         neighbor_type ='in',
                                                         shuffle = False,
                                                         num_hops = 2,
                                                         seed_nodes = GCN_seed,
                                                         add_self_loop = False)
        for node_flow in GCN_batch:
            break
        node_flow.copy_from_parent()
        node_unique, node_index = torch.unique(torch.cat([node_flow.layers[0].data['node_idx'], 
                                                          node_flow.layers[1].data['node_idx'], 
                                                          node_flow.layers[2].data['node_idx']]), sorted = False, return_inverse = True)
        node_emb = self.node_embedding_layer(node_unique.to(self.device_0))
        len0 = len(node_flow.layers[0].data['node_idx'])
        len1 = len(node_flow.layers[1].data['node_idx'])
        rel_unique, rel_index = torch.unique(torch.cat([node_flow.blocks[0].data['relation_idx'],
                                                        node_flow.blocks[1].data['relation_idx']]), sorted = False, return_inverse = True)
        len2 = len(node_flow.blocks[0].data['relation_idx'])
        self.rel_emb = self.edge_embedding_layer(rel_unique.to(self.device_0)).detach()
        node_flow.layers[0].data['node_emb'] = node_emb[node_index[:len0]]
        node_flow.blocks[0].data['unique_idx'] = rel_index[:len2]
        node_flow.layers[0].data['out_degree_sqrt'] = torch.sqrt(node_flow.layer_out_degree(0).type(torch.FloatTensor)).unsqueeze(  1).to(self.device_0) # degree normalization
        node_flow.layers[1].data['in_degree_sqrt'] = torch.sqrt(node_flow.layer_in_degree(1).type(torch.FloatTensor)).unsqueeze(1).to(self.device_0) # degree normalization
        node_flow.block_compute(block_id = 0, message_func=self.msg_GCN, reduce_func=self.reduce_GCN)
        node_flow.layers[1].data['node_emb'] = node_flow.layers[1].data['reduced'] + node_emb[node_index[len0:len0+len1]]
        node_flow.blocks[1].data['unique_idx'] = rel_index[len2:]
        node_flow.layers[1].data['out_degree_sqrt'] = torch.sqrt(node_flow.layer_out_degree(1).type(torch.FloatTensor)).unsqueeze(1).to(self.device_0) # degree normalization
        node_flow.layers[2].data['in_degree_sqrt'] = torch.sqrt(node_flow.layer_in_degree(2).type(torch.FloatTensor)).unsqueeze(1).to(self.device_0) # degree normalization
        node_flow.block_compute(block_id = 1, message_func=self.msg_GCN, reduce_func=self.reduce_GCN)
        n = node_flow.layers[2].data['reduced'] + node_emb[node_index[len0+len1:]]
        SSL_flow.layers[0].data['node_emb'] = n[GCN_seed_idx[seed_node_batch_size + len(SSL_seed):]]
        SSL_flow.layers[1].data['node_emb'] = n[GCN_seed_idx[seed_node_batch_size:seed_node_batch_size + len(SSL_seed)]]
        SSL_flow.block_compute(block_id = 0, message_func = self.msg_SSL, reduce_func = self.reduce_SSL)
        return n[GCN_seed_idx[:seed_node_batch_size]], SSL_flow.layers[1].data['sim'][SSL_seed_idx]
        
    def msg_SSL(self, edges):
        return {'window' : edges.src['node_emb']}
 
    def reduce_SSL(self, nodes):
        shape = torch.tensor([nodes.batch_size(), 10])
        similarity = torch.bmm(nodes.mailbox['window'], (nodes.data['node_emb']).unsqueeze(2)).squeeze(2)
        return {'sim' : F.pad(similarity, (0, (shape-torch.tensor(similarity.shape))[1], 0, 0), value = float("-inf"))}

    def msg_GCN(self,edges):  # out degree
        return {'m' : (edges.src['node_emb'] * self.rel_emb[edges.data['unique_idx']]) / (edges.src['out_degree_sqrt'] * edges.dst['in_degree_sqrt'])}

    def reduce_GCN(self,nodes): # in degree
        return {'reduced': nodes.mailbox['m'].sum(1)}

    def msg_Global(self,edges):  # out degree
        return {'m' : (edges.src['node_emb'] * self.Glob_rel_emb[edges.data['unique_idx']]) / (edges.src['out_degree_sqrt'] * edges.dst['in_degree_sqrt'])}

    def reduce_Global(self,nodes): # in degree
        return {'reduced': nodes.mailbox['m'].sum(1)}

    def msg_rel(self, edges):
        return {'m' : edges.src['rel_emb'] / (edges.src['out_degree_sqrt'] * edges.dst['in_degree_sqrt'])}

    def reduce_rel(self, nodes):
        return {'reduced': nodes.mailbox['m'].sum(1)}
