from typing import Dict
import pytorch_lightning as pl
import numpy as np
# from sklearn import linear_model as sk_lm
from sklearn import metrics as sk_mtr
# from sklearn import model_selection as sk_ms
# from sklearn import multiclass as sk_mc
from sklearn import preprocessing as sk_prep
import torch
from torch import nn
from torch_geometric.data import Data
from tqdm import tqdm

from torch_geometric.utils.nested import to_nested_tensor
import json

from scipy.spatial import KDTree


# LATEST version uses 
def recall_accuracy(args, data, embeddings):
    # Potential permutations
    # 1. KDTree of singlular nodes
    #    a. One node per GT node
    #    b. All node embeddings of each node from exhaustive walks
    # 2. Embedding size
    #    a. Single node embedding
    #    b. Stacked embeddings from walks
    embs_a_dict, embs_b_dict = {}, {}
    embs_a = [e[0] for e in embeddings]
    embs_b = [e[1] for e in embeddings]

    for idx, batch in enumerate(data):
        pos_s = batch.start_point.tolist()
        stps = [tuple([pos_s[i], pos_s[i+1]]) for i in range(0, len(pos_s), 2)]
        ptr = batch.ptr.tolist()
        for p in range(0, len(ptr)-1, 1):
            ptr_1, ptr_2 = ptr[p], ptr[p+1]        
            starting_point = tuple(stps[p][0])
            walk_emb_a = embs_a[idx][ptr_1:ptr_2].cpu().numpy()
            walk_emb_b = embs_b[idx][ptr_1:ptr_2].cpu().numpy()
            if starting_point in embs_a_dict.keys(): 
                embs_a_dict[starting_point].append(walk_emb_a) 
            else: embs_a_dict[starting_point] = [walk_emb_a]
            if starting_point in embs_b_dict.keys(): embs_b_dict[starting_point].append(walk_emb_b)
            else: embs_b_dict[starting_point] = [walk_emb_b]

    # Statements reorganise inputs to fit the above permutations
    if args.eval.single_walk: # one node embedding for each node in corpus
        if args.eval.single_node: # That node embedding is the start_point only
            for k in embs_a_dict.keys(): embs_a_dict[k] = [embs_a_dict[k][0][0]]
            for k in embs_b_dict.keys(): embs_b_dict[k] = [embs_b_dict[k][0][0]]
        else: # whole walk is embedding, flattened
            for k in embs_a_dict.keys(): 
                embs_a_dict[k] = [embs_a_dict[k][0].flatten()]
            for k in embs_b_dict.keys(): 
                embs_b_dict[k] = [embs_b_dict[k][0].flatten()]
                print(embs_b_dict[k])
    else:
        if args.eval.single_node: # all walks from all positions are reduced to their starting node
            for k in embs_a_dict.keys(): 
                for walk in range(len(embs_a_dict[k])): 
                    embs_a_dict[k][walk] = embs_a_dict[k][walk][0] # list as above?
                    embs_a_dict[k] = [l for l in embs_a_dict[k] if l is not None]
            for k in embs_b_dict.keys():
                for walk in range(len(embs_b_dict[k])):
                    embs_b_dict[k][walk] = embs_b_dict[k][walk][0]
                    embs_b_dict[k] = [l for l in embs_b_dict[k] if l is not None]
        else:
            for k in embs_a_dict.keys():
                for walk in range(len(embs_a_dict[k])):
                    embs_a_dict[k][walk] = [embs_a_dict[k][walk].flatten()]
            for k in embs_b_dict.keys():
                for walk in range(len(embs_b_dict[k])):
                    embs_b_dict[k][walk] = [embs_b_dict[k][walk].flatten()]
            
    # Build KDTree, Query, and Evaluate
    indices = []
    db_list, query_list = [], []
    index_counter = 0
    if args.eval.single_walk:
        for k in embs_a_dict.keys():
            length = len(embs_a_dict[k]) # number of embedding for this node
            db_list.extend(embs_a_dict[k])
            query_list.extend(embs_b_dict[k])
            indices.append(list(range(index_counter, index_counter+length)))
            index_counter += length
    else:
        for k in embs_a_dict.keys():
            sub_inds = []
            sub_query_list = []
            for emb_a, emb_b in zip(embs_a_dict[k], embs_b_dict[k]): # make numpy list of lists
                db_list.append(emb_a)
                sub_query_list.append(emb_b)
                latest_indice = len(db_list)-1
                sub_inds.append(latest_indice)
            query_list.append(np.array(sub_query_list))
            sub_i = len(sub_inds)
            indices.append(np.array(list(range(index_counter, index_counter+sub_i))))
            index_counter += sub_i

    db = np.array(db_list)
    query = np.array(query_list) if args.eval.single_walk else query_list
    db_length = len(db)
    one_percent = db_length//100 if db_length > 100 else 10
    tree = KDTree(db)

    top_1, top_5, top_10, top_one = 0, 0, 0, 0
    for idx, q in enumerate(query):
        _, ind = tree.query([q], k=one_percent)
        if args.eval.single_walk:
            idxes = indices[idx] # gets sublist of relevant indices
            if len(np.intersect1d(ind[0][:1], idxes)) > 0: top_1 += 1
            if len(np.intersect1d(ind[0][:5], idxes)) > 0: top_5 += 1
            if len(np.intersect1d(ind[0][:10], idxes)) > 0: top_10 += 1
            if len(np.intersect1d(ind[0], idxes)) > 0: top_one += 1
        else:
            idxes = indices[idx] # gets sublist of relevant indices
            ind = ind[0]
            for j in ind:
                if len(np.intersect1d(j[:1], idxes)) > 0: top_1 += 1
                if len(np.intersect1d(j[:5], idxes)) > 0: top_5 += 1
                if len(np.intersect1d(j[:10], idxes)) > 0: top_10 += 1
                if len(np.intersect1d(j, idxes)) > 0: top_one += 1

    top_1_acc = round((top_1/db_length)*100, 6)
    top_5_acc = round((top_5/db_length)*100, 6)
    top_10_acc = round((top_10/db_length)*100, 6)
    top_1_per = round((top_one/db_length)*100, 6)

    return top_1_acc, top_5_acc, top_10_acc, top_1_per

