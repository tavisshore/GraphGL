import random
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from scipy.spatial import KDTree
import pickle
from tqdm import tqdm
from pathlib import Path
# from torch_geometric.data import InMemoryDataset
# from torch_geometric.data import Data
from torch_geometric.utils.convert import from_networkx
import torch
import time


cwd = Path.cwd()


def random_walk(graph, start_node, length):
    walk = [start_node]
    while len(walk) < length:
        neighbours = list(graph.neighbors(walk[-1]))
        if len(walk) > 1:
            if walk[-2] in neighbours:
                neighbours.remove(walk[-2])

        if len(neighbours) == 0: break
        else: walk.append(random.choice(neighbours))
    return walk

def reset_index(sub_graphs):
    # Make indices incremental from 0 and return dict to original mapping
    sub_graphs_out = []
    for sub_graph in sub_graphs:
        relabel_dict = {}
        for idx, node in enumerate(sub_graph.nodes()): 
            relabel_dict[node] = idx

        sub_g = nx.relabel_nodes(sub_graph, relabel_dict)
        sub_g.start_point = sub_graph.start_point
        sub_graphs_out.append(sub_g)
    return sub_graphs_out

def visualise(graph, counter):
    # if type(graph) == nx.Graph():
    nx.draw(graph, nx.get_node_attributes(graph, 'pos'), with_labels=False)
    plt.savefig(f'walk_{counter}.png')
    plt.clf()

    # print(graph)
    # else:
        # for g in graph:
            # nx.draw(g, nx.get_node_attributes(g, 'pos'), with_labels=False)
            # plt.savefig(f'walk_{g.id}.png')


def random_k_walks(corpus_graph='src/data.py', walk_length=10, scale_pos=False, noise=0, attempt=20, relative_pos=False):
    nodes = list(corpus_graph.nodes)
    walks = []


    location_data = {}
    for idx, node in enumerate(corpus_graph.nodes):
        location_data[corpus_graph.nodes[node]['pos']] = {'id': idx}



    for node in nodes:
        attempts = 0
        walk = []
        while len(walk) < walk_length or attempts < attempt:
            walk = random_walk(corpus_graph, node, walk_length)
            attempts += 1
        if len(walk) == walk_length: 
            walks.append(walk)


    sub_graphs = []
    for indice, walk in enumerate(walks):
        sub_graph = nx.Graph()
        for node in walk:
            sub_graph.add_node(node, pos=corpus_graph.nodes[node]['pos'], x=corpus_graph.nodes[node]['x'])

        for i, node in enumerate(walk): 
            if i < len(walk) - 1: sub_graph.add_edge(node, walk[i + 1])
    
        # Making random walks relative to start position by removing positional info - start_point for later metrics
        start_position = nx.get_node_attributes(corpus_graph, 'pos')[walk[0]]
        # print(f'start position: {start_position}')


        # location_data[indice] = {'start_position': start_position} # allows for more

                        
        if relative_pos:
            for node in corpus_graph.nodes:
                pos = corpus_graph.nodes[node]['pos']
                # sub_graph.nodes[node]['original_pos'] = pos
                x, y = pos
                x -= start_position[0]
                y -= start_position[1]
                corpus_graph.nodes[node]['pos'] = (x, y)


        relabel_dict = {}
        for idx, node in enumerate(sub_graph.nodes()): 
            relabel_dict[node] = idx

        # if indice < 5:
        #     visualise(sub_graph, counter)
        #     counter += 1

        sub_g = nx.relabel_nodes(sub_graph, relabel_dict)



        pyg_sub = from_networkx(sub_g)
        pyg_sub.id = indice
        pyg_sub.start_point = start_position

        sub_graphs.append(pyg_sub)

    # sub_graphs = reset_index(sub_graphs)
    return sub_graphs, location_data

