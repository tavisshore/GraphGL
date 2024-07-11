import os
import shutil
import torch
import json

from models.sat.detr import build
from utils.agent import Agent
import time
import pickle
import networkx as nx

def create_directory(dir,delete=False):
    if os.path.isdir(dir) and delete: shutil.rmtree(dir)
    os.makedirs(dir, exist_ok=True)


class Vertex():
    def __init__(self,v,id):
        self.x = v[0]
        self.y = v[1]
        self.id = id
        self.neighbors = []


class Edge():
    def __init__(self,src,dst,id):
        self.src = src
        self.dst = dst
        self.id = id


class Graph():
    def __init__(self):
        self.vertices = {}
        self.edges = {}
        self.vertex_num = 0
        self.edge_num = 0

    def find_v(self,v_coord):
        if f'{v_coord[0]}_{v_coord[1]}' in self.vertices.keys():
            return self.vertices[f'{v_coord[0]}_{v_coord[1]}']
        return 

    def find_e(self,v1,v2):
        if f'{v1.id}_{v2.id}' in self.edges:
            return True
        return None

    def add(self,edge):
        v1_coord = edge[0]
        v2_coord = edge[1]
        v1 = self.find_v(v1_coord)
        if v1 is None:
            v1 = Vertex(v1_coord,self.vertex_num)
            self.vertex_num += 1
            self.vertices[f'{v1.x}_{v1.y}'] = v1
        
        v2 = self.find_v(v2_coord)
        if v2 is None:
            v2 = Vertex(v2_coord,self.vertex_num)
            self.vertex_num += 1
            self.vertices[f'{v2.x}_{v2.y}'] = v2

        if v1 not in v2.neighbors:
            v2.neighbors.append(v1)
        if v2 not in v1.neighbors:
            v1.neighbors.append(v2)
        e = self.find_e(v1,v2)
        if e is None:
            self.edges[f'{v1.id}_{v2.id}'] = Edge(v1,v2,self.edge_num)
            self.edge_num += 1
            self.edges[f'{v2.id}_{v1.id}'] = Edge(v2,v1,self.edge_num)
            self.edge_num += 1


def construct_graph(args, tile_name):
    RNGDetNet, _ = build(args)
    torch.cuda.set_device(args.device)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False 
    RNGDetNet.load_state_dict(torch.load(f'/scratch/sat/weights/RNGDet_best.pt',  map_location='cpu'))#./weights/RNGDetPP_best.pt', map_location='cpu'))
    RNGDetNet.cuda()
    RNGDetNet.eval()

    args.agent_savedir = f'./{args.savedir}/graphs/'
    
    create_directory(f'./{args.savedir}/graphs/graph',delete=False)
    create_directory(f'./{args.savedir}/graphs/skeleton',delete=False)
    create_directory(f'./{args.savedir}/graphs/json',delete=False)
    create_directory(f'./{args.savedir}/graphs/segmentation',delete=False)

    # sigmoid = nn.Sigmoid()
    
    # tile_name = 'guildford_2'        # LIST OF TILES IN CUSTOM DATASET?    guildford_2048
    time_start = time.time()
    agent = Agent(args, RNGDetNet, tile_name)

    while 1:
        agent.step_counter += 1
        # crop ROI
        sat_ROI, historical_ROI = agent.crop_ROI(agent.current_coord)
        sat_ROI = torch.FloatTensor(sat_ROI).permute(2,0,1).unsqueeze(0).cuda() / 255.0
        historical_ROI = torch.FloatTensor(historical_ROI).unsqueeze(0).unsqueeze(0).cuda() / 255.0
        # predict vertices in the next step
        outputs = RNGDetNet(sat_ROI, historical_ROI)
        # agent moves
        # alignment vertices
        pred_coords = outputs['pred_boxes']
        pred_probs = outputs['pred_logits']
        alignment_vertices = [[v[0]-agent.current_coord[0]+agent.crop_size//2, v[1]-agent.current_coord[1]+agent.crop_size//2] for v in agent.historical_vertices]
        pred_coords_ROI = agent.step(pred_probs, pred_coords,thr=args.logit_threshold)

        if agent.finish_current_image:
            graph = Graph()
            with open(f'./{args.savedir}/graphs/graph/{tile_name}.json','w') as jf: json.dump(agent.historical_edges, jf)

            for e in agent.historical_edges: graph.add(e)            
            output_graph = {}

            for _, v in graph.vertices.items(): output_graph[(v.y,v.x)] = [(n.y,n.x) for n in v.neighbors]

            pickle.dump(output_graph, open(f'./{args.savedir}/graphs/graph/{tile_name}.p','wb'),protocol=2)
            break
            
    # time_end = time.time()
    # print(f'Finish inference, time usage {round((time_end-time_start)/3600,3)}h')     
    
    # rename nodes and neighbours to indices, add old node name as pos
    graph = nx.Graph(output_graph)
    for i, node in enumerate(graph.nodes()):
        # graph.nodes[node]['id'] = i
        n = (node[1], node[0]) # swap otherwise rotated 90
        graph.nodes[node]['pos'] = n

    # reset node names to indices
    mapping = {n:i for i,n in enumerate(graph.nodes())}
    graph = nx.relabel_nodes(graph, mapping)
    
    return graph

