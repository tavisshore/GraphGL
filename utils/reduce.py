import networkx as nx


def reduce_graph(corpus_graph):
    junctions = [node for node in corpus_graph.nodes() if corpus_graph.degree(node) > 2]
    reduced_graph = nx.Graph(corpus_graph.subgraph(junctions))
    reduced_graph.remove_edges_from(corpus_graph.edges(junctions))

    corpus_graph_copy = corpus_graph.copy()

    all_nodes = []
    # Now determine which of these nodes are to be connected
    for idx, junction in enumerate(junctions):
        junctions_edges = []
        # 1. Get junctions immediate neighbours
        neighbours = corpus_graph_copy.neighbors(junction)
        previous_neighbour = junction

        for neigh in neighbours:
            # 2. Get neighbours neighbours
            neigh_neigh = list(corpus_graph_copy.neighbors(neigh))
            # 3. Remove previous neighbour from list
            neigh_save = previous_neighbour
            neigh_neigh.remove(previous_neighbour) # if not a junction, should be length one
            previous_neighbour = neigh

            while len(neigh_neigh) < 2: # if length is 0, side of graph
                
                if len(neigh_neigh) == 0:
                    junctions_edges.append(neigh_save)
                    break
                neigh_save = neigh_neigh[0]

                # 4. Next neighbours
                neigh_neigh = list(corpus_graph_copy.neighbors(neigh_neigh[0]))
                if len(neigh_neigh) > 2:
                    junctions_edges.append(neigh_save)
                    break
                neigh_neigh.remove(previous_neighbour)
                previous_neighbour = neigh_save
            previous_neighbour = junction

        for edge in junctions_edges:
            reduced_graph.add_edge(junction, edge)
        all_nodes.extend(junctions_edges)

    

    for node in all_nodes:
        pos = nx.get_node_attributes(corpus_graph, 'pos')
        print(pos)
        print(node)
        pos = pos[node]
        reduced_graph.add_node(node, pos=pos)

    for node in reduced_graph.nodes():
        try: reduced_graph.remove_edge(node, node)
        except: pass
    return reduced_graph
