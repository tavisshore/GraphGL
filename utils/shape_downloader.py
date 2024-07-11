import osmnx as ox
import networkx as nx

SIMPLE = True

def get_graph_osm(centre=(51.243594, -0.576837), dist=1000):
    g = ox.graph.graph_from_point(center_point=centre, dist=dist, dist_type='bbox', network_type='drive', 
                                        simplify=SIMPLE, retain_all=False, truncate_by_edge=False, clean_periphery=None, 
                                        custom_filter=None)
    g = ox.projection.project_graph(g, to_latlong=True)
    graph = nx.Graph()

    for n in g.nodes(data=True): 
        position = (n[1]['x'], n[1]['y'])
        graph.add_node(n[0], pos=position)

    for start, end in g.edges(): graph.add_edge(start, end)
    return graph



# positions = nx.get_node_attributes(graph, 'pos')
# nx.draw(graph, pos=positions, node_size=5)
# plt.savefig('graph_full.png')

# fig, ax = ox.plot.plot_graph(g, save=True)
# sat = download_sat(image_width=1000, image_height=1000, max_meters_per_pixel=None)
# get_graph_osm()