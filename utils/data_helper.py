###############################################################################
#
# Some code is adapted from https://github.com/JiaxuanYou/graph-generation
#
###############################################################################
import os
import torch
import pickle
import numpy as np
from scipy import sparse as sp
import networkx as nx
import torch.nn.functional as F

__all__ = [
    'save_graph_list', 'load_graph_list', 'graph_load_batch',
    'preprocess_graph_list', 'create_graphs'
]


# save a list of graphs
def save_graph_list(G_list, fname):
  with open(fname, "wb") as f:
    pickle.dump(G_list, f)


def pick_connected_component_new(G):
  # import pdb; pdb.set_trace()

  # adj_list = G.adjacency_list()
  # for id,adj in enumerate(adj_list):
  #     id_min = min(adj)
  #     if id<id_min and id>=1:
  #     # if id<id_min and id>=4:
  #         break
  # node_list = list(range(id)) # only include node prior than node "id"

  adj_dict = nx.to_dict_of_lists(G)
  for node_id in sorted(adj_dict.keys()):
    id_min = min(adj_dict[node_id])
    if node_id < id_min and node_id >= 1:
      # if node_id<id_min and node_id>=4:
      break
  node_list = list(
      range(node_id))  # only include node prior than node "node_id"

  G = G.subgraph(node_list)
  G = max(nx.connected_component_subgraphs(G), key=len)
  return G


def load_graph_list(fname, is_real=True):
  with open(fname, "rb") as f:
    graph_list = pickle.load(f)

  # import pdb; pdb.set_trace()
  for i in range(len(graph_list)):
    edges_with_selfloops = list(graph_list[i].selfloop_edges())
    if len(edges_with_selfloops) > 0:
      graph_list[i].remove_edges_from(edges_with_selfloops)
    if is_real:
      graph_list[i] = max(
          nx.connected_component_subgraphs(graph_list[i]), key=len)
      graph_list[i] = nx.convert_node_labels_to_integers(graph_list[i])
    else:
      graph_list[i] = pick_connected_component_new(graph_list[i])
  return graph_list


def preprocess_graph_list(graph_list):
  for i in range(len(graph_list)):
    edges_with_selfloops = list(graph_list[i].selfloop_edges())
    if len(edges_with_selfloops) > 0:
      graph_list[i].remove_edges_from(edges_with_selfloops)
    if is_real:
      graph_list[i] = max(
          nx.connected_component_subgraphs(graph_list[i]), key=len)
      graph_list[i] = nx.convert_node_labels_to_integers(graph_list[i])
    else:
      graph_list[i] = pick_connected_component_new(graph_list[i])
  return graph_list


def graph_load_batch(data_dir,
                     min_num_nodes=20,
                     max_num_nodes=1000,
                     name='ENZYMES',
                     node_attributes=True,
                     graph_labels=True):
  '''
    load many graphs, e.g. enzymes
    :return: a list of graphs
    '''
  print('Loading graph dataset: ' + str(name))
  G = nx.Graph()
  # load data
  path = os.path.join(data_dir, name)
  data_adj = np.loadtxt(
      os.path.join(path, '{}_A.txt'.format(name)), delimiter=',').astype(int)
  if node_attributes:
    data_node_att = np.loadtxt(
        os.path.join(path, '{}_node_attributes.txt'.format(name)),
        delimiter=',')
  data_node_label = np.loadtxt(
      os.path.join(path, '{}_node_labels.txt'.format(name)),
      delimiter=',').astype(int)
  data_graph_indicator = np.loadtxt(
      os.path.join(path, '{}_graph_indicator.txt'.format(name)),
      delimiter=',').astype(int)
  if graph_labels:
    data_graph_labels = np.loadtxt(
        os.path.join(path, '{}_graph_labels.txt'.format(name)),
        delimiter=',').astype(int)

  data_tuple = list(map(tuple, data_adj))
  # print(len(data_tuple))
  # print(data_tuple[0])

  # add edges
  G.add_edges_from(data_tuple)
  # add node attributes
  for i in range(data_node_label.shape[0]):
    if node_attributes:
      G.add_node(i + 1, feature=data_node_att[i])
    G.add_node(i + 1, label=data_node_label[i])
  G.remove_nodes_from(list(nx.isolates(G)))

  # remove self-loop
  G.remove_edges_from(nx.selfloop_edges(G))

  # print(G.number_of_nodes())
  # print(G.number_of_edges())

  # split into graphs
  graph_num = data_graph_indicator.max()
  node_list = np.arange(data_graph_indicator.shape[0]) + 1
  graphs = []
  max_nodes = 0
  for i in range(graph_num):
    # find the nodes for each graph
    nodes = node_list[data_graph_indicator == i + 1]
    G_sub = G.subgraph(nodes)
    if graph_labels:
      G_sub.graph['label'] = data_graph_labels[i]
    # print('nodes', G_sub.number_of_nodes())
    # print('edges', G_sub.number_of_edges())
    # print('label', G_sub.graph)
    if G_sub.number_of_nodes() >= min_num_nodes and G_sub.number_of_nodes(
    ) <= max_num_nodes:
      graphs.append(G_sub)
      if G_sub.number_of_nodes() > max_nodes:
        max_nodes = G_sub.number_of_nodes()
      # print(G_sub.number_of_nodes(), 'i', i)
      # print('Graph dataset name: {}, total graph num: {}'.format(name, len(graphs)))
      # logging.warning('Graphs loaded, total num: {}'.format(len(graphs)))
  print('Loaded')
  return graphs


def load_cora_graph(data_dir):
  return load_planetoid_graph('cora', data_dir)

def load_citeseer_graph(data_dir):
  return load_planetoid_graph('citeseer', data_dir)

def load_planetoid_graph(name, data_dir='data'):
    """
    Return [G] where G contains *all* N nodes of the Planetoid citation
    network (Cora = 2708, Citeseer = 3312, Pubmed = 19717).

    Parameters
    ----------
    name : str      'cora' | 'citeseer' | 'pubmed'  (case-insensitive)
    data_dir : str   folder that holds  data/<name>/ind.<name>.*  files
    """

    name = name.lower()
    path = lambda suf: os.path.join(data_dir, name, f'ind.{name}.{suf}')


    # ------------------------------------------------------------------
    # 1) read the adjacency dictionary that *does* exist
    # ------------------------------------------------------------------
    with open(path('graph'), 'rb') as f:
        adj_dict = pickle.load(f, encoding='latin1')

    # ------------------------------------------------------------------
    # 2) figure out how many *total* nodes there should be
    #    • use test.index (always present and includes the max id)
    #      fall back to the largest id seen in adj_dict otherwise
    # ------------------------------------------------------------------
    try:
        test_idx = np.loadtxt(path('test.index'), dtype=int)
        num_nodes = int(test_idx.max()) + 1
    except OSError:
        num_nodes = max(adj_dict.keys()) + 1       # rarely needed

    # ------------------------------------------------------------------
    # 3) build the full graph, adding isolated nodes explicitly
    # ------------------------------------------------------------------
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))            # stubs for *every* node
    for src, nbrs in adj_dict.items():
        G.add_edges_from((src, dst) for dst in nbrs)
    G.remove_edges_from(nx.selfloop_edges(G))     # optional: strip loops

    # **no** LCC chopping here!
    return [G]

def load_polblogs_graph(data_dir='data'):
    """
    Return a list with ONE NetworkX graph containing **all 1 490 nodes**
    of the NetSet/T-SV PolBlogs dataset.

    Expected files (tab-separated, no headers) in
      f"{data_dir}/polblogs/":  adjacency.tsv, names.tsv, labels.tsv
    """
    import os, networkx as nx, numpy as np

    folder = os.path.join(data_dir, 'polblogs')
    # ---------- 1 ) read node meta ----------
    names   = [line.strip()        for line in open(os.path.join(folder, 'names.tsv'))]
    labels  = [int(line.strip())   for line in open(os.path.join(folder, 'labels.tsv'))]
    n       = len(names)                       # 1 490

    # ---------- 2 ) build an empty graph with all nodes ----------
    G = nx.Graph()
    G.add_nodes_from(range(n))                 # 0 … 1489
    for i in range(n):
        G.nodes[i]['url']   = names[i]
        G.nodes[i]['label'] = labels[i]        # 0=liberal, 1=conservative

    # ---------- 3 ) add edges ----------
    with open(os.path.join(folder, 'adjacency.tsv')) as f:
        for line in f:
            u, v, _ = line.split('\t')
            G.add_edge(int(u), int(v))         # duplicates collapse automatically
    G.remove_edges_from(nx.selfloop_edges(G))

    return [G]                                 # GRAN expects a *list*

# ------------------------------------------------------------------
#  Single-graph loader for a pickled adjacency matrix
# ------------------------------------------------------------------
def load_adj_pkl_graph(fname_or_path, data_dir='data'):
    """
    Parameters
    ----------
    fname_or_path : str
        Either an absolute path or a file name relative to <data_dir>/poisoned/.
        The pickle can contain:
          • scipy.sparse matrix
          • numpy ndarray
          • torch Tensor
          • dict with key 'adj' holding one of the above
    Returns
    -------
    [G] : list(networkx.Graph)  – GRAN always wants a *list*.
    """
    import os, pickle, numpy as np, networkx as nx
    from scipy import sparse as sp
    try:
        import torch
        is_tensor = lambda x: torch.is_tensor(x)
    except ImportError:             # just in case
        is_tensor = lambda x: False

    # --------------------------------------------------------------
    # 1) resolve path
    # --------------------------------------------------------------
    if not os.path.isabs(fname_or_path):
        fname_or_path = os.path.join(data_dir, fname_or_path)

    # --------------------------------------------------------------
    # 2) un-pickle
    # --------------------------------------------------------------
    with open(fname_or_path, 'rb') as f:
        obj = pickle.load(f, encoding='latin1')

    if isinstance(obj, dict) and 'adj' in obj:
        obj = obj['adj']                        # unwrap common pattern

    # --------------------------------------------------------------
    # 3) dense / sparse / tensor → NetworkX
    # --------------------------------------------------------------
    if is_tensor(obj):                          # torch.Tensor
        obj = obj.cpu().numpy()
        G = nx.from_numpy_array(obj)
    elif sp.issparse(obj):                      # scipy.sparse
        G = nx.from_scipy_sparse_matrix(obj)
    else:                                       # assume NumPy array-like
        obj = np.asarray(obj)
        G = nx.from_numpy_array(obj)

    # optional clean-up
    G.remove_edges_from(nx.selfloop_edges(G))
    G = nx.convert_node_labels_to_integers(G)

    return [G]




def create_graphs(graph_type, graph_name, data_dir='data', noise=10.0, seed=1234):
  npr = np.random.RandomState(seed)
  ### load datasets
  graphs = []
  # synthetic graphs
  if graph_type == 'grid':
    graphs = []
    for i in range(10, 20):
      for j in range(10, 20):
        graphs.append(nx.grid_2d_graph(i, j))    
  elif graph_type == 'lobster':
    graphs = []
    p1 = 0.7
    p2 = 0.7
    count = 0
    min_node = 10
    max_node = 100
    max_edge = 0
    mean_node = 80
    num_graphs = 100

    seed_tmp = seed
    while count < num_graphs:
      G = nx.random_lobster(mean_node, p1, p2, seed=seed_tmp)
      if len(G.nodes()) >= min_node and len(G.nodes()) <= max_node:
        graphs.append(G)
        if G.number_of_edges() > max_edge:
          max_edge = G.number_of_edges()
        
        count += 1

      seed_tmp += 1
  elif graph_type == 'DD':
    graphs = graph_load_batch(
        data_dir,
        min_num_nodes=100,
        max_num_nodes=500,
        name='DD',
        node_attributes=False,
        graph_labels=True)
    # args.max_prev_node = 230
  elif graph_type == 'FIRSTMM_DB':
    graphs = graph_load_batch(
        data_dir,
        min_num_nodes=0,
        max_num_nodes=10000,
        name='FIRSTMM_DB',
        node_attributes=False,
        graph_labels=True)
  elif graph_type == 'raw':
    if graph_name == 'cora':
        graphs = load_cora_graph(data_dir)
    elif graph_name == 'citeseer':
        graphs = load_citeseer_graph(data_dir)
    elif graph_name == 'polblogs':
        graphs = load_polblogs_graph(data_dir)
  elif graph_type == 'poison':
    graphs = load_adj_pkl_graph(graph_name, data_dir)

  num_nodes = [gg.number_of_nodes() for gg in graphs]
  num_edges = [gg.number_of_edges() for gg in graphs]
  print('max # nodes = {} || mean # nodes = {}'.format(max(num_nodes), np.mean(num_nodes)))
  print('max # edges = {} || mean # edges = {}'.format(max(num_edges), np.mean(num_edges)))
   
  return graphs

