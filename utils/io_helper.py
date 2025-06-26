import os
import networkx as nx
import scipy.sparse as sp
from pathlib import Path


def save_graph_as_npz(G, fname):
    """
    Save a NetworkX graph to a compressed sparse .npz adjacency matrix.

    Parameters
    ----------
    G : networkx.Graph
    fname : str
        Path like "exp/GRAN_Cora/generated/graph_000.npz"
    """
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    A_csr = nx.to_scipy_sparse_array(G, dtype='float32', format='csr')
    sp.save_npz(fname, A_csr, compressed=True)

def save_graph_as_npz_strict(G, fname, total_nodes):
    """
    Save `G` to .npz, *padding* with degree-0 nodes so the matrix is
    exactly `total_nodes` × `total_nodes`.

    Parameters
    ----------
    G : networkx.Graph
    fname : str
    total_nodes : int
        Expected number of nodes (2708 for raw Cora).
    """
    # 0) Make dir
    os.makedirs(os.path.dirname(fname), exist_ok=True)

    # 1) Ensure contiguous integer labels 0 … total_nodes-1
    G = nx.convert_node_labels_to_integers(G, ordering="sorted")

    # 2) Add missing isolates
    if G.number_of_nodes() < total_nodes:
        missing = set(range(total_nodes)) - set(G.nodes)
        G.add_nodes_from(missing)

    assert G.number_of_nodes() == total_nodes, \
        f"padding failed: {G.number_of_nodes()} nodes"

    # 3) Build CSR with a *fixed nodelist* so the row/col order is stable
    A_csr = nx.to_scipy_sparse_array(
        G,
        nodelist=range(total_nodes),   # fixes shape & ordering
        format="csr",
        dtype="float32"
    )

    sp.save_npz(fname, A_csr, compressed=True)

def sanity_check_npz(fname, expected_n=None, expected_m=None, verbose=True):
    """
    Fast integrity checks on a graph stored as a .npz adjacency.

    Parameters
    ----------
    fname : str | Path
        Path to a .npz file produced by save_graph_as_npz.
    expected_n : int or None
        Assert the number of nodes equals this (optional).
    expected_m : int or None
        Assert the number of *undirected* edges equals this (optional).
    verbose : bool
        Print a short pass/fail summary.
    """
    fname = Path(fname)
    assert fname.is_file(), f"{fname} does not exist"

    A = sp.load_npz(fname)           # ❶ load
    assert A.format == "csr", "matrix is not CSR"
    n, m = A.shape
    assert n == m, f"matrix not square: {n}×{m}"
    if expected_n is not None:
        assert n == expected_n, f"#nodes {n} ≠ expected {expected_n}"

    # ❷ binary values
    assert set(A.data) <= {0, 1}, "matrix contains values other than 0/1"

    # ❸ no self-loops
    assert A.diagonal().sum() == 0, "non-zero diagonal (self-loops)"

    # ❹ symmetry (undirected)
    assert (A != A.T).nnz == 0, "matrix not symmetric"

    # ❺ edge count
    m_undirected = A.nnz // 2        # each edge appears twice
    if expected_m is not None:
        assert m_undirected == expected_m, \
               f"#edges {m_undirected} ≠ expected {expected_m}"

    if verbose:
        print(f"[OK] {fname.name}: n={n}, m={m_undirected}")

    # (optional) return as NetworkX graph for further stats
    return nx.from_scipy_sparse_array(A)