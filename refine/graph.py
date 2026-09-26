import networkx as nx

LIFTING_MODES = ("all_to_all", "direct")
H_INTRA_SCC = "intra_scc_direct"
H_LIFTED = "lifted_all_to_all"
H_CROSS_DIRECT = "cross_direct"

def refinement_graph_from_judgments(n, judgments):
    D = nx.DiGraph()
    D.add_nodes_from(range(n))
    for jd in judgments:
        i, j = jd["source"], jd["target"]
        if i == j:
            raise ValueError(f"self-judgment ({i},{j}) is not part of R")
        if jd["refines"] is True:
            D.add_edge(i, j)
    return D

def compute_scc_quotient(D):
    sccs = sorted((sorted(c) for c in nx.strongly_connected_components(D)), key=lambda c: c[0])
    C_full = nx.condensation(D, scc=[set(c) for c in sccs])
    agent_to_component = {i: C_full.graph["mapping"][i] for i in D.nodes}
    C = nx.DiGraph()
    C.add_nodes_from(range(len(sccs)))
    C.add_edges_from(C_full.edges)
    return sccs, agent_to_component, C

def compute_component_topology(C, transitive_reduction=True):
    G = nx.DiGraph()
    G.add_nodes_from(C.nodes)
    if transitive_reduction:
        G.add_edges_from(nx.transitive_reduction(C).edges)
    else:
        G.add_edges_from(C.edges)
    return G

def build_execution_graph(D, G, sccs, agent_to_component, lifting="all_to_all"):
    if lifting not in LIFTING_MODES:
        raise ValueError(f"cross_component_lifting must be one of {LIFTING_MODES}; got {lifting!r}")
    H = nx.DiGraph()
    H.add_nodes_from(D.nodes)
    for i, j in D.edges:
        if agent_to_component[i] == agent_to_component[j]:
            H.add_edge(i, j, kind=H_INTRA_SCC)
    for a, b in G.edges:
        for i in sccs[a]:
            for j in sccs[b]:
                if lifting == "all_to_all":
                    H.add_edge(i, j, kind=H_LIFTED)
                elif D.has_edge(i, j):
                    H.add_edge(i, j, kind=H_CROSS_DIRECT)
    return H

def maximal_components(G):
    return sorted(c for c in G.nodes if G.in_degree(c) == 0)

def edges(g):
    return sorted([int(u), int(v)] for u, v in g.edges)

def topology_statistics(D, C, G, H, sccs):
    eC, eG = C.number_of_edges(), G.number_of_edges()
    return {
        "E_D": D.number_of_edges(),
        "E_C": eC,
        "E_G": eG,
        "E_H": H.number_of_edges(),
        "num_sccs": len(sccs),
        "scc_sizes": sorted((len(c) for c in sccs), reverse=True),
        "num_maximal_components": len(maximal_components(G)),
        "transitive_reduction_ratio": (1 - eG / eC) if eC > 0 else None,
    }

def check_invariants(D, sccs, agent_to_component, C, G, H, lifting="all_to_all"):
    assert nx.is_directed_acyclic_graph(C), "condensation C is not acyclic"
    assert nx.is_directed_acyclic_graph(G), "component topology G is not acyclic"
    for u in C.nodes:
        assert nx.descendants(C, u) == nx.descendants(G, u), f"G changes reachability from component {u}"
    for c, members in enumerate(sccs):
        for i in members:
            assert agent_to_component[i] == c, f"agent {i} mapped inconsistently"
    for i in D.nodes:
        for j in D.nodes:
            if i == j:
                continue
            same = agent_to_component[i] == agent_to_component[j]
            if same:
                assert H.has_edge(i, j) == D.has_edge(i, j), f"intra-SCC edge ({i},{j}) differs between D and H"
    for a, b in G.edges:
        for i in sccs[a]:
            for j in sccs[b]:
                if lifting == "all_to_all":
                    assert H.has_edge(i, j), f"G edge {a}->{b} not lifted to ({i},{j})"
    if lifting == "all_to_all":
        for i in D.nodes:
            assert nx.descendants(D, i) == nx.descendants(H, i), f"H changes reachability from agent {i}"
    else:
        assert set(H.edges) <= set(D.edges), "direct lifting introduced an unwitnessed edge"
