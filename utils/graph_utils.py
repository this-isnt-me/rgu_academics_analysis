import os
import networkx as nx
import pandas as pd
import streamlit as st

GRAPH_PATH = "data/rgu_collaborations_refined.graphml"


@st.cache_resource
def load_graph():
    """Load the full graph from disk, convert to undirected, remove incomplete nodes."""
    if not os.path.exists(GRAPH_PATH):
        return None, (
            f"Graph file not found at: **{GRAPH_PATH}**. "
            "Please place `rgu_collaborations_refined.graphml` in the `data/` directory "
            "and restart the app."
        )
    try:
        G = nx.read_graphml(GRAPH_PATH)
        # Convert to simple undirected graph (handles MultiGraph from some GraphML files)
        if isinstance(G, nx.MultiGraph):
            G = nx.Graph(G)
        else:
            G = G.to_undirected()

        # Remove self-loops — nx.core_number and several other algorithms require a simple graph
        G.remove_edges_from(list(nx.selfloop_edges(G)))

        incomplete = [
            n for n, d in G.nodes(data=True)
            if not d.get("school") or not d.get("job_title")
        ]
        G.remove_nodes_from(incomplete)

        # Remove isolates created by incomplete-node removal or self-loop stripping
        G.remove_nodes_from(list(nx.isolates(G)))

        return G, None
    except Exception as exc:
        return None, f"Error loading graph: {exc}"


def get_unique_schools(G):
    return sorted({d.get("school", "") for _, d in G.nodes(data=True) if d.get("school")})


def get_unique_titles(G):
    return sorted({d.get("job_title", "") for _, d in G.nodes(data=True) if d.get("job_title")})


def build_filtered_subgraph(G, selected_schools, selected_titles):
    """Return a clean undirected subgraph for the chosen filters."""
    nodes = [
        n for n, d in G.nodes(data=True)
        if d.get("school") in selected_schools and d.get("job_title") in selected_titles
    ]
    return G.subgraph(nodes).copy()


def get_node_label(G, n: str) -> str:
    """Return the human-readable label for a node (falls back to node ID)."""
    return G.nodes[n].get("label", n)


def graph_to_cache_args(G):
    """
    Convert G to a pair of hashable tuples suitable for use as @st.cache_data keys.
    Returns (nodes_data, edges_data).
    nodes_data: tuple of (node_id, school, job_title, label)
    edges_data: tuple of (u, v, weight)  — u < v lexicographically to avoid duplicates
    """
    nodes_data = tuple(sorted(
        (n,
         G.nodes[n].get("school", ""),
         G.nodes[n].get("job_title", ""),
         G.nodes[n].get("label", n))
        for n in G.nodes()
    ))
    edges_data = tuple(sorted(
        (min(u, v), max(u, v), G[u][v].get("weight", 1))
        for u, v in G.edges()
        if u != v  # safety: exclude any self-loops not yet stripped
    ))
    return nodes_data, edges_data


def rebuild_graph(nodes_data, edges_data):
    """Reconstruct a nx.Graph from the serialised cache-arg tuples."""
    G = nx.Graph()
    for row in nodes_data:
        if len(row) == 4:
            node, school, job_title, label = row
        else:
            node, school, job_title = row
            label = node
        G.add_node(node, school=school, job_title=job_title, label=label)
    for u, v, weight in edges_data:
        G.add_edge(u, v, weight=weight)
    return G


def get_node_dataframe(G):
    rows = []
    for n, d in G.nodes(data=True):
        rows.append({
            "Name": d.get("label", n),
            "School": d.get("school", ""),
            "Job Title": d.get("job_title", ""),
            "Number of Co-authors": G.degree(n),
            "Total Co-authored Papers": G.degree(n, weight="weight"),
        })
    return pd.DataFrame(rows)