import networkx as nx
import pandas as pd
import streamlit as st

from utils.graph_utils import rebuild_graph


@st.cache_data
def compute_degree_centrality(nodes_data, edges_data):
    G = rebuild_graph(nodes_data, edges_data)
    return nx.degree_centrality(G)


@st.cache_data
def compute_weighted_degree(nodes_data, edges_data):
    G = rebuild_graph(nodes_data, edges_data)
    return dict(G.degree(weight="weight"))


@st.cache_data
def compute_betweenness_centrality(nodes_data, edges_data):
    G = rebuild_graph(nodes_data, edges_data)
    return nx.betweenness_centrality(G, weight="weight", normalized=True)


@st.cache_data
def compute_eigenvector_centrality(nodes_data, edges_data):
    G = rebuild_graph(nodes_data, edges_data)
    try:
        return nx.eigenvector_centrality(G, weight="weight", max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        try:
            return nx.eigenvector_centrality_numpy(G, weight="weight")
        except Exception:
            return {n: 0.0 for n in G.nodes()}


@st.cache_data
def compute_hits(nodes_data, edges_data):
    G = rebuild_graph(nodes_data, edges_data)
    try:
        hubs, authorities = nx.hits(G, max_iter=1000, normalized=True)
        return hubs, authorities
    except Exception:
        zero = {n: 0.0 for n in G.nodes()}
        return zero, zero


def centrality_to_df(scores: dict, G, score_label: str = "Score") -> pd.DataFrame:
    """Convert a {node: score} dict to a ranked DataFrame with Name/School/Job Title."""
    rows = [
        {
            "Name": G.nodes[n].get("label", n),
            "School": G.nodes[n].get("school", ""),
            "Job Title": G.nodes[n].get("job_title", ""),
            score_label: round(score, 6),
        }
        for n, score in scores.items()
    ]
    df = (
        pd.DataFrame(rows)
        .sort_values(score_label, ascending=False)
        .reset_index(drop=True)
    )
    df.index = df.index + 1
    df.index.name = "Rank"
    return df
