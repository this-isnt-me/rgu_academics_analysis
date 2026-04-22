"""
Chart builders for the RGU Research Network Analyser.
All functions return Plotly figures or PyVis HTML strings.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Qualitative colour palette (20 colours)
PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
]


def get_school_color_map(G) -> dict[str, str]:
    schools = sorted({d.get("school", "") for _, d in G.nodes(data=True) if d.get("school")})
    return {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(schools)}


def build_pyvis_network(
    G,
    color_by: str = "school",
    color_map: dict | None = None,
    partition: dict | None = None,
    large_threshold: int = 500,
    show_full: bool = False,
    height: int = 650,
) -> str:
    """Render G as a PyVis network and return the HTML string."""
    from pyvis.network import Network

    # Subset for large graphs
    if G.number_of_nodes() > large_threshold and not show_full:
        wd = dict(G.degree(weight="weight"))
        cutoff = np.percentile(list(wd.values()), 50)
        vis_nodes = [n for n, w in wd.items() if w >= cutoff]
        display_G = G.subgraph(vis_nodes).copy()
    else:
        display_G = G

    net = Network(
        height=f"{height}px",
        width="100%",
        bgcolor="#f8f9fa",
        font_color="#222222",
        directed=False,
    )

    # Colour assignment
    if color_by == "community" and partition:
        comm_ids = sorted(set(partition.values()))
        comm_colors = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(comm_ids)}
        def _color(n):
            return comm_colors.get(partition.get(n, 0), "#aaaaaa")
    else:
        if color_map is None:
            color_map = get_school_color_map(G)
        def _color(n):
            return color_map.get(G.nodes[n].get("school", ""), "#aaaaaa")

    # Node sizing by weighted degree
    wd_all = dict(display_G.degree(weight="weight"))
    wd_vals = list(wd_all.values())
    min_wd = min(wd_vals) if wd_vals else 0
    max_wd = max(wd_vals) if wd_vals else 1
    rng = max_wd - min_wd if max_wd != min_wd else 1

    for n in display_G.nodes():
        d = G.nodes[n]
        name = d.get("label", n)
        size = 8 + 32 * (wd_all.get(n, 0) - min_wd) / rng
        title = (
            f"<b>{name}</b><br>"
            f"School: {d.get('school', '')}<br>"
            f"Job Title: {d.get('job_title', '')}<br>"
            f"Co-authors: {display_G.degree(n)}<br>"
            f"Total papers: {display_G.degree(n, weight='weight')}"
        )
        net.add_node(str(n), label=str(name), title=title, color=_color(n), size=float(size))

    # Edge width by weight
    weights = [d.get("weight", 1) for _, _, d in display_G.edges(data=True)]
    max_w = max(weights) if weights else 1
    for u, v, d in display_G.edges(data=True):
        w = d.get("weight", 1)
        width = 1 + 5 * (w / max_w)
        net.add_edge(str(u), str(v), width=float(width), title=f"{w} co-authored paper(s)")

    net.set_options("""{
      "physics": {
        "enabled": true,
        "stabilization": {"iterations": 120},
        "barnesHut": {
          "gravitationalConstant": -8000,
          "centralGravity": 0.3,
          "springLength": 200,
          "springConstant": 0.04,
          "damping": 0.09
        }
      },
      "edges": {"smooth": {"type": "continuous"}},
      "interaction": {"tooltipDelay": 100}
    }""")

    return net.generate_html()


def plot_centrality_bar(df: pd.DataFrame, score_col: str, title: str, top_n: int = 20) -> go.Figure:
    plot_df = df.head(top_n).copy()
    fig = px.bar(
        plot_df,
        x=score_col,
        y="Name",
        orientation="h",
        color="School",
        title=title,
        hover_data=["School", "Job Title"],
        color_discrete_sequence=PALETTE,
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=520, margin=dict(l=200))
    fig.update_xaxes(title=score_col)
    fig.update_yaxes(title="Academic")
    return fig


def plot_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    size_col: str,
    color_col: str,
    title: str,
    hover_name: str = "Name",
    hover_data: list | None = None,
) -> go.Figure:
    hover_data = hover_data or ["School", "Job Title"]
    # Normalise size to avoid tiny/huge dots
    sizes = df[size_col].clip(lower=0)
    if sizes.max() > 0:
        sizes = 5 + 40 * (sizes / sizes.max())
    else:
        sizes = pd.Series([10] * len(df))

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        size=sizes,
        color=color_col,
        title=title,
        hover_name=hover_name,
        hover_data=hover_data,
        color_discrete_sequence=PALETTE,
    )
    fig.update_layout(height=520)
    return fig


def plot_heatmap(matrix_df: pd.DataFrame, title: str, colorscale: str = "Blues") -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=matrix_df.values,
        x=list(matrix_df.columns),
        y=list(matrix_df.index),
        colorscale=colorscale,
        hoverongaps=False,
        hovertemplate="%{y} → %{x}: %{z} papers<extra></extra>",
    ))
    fig.update_layout(title=title, height=520, xaxis_tickangle=-45)
    return fig


def plot_school_network(meta_G, school_sizes: dict) -> go.Figure:
    if meta_G.number_of_nodes() == 0:
        fig = go.Figure()
        fig.update_layout(title="No inter-school co-authorship data available.")
        return fig

    pos = nx.spring_layout(meta_G, weight="weight", seed=42)
    schools = list(meta_G.nodes())
    color_map = {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(sorted(schools))}

    edge_traces = []
    for u, v, d in meta_G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        w = d.get("weight", 1)
        edge_traces.append(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode="lines",
            line=dict(width=max(1, np.log1p(w)), color="#aaaaaa"),
            hoverinfo="text",
            text=f"{u} — {v}: {w} co-authored papers",
            showlegend=False,
        ))
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        edge_traces.append(go.Scatter(
            x=[mx], y=[my],
            mode="text",
            text=[str(w)],
            textfont=dict(size=9, color="#666666"),
            showlegend=False,
            hoverinfo="none",
        ))

    node_x, node_y, node_text, node_hover, node_sizes, node_colors = [], [], [], [], [], []
    for s in schools:
        x, y = pos[s]
        sz = school_sizes.get(s, 1)
        node_x.append(x)
        node_y.append(y)
        node_text.append(s)
        node_hover.append(f"<b>{s}</b><br>Academics: {sz}")
        node_sizes.append(20 + 4 * sz)
        node_colors.append(color_map[s])

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        hovertext=node_hover,
        hoverinfo="text",
        marker=dict(size=node_sizes, color=node_colors, line=dict(width=2, color="white")),
        showlegend=False,
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        title="School-Level Co-authorship Map",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=620,
    )
    return fig


def plot_stacked_bar(df: pd.DataFrame, label_col: str, role_cols: list, title: str) -> go.Figure:
    plot_df = df[[label_col] + role_cols].head(20)
    melted = plot_df.melt(id_vars=label_col, value_vars=role_cols, var_name="Role", value_name="Score")
    fig = px.bar(melted, x=label_col, y="Score", color="Role", barmode="stack", title=title)
    fig.update_layout(xaxis_tickangle=-45, height=500)
    return fig


def plot_boxplot(df: pd.DataFrame, x_col: str, y_col: str, title: str) -> go.Figure:
    fig = px.box(df, x=x_col, y=y_col, title=title, color=x_col,
                 color_discrete_sequence=PALETTE, points="outliers")
    fig.update_layout(xaxis_tickangle=-45, height=450, showlegend=False)
    fig.update_xaxes(title=x_col)
    fig.update_yaxes(title=y_col)
    return fig


def plot_grouped_bar(df: pd.DataFrame, x_col: str, value_cols: list, title: str) -> go.Figure:
    melted = df.melt(id_vars=x_col, value_vars=value_cols, var_name="Category", value_name="Value")
    fig = px.bar(melted, x=x_col, y="Value", color="Category", barmode="group",
                 title=title, color_discrete_sequence=PALETTE)
    fig.update_layout(xaxis_tickangle=-45, height=420)
    return fig


def school_legend_html(color_map: dict) -> str:
    items = [
        f'<span style="display:inline-block;width:12px;height:12px;background:{c};'
        f'border-radius:2px;margin-right:4px;"></span>{s}'
        for s, c in color_map.items()
    ]
    return "  &nbsp;|&nbsp;  ".join(items)
