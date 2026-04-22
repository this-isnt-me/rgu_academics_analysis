"""
ONA metrics: brokerage, constraint, assortativity, E-I index, resilience,
community detection, school meta-graph.
All functions operate on undirected graphs only.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st

from utils.graph_utils import rebuild_graph

# ---------------------------------------------------------------------------
# Seniority mapping
# ---------------------------------------------------------------------------
SENIORITY_MAP = {
    "Professor": 4,
    "Reader": 3,
    "Senior Lecturer": 3,
    "Associate Professor": 3,
    "Lecturer": 2,
    "Research Fellow": 2,
    "Research Assistant": 1,
}


def _seniority(job_title: str) -> int:
    jt_lower = job_title.lower()
    for key, val in SENIORITY_MAP.items():
        if key.lower() in jt_lower:
            return val
    return 1


# ---------------------------------------------------------------------------
# Gould-Fernandez brokerage
# ---------------------------------------------------------------------------
def _classify_triple(gi, gj, gk) -> str:
    """Classify a directed triple (j → i → k) by group membership."""
    if gi == gj == gk:
        return "coordinator"
    if gj == gk and gj != gi:
        return "consultant"
    if gj != gi and gi == gk:
        return "gatekeeper"
    if gj == gi and gi != gk:
        return "representative"
    return "liaison"  # all different


@st.cache_data
def compute_brokerage(nodes_data, edges_data) -> pd.DataFrame:
    G = rebuild_graph(nodes_data, edges_data)
    group = {n: G.nodes[n].get("school", "") for n in G.nodes()}
    roles = ["coordinator", "consultant", "gatekeeper", "representative", "liaison"]
    counts: dict[str, dict[str, float]] = {n: {r: 0 for r in roles} for n in G.nodes()}

    for i in G.nodes():
        gi = group[i]
        nbrs = list(G.neighbors(i))
        d = len(nbrs)
        if d < 2:
            continue
        norm = d * (d - 1)
        for j, k in combinations(nbrs, 2):
            gj, gk = group[j], group[k]
            counts[i][_classify_triple(gi, gj, gk)] += 1
            counts[i][_classify_triple(gi, gk, gj)] += 1
        for r in roles:
            counts[i][r] /= norm

    rows = []
    for n, sc in counts.items():
        d = G.nodes[n]
        total = sum(sc.values())
        rows.append({
            "Name": d.get("label", n),
            "School": d.get("school", ""),
            "Job Title": d.get("job_title", ""),
            "Coordinator": round(sc["coordinator"], 4),
            "Consultant": round(sc["consultant"], 4),
            "Gatekeeper": round(sc["gatekeeper"], 4),
            "Representative": round(sc["representative"], 4),
            "Liaison": round(sc["liaison"], 4),
            "Total Brokerage": round(total, 4),
        })
    return pd.DataFrame(rows).sort_values("Total Brokerage", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Burt's Constraint
# ---------------------------------------------------------------------------
@st.cache_data
def compute_constraint(nodes_data, edges_data) -> pd.DataFrame:
    G = rebuild_graph(nodes_data, edges_data)
    try:
        constraint = nx.constraint(G, weight="weight")
    except Exception:
        constraint = {n: None for n in G.nodes()}

    rows = []
    for n, score in constraint.items():
        d = G.nodes[n]
        rows.append({
            "Name": d.get("label", n),
            "School": d.get("school", ""),
            "Job Title": d.get("job_title", ""),
            "Constraint Score": round(score, 4) if score is not None else None,
            "Weighted Degree": G.degree(n, weight="weight"),
        })
    df = pd.DataFrame(rows).dropna(subset=["Constraint Score"])
    return df.sort_values("Constraint Score").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Assortativity
# ---------------------------------------------------------------------------
@st.cache_data
def compute_assortativity(nodes_data, edges_data):
    G = rebuild_graph(nodes_data, edges_data)
    try:
        school_assort = nx.attribute_assortativity_coefficient(G, "school")
    except Exception:
        school_assort = None

    for n in G.nodes():
        G.nodes[n]["seniority_rank"] = _seniority(G.nodes[n].get("job_title", ""))
    try:
        rank_assort = nx.numeric_assortativity_coefficient(G, "seniority_rank")
    except Exception:
        rank_assort = None

    return school_assort, rank_assort


@st.cache_data
def compute_school_heatmap(nodes_data, edges_data) -> pd.DataFrame:
    G = rebuild_graph(nodes_data, edges_data)
    schools = sorted({G.nodes[n].get("school", "") for n in G.nodes() if G.nodes[n].get("school")})
    mat = pd.DataFrame(0, index=schools, columns=schools, dtype=int)
    for u, v, d in G.edges(data=True):
        su, sv = G.nodes[u].get("school", ""), G.nodes[v].get("school", "")
        w = int(d.get("weight", 1))
        if su and sv:
            mat.at[su, sv] += w
            if su != sv:
                mat.at[sv, su] += w
    return mat


@st.cache_data
def compute_title_heatmap(nodes_data, edges_data) -> pd.DataFrame:
    G = rebuild_graph(nodes_data, edges_data)
    titles = sorted({G.nodes[n].get("job_title", "") for n in G.nodes() if G.nodes[n].get("job_title")})
    mat = pd.DataFrame(0, index=titles, columns=titles, dtype=int)
    for u, v, d in G.edges(data=True):
        tu, tv = G.nodes[u].get("job_title", ""), G.nodes[v].get("job_title", "")
        w = int(d.get("weight", 1))
        if tu and tv:
            mat.at[tu, tv] += w
            if tu != tv:
                mat.at[tv, tu] += w
    return mat


# ---------------------------------------------------------------------------
# Resilience: articulation points, k-core, school fragmentation
# ---------------------------------------------------------------------------
@st.cache_data
def compute_articulation_points(nodes_data, edges_data) -> pd.DataFrame:
    G = rebuild_graph(nodes_data, edges_data)
    try:
        aps = list(nx.articulation_points(G))
    except Exception:
        aps = []
    rows = []
    for n in aps:
        d = G.nodes.get(n, {})
        rows.append({
            "Name": d.get("label", n),
            "School": d.get("school", ""),
            "Job Title": d.get("job_title", ""),
            "Risk Description": (
                "Removing this academic from the co-authorship network would split "
                "the recorded collaboration record into two or more disconnected components."
            ),
        })
    return pd.DataFrame(rows)


@st.cache_data
def compute_kcore(nodes_data, edges_data) -> pd.DataFrame:
    G = rebuild_graph(nodes_data, edges_data)
    try:
        core_num = nx.core_number(G)
    except Exception:
        core_num = {n: 0 for n in G.nodes()}
    rows = [
        {
            "Name": G.nodes[n].get("label", n),
            "School": G.nodes[n].get("school", ""),
            "Job Title": G.nodes[n].get("job_title", ""),
            "K-Core Number": k,
        }
        for n, k in core_num.items()
    ]
    return pd.DataFrame(rows).sort_values("K-Core Number", ascending=False).reset_index(drop=True)


@st.cache_data
def compute_school_fragmentation(nodes_data, edges_data) -> pd.DataFrame:
    G = rebuild_graph(nodes_data, edges_data)
    schools = sorted({G.nodes[n].get("school", "") for n in G.nodes() if G.nodes[n].get("school")})
    rows = []
    for school in schools:
        school_nodes = [n for n in G.nodes() if G.nodes[n].get("school", "") == school]
        subG = G.subgraph(school_nodes).copy()
        n_acad = len(school_nodes)
        if n_acad == 0:
            continue
        try:
            aps = list(nx.articulation_points(subG))
            n_aps = len(aps)
        except Exception:
            n_aps = 0
        comps = list(nx.connected_components(subG))
        largest_cc = max((len(c) for c in comps), default=0)
        pct = round(100 * largest_cc / n_acad, 1) if n_acad else 0
        rows.append({
            "School": school,
            "Academics": n_acad,
            "Internal Articulation Points": n_aps,
            "Largest Connected Component (%)": pct,
            "High Risk": n_aps > 0,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Community detection
# ---------------------------------------------------------------------------
@st.cache_data
def compute_communities(nodes_data, edges_data):
    """Returns (partition_dict, modularity, method_name)."""
    G = rebuild_graph(nodes_data, edges_data)
    partition = {}
    method = "unknown"
    modularity = None

    try:
        import community as community_louvain  # python-louvain
        partition = community_louvain.best_partition(G, weight="weight")
        method = "Louvain"
        try:
            modularity = community_louvain.modularity(partition, G, weight="weight")
        except Exception:
            pass
    except Exception:
        try:
            comms = list(nx.community.greedy_modularity_communities(G, weight="weight"))
            for i, comm in enumerate(comms):
                for n in comm:
                    partition[n] = i
            method = "Greedy Modularity"
            try:
                modularity = nx.community.modularity(G, [set(c) for c in comms], weight="weight")
            except Exception:
                pass
        except Exception:
            for n in G.nodes():
                partition[n] = 0
            method = "Fallback (single community)"

    return partition, modularity, method


@st.cache_data
def compute_community_summary(nodes_data, edges_data, partition_tuple) -> pd.DataFrame:
    G = rebuild_graph(nodes_data, edges_data)
    partition = dict(partition_tuple)
    communities: dict[int, list] = {}
    for n, c in partition.items():
        communities.setdefault(c, []).append(n)

    rows = []
    for comm_id, members in sorted(communities.items(), key=lambda x: -len(x[1])):
        subG = G.subgraph(members)
        schools = [G.nodes[m].get("school", "") for m in members]
        titles = [G.nodes[m].get("job_title", "") for m in members]
        sch_cnt = Counter(schools)
        ttl_cnt = Counter(titles)
        top_school = sch_cnt.most_common(1)[0][0] if sch_cnt else ""
        top_school_pct = sch_cnt.most_common(1)[0][1] / len(members) if members else 0
        top_title = ttl_cnt.most_common(1)[0][0] if ttl_cnt else ""
        int_deg = dict(subG.degree())
        top3_ids = sorted(int_deg, key=int_deg.get, reverse=True)[:3]
        top3_names = [G.nodes[m].get("label", m) for m in top3_ids]
        alignment = "Single-school community" if top_school_pct > 0.8 else "Cross-school community"
        rows.append({
            "Community": comm_id,
            "Size": len(members),
            "Schools": ", ".join(sorted(set(schools))),
            "Dominant School (%)": f"{top_school} ({round(100*top_school_pct)}%)",
            "Most Common Title": top_title,
            "Top Academics (by internal degree)": ", ".join(top3_names),
            "School Alignment": alignment,
        })
    return pd.DataFrame(rows)


def classify_community_roles(G, partition: dict) -> dict[str, str]:
    """
    Assign Hub / Bridge / Peripheral / Member to each node.
    Hub: top-20% internal degree in their community
    Bridge: edges to >= 2 other communities
    Peripheral: bottom-20% internal degree, no cross-community edges
    """
    communities: dict[int, list] = {}
    for n, c in partition.items():
        communities.setdefault(c, []).append(n)

    comm_sets = {c: set(members) for c, members in communities.items()}

    roles = {}
    for n in G.nodes():
        if n not in partition:
            roles[n] = "Member"
            continue
        my_comm = partition[n]
        my_set = comm_sets[my_comm]
        int_nbrs = [nb for nb in G.neighbors(n) if nb in my_set]
        ext_comms = {partition[nb] for nb in G.neighbors(n) if nb not in my_set and nb in partition}

        if len(ext_comms) >= 2:
            roles[n] = "Bridge"
            continue

        # Compute percentiles within community
        all_int_deg = [
            sum(1 for nb in G.neighbors(m) if nb in my_set)
            for m in my_set
        ]
        p80 = float(np.percentile(all_int_deg, 80)) if all_int_deg else 0
        p20 = float(np.percentile(all_int_deg, 20)) if all_int_deg else 0
        my_int_deg = len(int_nbrs)

        if my_int_deg >= p80:
            roles[n] = "Hub"
        elif my_int_deg <= p20 and not ext_comms:
            roles[n] = "Peripheral"
        else:
            roles[n] = "Member"

    return roles


# ---------------------------------------------------------------------------
# E-I Index
# ---------------------------------------------------------------------------
@st.cache_data
def compute_ei_index(nodes_data, edges_data):
    """Returns (global_ei, school_df, title_df)."""
    G = rebuild_graph(nodes_data, edges_data)

    # --- Institution-wide ---
    g_int, g_ext = 0, 0
    for u, v, d in G.edges(data=True):
        w = d.get("weight", 1)
        if G.nodes[u].get("school", "") == G.nodes[v].get("school", ""):
            g_int += w
        else:
            g_ext += w
    total = g_int + g_ext
    ei_global = round((g_ext - g_int) / total, 4) if total else 0.0

    # --- Per school ---
    schools = sorted({G.nodes[n].get("school", "") for n in G.nodes() if G.nodes[n].get("school")})
    sch_rows = []
    for school in schools:
        school_set = {n for n in G.nodes() if G.nodes[n].get("school", "") == school}
        s_int, s_ext = 0, 0
        for u, v, d in G.edges(data=True):
            w = d.get("weight", 1)
            u_in = u in school_set
            v_in = v in school_set
            if u_in and v_in:
                s_int += w
            elif u_in or v_in:
                s_ext += w
        s_total = s_int + s_ext
        ei = round((s_ext - s_int) / s_total, 4) if s_total else 0.0
        if ei > 0.2:
            interp = "Interdisciplinary Hub"
        elif ei < -0.2:
            interp = "Insular — strong internal co-authorship focus"
        else:
            interp = "Balanced"
        sch_rows.append({
            "School": school,
            "Internal Co-authorships": s_int,
            "External Co-authorships": s_ext,
            "E-I Score": ei,
            "Interpretation": interp,
        })
    school_df = pd.DataFrame(sch_rows).sort_values("E-I Score", ascending=False).reset_index(drop=True)

    # --- Per job title ---
    title_counts: dict[str, dict] = {}
    for n in G.nodes():
        jt = G.nodes[n].get("job_title", "")
        if jt not in title_counts:
            title_counts[jt] = {"internal": 0, "external": 0}

    for u, v, d in G.edges(data=True):
        w = d.get("weight", 1)
        su, sv = G.nodes[u].get("school", ""), G.nodes[v].get("school", "")
        tu, tv = G.nodes[u].get("job_title", ""), G.nodes[v].get("job_title", "")
        is_ext = su != sv
        key = "external" if is_ext else "internal"
        if tu in title_counts:
            title_counts[tu][key] += w
        if tv in title_counts:
            title_counts[tv][key] += w

    ttl_rows = []
    for jt, cnts in title_counts.items():
        t = cnts["internal"] + cnts["external"]
        ei = round((cnts["external"] - cnts["internal"]) / t, 4) if t else 0.0
        ttl_rows.append({
            "Job Title": jt,
            "Internal Co-authorships": cnts["internal"],
            "External Co-authorships": cnts["external"],
            "E-I Score": ei,
        })
    title_df = pd.DataFrame(ttl_rows).sort_values("E-I Score", ascending=False).reset_index(drop=True)

    return ei_global, school_df, title_df


# ---------------------------------------------------------------------------
# School meta-graph
# ---------------------------------------------------------------------------
@st.cache_data
def compute_school_metagraph(nodes_data, edges_data):
    """Returns (meta_G, school_sizes_dict, pair_df, bc_df)."""
    G = rebuild_graph(nodes_data, edges_data)
    schools = sorted({G.nodes[n].get("school", "") for n in G.nodes() if G.nodes[n].get("school")})
    school_sizes = {s: sum(1 for n in G.nodes() if G.nodes[n].get("school", "") == s) for s in schools}

    meta_G = nx.Graph()
    for s in schools:
        meta_G.add_node(s, size=school_sizes[s])

    for u, v, d in G.edges(data=True):
        su, sv = G.nodes[u].get("school", ""), G.nodes[v].get("school", "")
        if su and sv and su != sv:
            w = d.get("weight", 1)
            if meta_G.has_edge(su, sv):
                meta_G[su][sv]["weight"] += w
            else:
                meta_G.add_edge(su, sv, weight=w)

    # Betweenness on meta-graph
    try:
        bc = nx.betweenness_centrality(meta_G, weight="weight", normalized=True)
    except Exception:
        bc = {s: 0.0 for s in schools}

    # Pair table
    pairs = [
        {"School A": u, "School B": v, "Co-authored Papers": d["weight"]}
        for u, v, d in meta_G.edges(data=True)
    ]
    if pairs:
        pair_df = pd.DataFrame(pairs).sort_values("Co-authored Papers", ascending=False).reset_index(drop=True)
        q80 = pair_df["Co-authored Papers"].quantile(0.8)
        q20 = pair_df["Co-authored Papers"].quantile(0.2)

        def _classify(w):
            if w >= q80:
                return "Strategic co-authorship partnership"
            if w <= q20:
                return "Emerging or dormant"
            return "Active co-authorship relationship"

        pair_df["Interpretation"] = pair_df["Co-authored Papers"].apply(_classify)
    else:
        pair_df = pd.DataFrame(columns=["School A", "School B", "Co-authored Papers", "Interpretation"])

    def _bc_interp(score):
        if score > 0.3:
            return "High bridge — many inter-school paths pass through here"
        if score > 0.1:
            return "Moderate bridge — connects several schools"
        return "Peripheral in school-level co-authorship network"

    bc_df = pd.DataFrame([
        {"School": s, "Betweenness Score": round(bc.get(s, 0), 4), "Interpretation": _bc_interp(bc.get(s, 0))}
        for s in schools
    ]).sort_values("Betweenness Score", ascending=False).reset_index(drop=True)

    return meta_G, school_sizes, pair_df, bc_df


# ---------------------------------------------------------------------------
# School bridges
# ---------------------------------------------------------------------------

@st.cache_data
def compute_school_bridges(nodes_data, edges_data, betweenness_tuple) -> pd.DataFrame:
    """
    Identify the top bridging academic between every pair of schools that share
    at least one co-authorship edge in the filtered graph.

    For each school pair (A, B) the function:
    1. Collects all cross-boundary edges between academics in A and academics in B.
    2. Builds a candidate pool of academics incident to those edges.
    3. Ranks candidates by their pre-computed global betweenness centrality and
       selects the top-ranked node as the primary bridge.
    4. Computes bridge_strength = (weight of cross-boundary edges incident to the
       bridge node) / (total weight of all cross-boundary edges for that pair).

    Parameters
    ----------
    nodes_data : tuple
        Serialised node data from graph_to_cache_args(). Used as the cache key.
    edges_data : tuple
        Serialised edge data from graph_to_cache_args(). Used as the cache key.
    betweenness_tuple : tuple
        Pre-computed global betweenness centrality expressed as
        ``tuple(sorted(bc.items()))`` so it is hashable for caching.
        Must have been computed on the same filtered graph represented by
        nodes_data / edges_data.  The caller is responsible for ensuring this;
        typically obtained via compute_betweenness_centrality(nodes_data, edges_data)
        from utils.centrality, which is itself cached.

    Returns
    -------
    pd.DataFrame
        Columns: school_a, school_b, bridge_academic, bridge_job_title,
        bridge_school, betweenness_score, bridge_strength, total_cross_papers,
        bridge_papers, num_candidates, fragility_flag.
        Sorted by bridge_strength descending.

    Notes
    -----
    Relies entirely on the undirected graph representation.  Each cross-boundary
    edge is treated as mutual; a node is a candidate if it is incident to at least
    one edge crossing the school boundary for that pair.  The fragility threshold
    of 0.5 (50 %) is a heuristic and should be considered alongside the absolute
    paper counts — in small school pairs a high bridge_strength may simply reflect
    a naturally limited collaboration pool.
    """
    G = rebuild_graph(nodes_data, edges_data)
    betweenness = dict(betweenness_tuple)

    # Index nodes by school for fast lookup
    school_nodes: dict[str, set] = {}
    for n in G.nodes():
        s = G.nodes[n].get("school", "")
        if s:
            school_nodes.setdefault(s, set()).add(n)

    schools = sorted(school_nodes.keys())
    rows = []

    for idx_a, school_a in enumerate(schools):
        for school_b in schools[idx_a + 1:]:
            nodes_a = school_nodes[school_a]
            nodes_b = school_nodes[school_b]

            # Collect every edge that crosses this specific school boundary
            cross_edges: list[tuple] = []
            for u, v, d in G.edges(data=True):
                a_to_b = (u in nodes_a and v in nodes_b)
                b_to_a = (u in nodes_b and v in nodes_a)
                if a_to_b or b_to_a:
                    cross_edges.append((u, v, d.get("weight", 1)))

            if not cross_edges:
                continue

            total_cross_papers = sum(w for _, _, w in cross_edges)

            # Candidate pool: every node incident to a cross-boundary edge
            candidates: set = set()
            for u, v, _ in cross_edges:
                candidates.add(u)
                candidates.add(v)

            # Rank by global betweenness; tie-break alphabetically on label for
            # determinism
            def _sort_key(n):
                return (-betweenness.get(n, 0.0), G.nodes[n].get("label", n))

            ranked = sorted(candidates, key=_sort_key)
            bridge_node = ranked[0]

            d_node = G.nodes[bridge_node]
            bridge_name = d_node.get("label", bridge_node)
            bridge_jt = d_node.get("job_title", "")
            bridge_sch = d_node.get("school", "")
            bc_score = betweenness.get(bridge_node, 0.0)

            # Bridge strength: share of cross-boundary weight incident to bridge node
            bridge_papers = sum(
                w for u, v, w in cross_edges
                if u == bridge_node or v == bridge_node
            )
            bridge_strength = (
                bridge_papers / total_cross_papers if total_cross_papers > 0 else 0.0
            )

            rows.append({
                "school_a": school_a,
                "school_b": school_b,
                "bridge_academic": bridge_name,
                "bridge_job_title": bridge_jt,
                "bridge_school": bridge_sch,
                "betweenness_score": round(bc_score, 4),
                "bridge_strength": round(bridge_strength, 4),
                "total_cross_papers": int(total_cross_papers),
                "bridge_papers": int(bridge_papers),
                "num_candidates": len(candidates),
                "fragility_flag": bool(bridge_strength > 0.5),
            })

    if not rows:
        return pd.DataFrame(columns=[
            "school_a", "school_b", "bridge_academic", "bridge_job_title",
            "bridge_school", "betweenness_score", "bridge_strength",
            "total_cross_papers", "bridge_papers", "num_candidates", "fragility_flag",
        ])

    return (
        pd.DataFrame(rows)
        .sort_values("bridge_strength", ascending=False)
        .reset_index(drop=True)
    )
