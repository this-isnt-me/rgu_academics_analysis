"""
RGU Research Network Analyser — main Streamlit application.
All analyses operate on undirected co-authorship graphs only.
"""
from __future__ import annotations

import io
import os
import traceback

import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from utils.graph_utils import (
    load_graph,
    get_unique_schools,
    get_unique_titles,
    build_filtered_subgraph,
    graph_to_cache_args,
    get_node_dataframe,
)
from utils.centrality import (
    compute_degree_centrality,
    compute_weighted_degree,
    compute_betweenness_centrality,
    compute_eigenvector_centrality,
    compute_hits,
    centrality_to_df,
)
from utils.ona_metrics import (
    compute_brokerage,
    compute_constraint,
    compute_assortativity,
    compute_school_heatmap,
    compute_title_heatmap,
    compute_articulation_points,
    compute_kcore,
    compute_school_fragmentation,
    compute_communities,
    compute_community_summary,
    classify_community_roles,
    compute_ei_index,
    compute_school_metagraph,
    compute_school_bridges,
)
from utils.visualisation import (
    get_school_color_map,
    build_pyvis_network,
    plot_centrality_bar,
    plot_scatter,
    plot_heatmap,
    plot_school_network,
    plot_school_bridge_network,
    plot_stacked_bar,
    plot_boxplot,
    plot_grouped_bar,
    school_legend_html,
    PALETTE,
)

# ============================================================
# Page configuration
# ============================================================
st.set_page_config(
    layout="wide",
    page_title="RGU Research Network Analyser",
    page_icon="🔬",
)

# ============================================================
# Authentication — config and initialisation
# ============================================================
if not os.path.exists("config.yaml"):
    st.error(
        "**config.yaml not found.** Please create it from the project template "
        "before running the app. See the project README for instructions."
    )
    st.stop()

with open("config.yaml") as _f:
    _config = yaml.load(_f, Loader=SafeLoader)

# SECURITY NOTES — MUST READ BEFORE DEPLOYMENT
# 1. Run generate_passwords.py to hash all passwords before deploying.
# 2. Change the cookie key in config.yaml to a long random string before deploying.
# 3. Never commit config.yaml to a public Git repository — it is in .gitignore.
# 4. Set cookie expiry_days to 0 in config.yaml for session-only login (no persistent cookie).
authenticator = stauth.Authenticate(
    _config["credentials"],
    _config["cookie"]["name"],
    _config["cookie"]["key"],
    _config["cookie"]["expiry_days"],
)

# ============================================================
# Auth gate — show login form if not yet authenticated
# ============================================================
if st.session_state.get("authentication_status") is not True:
    _, _login_col, _ = st.columns([3, 4, 3])
    with _login_col:
        authenticator.login()
    if st.session_state.get("authentication_status") is False:
        st.error("Username or password is incorrect.")
    else:
        st.warning(
            "Please enter your credentials to access the RGU Research Network Analyser."
        )
    st.stop()

# ============================================================
# Load graph
# ============================================================
G_full, load_error = load_graph()

if load_error:
    st.error(f"**Graph loading error:** {load_error}")
    st.stop()

if G_full.is_directed():
    st.error(
        "The loaded graph is directed. All analyses require an undirected graph. "
        "Please check the source file and ensure it represents undirected co-authorship relationships."
    )
    st.stop()

# ============================================================
# Sidebar — navigation & filters
# ============================================================
st.sidebar.markdown(f"Logged in as **{st.session_state['name']}**")
authenticator.logout("Logout", "sidebar")
st.sidebar.markdown("---")
st.sidebar.title("🔬 RGU Network Analyser")
st.sidebar.markdown("---")

PAGES = [
    "1. Introduction & How to Use",
    "2. Network Overview",
    "3. Centrality Analysis",
    "4. Brokerage & Structural Roles",
    "5. Structural Holes & Innovation",
    "6. Collaboration Culture",
    "7. Network Resilience & Key-Person Risk",
    "8. Community Detection & Research Tribes",
    "9. Interdepartmental Synergy (E-I Index)",
    "10. School-Level Collaboration Map",
    "11. School Bridges & Key Connectors",
    "12. HITS Analysis — Hubs & Authorities",
    "13. Download & Export",
]

page = st.sidebar.radio("Navigate to:", PAGES, label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")

all_schools = get_unique_schools(G_full)
all_titles = get_unique_titles(G_full)

if st.sidebar.button("↺  Reset filters"):
    for k in ("school_ms", "title_ms"):
        if k in st.session_state:
            del st.session_state[k]
    st.rerun()

selected_schools = st.sidebar.multiselect(
    "Filter by School",
    options=all_schools,
    default=all_schools,
    key="school_ms",
)
selected_titles = st.sidebar.multiselect(
    "Filter by Job Title",
    options=all_titles,
    default=all_titles,
    key="title_ms",
)

if not selected_schools or not selected_titles:
    st.sidebar.warning("Select at least one school and one job title.")
    G = nx.Graph()
else:
    G = build_filtered_subgraph(G_full, selected_schools, selected_titles)

n_nodes = G.number_of_nodes()
n_schools_shown = len({G.nodes[n].get("school", "") for n in G.nodes() if G.nodes[n].get("school")})
st.sidebar.info(f"**Showing {n_nodes} academics across {n_schools_shown} schools**")

school_color_map = get_school_color_map(G_full)


# ============================================================
# Helpers
# ============================================================
def min_nodes_ok(G, minimum: int = 5) -> bool:
    if G.number_of_nodes() < minimum:
        st.warning(
            f"The current filter selection contains only **{G.number_of_nodes()} academics**. "
            f"Most analyses require at least {minimum} academics to be meaningful. "
            "Please broaden your filters."
        )
        return False
    return True


def safe_run(fn, *args, label="metric", **kwargs):
    """Run fn(*args, **kwargs); on exception show a warning and return None."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        st.warning(f"Could not compute {label}: {exc}")
        return None


# ============================================================
# Page 1 — Introduction & How to Use
# ============================================================
def page_introduction():
    st.header("Introduction & How to Use This Tool")
    st.caption(
        "An editorial guide for RGU leadership explaining what this tool reveals "
        "about co-authorship patterns and how to interpret its outputs."
    )

    st.markdown(
        """
        This tool maps the co-authorship relationships between RGU academic staff to surface strategic
        insights about research collaboration, knowledge flow, and institutional resilience.
        The underlying data is drawn **exclusively from published co-authored papers**.
        A connection between two academics exists only because they have jointly published at least one
        paper together — nothing more and nothing less. The strength of a connection reflects how many
        papers they have co-authored, not the depth of their working relationship in any broader sense.
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("What this tool can tell you")
        st.markdown(
            """
            - Which academics are the most collaborative in the published record
            - Which academics are the most strategically positioned in the co-authorship network
            - Where schools are publishing jointly and where they are not
            - Which individuals or relationships represent a fragility risk to RGU's research portfolio
            """
        )
    with col2:
        st.subheader("What this tool cannot tell you")
        st.markdown(
            """
            - Whether two academics work closely together but have not yet co-published
            - The quality or impact of the papers behind each connection
            - Relationships formed through teaching, supervision, or committee work
            - Collaborations with researchers outside RGU (unless those co-authors appear as graph nodes)
            """
        )

    st.markdown("---")
    st.subheader("How to read this tool")
    st.markdown(
        """
        - **Each dot** in the network is an RGU academic with a known school and job title.
        - **Each line** between two dots means those two academics have at least one jointly published paper.
        - **A thicker or heavier line** means more co-authored papers between that pair.
        - **The sidebar filters** let you focus on a specific school or job title band. Filtering to a
          single school shows only the co-authorship relationships within that school. Filtering by job
          title allows comparison of collaboration patterns across career levels.
        - **All metrics on every page update instantly** when you change the filters.
        """
    )

    st.markdown("---")
    st.subheader("Key questions this tool can help answer")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info(
            "**Head of School**\n\n"
            "\"Who in my school is a critical connector through co-authorship, "
            "and what happens to our research network if they leave?\""
        )
    with c2:
        st.info(
            "**VP Research**\n\n"
            "\"Which schools are publishing interdisciplinarily and which are "
            "co-authoring only internally?\""
        )
    with c3:
        st.info(
            "**Vice Chancellor**\n\n"
            "\"Where are the strategic co-authorship partnership opportunities "
            "across RGU's research portfolio, and where are the gaps?\""
        )

    st.markdown("---")
    st.subheader("Metric Glossary")

    with st.expander("Network Structure"):
        st.markdown(
            """
            - **Degree / Co-authors:** The number of different academics an individual has co-published with.
            - **Weighted degree:** The total number of co-authored papers across all collaborative relationships.
            - **Connected component:** A group of academics linked, directly or indirectly, through co-authorship ties.
            - **Edge weight:** The number of jointly published papers between a specific pair of academics.
            """
        )
    with st.expander("Centrality Metrics"):
        st.markdown(
            """
            - **Degree centrality:** Normalised count of how many different co-authors an academic has.
            - **Betweenness centrality:** How often an academic appears on the shortest co-authorship path between other researchers — a bridge measure.
            - **Eigenvector centrality:** Being connected to other well-connected researchers — network prestige.
            - **HITS score:** Embeddedness in the most interconnected part of the co-authorship network.
            """
        )
    with st.expander("Brokerage & Structural Holes"):
        st.markdown(
            """
            - **Gould-Fernandez brokerage:** Classifies how each academic bridges schools in the co-authorship record — as Coordinator, Consultant, Gatekeeper, Representative, or Liaison.
            - **Burt's Constraint:** Measures how much an academic's co-authors also publish with each other. Low constraint = more diverse co-authorship position.
            """
        )
    with st.expander("Network Culture & Resilience"):
        st.markdown(
            """
            - **School assortativity:** Whether co-authored papers tend to involve academics from the same school (positive) or cross schools (negative/zero).
            - **Rank assortativity:** Whether co-authorship tends to occur between academics at the same career level.
            - **Articulation point:** An academic whose removal from the record would disconnect the co-authorship network.
            - **K-core:** The innermost, most mutually connected layer of the co-authorship network.
            """
        )
    with st.expander("Community & Interdisciplinarity"):
        st.markdown(
            """
            - **Community detection:** Identifies naturally forming co-authorship clusters independent of the formal school structure.
            - **E-I Index:** Measures whether co-authorship tends to cross school boundaries (external) or stay within them (internal). Ranges from −1 (fully insular) to +1 (fully interdisciplinary).
            """
        )


# ============================================================
# Page 2 — Network Overview
# ============================================================
def page_network_overview():
    st.header("Network Overview")
    st.caption(
        "A high-level view of the shape and scale of RGU's co-authorship network "
        "within the current filter selection."
    )

    if not min_nodes_ok(G):
        return

    n_edges = G.number_of_edges()
    n_sch = len({G.nodes[n].get("school", "") for n in G.nodes()})
    wd_all = dict(G.degree(weight="weight"))
    most_collab_id = max(wd_all, key=wd_all.get) if wd_all else None
    if most_collab_id:
        most_collab = G.nodes[most_collab_id].get("label", most_collab_id)
        most_collab_school = G.nodes[most_collab_id].get("school", "")
    else:
        most_collab, most_collab_school = "—", ""

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Academics", n_nodes)
    c2.metric("Total Co-authorship Ties", n_edges)
    c3.metric("Schools Represented", n_sch)
    c4.metric(
        "Most Collaborative Academic",
        most_collab,
        delta=most_collab_school,
        delta_color="off",
    )

    st.info(
        "This map shows the full shape of RGU's co-authorship network — who has published with whom, "
        "and how often. Dense clusters indicate schools or groups where joint publication is a common "
        "working practice. Isolated nodes on the periphery are academics with few or no recorded "
        "co-authorship ties in the dataset; this may reflect a preference for sole-authored work, an "
        "early-career stage, or a research profile not yet captured in the graph. It does not "
        "necessarily mean those individuals are not collaborating — only that the collaboration has "
        "not yet produced a jointly published paper included in this dataset."
    )

    # Large-graph toggle
    is_large = G.number_of_nodes() > 500
    show_full = False
    if is_large:
        show_full = st.checkbox(
            "Show full network (may be slow — currently filtering to top 50% by co-authored papers)",
            value=False,
        )

    # School legend
    st.markdown(
        f"**School colour legend:**  {school_legend_html(school_color_map)}",
        unsafe_allow_html=True,
    )

    physics_on = st.checkbox("Enable physics simulation (uncheck to freeze layout)", value=True)

    with st.spinner("Rendering network — this may take a moment for large graphs…"):
        html = safe_run(
            build_pyvis_network,
            G,
            color_by="school",
            color_map=school_color_map,
            large_threshold=500,
            show_full=show_full,
            label="network visualisation",
        )
    if html:
        # Inject physics toggle
        if not physics_on:
            html = html.replace('"enabled": true', '"enabled": false')
        components.html(html, height=680, scrolling=False)

    st.markdown("---")
    st.subheader("All Academics in Current Selection")
    node_df = get_node_dataframe(G)
    st.dataframe(
        node_df.sort_values("Total Co-authored Papers", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Technical notes"):
        st.markdown(
            """
            - The network is undirected. Every edge represents a mutual co-authorship — lines have no directionality.
            - Node size encodes weighted degree (total co-authored papers across all co-authors).
            - Edge thickness encodes edge weight (number of co-authored papers between that specific pair).
            - PyVis is initialised with `directed=False` to ensure no arrows are rendered.
            - For graphs exceeding 500 nodes, only the top 50% by weighted degree are shown by default to maintain browser performance.
            """
        )


# ============================================================
# Page 3 — Centrality Analysis
# ============================================================
def page_centrality():
    st.header("Centrality Analysis")
    st.caption(
        "Four complementary measures of each academic's position and influence "
        "within the co-authorship network."
    )

    if not min_nodes_ok(G):
        return

    nd, ed = graph_to_cache_args(G)

    # --- Degree centrality ---
    st.subheader("1. Degree Centrality — Breadth of co-authorship")
    st.info(
        "Degree centrality counts how many different colleagues an academic has co-authored with, "
        "expressed as a proportion of all possible co-authors in the network. A high score means this "
        "person has published jointly with a wide range of colleagues across RGU. This is a measure of "
        "breadth of co-authorship — how many different people this academic has worked with, regardless "
        "of how many papers each relationship has produced."
    )
    dc = safe_run(compute_degree_centrality, nd, ed, label="degree centrality")
    if dc:
        dc_df = centrality_to_df(dc, G, "Degree Centrality")
        st.dataframe(dc_df.head(20), use_container_width=True)
        with st.expander("Full ranked list"):
            st.dataframe(dc_df, use_container_width=True)
        st.plotly_chart(
            plot_centrality_bar(dc_df.reset_index(), "Degree Centrality", "Top 20 — Degree Centrality"),
            use_container_width=True,
        )

    # --- Weighted degree ---
    st.subheader("2. Weighted Degree (Strength) — Volume of collaborative output")
    st.info(
        "Weighted degree measures the total number of co-authored papers an academic has produced across "
        "all their collaborative relationships. Where degree centrality counts the number of distinct "
        "co-authors, weighted degree captures the volume of collaborative output. An academic with a high "
        "weighted degree has not only worked with many colleagues but has sustained those relationships "
        "across multiple publications. This is the closest measure in this dataset to raw collaborative "
        "research productivity."
    )
    wd = safe_run(compute_weighted_degree, nd, ed, label="weighted degree")
    if wd:
        wd_df = centrality_to_df(wd, G, "Weighted Degree")
        st.dataframe(wd_df.head(20), use_container_width=True)
        with st.expander("Full ranked list"):
            st.dataframe(wd_df, use_container_width=True)
        st.plotly_chart(
            plot_centrality_bar(wd_df.reset_index(), "Weighted Degree", "Top 20 — Weighted Degree (Total Co-authored Papers)"),
            use_container_width=True,
        )

    # --- Betweenness centrality ---
    st.subheader("3. Betweenness Centrality — Bridge importance")
    st.info(
        "Betweenness centrality identifies the academics who appear most often on the shortest "
        "co-authorship paths between other pairs of researchers. An academic with high betweenness is a "
        "bridge: many other researchers are connected to the rest of the network through this person's "
        "publication record. If this academic were to leave RGU or cease publishing collaboratively, the "
        "paths connecting many other researchers would lengthen or disappear entirely. These individuals "
        "are the connective tissue of RGU's co-authorship network and represent key-person risks from a "
        "research continuity perspective."
    )
    st.markdown(
        "_Note: Betweenness centrality on an undirected graph counts each shortest path once, "
        "treating co-authorship links as bidirectional (which they are — a jointly published paper "
        "credits all authors equally)._"
    )
    bc = safe_run(compute_betweenness_centrality, nd, ed, label="betweenness centrality")
    if bc:
        bc_df = centrality_to_df(bc, G, "Betweenness Centrality")
        st.dataframe(bc_df.head(20), use_container_width=True)
        with st.expander("Full ranked list"):
            st.dataframe(bc_df, use_container_width=True)
        st.plotly_chart(
            plot_centrality_bar(bc_df.reset_index(), "Betweenness Centrality", "Top 20 — Betweenness Centrality"),
            use_container_width=True,
        )

    # --- Eigenvector centrality ---
    st.subheader("4. Eigenvector Centrality — Research network prestige")
    st.info(
        "Eigenvector centrality measures not just how many co-authors an academic has, but how "
        "well-connected and productive those co-authors are. Being linked by co-authorship to other "
        "highly connected researchers amplifies your score. An academic with a high eigenvector score "
        "is embedded in the most active and mutually interconnected part of RGU's co-authorship network "
        "— they publish with people who themselves publish widely. This is a measure of research network prestige."
    )
    ec = safe_run(compute_eigenvector_centrality, nd, ed, label="eigenvector centrality")
    if ec:
        ec_df = centrality_to_df(ec, G, "Eigenvector Centrality")
        st.dataframe(ec_df.head(20), use_container_width=True)
        with st.expander("Full ranked list"):
            st.dataframe(ec_df, use_container_width=True)
        st.plotly_chart(
            plot_centrality_bar(ec_df.reset_index(), "Eigenvector Centrality", "Top 20 — Eigenvector Centrality"),
            use_container_width=True,
        )

    # --- Combined scatter ---
    if dc and bc and wd:
        st.subheader("Co-authorship Influence vs. Connectivity Map")
        st.info(
            "Academics in the top-right of this chart are both widely collaborative and structurally "
            "important to the network's connectivity. Those with high betweenness but low degree are "
            "narrow bridges — they connect specific communities through a small number of key "
            "relationships, making their structural position fragile. A single disruption to their "
            "co-authorship activity could disconnect parts of the network."
        )
        combo = pd.DataFrame({
            "Name": list(dc.keys()),
            "Degree Centrality": [dc[n] for n in dc],
            "Betweenness Centrality": [bc.get(n, 0) for n in dc],
            "Weighted Degree": [wd.get(n, 0) for n in dc],
            "School": [G.nodes[n].get("school", "") for n in dc],
            "Job Title": [G.nodes[n].get("job_title", "") for n in dc],
        })
        st.plotly_chart(
            plot_scatter(
                combo,
                x_col="Degree Centrality",
                y_col="Betweenness Centrality",
                size_col="Weighted Degree",
                color_col="School",
                title="Co-authorship Influence vs. Connectivity Map",
            ),
            use_container_width=True,
        )

    with st.expander("Technical notes"):
        st.markdown(
            """
            - All centrality measures use undirected NetworkX functions only.
            - `nx.degree_centrality(G)` — normalises by (n-1).
            - `G.degree(weight='weight')` — sum of edge weights per node.
            - `nx.betweenness_centrality(G, weight='weight', normalized=True)` — shortest paths weighted by inverse co-authorship volume.
            - `nx.eigenvector_centrality(G, weight='weight', max_iter=1000)` — power iteration; falls back to numpy solver on convergence failure.
            - Using `weight` in betweenness causes NetworkX to interpret higher weight as shorter distance. In a co-authorship graph this means frequently co-authored pairs are treated as more directly connected, which is appropriate.
            """
        )


# ============================================================
# Page 4 — Brokerage & Structural Roles
# ============================================================
def page_brokerage():
    st.header("Brokerage & Structural Roles (Gould-Fernandez)")
    st.caption(
        "Classifies each academic's role as a bridge between schools in the co-authorship record."
    )

    if not min_nodes_ok(G):
        return

    nd, ed = graph_to_cache_args(G)
    st.info(
        "These brokerage roles describe how each academic functions as a bridge within RGU's "
        "co-authorship network. The roles are defined purely by co-publication patterns — an academic "
        "identified as a Gatekeeper controls external co-authorship access to their school only in the "
        "sense that the published record shows all cross-school papers passing through them. This may "
        "reflect deliberate coordination or simply the natural result of that individual's research "
        "profile spanning multiple fields."
    )

    with st.expander("Role definitions — expand for leadership explanations"):
        st.markdown(
            """
            - **Coordinator:** Co-authors within the same school as a bridge between two colleagues who are also in the same school. The glue of a tightly knit publishing cluster within a school.
            - **Consultant:** Brought into co-authorship from outside the school but does not introduce the two groups to each other through subsequent joint publication.
            - **Gatekeeper:** All recorded co-authorship between their school and another school passes through this one academic. If this person stops publishing collaboratively, that inter-school co-authorship link disappears from the record entirely.
            - **Representative:** Takes their school's research outward into another school's publications but does not draw that school into broader collaboration with their home group.
            - **Liaison:** Co-authors with academics from two different schools who do not themselves co-author with each other. This is the highest-value brokerage role — the academic is the sole recorded link between two otherwise disconnected research communities. If they leave, that bridge exists nowhere else in the published record.
            """
        )

    brok_df = safe_run(compute_brokerage, nd, ed, label="brokerage")
    if brok_df is None:
        return

    st.subheader("Individual brokerage scores")
    st.dataframe(brok_df, use_container_width=True, hide_index=True)

    role_cols = ["Coordinator", "Consultant", "Gatekeeper", "Representative", "Liaison"]

    st.subheader("Brokerage role composition — top 20 by total brokerage")
    st.plotly_chart(
        plot_stacked_bar(brok_df, "Name", role_cols, "Brokerage Role Composition — Top 20"),
        use_container_width=True,
    )

    st.subheader("School-level mean brokerage by role")
    school_brok = brok_df.groupby("School")[role_cols].mean().reset_index()
    fig = px.bar(
        school_brok.melt(id_vars="School", value_vars=role_cols, var_name="Role", value_name="Mean Score"),
        x="School",
        y="Mean Score",
        color="Role",
        barmode="group",
        title="Mean Brokerage Score by Role and School",
        color_discrete_sequence=PALETTE,
    )
    fig.update_layout(xaxis_tickangle=-45, height=450)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Technical notes"):
        st.markdown(
            """
            - Implemented manually using the undirected neighbourhood of each node.
            - For each node *i*, all unordered pairs of neighbours (*j*, *k*) are enumerated.
            - Each unordered pair produces two ordered triples — (*j* → *i* → *k*) and (*k* → *i* → *j*) — both classified.
            - Classification follows Gould & Fernandez (1989): group partition = `school` attribute.
            - Scores are normalised by *d(d−1)* where *d* = degree of node *i*, so scores sum to 1.
            - Nodes with degree < 2 receive zero brokerage across all roles.
            """
        )


# ============================================================
# Page 5 — Structural Holes & Innovation Potential
# ============================================================
def page_structural_holes():
    st.header("Structural Holes & Innovation Potential (Burt's Constraint)")
    st.caption(
        "Identifies academics whose co-authorship spans communities that do not otherwise publish together."
    )

    if not min_nodes_ok(G):
        return

    nd, ed = graph_to_cache_args(G)
    st.info(
        "Burt's Constraint is computed from the co-authorship network and measures how much an "
        "academic's co-authors have also co-authored with each other. A **low score** means this "
        "person's published collaborations span groups that do not otherwise publish together — they "
        "are a cross-disciplinary bridge in the recorded literature. A **high score** means the "
        "academic's co-authors form a tight mutual publishing cluster. Both positions have value: "
        "low-constraint academics are positioned to generate novel cross-disciplinary ideas; "
        "high-constraint academics are embedded in productive, trust-based publishing teams. The "
        "strategic question for RGU is whether low constraint is distributed across the institution "
        "or concentrated only at senior levels."
    )

    const_df = safe_run(compute_constraint, nd, ed, label="Burt's constraint")
    if const_df is None:
        return

    st.subheader("Academics with the most structural holes (lowest constraint)")
    st.dataframe(const_df.head(20), use_container_width=True, hide_index=True)
    with st.expander("Full constraint table"):
        st.dataframe(const_df, use_container_width=True, hide_index=True)

    # Box plots
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            plot_boxplot(const_df, "Job Title", "Constraint Score",
                         "Constraint Score by Job Title"),
            use_container_width=True,
        )
    with col2:
        st.plotly_chart(
            plot_boxplot(const_df, "School", "Constraint Score",
                         "Constraint Score by School"),
            use_container_width=True,
        )

    # Scatter
    st.subheader("Constraint vs. Collaborative Output")
    st.plotly_chart(
        plot_scatter(
            const_df,
            x_col="Constraint Score",
            y_col="Weighted Degree",
            size_col="Weighted Degree",
            color_col="School",
            title="Burt's Constraint vs. Weighted Degree",
        ),
        use_container_width=True,
    )

    # Per-school lowest constraint
    st.subheader("Structural hole holders by school")
    st.markdown(
        "These are the academics whose co-authorship record spans the most diverse range of communities "
        "— the published evidence of cross-disciplinary boundary-spanning in each school."
    )
    if not const_df.empty:
        best_per_school = (
            const_df.sort_values("Constraint Score")
            .groupby("School")
            .first()
            .reset_index()[["School", "Name", "Job Title", "Constraint Score"]]
        )
        st.dataframe(best_per_school, use_container_width=True, hide_index=True)

    with st.expander("Technical notes"):
        st.markdown(
            """
            - Computed with `nx.constraint(G, weight='weight')` on the undirected filtered graph.
            - Constraint ranges from near 0 (maximum structural holes) to 1 (fully embedded in a clique).
            - Isolate nodes (degree 0) and nodes with a single neighbour receive a constraint of 1 by convention.
            - Lower is more advantaged in the sense of bridging diverse co-authorship communities.
            """
        )


# ============================================================
# Page 6 — Collaboration Culture (Assortativity)
# ============================================================
def page_assortativity():
    st.header("Collaboration Culture (Assortativity)")
    st.caption(
        "Measures whether co-authorship preferentially occurs within school boundaries "
        "and across career levels."
    )

    if not min_nodes_ok(G):
        return

    nd, ed = graph_to_cache_args(G)
    result = safe_run(compute_assortativity, nd, ed, label="assortativity")
    if result is None:
        return
    school_assort, rank_assort = result

    c1, c2 = st.columns(2)
    with c1:
        st.metric("School Assortativity", f"{school_assort:.4f}" if school_assort is not None else "N/A")
        st.info(
            "This score is calculated from the co-authorship record only. A score close to **+1** "
            "means the vast majority of joint publications involve two academics from the same school "
            "— the published record shows RGU operating as a set of disciplinary silos. A score near "
            "**0** means co-authorship crosses school boundaries as often as it stays within them. A "
            "**negative score** would mean academics are more likely to co-publish with someone from a "
            "different school than their own, indicating an unusually open interdisciplinary publishing culture."
        )
    with c2:
        st.metric("Career-Level (Rank) Assortativity", f"{rank_assort:.4f}" if rank_assort is not None else "N/A")
        st.info(
            "This score measures whether academics tend to co-publish with others at a similar career "
            "level. A **positive score** means Professors predominantly co-author with Professors and "
            "Lecturers with Lecturers — the published record shows a rank-segregated culture. A "
            "**negative score** means joint publications frequently cross career levels, which the "
            "published record suggests a mentoring or cross-rank collaborative culture. Note that this "
            "reflects publication patterns only — informal mentoring and supervision that has not "
            "produced a joint paper is not captured here."
        )

    # Heatmaps
    st.subheader("Co-authorship Volume Between Schools")
    school_mat = safe_run(compute_school_heatmap, nd, ed, label="school heatmap")
    if school_mat is not None and not school_mat.empty:
        st.plotly_chart(
            plot_heatmap(school_mat, "Co-authorship Volume Between Schools"),
            use_container_width=True,
        )
        st.markdown(
            "_A blank or near-blank cell between two schools that share obvious research themes is an "
            "actionable gap in the co-authorship record. It may mean collaboration exists but has not "
            "yet produced joint publications, or that no meaningful research relationship exists at all "
            "— both interpretations warrant a conversation._"
        )

    st.subheader("Co-authorship Volume Between Job Title Bands")
    title_mat = safe_run(compute_title_heatmap, nd, ed, label="title heatmap")
    if title_mat is not None and not title_mat.empty:
        st.plotly_chart(
            plot_heatmap(title_mat, "Co-authorship Volume Between Job Title Bands", colorscale="Oranges"),
            use_container_width=True,
        )

    with st.expander("Technical notes"):
        st.markdown(
            """
            - School assortativity: `nx.attribute_assortativity_coefficient(G, 'school')`. Ranges −1 to +1.
            - Rank assortativity: `nx.numeric_assortativity_coefficient(G, 'seniority_rank')`. Seniority mapping: Professor=4, Reader/Senior Lecturer/Associate Professor=3, Lecturer/Research Fellow=2, Research Assistant/default=1.
            - Both computations treat the graph as undirected — each edge is counted once.
            - Heatmap cells show total co-authored papers (sum of edge weights) between academics in each school or title band. Diagonal = intra-group co-authorship.
            """
        )


# ============================================================
# Page 7 — Network Resilience & Key-Person Risk
# ============================================================
def page_resilience():
    st.header("Network Resilience & Key-Person Risk")
    st.caption(
        "Identifies academics and structural features whose loss would fragment "
        "the co-authorship record."
    )

    if not min_nodes_ok(G):
        return

    nd, ed = graph_to_cache_args(G)

    st.markdown(
        "_All resilience analysis is based on the undirected co-authorship graph. Fragmentation here "
        "means fragmentation of the **recorded co-authorship network** — it does not imply that "
        "professional relationships between academics would break down, only that the published "
        "collaborative record would become disconnected._"
    )

    # --- Articulation points ---
    st.subheader("Articulation Points — Single Points of Co-authorship Failure")
    st.warning(
        "Articulation points in the co-authorship network are academics whose published collaboration "
        "record is the sole recorded link between two or more groups. If they were to leave RGU, "
        "retire, or cease collaborative publishing, those groups would have no remaining direct or "
        "indirect co-authorship path between them in the published record. These academics should be "
        "considered for succession planning, and their schools should be encouraged to build additional "
        "co-authorship bridges so that institutional connectivity does not rest on a single individual's "
        "publication activity."
    )
    ap_df = safe_run(compute_articulation_points, nd, ed, label="articulation points")
    if ap_df is not None:
        if ap_df.empty:
            st.success("No articulation points found in the current selection — the co-authorship network has no single points of failure.")
        else:
            st.dataframe(ap_df, use_container_width=True, hide_index=True)

    # --- K-core ---
    st.subheader("K-Core Decomposition — The Resilient Core")
    st.info(
        "K-core decomposition peels the co-authorship network layer by layer, removing at each step "
        "any academic with fewer than *k* co-authors in the remaining graph. The innermost core — "
        "those with the highest k-core number — are academics who are so mutually interconnected "
        "through joint publication that the network around them is highly stable. This is RGU's most "
        "resilient co-authorship cluster. Academics in the outermost shells (k-core = 1) have only a "
        "single recorded co-authorship tie and sit at the periphery of the published collaboration network."
    )
    kcore_df = safe_run(compute_kcore, nd, ed, label="k-core")
    if kcore_df is not None and not kcore_df.empty:
        fig_hist = px.histogram(
            kcore_df,
            x="K-Core Number",
            nbins=kcore_df["K-Core Number"].nunique(),
            title="Distribution of K-Core Shell Numbers",
            labels={"K-Core Number": "K-Core Shell", "count": "Number of Academics"},
        )
        fig_hist.update_layout(height=350)
        st.plotly_chart(fig_hist, use_container_width=True)

        st.subheader("Top 20 — Deepest K-Core Members (most resilient)")
        st.dataframe(kcore_df.head(20), use_container_width=True, hide_index=True)
        with st.expander("Full k-core table"):
            st.dataframe(kcore_df, use_container_width=True, hide_index=True)

    # --- School fragmentation ---
    st.subheader("School-Level Co-authorship Fragmentation")
    frag_df = safe_run(compute_school_fragmentation, nd, ed, label="school fragmentation")
    if frag_df is not None and not frag_df.empty:
        # Highlight high-risk schools
        def style_risk(row):
            if row["High Risk"]:
                return ["background-color: #ffe4e1"] * len(row)
            return [""] * len(row)

        display_df = frag_df.drop(columns=["High Risk"])
        st.dataframe(
            frag_df.style.apply(style_risk, axis=1),
            use_container_width=True,
            hide_index=True,
        )
        high_risk = frag_df[frag_df["High Risk"]]
        if not high_risk.empty:
            st.warning(
                f"Schools highlighted in red have internal articulation points — a single academic's "
                f"departure would fragment their school's own co-authorship sub-network: "
                f"**{', '.join(high_risk['School'].tolist())}**"
            )

    with st.expander("Technical notes"):
        st.markdown(
            """
            - Articulation points: `nx.articulation_points(G)` — operates on undirected graphs only.
            - K-core: `nx.core_number(G)` — iterative peeling of nodes with degree < *k*.
            - School fragmentation: each school's induced subgraph (edges only between school members) is analysed independently for articulation points and connected component size.
            - These measures reflect the co-authorship record only, not organisational resilience more broadly.
            """
        )


# ============================================================
# Page 8 — Community Detection & Research Tribes
# ============================================================
def page_communities():
    st.header("Community Detection & Research Tribes")
    st.caption(
        "Reveals the organic co-authorship clusters that have formed independently "
        "of RGU's formal school structure."
    )

    if not min_nodes_ok(G):
        return

    nd, ed = graph_to_cache_args(G)
    result = safe_run(compute_communities, nd, ed, label="community detection")
    if result is None:
        return
    partition, modularity, method = result

    n_comms = len(set(partition.values()))
    c1, c2, c3 = st.columns(3)
    c1.metric("Detected Communities", n_comms)
    c2.metric("Modularity Score", f"{modularity:.4f}" if modularity is not None else "N/A")
    c3.metric("Detection Method", method)

    st.info(
        "Community detection reveals the research publishing tribes that have formed organically "
        "through joint publication — independent of RGU's formal school structure. When a detected "
        "community spans multiple schools, it means academics from those schools have built a genuine "
        "co-authorship cluster through their published work. This may represent an emerging "
        "interdisciplinary research group that could benefit from formal recognition, a dedicated "
        "research institute designation, or targeted funding. When a detected community maps perfectly "
        "onto a single school, it reflects a coherent internal publishing culture — which may be a "
        "strength or, depending on strategic context, a missed opportunity for external co-authorship."
    )

    # Colour toggle
    colour_mode = st.radio(
        "Node colour represents:",
        ["Detected community", "Formal school"],
        horizontal=True,
    )
    show_full = False
    if G.number_of_nodes() > 500:
        show_full = st.checkbox("Show full network (may be slow)", value=False)

    with st.spinner("Rendering community network…"):
        html = safe_run(
            build_pyvis_network,
            G,
            color_by="community" if colour_mode == "Detected community" else "school",
            color_map=school_color_map,
            partition=partition,
            large_threshold=500,
            show_full=show_full,
            label="community network",
        )
    if html:
        components.html(html, height=680, scrolling=False)

    # Community summary table
    st.subheader("Community Composition")
    part_tuple = tuple(sorted(partition.items()))
    comm_df = safe_run(compute_community_summary, nd, ed, part_tuple, label="community summary")
    if comm_df is not None:
        st.dataframe(comm_df, use_container_width=True, hide_index=True)

    # Member roles
    st.subheader("Academic Roles within Communities")
    roles = classify_community_roles(G, partition)
    role_df = pd.DataFrame([
        {
            "Name": G.nodes[n].get("label", n),
            "School": G.nodes[n].get("school", ""),
            "Job Title": G.nodes[n].get("job_title", ""),
            "Community": partition.get(n, "—"),
            "Role": roles.get(n, "Member"),
        }
        for n in G.nodes()
    ]).sort_values(["Community", "Role"])
    st.dataframe(role_df, use_container_width=True, hide_index=True)

    with st.expander("Technical notes"):
        st.markdown(
            """
            - Primary method: `community.best_partition(G, weight='weight')` (python-louvain).
            - Fallback: `nx.community.greedy_modularity_communities(G, weight='weight')`.
            - Modularity measures how much more densely connected community members are to each other than to the rest of the network. Values > 0.3 indicate meaningful community structure.
            - Hub: top-20% internal degree within community. Bridge: edges to ≥ 2 other communities. Peripheral: bottom-20% internal degree with no cross-community ties.
            - Communities are stochastic for Louvain — results may vary slightly between runs.
            """
        )


# ============================================================
# Page 9 — E-I Index (Interdepartmental Synergy)
# ============================================================
def page_ei_index():
    st.header("Interdepartmental Synergy (E-I Index)")
    st.caption(
        "Quantifies how much co-authorship crosses school boundaries versus staying within them."
    )

    if not min_nodes_ok(G):
        return

    nd, ed = graph_to_cache_args(G)
    result = safe_run(compute_ei_index, nd, ed, label="E-I index")
    if result is None:
        return
    ei_global, school_df, title_df = result

    st.metric("Institution-Wide E-I Index", f"{ei_global:.4f}")
    if ei_global > 0.2:
        st.success("The co-authorship record shows an interdisciplinary publishing culture overall.")
    elif ei_global < -0.2:
        st.warning("The co-authorship record shows a predominantly within-school publishing pattern overall.")
    else:
        st.info("The institution-wide co-authorship record is broadly balanced between intra- and inter-school publishing.")

    st.markdown(
        "The E-I Index ranges from **−1** (all co-authorships within school) to **+1** (all "
        "co-authorships across school boundaries). A score of 0 means external and internal "
        "co-authorships are equally common."
    )

    st.subheader("E-I Index by School")
    st.dataframe(school_df, use_container_width=True, hide_index=True)

    if not school_df.empty:
        st.plotly_chart(
            plot_grouped_bar(
                school_df,
                x_col="School",
                value_cols=["Internal Co-authorships", "External Co-authorships"],
                title="Internal vs. External Co-authorships by School",
            ),
            use_container_width=True,
        )

    st.subheader("E-I Index by Job Title")
    st.info(
        "This table shows whether interdisciplinary co-authorship — publishing with colleagues from "
        "other schools — is evenly distributed across career levels or concentrated among senior "
        "academics. If Professors have a substantially higher E-I score than Lecturers, the "
        "co-authorship record suggests that cross-school publishing is a privilege of seniority. "
        "This has implications for how early-career researchers are introduced to cross-disciplinary "
        "publishing opportunities."
    )
    st.dataframe(title_df, use_container_width=True, hide_index=True)

    with st.expander("Technical notes"):
        st.markdown(
            """
            - E-I Index = (External edges − Internal edges) / (External edges + Internal edges).
            - Each edge is counted once (undirected graph). An edge is *internal* if both endpoints share the same `school` attribute.
            - Per-school calculation: all edges incident to the school's nodes; internal = both endpoints in the school.
            - Per-title calculation: each edge contributes once to each endpoint's job title count.
            """
        )


# ============================================================
# Page 10 — School-Level Collaboration Map
# ============================================================
def page_school_map():
    st.header("School-Level Collaboration Map")
    st.caption(
        "Collapses the full academic graph into a school-to-school view of co-authorship volume."
    )

    if not min_nodes_ok(G):
        return

    nd, ed = graph_to_cache_args(G)
    result = safe_run(compute_school_metagraph, nd, ed, label="school meta-graph")
    if result is None:
        return
    meta_G, school_sizes, pair_df, bc_df = result

    st.info(
        "This map collapses RGU's entire co-authorship record into a school-level view. The thickness "
        "of each line is proportional to the total number of papers co-authored between academics in "
        "those two schools. It is important to note that a thick line means many joint publications "
        "have been recorded — it does not independently verify the strategic importance or quality of "
        "that relationship. A missing line between two schools means no joint papers exist in the "
        "dataset between academics in those schools, which may reflect an absence of collaboration or "
        "simply an absence of co-publication."
    )

    st.plotly_chart(plot_school_network(meta_G, school_sizes), use_container_width=True)

    st.subheader("Ranked School Co-authorship Partnerships")
    st.dataframe(pair_df, use_container_width=True, hide_index=True)

    st.subheader("School Betweenness Centrality in the Co-authorship Meta-Graph")
    st.markdown(
        "A school with high betweenness in the school meta-graph sits on many inter-school "
        "co-authorship paths — it is a connector between otherwise separate parts of RGU's "
        "inter-school publishing network."
    )
    st.dataframe(bc_df, use_container_width=True, hide_index=True)

    with st.expander("Technical notes"):
        st.markdown(
            """
            - School meta-graph: each school becomes a node. Edge weight = sum of individual co-authorship edge weights between academics in the two schools.
            - Intra-school edges (both endpoints in the same school) are excluded from the meta-graph — only inter-school edges are shown.
            - Node size on the chart encodes the number of academics in that school in the filtered graph.
            - Meta-graph betweenness: `nx.betweenness_centrality(meta_G, weight='weight', normalized=True)`.
            """
        )


# ============================================================
# Page 11 — HITS Analysis
# ============================================================
def page_hits():
    st.header("HITS Analysis — Hubs & Authorities")
    st.caption(
        "Measures embeddedness in the most mutually interconnected part of RGU's co-authorship network."
    )

    if not min_nodes_ok(G):
        return

    nd, ed = graph_to_cache_args(G)

    st.info(
        "In a co-authorship network where all relationships are mutual — because a jointly published "
        "paper credits all authors equally — Hub and Authority scores from the HITS algorithm will be "
        "**identical for every academic**. This is the mathematically correct result for an undirected "
        "graph, not a data error. The scores still provide a useful ranking of the most mutually "
        "well-connected researchers in the network. For a meaningful distinction between Hub and "
        "Authority roles in a co-authorship context, see the interpretation below."
    )

    with st.expander("How to interpret HITS in a co-authorship network"):
        st.markdown(
            """
            In a directed citation network, **Hubs** are papers that cite many important papers, and
            **Authorities** are papers that are cited by many important papers — these roles are distinct.

            In an **undirected co-authorship network**, every relationship is mutual: if Academic A
            co-authors with Academic B, then B also co-authors with A. There is no directionality.
            The HITS score therefore reflects overall embeddedness in the mutually well-connected core
            of the co-authorship network. A high HITS score means this academic co-publishes with other
            academics who are themselves widely and mutually connected.

            Think of it as a measure of **being in the right rooms** — this academic's co-authorship
            record places them at the centre of RGU's most interconnected publishing clusters.

            NetworkX's HITS implementation treats each undirected edge as a pair of directed edges in
            both directions, so hub and authority scores converge to the same value — this is expected
            and mathematically inevitable for a symmetric adjacency matrix.
            """
        )

    result = safe_run(compute_hits, nd, ed, label="HITS")
    if result is None:
        return
    hubs, _ = result  # hub == authority for undirected

    hits_df = centrality_to_df(hubs, G, "HITS Score")
    st.subheader("HITS Ranking")
    st.dataframe(hits_df.head(20), use_container_width=True)
    with st.expander("Full HITS ranked list"):
        st.dataframe(hits_df, use_container_width=True)

    st.plotly_chart(
        plot_centrality_bar(
            hits_df.reset_index(),
            "HITS Score",
            "Top 20 — HITS Score (Co-authorship Network Embeddedness)",
        ),
        use_container_width=True,
    )

    # School-level mean HITS
    school_hits = (
        hits_df.groupby("School")["HITS Score"]
        .mean()
        .reset_index()
        .sort_values("HITS Score", ascending=False)
    )
    fig_sch = px.bar(
        school_hits,
        x="School",
        y="HITS Score",
        title="Mean HITS Score by School",
        color="School",
        color_discrete_sequence=PALETTE,
    )
    fig_sch.update_layout(xaxis_tickangle=-45, height=400, showlegend=False)
    st.plotly_chart(fig_sch, use_container_width=True)

    st.subheader("HITS Score Distribution by School")
    st.plotly_chart(
        plot_boxplot(hits_df, "School", "HITS Score", "HITS Score Distribution by School"),
        use_container_width=True,
    )

    with st.expander("Technical notes"):
        st.markdown(
            """
            - Computed with `nx.hits(G, max_iter=1000, normalized=True)` on the undirected graph.
            - For an undirected graph the adjacency matrix is symmetric, so the HITS power iteration converges to identical hub and authority vectors. Only one score is displayed.
            - This is mathematically expected behaviour, not a bug.
            - The score is related to eigenvector centrality in the undirected case and similarly rewards connection to well-connected nodes.
            """
        )


# ============================================================
# Page 12 — Download & Export
# ============================================================
def page_download():
    st.header("Download & Export")
    st.caption("Export the current filtered graph and computed metrics as CSV files.")

    if not min_nodes_ok(G, minimum=1):
        return

    nd, ed = graph_to_cache_args(G)

    # Node list
    node_df = get_node_dataframe(G)
    st.subheader("Node List")
    st.dataframe(node_df, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇ Download Node List (CSV)",
        data=node_df.to_csv(index=False).encode(),
        file_name="rgu_nodes.csv",
        mime="text/csv",
    )

    # Edge list
    edge_rows = [
        {
            "Source": u,
            "Target": v,
            "Source School": G.nodes[u].get("school", ""),
            "Target School": G.nodes[v].get("school", ""),
            "Co-authored Papers (Weight)": d.get("weight", 1),
        }
        for u, v, d in G.edges(data=True)
    ]
    edge_df = pd.DataFrame(edge_rows)
    st.subheader("Edge List")
    st.dataframe(edge_df, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇ Download Edge List (CSV)",
        data=edge_df.to_csv(index=False).encode(),
        file_name="rgu_edges.csv",
        mime="text/csv",
    )

    # Centrality metrics
    st.subheader("Centrality Metrics")
    with st.spinner("Computing centrality metrics for export…"):
        dc = safe_run(compute_degree_centrality, nd, ed, label="degree centrality")
        wd = safe_run(compute_weighted_degree, nd, ed, label="weighted degree")
        bc = safe_run(compute_betweenness_centrality, nd, ed, label="betweenness centrality")
        ec = safe_run(compute_eigenvector_centrality, nd, ed, label="eigenvector centrality")
        hits_res = safe_run(compute_hits, nd, ed, label="HITS")

    if all(x is not None for x in [dc, wd, bc, ec, hits_res]):
        hubs, _ = hits_res
        cent_rows = []
        for n in G.nodes():
            d = G.nodes[n]
            cent_rows.append({
                "Name": d.get("label", n),
                "School": d.get("school", ""),
                "Job Title": d.get("job_title", ""),
                "Degree Centrality": round(dc.get(n, 0), 6),
                "Weighted Degree": wd.get(n, 0),
                "Betweenness Centrality": round(bc.get(n, 0), 6),
                "Eigenvector Centrality": round(ec.get(n, 0), 6),
                "HITS Score": round(hubs.get(n, 0), 6),
            })
        cent_df = pd.DataFrame(cent_rows).sort_values("Weighted Degree", ascending=False)
        st.dataframe(cent_df, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇ Download Centrality Metrics (CSV)",
            data=cent_df.to_csv(index=False).encode(),
            file_name="rgu_centrality.csv",
            mime="text/csv",
        )

    # Community assignments
    st.subheader("Community Assignments")
    with st.spinner("Computing community assignments…"):
        comm_result = safe_run(compute_communities, nd, ed, label="communities")
    if comm_result:
        partition, modularity, method = comm_result
        comm_df = pd.DataFrame([
            {
                "Name": G.nodes[n].get("label", n),
                "School": G.nodes[n].get("school", ""),
                "Job Title": G.nodes[n].get("job_title", ""),
                "Community": partition.get(n, -1),
            }
            for n in G.nodes()
        ]).sort_values("Community")
        st.dataframe(comm_df, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇ Download Community Assignments (CSV)",
            data=comm_df.to_csv(index=False).encode(),
            file_name="rgu_communities.csv",
            mime="text/csv",
        )


# ============================================================
# Page 11 — School Bridges & Key Connectors
# ============================================================
def page_school_bridges():
    st.header("School Bridges & Key Connectors")
    st.caption(
        "Identifying the academics whose co-authorship record is the primary link between each pair "
        "of schools — and quantifying the institutional risk if that link were lost."
    )

    # Guard: need ≥ 2 schools
    present_schools = {G.nodes[n].get("school", "") for n in G.nodes() if G.nodes[n].get("school")}
    if len(present_schools) < 2:
        st.warning(
            "Fewer than two schools are present in the current filter selection. "
            "Bridge analysis requires at least two schools. Please broaden your filters."
        )
        return

    if not min_nodes_ok(G):
        return

    nd, ed = graph_to_cache_args(G)

    st.info(
        "Every line connecting two schools in RGU's co-authorship network passes through individual "
        "academics. This page identifies, for each pair of connected schools, the single person whose "
        "publication record contributes most to that inter-school link — and measures how much of that "
        "connection depends on them alone. A Bridge Strength of 100% means every co-authored paper "
        "between those two schools involves this one individual. If they were to leave RGU or reduce "
        "their collaborative activity, that inter-school co-authorship connection would disappear "
        "entirely from the published record. These findings are not a criticism of those individuals "
        "— they are a structural signal that RGU needs to build additional co-authorship bridges "
        "alongside the ones that already exist."
    )

    # Obtain betweenness from the existing cache — do not recompute
    bc = safe_run(compute_betweenness_centrality, nd, ed, label="betweenness centrality")
    if bc is None:
        return
    betweenness_tuple = tuple(sorted(bc.items()))

    bridge_df = safe_run(
        compute_school_bridges, nd, ed, betweenness_tuple, label="school bridges"
    )
    if bridge_df is None or bridge_df.empty:
        st.info("No inter-school co-authorship connections were found for the current filter selection.")
        return

    # ---- Key metric cards ----
    n_pairs = len(bridge_df)
    n_fragile = int(bridge_df["fragility_flag"].sum())
    unique_bridges = bridge_df["bridge_academic"].nunique()

    bridge_counts = bridge_df["bridge_academic"].value_counts()
    top_bridge_name = bridge_counts.index[0]
    top_bridge_count = int(bridge_counts.iloc[0])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("School pairs with co-authorship", n_pairs)
    c2.metric("High-fragility bridges", n_fragile)
    c3.metric("Academics acting as school bridges", unique_bridges)
    c4.metric(
        "Most frequent bridge",
        top_bridge_name,
        delta=f"{top_bridge_count} school-pair connection{'s' if top_bridge_count != 1 else ''}",
        delta_color="off",
    )

    # ---- Bridge network visualisation ----
    st.subheader("Bridge Network")
    st.markdown(
        "**Colour key:** "
        '<span style="color:#d62728;font-weight:bold">■ Red</span> = high fragility (&gt;50% of cross-school papers through one person)&nbsp;&nbsp;'
        '<span style="color:#ff7f0e;font-weight:bold">■ Amber</span> = moderate (30–50%)&nbsp;&nbsp;'
        '<span style="color:#2ca02c;font-weight:bold">■ Green</span> = distributed (&lt;30%)',
        unsafe_allow_html=True,
    )

    meta_result = safe_run(compute_school_metagraph, nd, ed, label="school meta-graph")
    if meta_result:
        meta_G, school_sizes, _, _ = meta_result
        fig_bridge = safe_run(
            plot_school_bridge_network, bridge_df, meta_G, label="bridge network chart"
        )
        if fig_bridge:
            st.plotly_chart(fig_bridge, use_container_width=True)

    # ---- Fragility alert ----
    fragile_rows = bridge_df[bridge_df["fragility_flag"]]
    if not fragile_rows.empty:
        bullets = []
        for _, row in fragile_rows.iterrows():
            pct = round(row["bridge_strength"] * 100, 1)
            bullets.append(
                f"**{row['school_a']} ↔ {row['school_b']}:** "
                f"{row['bridge_academic']} ({row['bridge_job_title']}) accounts for "
                f"**{pct}%** of all co-authored papers between these schools "
                f"({row['bridge_papers']} of {row['total_cross_papers']} papers)."
            )
        st.warning(
            "**High-fragility bridges detected** — the following inter-school co-authorship "
            "connections are dependent on a single individual:\n\n" + "\n\n".join(f"- {b}" for b in bullets)
        )
    else:
        st.success(
            "No single-point-of-failure bridges detected in the current filter selection."
        )

    # ---- Full bridge table ----
    st.subheader("Full Bridge Table")

    display_df = bridge_df.rename(columns={
        "school_a": "School A",
        "school_b": "School B",
        "bridge_academic": "Bridge Academic",
        "bridge_job_title": "Job Title",
        "bridge_school": "Bridge School",
        "betweenness_score": "Betweenness Score",
        "bridge_strength": "Bridge Strength",
        "total_cross_papers": "Total Cross-School Papers",
        "bridge_papers": "Bridge Papers",
        "num_candidates": "Candidates Considered",
        "fragility_flag": "High Fragility",
    }).copy()

    # Format bridge strength as percentage
    display_df["Bridge Strength"] = display_df["Bridge Strength"].apply(
        lambda x: f"{x * 100:.1f}%"
    )

    def _highlight_fragile(row):
        return (
            ["background-color: #ffe4e1"] * len(row)
            if row["High Fragility"]
            else [""] * len(row)
        )

    st.dataframe(
        display_df.style.apply(_highlight_fragile, axis=1),
        use_container_width=True,
        hide_index=True,
    )

    # ---- Per-school bridge profile ----
    st.subheader("Per-School Bridge Profile")
    all_schools_in_bridges = sorted(
        set(bridge_df["school_a"].tolist() + bridge_df["school_b"].tolist())
    )
    selected_school = st.selectbox("Explore bridges for a specific school:", all_schools_in_bridges)

    school_mask = (
        (bridge_df["school_a"] == selected_school) |
        (bridge_df["school_b"] == selected_school)
    )
    school_bridge_df = bridge_df[school_mask].copy()

    n_connections = len(school_bridge_df)
    n_fragile_school = int(school_bridge_df["fragility_flag"].sum())

    # Most frequent bridge academic representing this school (bridge_school == selected_school)
    rep_mask = school_bridge_df["bridge_school"] == selected_school
    if rep_mask.any():
        rep_counts = school_bridge_df.loc[rep_mask, "bridge_academic"].value_counts()
        most_freq_rep = rep_counts.index[0]
        most_freq_rep_jt = school_bridge_df.loc[
            school_bridge_df["bridge_academic"] == most_freq_rep, "bridge_job_title"
        ].iloc[0]
        # Count across ALL bridge_df rows (global)
        global_count = int((bridge_df["bridge_academic"] == most_freq_rep).sum())
        rep_summary = (
            f"The most frequent bridge academic representing **{selected_school}** is "
            f"**{most_freq_rep}** ({most_freq_rep_jt}), appearing in "
            f"**{global_count}** school-pair connection{'s' if global_count != 1 else ''} globally."
        )
    else:
        rep_summary = (
            f"No academics from **{selected_school}** are identified as the top bridge "
            "for any school pair in the current selection."
        )

    st.markdown(
        f"**{selected_school}** appears in **{n_connections}** inter-school co-authorship "
        f"connection{'s' if n_connections != 1 else ''}. "
        f"It has **{n_fragile_school}** high-fragility bridge{'s' if n_fragile_school != 1 else ''}. "
        + rep_summary
    )

    school_display_df = school_bridge_df.rename(columns={
        "school_a": "School A",
        "school_b": "School B",
        "bridge_academic": "Bridge Academic",
        "bridge_job_title": "Job Title",
        "bridge_school": "Bridge School",
        "betweenness_score": "Betweenness Score",
        "bridge_strength": "Bridge Strength",
        "total_cross_papers": "Total Cross-School Papers",
        "bridge_papers": "Bridge Papers",
        "num_candidates": "Candidates Considered",
        "fragility_flag": "High Fragility",
    }).copy()
    school_display_df["Bridge Strength"] = school_display_df["Bridge Strength"].apply(
        lambda x: f"{x * 100:.1f}%"
    )
    st.dataframe(
        school_display_df.style.apply(_highlight_fragile, axis=1),
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Technical notes"):
        st.markdown(
            """
            - **Bridge identification:** for each school pair the candidate pool is every academic
              incident to at least one cross-boundary edge between those two schools. Candidates are
              ranked by their global betweenness centrality (pre-computed on the full filtered graph
              using `nx.betweenness_centrality(G, weight='weight', normalized=True)`). The
              highest-ranked candidate is selected as the primary bridge.
            - **Bridge strength:** the share of total cross-boundary edge weight (co-authored papers)
              between the two schools that is incident to the bridge node.
              `bridge_strength = bridge_papers / total_cross_papers`.
              A score of 1.0 means every co-authored paper between those schools involves this person.
            - **Fragility threshold:** the 50% threshold used for `fragility_flag` is a heuristic.
              In school pairs with very few co-authored papers, high bridge strength may reflect a
              naturally small collaboration pool rather than genuine structural dependency. Always
              consider `total_cross_papers` alongside `bridge_strength`.
            - **Tie-breaking:** when multiple candidates share the highest betweenness score, the
              tie is broken alphabetically by display name for determinism.
            - All computations use the undirected graph only — each co-authored paper between a
              pair of academics is represented as a single undirected edge with a weight equal to
              the number of jointly published papers.
            """
        )


# ============================================================
# Router
# ============================================================
ROUTE_MAP = {
    "1. Introduction & How to Use": page_introduction,
    "2. Network Overview": page_network_overview,
    "3. Centrality Analysis": page_centrality,
    "4. Brokerage & Structural Roles": page_brokerage,
    "5. Structural Holes & Innovation": page_structural_holes,
    "6. Collaboration Culture": page_assortativity,
    "7. Network Resilience & Key-Person Risk": page_resilience,
    "8. Community Detection & Research Tribes": page_communities,
    "9. Interdepartmental Synergy (E-I Index)": page_ei_index,
    "10. School-Level Collaboration Map": page_school_map,
    "11. School Bridges & Key Connectors": page_school_bridges,
    "12. HITS Analysis — Hubs & Authorities": page_hits,
    "13. Download & Export": page_download,
}

ROUTE_MAP[page]()
