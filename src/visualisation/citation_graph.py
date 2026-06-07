import pandas as pd
import numpy as np
import plotly.graph_objects as go
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity


def build_similarity_network(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    similarity_threshold: float = 0.50,
    max_edges: int = 200
) -> nx.Graph:
    """
    Builds a network graph where:
    - Each NODE is a paper
    - Each EDGE connects two papers that are semantically similar
    - Edge weight = similarity score

    We use semantic similarity as a proxy for citation relationships
    since we don't have actual citation data from the free APIs.
    """

    G = nx.Graph()

    # Add nodes
    for idx, row in df.iterrows():
        G.add_node(idx,
                   title=row.get("title", "Unknown")[:80],
                   year=str(row.get("year", "?")),
                   source=row.get("source", "?"),
                   cluster=int(row.get("cluster", -1)),
                   abstract_preview=str(row.get("abstract", ""))[:200]
                   )

    # Add edges based on semantic similarity
    sim_matrix = cosine_similarity(embeddings)
    n = len(df)
    edges = []

    for i in range(n):
        for j in range(i + 1, n):
            sim = float(sim_matrix[i][j])
            if sim >= similarity_threshold:
                edges.append((i, j, sim))

    # Sort by similarity and cap at max_edges to keep graph readable
    edges.sort(key=lambda x: x[2], reverse=True)
    edges = edges[:max_edges]

    for i, j, sim in edges:
        G.add_edge(i, j, weight=sim)

    print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def compute_layout(G: nx.Graph) -> dict:
    """
    Computes 2D positions for each node using spring layout.
    Nodes with more connections (more cited/related) end up in the centre.
    """
    if len(G.nodes) == 0:
        return {}

    # spring_layout simulates a physical system where edges are springs
    # and nodes repel each other-densely connected nodes cluster together
    pos = nx.spring_layout(
        G,
        k=2.0,           # optimal distance between nodes
        iterations=100,  # more iterations = more stable layout
        seed=42
    )
    return pos


def render_citation_network(
    G: nx.Graph,
    pos: dict,
    df: pd.DataFrame,
    keywords: dict
) -> go.Figure:
    """
    Renders the network as an interactive Plotly figure.
    - Node size = number of connections (more connected = bigger)
    - Node colour = cluster
    - Hover = paper title + year
    """

    if len(G.nodes) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No network to display", showarrow=False)
        return fig

    # Colour palette for clusters
    colours = [
        "#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
        "#1abc9c", "#e67e22", "#34495e", "#e91e63", "#00bcd4"
    ]

    # ── Edge traces ──────────────────────────────────────────────────────
    edge_x, edge_y = [], []
    for u, v in G.edges():
        if u in pos and v in pos:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=0.5, color="#cccccc"),
        hoverinfo="none",
        showlegend=False
    )

    # ── Node traces-one per cluster for legend ─────────────────────────
    cluster_ids = sorted(set(
        G.nodes[n]["cluster"] for n in G.nodes
    ))

    node_traces = []

    for cid in cluster_ids:
        cluster_nodes = [n for n in G.nodes if G.nodes[n]["cluster"] == cid]
        if not cluster_nodes:
            continue

        node_x, node_y, node_size = [], [], []
        hover_texts = []

        for node in cluster_nodes:
            if node not in pos:
                continue
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

            degree = G.degree(node)
            node_size.append(max(8, min(30, 8 + degree * 3)))

            kws = ", ".join(keywords.get(cid, ["unknown"])[:2])
            hover_texts.append(
                f"<b>{G.nodes[node]['title']}</b><br>"
                f"Year: {G.nodes[node]['year']}<br>"
                f"Source: {G.nodes[node]['source']}<br>"
                f"Cluster: {'Unclustered' if cid == -1 else f'Cluster {cid} ({kws})'}<br>"
                f"Connections: {degree}"
            )

        colour = colours[cid % len(colours)] if cid != -1 else "#aaaaaa"
        label = "Unclustered" if cid == -1 else f"Cluster {cid}"

        trace = go.Scatter(
            x=node_x, y=node_y,
            mode="markers",
            marker=dict(
                size=node_size,
                color=colour,
                line=dict(width=1, color="white")
            ),
            text=hover_texts,
            hoverinfo="text",
            name=label
        )
        node_traces.append(trace)

    # ── Assemble figure ──────────────────────────────────────────────────
    fig = go.Figure(
        data=[edge_trace] + node_traces,
        layout=go.Layout(
            title="Semantic Citation Network-node size = number of connections",
            showlegend=True,
            hovermode="closest",
            height=600,
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font=dict(color="white"),
            legend=dict(
                bgcolor="#1a1a2e",
                bordercolor="#444",
                borderwidth=1
            )
        )
    )

    return fig


def get_network_stats(G: nx.Graph) -> dict:
    """
    Computes key network metrics that reveal structural gaps.

    - High degree nodes = heavily connected papers = core of the field
    - Low degree nodes = isolated papers = potential gap areas
    - Bridges = papers connecting two otherwise separate subtopics
    """

    if len(G.nodes) == 0:
        return {}

    degrees = dict(G.degree())
    avg_degree = sum(degrees.values()) / len(degrees) if degrees else 0

    # Most connected papers-these are the "canonical" papers in the field
    top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]

    # Isolated nodes-papers with no strong similarity to others = gaps
    isolated = [n for n, d in degrees.items() if d == 0]

    # Bridge detection-nodes whose removal disconnects the graph
    bridges = []
    try:
        if nx.is_connected(G):
            bridges = list(nx.bridges(G))
    except Exception:
        pass

    return {
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "avg_connections": round(avg_degree, 2),
        "isolated_papers": len(isolated),
        "bridge_count": len(bridges),
        "top_connected": top_nodes
    }
