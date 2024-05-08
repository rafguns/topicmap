import altair as alt
import networkx as nx
from sklearn.preprocessing import MinMaxScaler

from .topics import umap_reduce_embeddings


def to_nx_graph(df_topics, topic_model=None, attrs=None):
    graph = nx.Graph()

    # Add nodes with attributes
    if attrs is None:
        attrs = ["x", "y", "count", "description"]
    for _, row in df_topics.iterrows():
        graph.add_node(row["short_label"], **{attr: row[attr] for attr in attrs})

    if topic_model is None:
        # Return graph without edges
        return graph

    # Determine edge weights
    embeddings = topic_model.c_tf_idf_.toarray()
    embeddings = MinMaxScaler().fit_transform(embeddings)
    sorted_topics = df_topics.sort_values(by="topic")["short_label"]
    coocc = embeddings @ embeddings.T

    # Add edges
    edges = ((u, v) for u, v in zip(*coocc.nonzero(), strict=True))
    triples = (
        (sorted_topics[u], sorted_topics[v], {"weight": coocc[u, v]}) for u, v in edges
    )
    graph.add_edges_from(triples)

    return graph


def write_project_map(embeddings, df_proj, df_topics, fname):
    reduced_embeddings_2d = umap_reduce_embeddings(embeddings, n_components=2)
    df_proj = df_proj.assign(
        x=reduced_embeddings_2d[:, 0],
        y=reduced_embeddings_2d[:, 1],
    )
    df_proj = df_proj.merge(
        df_topics[
            ["topic", "count", "representation", "summary", "short_label", "long_label"]
        ],
        on="topic",
        how="left",
    )

    # Get centroids of clusters
    mean_df = (
        df_proj.groupby("topic")
        .agg(
            {
                "x": "mean",
                "y": "mean",
                "short_label": "first",
            }
        )
        .reset_index()
        .sort_values(by="topic")
    )

    dots = (
        alt.Chart(df_proj)
        .mark_point(size=60, opacity=0.4, filled=True)
        .encode(
            x=alt.X("x:Q", axis=None),
            y=alt.Y("y:Q", axis=None),
            color=alt.Color(
                "short_label:N",
                legend=None,
                scale=alt.Scale(scheme="tableau20"),
            ),
            tooltip=["acronym:N", "title:N", "short_label:N", "long_label:N"],
            shape="ssh_project",
            href="url",
        )
    )

    text = (
        alt.Chart(mean_df)
        .mark_text(fontWeight="bold", fontSize=16, opacity=1)
        .encode(
            x=alt.X("x:Q", axis=None),
            y=alt.Y("y:Q", axis=None),
            text="short_label:N",
            color=alt.Color("short_label:N", legend=None),
        )
    )

    chart = (
        ((dots + text).interactive().properties(width=1000, height=700))
        .configure_axis(grid=False)
        .configure_view(strokeWidth=0)
    )
    chart.save(fname)
