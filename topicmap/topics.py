import collections
import json

import hdbscan
import joblib
import matplotlib.pyplot as plt
import numpy as np
import openai
import pandas as pd
import seaborn as sns
import umap
from bertopic import BERTopic
from nltk.corpus import stopwords
from rich.progress import track
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

memory = joblib.Memory(".cache", verbose=0)


def openai_client():
    """Singleton OpenAI client

    We use this to ensure that chatgpt_data_from_titles can be cached.
    """
    if not hasattr(openai_client, "client"):
        with open("apikey.txt") as fh:
            api_key = fh.read().strip()
        client = openai.OpenAI(api_key=api_key)
        openai_client.client = client
    return openai_client.client


prompt = """
You will be provided with the titles of a representative sample of papers from a larger cluster of related scientific papers.
Titles may be in different languages.

Your task is to identify the topic of the entire cluster based on the titles of the representative papers.

Output the following items (in English) that describe the topic of the cluster: 'short_label'
(at most 3 words and format in Title Case), 'long_label' (at most 8 words and format in Title Case),
list of 10 'keywords' (ordered by relevance and format in Title Case), 'summary' (few sentences),
and 'wikipedia_page' (URL).
Do not start short and long labels with the word "The".
Do not use the word 'Cluster' in short or long labels.
Start each summary with "This cluster of papers".
Format the output in JSON.

The titles are:
"""  # noqa: E501


@memory.cache(ignore=["client"])
def chatgpt_data_from_titles(
    titles: str, client: openai.OpenAI | None = None, model="gpt-4o-mini"
) -> dict:
    client = client or openai_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt + titles},
        ],
        response_format={"type": "json_object"},
    )
    data = json.loads(response.choices[0].message.content)
    return {key.lower().replace(" ", "_"): val for key, val in data.items()}


def stop_word_list(languages=None):
    """Make stop word list for specified languages

    default: English, Dutch, French, Spanish, German, Italian

    """
    stop_words = []
    languages = languages or [
        "english",
        "dutch",
        "french",
        "spanish",
        "german",
        "italian",
    ]
    for lang in languages:
        stop_words.extend(stopwords.words(lang))
    return list(set(stop_words))


@memory.cache
def sentence_transformer_embeddings(documents, embedding_model_name):
    model = SentenceTransformer(embedding_model_name)
    return model.encode(documents, show_progress_bar=True)


@memory.cache
def umap_reduce_embeddings(embeddings, n_components=5):
    model = umap.UMAP(
        n_components=n_components,
        n_neighbors=20,
        random_state=42,
        metric="cosine",
        verbose=True,
    )
    return model.fit_transform(embeddings)


# This is used to allow for precalculating/caching UMAP
class Dimensionality:
    """Use this for pre-calculated reduced embeddings"""

    def __init__(self, reduced_embeddings):
        self.reduced_embeddings = reduced_embeddings

    def fit(self, x):  # noqa: ARG002 (unused argument)
        return self

    def transform(self, x):  # noqa: ARG002 (unused argument)
        return self.reduced_embeddings


def load_topic_model(
    documents,
    cluster_name,
    embedding_model_name,
    *,
    stop_words=None,
    from_cache=True,
    min_word_frequency=1,
    min_cluster_size=80,
):
    # Precalculate and cache embeddings
    # Note: I did this on Google Colab, which was about 20x times faster.
    model_cachedir = f".cache/{cluster_name}/{embedding_model_name}"
    embeddings = sentence_transformer_embeddings(documents, embedding_model_name)
    reduced_embeddings = umap_reduce_embeddings(embeddings)

    if from_cache:
        topic_model = BERTopic.load(
            model_cachedir, embedding_model=embedding_model_name
        )
        # Representative docs are not saved, so recalculate those:
        df_docs = topic_model.get_document_info(documents)
        topic_model._save_representative_docs(df_docs.drop("Representation", axis=1))
        return topic_model, embeddings

    # Extract vocab to be used in BERTopic
    stop_words = stop_words or stop_word_list()
    vocab = collections.Counter()
    analyzer = CountVectorizer(stop_words=stop_words).build_analyzer()
    for doc in documents:
        vocab.update(analyzer(doc.lower()))
    vocab = [
        word for word, frequency in vocab.items() if frequency >= min_word_frequency
    ]

    # Prepare sub-models
    embedding_model = SentenceTransformer(embedding_model_name)
    umap_model = Dimensionality(reduced_embeddings)
    hdbscan_model = hdbscan.HDBSCAN(
        min_samples=5,
        gen_min_span_tree=True,
        prediction_data=True,
        min_cluster_size=min_cluster_size,
        cluster_selection_method="leaf",
    )
    vectorizer_model = CountVectorizer(vocabulary=vocab, stop_words=stop_words)

    # Fit BERTopic
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=True,
        verbose=True,
    )
    topics, probs = topic_model.fit_transform(documents, embeddings=embeddings)

    # Reduce outliers
    new_topics = topic_model.reduce_outliers(
        documents, topic_model.topics_, strategy="probabilities", probabilities=probs
    )
    topic_model.update_topics(
        documents,
        topics=new_topics,
        vectorizer_model=vectorizer_model,
    )

    topic_model.save(
        model_cachedir,
        serialization="safetensors",
        save_ctfidf=True,
        save_embedding_model=embedding_model_name,
    )
    return topic_model, embeddings


def topic_coordinates(topic_model, embedding_base="topic_embeddings"):
    """Determine x and y coordinates for topics

    Valid values for `embedding_base` are "topic_embeddings" and "c_tf_idf".
    """
    match embedding_base:
        case "topic_embeddings" if topic_model.topic_embeddings_ is not None:
            embeddings = umap.UMAP(
                n_neighbors=3, n_components=2, metric="cosine", random_state=42
            ).fit_transform(topic_model.topic_embeddings_)
        case "c_tf_idf":
            embeddings = topic_model.c_tf_idf_.toarray()
            embeddings = MinMaxScaler().fit_transform(embeddings)
            embeddings = umap.UMAP(
                n_neighbors=10, n_components=2, metric="hellinger", random_state=42
            ).fit_transform(embeddings)
        case _:
            msg = (
                f"Unsupported value '{embedding_base}' for parameter embedding_base, "
                "or topic model has no calculated topic embeddings."
            )
            raise ValueError(msg)

    res = pd.DataFrame(
        {
            "Topic": list(topic_model.get_topics().keys()),
            "x": embeddings[:, 0],
            "y": embeddings[:, 1],
        }
    )
    # Ensure that map is more wide than high
    return (
        res
        if res.x.max() - res.x.min() >= res.y.max() - res.y.min()
        else res.rename(columns={"x": "y", "y": "x"})
    )


def plot_map(
    df,
    hue,
    filename=None,
    figsize=(16, 9),
    dpi=70,
    **kwargs,
):
    fig, ax = plt.subplots(figsize=figsize)

    plot_kwargs = {"x": "x", "y": "y", "alpha": 0.6, "s": 10}
    plot_kwargs.update(kwargs)
    sns.scatterplot(data=df, hue=hue, ax=ax, **plot_kwargs)
    ax.get_xaxis().set_visible(False)  # noqa: FBT003
    ax.get_yaxis().set_visible(False)  # noqa: FBT003
    plt.tight_layout()

    if filename:
        fig.savefig(filename, dpi=dpi)

    return fig, ax


def plot_map_highlight_clusters(
    df: pd.DataFrame,
    clusters: list[int],
    **kwargs,
):
    df_tmp = df.copy()

    cluster_dict = dict(
        zip(clusters, map(str, range(1, len(clusters) + 1)), strict=True)
    )
    df_tmp["highlight"] = df_tmp.Topic.apply(lambda x: cluster_dict.get(x, "0"))

    palette = sns.color_palette()
    palette[0] = "#aaa"
    return plot_map(
        df_tmp.sort_values(by="highlight"),
        "highlight",
        legend=False,
        palette=palette,
        **kwargs,
    )


def label_topics(topic_model, documents, titles, max_num_titles=100, random_state=42):
    # Link each document title to its topic
    df_titles_all = topic_model.get_document_info(documents).assign(title=titles)

    # Per topic, compile list of titles and label with ChatGPT
    labels_json = {}
    for topic in track(df_titles_all.Topic.unique(), "Getting topic labels..."):
        df_titles_topic = df_titles_all[df_titles_all.Topic == topic]
        # We use max number of titles per topic:
        # the representative ones and a sample of the rest
        titles = list(
            df_titles_topic.loc[df_titles_topic.Representative_document, "title"]
        )
        titles += list(
            df_titles_topic.loc[
                ~df_titles_topic.Representative_document, "title"
            ].sample(
                min(max_num_titles, len(df_titles_topic)) - len(titles),
                random_state=random_state,
            )
        )
        labels_json[topic] = chatgpt_data_from_titles(
            "\n".join(f"- {title}" for title in titles)
        )

    # Transform into dataframe
    json_data = pd.Series(labels_json, name="labels_json").sort_index()
    return pd.json_normalize(json_data)


@memory.cache
def find_closest(topic_model, search_terms: list[str], min_sim: float = 0.3):
    """Find closest topics to list of search terms

    Based on `BERTopic.find_topics`

    """
    # TODO generalize to find topics or documents/projects
    topic_list = list(topic_model.topic_representations_.keys())
    topic_list.sort()

    search_embeddings = []
    for search_term in search_terms:
        search_embeddings.append(
            topic_model._extract_embeddings(
                [search_term], method="word", verbose=False
            ).flatten()
        )
    # Per topic, we define its similarity based on the most similar search term
    sims = cosine_similarity(
        np.array(search_embeddings), topic_model.topic_embeddings_
    ).max(axis=0)

    # Find the number of items with the desired minimum similarity
    top_n = len(sims[sims > min_sim])

    # Extract topics most similar to search_term
    ids = np.argsort(sims)[-top_n:]
    similarity = [sims[i] for i in ids][::-1]
    similar_topics = [topic_list[index] for index in ids][::-1]

    return similar_topics, similarity
