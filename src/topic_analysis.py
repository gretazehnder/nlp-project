"""Groups abstracts into topics using BERTopic, to interpret why certain
articles are outliers, and to check whether the
flat temporal trend hides a shift in topic composition over time.

Reuses the same embeddings already computed by model_interface.py, so
BERTopic's own internal embedding step is skipped.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from bertopic import BERTopic


def fit_topics(abstracts: list[str], embeddings: np.ndarray, random_state: int = 42) -> BERTopic:
    """Fits a BERTopic model on the abstracts, using our own precomputed
    embeddings instead of letting BERTopic compute its own.

    random_state is fixed for reproducibility --> BERTopic's underlying
    UMAP step is stochastic by default.

    Uses a CountVectorizer with English stopwords removed for the
    representative-keyword step (c-TF-IDF) --> only affects which words are shown to represent each
    topic, it does not change cluster assignment (that depends on the
    embeddings + UMAP + HDBSCAN, computed independently), so topic
    counts and per-topic alignment scores are unaffected by this fix.
    """
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    # UMAP's randomness is what makes BERTopic non-reproducible by defaults --> fix its random_state explicitly
    umap_model = UMAP(random_state=random_state)
    vectorizer_model = CountVectorizer(stop_words="english")

    topic_model = BERTopic(
        umap_model=umap_model,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=False,
    )
    topic_model.fit_transform(abstracts, embeddings=embeddings)

    return topic_model


def find_articles_by_title_fragment(articles: list, topics: list, topic_model: BERTopic, title_fragments: list[str]) -> None:
    """Prints which topic each article of interest was assigned to, and
    the topic's representative keywords.
    """
    for i, article in enumerate(articles):
        for fragment in title_fragments:
            if fragment.lower() in article.title.lower():
                topic_id = topics[i]
                if topic_id == -1:
                    print(f"[NOISE] {article.title}")
                else:
                    keywords = [word for word, _ in topic_model.get_topic(topic_id)][:6]
                    print(f"[Topic {topic_id}: {', '.join(keywords)}] {article.title}")


def compute_topic_alignment_stats(
    topics: list[int],
    scores: np.ndarray,
    topic_model: BERTopic,
    output_dir: Path,
) -> None:
    """Prints and saves the mean alignment score for each topic,
    compared to the overall corpus mean.
    """
    import pandas as pd

    overall_mean = scores.mean()
    print(f"\nOverall corpus mean alignment score: {overall_mean:.4f}")

    unique_topics = sorted(set(topics))
    rows = []
    print(f"\n{'Topic':<8}{'Count':<8}{'Mean score':<14}{'Diff from overall':<20}Keywords")
    for topic_id in unique_topics:
        indices = [i for i, t in enumerate(topics) if t == topic_id]
        topic_scores = scores[indices]
        topic_mean = topic_scores.mean()
        diff = topic_mean - overall_mean

        if topic_id == -1:
            keywords = "(noise)"
        else:
            keywords = ", ".join(word for word, _ in topic_model.get_topic(topic_id)[:5])

        print(f"{topic_id:<8}{len(indices):<8}{topic_mean:<14.4f}{diff:+.4f}{'':<12}{keywords}")

        rows.append({
            "topic": topic_id,
            "count": len(indices),
            "mean_score": topic_mean,
            "diff_from_overall": diff,
            "keywords": keywords,
        })

    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tables_dir / "topic_summary.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\nSaved full topic summary (all {len(rows)} topics) to {csv_path}")


def check_noise_diagnostics(
    articles: list,
    abstracts: list[str],
    topics: list[int],
    scores: np.ndarray,
) -> None:
    """Checks whether the noise cluster (-1, unclustered articles) is
    mainly made of short abstracts (too little text to share vocabulary
    with a dense cluster) or a heterogeneous group (varied topics, each
    too small/unique to cluster).
    """
    noise_lengths = [len(abstracts[i]) for i, t in enumerate(topics) if t == -1]
    other_lengths = [len(abstracts[i]) for i, t in enumerate(topics) if t != -1]
    noise_scores = scores[[i for i, t in enumerate(topics) if t == -1]]
    other_scores = scores[[i for i, t in enumerate(topics) if t != -1]]

    print(f"\n--- Noise cluster diagnostics ({len(noise_lengths)} articles, {100 * len(noise_lengths) / len(topics):.1f}% of corpus) ---")
    print(f"Abstract length (chars) - noise: mean={np.mean(noise_lengths):.0f}, median={np.median(noise_lengths):.0f}")
    print(f"Abstract length (chars) - rest:  mean={np.mean(other_lengths):.0f}, median={np.median(other_lengths):.0f}")
    print(f"Alignment score - noise: mean={noise_scores.mean():.4f}, std={noise_scores.std():.4f}")
    print(f"Alignment score - rest:  mean={other_scores.mean():.4f}, std={other_scores.std():.4f}")

    length_diff = np.mean(noise_lengths) - np.mean(other_lengths)
    if abs(length_diff) < 100:
        print("-> Noise abstracts are NOT systematically shorter/longer than the rest "
              "(similar mean length) --> consistent with a heterogeneous group of "
              "varied, individually too unique to cluster topics, not a length artifact.")
    else:
        direction = "shorter" if length_diff < 0 else "longer"
        print(f"-> Noise abstracts are on average {direction} than the rest by "
              f"{abs(length_diff):.0f} characters, length may be a contributing factor.")


def save_outlier_topic_assignments(
    articles: list,
    scores: np.ndarray,
    outliers_z: np.ndarray,
    topics: list[int],
    topic_model: BERTopic,
    output_dir: Path,
) -> None:
    """Cross-references the z-score outlier list with topic assignment,
    to check how much of the outlier set is actually explained by the
    two systematically low-scoring topics (2 and 4) versus scattered
    elsewhere.
    """
    import pandas as pd

    outlier_indices = np.where(outliers_z)[0]
    rows = []
    for i in outlier_indices:
        topic_id = topics[i]
        keywords = "(noise)" if topic_id == -1 else ", ".join(
            word for word, _ in topic_model.get_topic(topic_id)[:5]
        )
        rows.append({
            "pmid": articles[i].pmid,
            "year": articles[i].year,
            "score": scores[i],
            "topic": topic_id,
            "topic_keywords": keywords,
            "title": articles[i].title,
        })

    df = pd.DataFrame(rows).sort_values("score")
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tables_dir / "outlier_topic_assignments.csv"
    df.to_csv(csv_path, index=False)

    print(f"\n--- Outlier-topic cross-tabulation ({len(df)} z-score outliers) ---")
    print(df["topic"].value_counts().sort_index().to_string())
    print(f"Saved to {csv_path}")


def plot_all_topics_heatmap(output_dir: Path) -> None:
    """Heatmap of every topic's article frequency across years, to check
    whether topics OTHER than Topic 4 also show meaningful temporal
    change.
    
    Reads topic_over_time.csv and topic_summary.csv, both already saved
    by earlier steps --> no BERTopic recomputation needed.

    The noise topic (-1) is excluded, and each row is
    min-max normalized independently, so a small topic's own temporal
    pattern is visible on its own scale, instead of being washed out
    by comparison to much larger topics.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    tables_dir = output_dir / "tables"
    topics_over_time_df = pd.read_csv(tables_dir / "topic_over_time.csv")
    topic_summary_df = pd.read_csv(tables_dir / "topic_summary.csv")

    pivot = topics_over_time_df.pivot_table(
        index="Topic", columns="Timestamp", values="Frequency", fill_value=0
    )
    pivot = pivot[pivot.index != -1].sort_index()  # drop noise, dominates the scale and isn't a coherent topic

    # row-wise min-max normalization
    row_min = pivot.min(axis=1).values.reshape(-1, 1)
    row_max = pivot.max(axis=1).values.reshape(-1, 1)
    row_range = np.where(row_max - row_min == 0, 1, row_max - row_min)  # avoid divide by zero for flat rows
    normalized = (pivot.values - row_min) / row_range

    # label each row with its top keywords (from topic_summary.csv)
    keywords_by_topic = topic_summary_df.set_index("topic")["keywords"].to_dict()
    row_labels = [
        f"{t}: {keywords_by_topic.get(t, '')[:30]}" for t in pivot.index
    ]

    fig, ax = plt.subplots(figsize=(9, 9))
    im = ax.imshow(normalized, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(int(y)) for y in pivot.columns], rotation=45)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_xlabel("Year")
    ax.set_title("Article frequency per topic, by year\n(each row independently normalized 0-1; noise excluded)")
    fig.colorbar(im, ax=ax, label="Relative frequency within topic (0=min, 1=max)")
    plt.tight_layout()

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig_path = figures_dir / "all_topics_over_time_heatmap.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved heatmap to {fig_path}")


def check_topic_over_time(
    topic_model: BERTopic,
    abstracts: list[str],
    years: list[int],
    output_dir: Path,
    topic_of_interest: int = 4,
) -> None:
    """Uses BERTopic's built-in topics_over_time() to check whether a
    topic's presence changes across years --> specifically testing the
    hand-noticed pattern that the AI/ML topic's example articles were
    both from 2025, not spread evenly across the corpus's time range.

    """
    # BERTopic expects a timestamp per document, aligned with the abstracts list order --> we use publication year as the timestamp
    topics_over_time = topic_model.topics_over_time(
        docs=abstracts,
        timestamps=years,
    )

    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "topic_over_time.csv"
    topics_over_time.to_csv(csv_path, index=False)
    print(f"Saved full topics_over_time table to {csv_path}")

    topic_rows = topics_over_time[topics_over_time["Topic"] == topic_of_interest]
    print(f"\n--- Topic {topic_of_interest} presence over time ---")
    print(topic_rows[["Topic", "Words", "Frequency", "Timestamp"]].to_string(index=False))

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4.5))
    plt.bar(topic_rows["Timestamp"].astype(str), topic_rows["Frequency"])
    plt.xlabel("Year")
    plt.ylabel("Number of articles")
    plt.title(f"Topic {topic_of_interest} (AI/ML methodology) frequency over time")
    plt.tight_layout()
    fig_path = figures_dir / "topic4_frequency_over_time.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved chart to {fig_path}")


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    from data_loader import load_articles, filter_research_articles, filter_by_year_range, load_aims_scope
    from preprocessing import clean_text
    from model_interface import EmbeddingModel
    from evaluation import compute_alignment_scores, detect_outliers

    print("Loading data...")
    articles = load_articles("data/raw/articles.csv")
    articles = filter_research_articles(articles)
    articles = filter_by_year_range(articles, min_year=2015, max_year=2025)
    aims_scope_raw = load_aims_scope("data/raw/aims_scope.txt")
    abstracts = [clean_text(a.abstract) for a in articles]
    aims_scope = clean_text(aims_scope_raw)

    print(f"Computing embeddings for {len(abstracts)} abstracts (reused for BERTopic)...")
    model = EmbeddingModel()
    embeddings = model.encode(abstracts)
    aims_scope_embedding = model.encode_one(aims_scope)
    scores = compute_alignment_scores(embeddings, aims_scope_embedding)
    outliers_z = detect_outliers(scores)  

    print("Fitting BERTopic...")
    topic_model = fit_topics(abstracts, embeddings)

    print("\n--- Topic overview (top 15 by size) ---")
    print(topic_model.get_topic_info().head(15).to_string())

    print("\n--- Per-topic alignment score comparison ---")
    results_dir = PROJECT_ROOT / "data" / "results"
    compute_topic_alignment_stats(topic_model.topics_, scores, topic_model, results_dir)

    check_noise_diagnostics(articles, abstracts, topic_model.topics_, scores)

    print("\n--- Checking the 3 previously noticed low-scoring articles ---")
    titles_of_interest = [
        "Performance and Reproducibility of Large Language Models in Named Entity Recognition",
        "Narrative Search Engine for Case Series Assessment Supported by Artificial Intelligence",
        "Suspected Adverse Effects After Human Papillomavirus Vaccination",
    ]
    find_articles_by_title_fragment(articles, topic_model.topics_, topic_model, titles_of_interest)

    # checks how many of the outliers are explained by the two systematically low topics (2 and 4)
    save_outlier_topic_assignments(articles, scores, outliers_z, topic_model.topics_, topic_model, results_dir)

    print("\n--- Checking if Topic 4 (AI/ML) is concentrated in recent years ---")
    years = [a.year for a in articles]
    check_topic_over_time(topic_model, abstracts, years, results_dir, topic_of_interest=4)

    print("\n--- Checking if OTHER topics also show temporal change (not just Topic 4) ---")
    plot_all_topics_heatmap(results_dir)