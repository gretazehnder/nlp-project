# Analyzing Thematic Alignment in Drug Safety

NLP course project analyzing how closely article abstracts published in
*Drug Safety* between 2015 and 2025 align with the journal's stated
Aims & Scope, using sentence embeddings, cosine similarity, and topic modeling.


## Research Question

To what extent are articles published in *Drug Safety* thematically aligned
with the journal's stated Aims & Scope, and how has this alignment evolved
between 2015 and 2025?



## Repository Structure

```text
nlp-project/
├── src/ # core modules 
│ ├── data_loader.py # load + validate CSV, filtering functions
│ ├── preprocessing.py # text whitespace normalization
│ ├── eda.py # exploratory data analysis
│ ├── model_interface.py # SentenceTransformer + sentence-based chunking
│ ├── evaluation.py # cosine similarity, outlier detection 
│ ├── pipeline.py # end-to-end orchestration + result exports
│ ├── visualization.py # yearly trend figures
│ ├── bert_comparison.py # naive BERT vs SentenceTransformer ablation
│ └── topic_analysis.py # BERTopic clustering + temporal analysis
├── scripts/
│ ├── fetch_articles.py # downloads the raw corpus from PubMed
│ └── run_pipeline.py # CLI wrapper around pipeline.py
├── tests/ #automated tests
│ ├── test_evaluation.py
│ ├── test_preprocessing.py
│ └── test_model_interface.py
├── notebooks/
│ └── demo_analysis.ipynb # demonstration notebook 
├── data/
│ ├── raw/ # articles.csv (not versioned), aims_scope.txt
│ └── results/
│ ├── tables/ # CSV outputs
│ └── figures/ # PNG figures
├── report.pdf 
├── slides.pdf
├── requirements.txt
├── LICENSE
└── README.md

```

## Setup

Create and activate a virtual environment, then install the required
dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

NLTK's sentence tokenizer data must be downloaded once:

```bash
python3 -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```


## Running the Pipeline

Run the following commands from the repository root, in this order:

```bash
# 1. Download the raw corpus from PubMed (only needed once)
python3 scripts/fetch_articles.py

# 2. Exploratory data analysis
python3 src/eda.py

# 3. Main pipeline: embeddings, alignment scores, outliers, and result exports
python3 src/pipeline.py

# 4. Temporal alignment analysis and figures
python3 src/visualization.py

# 5. Additional analyses
python3 src/bert_comparison.py
python3 src/topic_analysis.py
```

`scripts/run_pipeline.py` provides the same core pipeline as step 3 with
command-line arguments such as `--z-threshold` and `--min-year`. This can be
used to test alternative parameters without modifying the source code.

Run the test suite with:

```bash
pytest tests/
```


## Key Results

- The final corpus contains **1,074 articles** published between 2015 and 2025,
  after removing correction notices and records from the incomplete 2026
  publication year.

- The mean alignment score is **0.461** with a standard deviation of **0.104**.
  No clear systematic change in overall alignment is observed over time.

- **49 articles** are flagged by the z-score criterion with a threshold of
  2.0: 28 at the low end and 21 at the high end of the alignment distribution.
  These represent statistical extremes and should not automatically be
  interpreted as thematically misaligned articles.

- **BERTopic** identifies 24 topics plus a noise cluster. The topic analysis
  shows that relatively stable overall alignment can coexist with changes in
  the journal's thematic composition over time. Topic 4, mainly associated
  with AI and machine-learning methods, becomes more prominent in the later
  years of the corpus and contains two of the three least-aligned articles.
  Its representative terms include `model`, `learning`, `machine`,
  `pharmacovigilance`, and `performance`, showing that lower similarity does
  not necessarily imply a lack of connection with the journal's field.

- A naive-BERT baseline produces substantially higher and more concentrated
  cosine similarities than Sentence-BERT. This comparison supports the use
  of Sentence-BERT for the main semantic similarity analysis, where greater
  differentiation between article representations is required.


## Outputs

Generated results, tables, and figures are stored in:

```text
data/results/
```

The accompanying paper provides the complete methodology, analysis, and
interpretation of the results.


## License

MIT — see [LICENSE](LICENSE).
