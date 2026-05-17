# Transformer-Based Sentiment Analysis: Semantic Search, Embeddings & LLMs

A hands-on exploration of **Transformer models** for NLP, covering sentence embeddings, semantic search, review categorization, and sentiment analysis using multiple approaches -- from classical ML on top of embeddings to pre-trained HuggingFace pipelines to prompting an LLM (Google FLAN-T5).

---

## Problem Statement

In the fast-evolving entertainment industry, gauging audience sentiments towards movie releases is crucial for marketing strategies, content creation, and enhancing the overall viewer experience. Manually analyzing an extensive volume of reviews is time-consuming and doesn't capture nuanced sentiments at scale. This project builds an **ML-powered sentiment analyzer** that automatically classifies movie reviews as positive or negative.

## Dataset

| Property | Detail |
|---|---|
| Total reviews | 10,000 |
| Unique reviews | 9,982 (after dropping duplicates) |
| Columns | `review` (text), `sentiment` (0 = negative, 1 = positive) |
| Class balance | Roughly balanced between positive and negative |

---

## Project Structure

```
.
├── Hands_on_Transformers_Notebook (1).ipynb   # Main notebook with all code & analysis
├── movie_reviews.csv                          # Dataset (required, not included in repo)
└── README.md
```

---

## What I Did

### 1. Data Exploration & Preprocessing
- Loaded and explored the dataset using Pandas.
- Identified and handled **18 duplicate reviews** (10,000 -> 9,982 unique).
- Checked data types, null values, and shape.
- Visualized sentiment distribution using Seaborn bar charts.

### 2. Sentence Embeddings with `all-MiniLM-L6-v2`

The `all-MiniLM-L6-v2` model is an all-round sentence embedding model trained on **1 billion+ training samples**, generating **384-dimensional** embeddings.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embedding_matrix = model.encode(data['review'], device=device, show_progress_bar=True)
# Output shape: (9982, 384)
```

**Cosine similarity demos to verify semantic understanding:**

| Sentence Pair | Cosine Score | Interpretation |
|---|---|---|
| "The cat is on the mat." vs "The mat has a cat on it." | **0.962** | Highly similar (same meaning) |
| "Roses are red, violets are blue." vs "The Earth orbits the Sun." | **0.127** | Dissimilar |
| "My name is Mark and I love football." vs "A strange object was found in the Mariana Trench." | **-0.062** | Very dissimilar |

### 3. Semantic Search

Built a **top-k similarity search** function using dot-product on the embedding matrix:

```python
def top_k_similar_sentences(embedding_matrix, query_text, k):
    query_embedding = model.encode(query_text)
    score_vector = np.dot(embedding_matrix, query_embedding)
    top_k_indices = np.argsort(score_vector)[::-1][:k]
    return data.loc[list(top_k_indices), 'review']
```

**Example queries tested:**
- `"Horror movies"` -- returned classic horror film reviews (Halloween, slasher flicks, etc.)
- `"Action movie with lots of car chases"` -- surfaced reviews about Bullitt, The French Connection, The Seven Ups, etc.
- `"The movie wasn't great but it delivered as per the expectations."` -- found mixed/mediocre reviews

### 4. Review Categorization via Semantic Similarity

Used four natural language queries as **category identifiers** and assigned the top-500 most similar reviews to each:

| Category | Query Template |
|---|---|
| Highly Positive | "Overall a great movie that ticks all the right boxes." |
| Moderately Positive | "The movie wasn't that great but it delivered as per the expectations." |
| Highly Negative | "The movie was a bad experience with bad direction and poor story." |
| Moderately Negative | "The plot was confusing but the acting performances were okay. The movie was mediocre at best." |

> **Key insight:** Since the model was not fine-tuned on this data, some reviews fell into multiple categories -- this is expected with semantic search vs hard clustering.

### 5. Sentiment Classification with Random Forest + Transformer Embeddings

Used the 384-dim embedding vectors as features for a **Random Forest Classifier**:

```python
rf_transformer = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)
rf_transformer.fit(X_train, y_train)
```

- **Train/Test split:** 75% / 25% (`random_state=42`)
- Evaluated with confusion matrices on both train and test sets.

### 6. Zero-Shot Sentiment Analysis with HuggingFace Pipeline

Used the default HuggingFace sentiment pipeline (`distilbert-base-uncased-finetuned-sst-2-english`) -- **no fine-tuning, no training**:

```python
sentiment_hf = pipeline("sentiment-analysis")
```

Quick demo on trial data:

| Input | Label | Confidence |
|---|---|---|
| "I love this movie" | POSITIVE | 0.9999 |
| "This movie is not very good at all!" | NEGATIVE | 0.9998 |
| "There is a cat outside." | NEGATIVE | 0.6246 |

> **Limitation:** Reviews longer than 512 tokens get truncated, which can affect accuracy.

### 7. Sentiment Analysis with Google FLAN-T5 (LLM Prompting)

Loaded **FLAN-T5 Large** in 8-bit quantized format:

```python
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
model = T5ForConditionalGeneration.from_pretrained(
    "google/flan-t5-large", load_in_8bit=True, device_map="auto"
)
```

**Prompt used:**
```
Categorize the sentiment of the review as positive or negative.
Return 1 for positive and 0 for negative.
```

**Generation parameters:**
| Parameter | Value | Purpose |
|---|---|---|
| `max_length` | 16 | Max tokens in generated output |
| `temperature` | 0.001 | Near-deterministic sampling |
| `do_sample` | True | Enable sampling from token distribution |

> **Gotcha encountered:** The LLM sometimes returned `"1 for positive"` instead of just `"1"`. Handled by extracting only the first character: `int(predict_sentiment(X[item])[0])`.

---

## What I Learned & Explored

### Concepts
- **Sentence Transformers** -- pre-trained models that convert text into dense vector representations capturing semantic meaning.
- **Cosine Similarity** -- measuring how semantically close two texts are in embedding space (range: -1 to 1).
- **Semantic Search** -- finding relevant documents by meaning rather than keyword matching, using dot product over embeddings.
- **Transfer Learning** -- using pre-trained transformer embeddings as features for a classical ML model (Random Forest).

### Tools & Techniques
- **HuggingFace Pipelines** -- ready-made, production-grade NLP pipelines for zero-shot inference.
- **Large Language Models (FLAN-T5)** -- prompting an LLM for classification tasks, understanding tokenization, decoding strategies, temperature, and sampling.
- **8-bit Quantization** -- loading large models efficiently on GPU with reduced memory using `bitsandbytes` + `accelerate`.
- **Prompt Engineering** -- crafting effective instructions for LLMs, handling unpredictable output formats, and learning why LLMs don't always follow formatting rules.

### Key Takeaways
- Pre-trained embeddings are **powerful feature extractors** even without fine-tuning.
- Zero-shot models like DistilBERT perform remarkably well out of the box for sentiment analysis.
- LLMs (FLAN-T5) can be used for classification via **prompt engineering**, but their outputs need careful post-processing.
- There is a trade-off between **model size/inference cost** and **accuracy** across all three approaches.

---

## Tech Stack

| Category | Tools / Libraries |
|---|---|
| Language | Python |
| Deep Learning | PyTorch (`2.8.0+cu126`) |
| Transformers | HuggingFace Transformers (`4.57.1`), Sentence-Transformers (`5.1.1`) |
| LLM | Google FLAN-T5 Large (8-bit quantized) |
| Classical ML | scikit-learn (`1.6.1`) -- RandomForestClassifier |
| Data | Pandas (`2.2.2`), NumPy (`2.0.2`) |
| Visualization | Matplotlib (`3.10.0`), Seaborn (`0.13.2`) |
| Quantization | bitsandbytes (`0.48.1`), accelerate (`1.11.0`) |
| Tokenization | sentencepiece (`0.2.1`) |
| Environment | Google Colab (T4 GPU) |

---

## Results Summary

| # | Approach | Model Used | Type | Key Observation |
|---|---|---|---|---|
| 1 | Embeddings + Random Forest | `all-MiniLM-L6-v2` + RF | Supervised (transfer learning) | Solid baseline; 384-dim embeddings as features |
| 2 | HuggingFace Pipeline | `distilbert-base-uncased-finetuned-sst-2-english` | Zero-shot | Strong out-of-the-box; limited by 512-token truncation |
| 3 | LLM Prompting | `google/flan-t5-large` (8-bit) | Prompt-based | Most flexible; needs careful output parsing |

All three approaches were evaluated using **confusion matrices** against ground-truth labels.

---

## How to Run

1. Open the notebook in **Google Colab** (GPU recommended -- T4 or better).
2. Run the first cell to install all dependencies:
   ```bash
   pip install -U -q \
       sentence-transformers==5.1.1 \
       transformers==4.57.1 \
       bitsandbytes==0.48.1 \
       accelerate==1.11.0 \
       sentencepiece==0.2.1 \
       pandas==2.2.2 numpy==2.0.2 \
       matplotlib==3.10.0 seaborn==0.13.2 \
       torch==2.8.0+cu126 scikit-learn==1.6.1
   ```
3. **Restart the runtime** after installation (required for dependency resolution).
4. Upload `movie_reviews.csv` to the working directory (or mount Google Drive).
5. Run all cells sequentially from the top.

---

## Suggested Repository Name

```
transformer-sentiment-analysis
```

Other good alternatives:
- `nlp-transformers-hands-on`
- `sentiment-analysis-with-transformers`
- `movie-review-sentiment-transformers`
