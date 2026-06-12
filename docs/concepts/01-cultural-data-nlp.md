# 🧠 Concept: Cultural Data Enrichment & NLP

This guide explores the NLP concepts required for building enrichment models across the cultural data pipeline, handling entity extraction, topic modeling, stance, sentiment, and brand safety.

---

## 1. Named Entity Recognition (NER)
Named Entity Recognition is the process of identifying and categorizing key entities (people, organizations, locations, products) within unstructured text.

*   **Significance in Brand Mentality**: Allows a brand to detect whenever they, their competitors, or key trending subjects are mentioned.
*   **Techniques**:
    *   *Rule-Based & Gazetteers*: Using pre-compiled lists of entities (high precision, low recall).
    *   *Statistical Sequence Labeling*: CRF (Conditional Random Fields) or BiLSTM-CRF models.
    *   *Transformer-Based NER*: Fine-tuned BERT or RoBERTa models using BIO (Beginning, Inside, Outside) tagging schemes.

---

## 2. Topic and Stance Classification
Understanding *what* a post is about (Topic) and *where the author stands* on that topic (Stance) is critical for matching ads with appropriate cultural context.

*   **Topic Classification**: Categorizing content into taxonomies (e.g. Sports, Finance, Politics).
    *   *Zero-Shot Classification*: Using NLI (Natural Language Inference) models (e.g. `facebook/bart-large-mnli`) to classify text into arbitrary categories without retraining.
*   **Stance Detection**: Determines whether a piece of text is *Favor*, *Against*, or *Neutral* towards a target entity or concept.
    *   Unlike sentiment, which determines general positive/negative emotion, stance is target-specific. A post can have negative sentiment but a favorable stance toward a specific competitor.

---

## 3. Representation Learning & Clustering
To handle billions of social records, we compress text into dense vector representations (embeddings) and cluster them to discover emerging trends.

### Mathematical Representation
Given a text document $d$, a sentence transformer maps the document to a dense vector:
$$v = f(d) \in \mathbb{R}^D$$

We calculate semantic similarity between two documents $d_1$ and $d_2$ using **Cosine Similarity**:
$$\text{sim}(d_1, d_2) = \frac{v_1 \cdot v_2}{\|v_1\| \|v_2\|}$$

### Clustering Algorithms
*   **K-Means**: Partitions data into $K$ distinct clusters by minimizing the inertia (within-cluster sum-of-squares).
*   **HDBSCAN**: Density-based spatial clustering that handles noise and discovers clusters of varying densities. Useful for finding organic conversation clusters in social media feeds.

---

## 4. Sentiment & Brand Safety
*   **Sentiment Analysis**: Computing emotional polarity scores (e.g., VADER for heuristic social text, or Fine-tuned DistilBERT for deep contextual sentiment).
*   **Brand Safety**: Determining if content violates safe-advertising criteria (e.g., hate speech, violence, explicit content, or brand-specific sensitive topics).
    *   *Heuristic Safety*: Keyword blacklists and regular expressions.
    *   *Model-Based Safety*: Text classifiers trained on datasets like Toxicity or Hate Speech corpuses to score risk:
        $$P(\text{unsafe} \mid \text{text}) > \tau$$
        where $\tau$ is a customizable safety threshold.

---

## 🛠️ Sandbox Implementation
Check out the practical implementation of these NLP models in [`projects/cultural-enrichment-pipeline`](file:///Users/rahulsingh/.gemini/antigravity/scratch/ml-sandbox/projects/cultural-enrichment-pipeline).
