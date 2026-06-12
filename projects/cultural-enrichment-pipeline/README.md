# 🎨 Cultural Data & Multimodal Enrichment Pipeline

This project contains Python implementations for extracting named entities, topic classifications, stance, sentiment, and visual/multimodal properties from social records and creative assets. It models the core enrichment systems of the Brand Mentality® platform.

---

## 📁 File Structure

*   [`text_enrichment.py`](text_enrichment.py): Performs Named Entity Recognition (NER) (via `dbmdz/bert-large-cased-finetuned-conll03-english`), topic taxonomy mapping (BART-large-MNLI zero-shot), stance detection (BART zero-shot), sentiment scoring (Twitter-RoBERTa), and toxic brand safety filtering (Unitary Toxic BERT). Contains robust fallback heuristics when neural pipelines are disabled.
*   [`multimodal_enrichment.py`](multimodal_enrichment.py): Extracts video frames using OpenCV, embeds visual/text representations via a pre-trained CLIP model (`openai/clip-vit-base-patch32`), and aggregates temporal frame signals with mean pooling to perform zero-shot visual safety classification.
*   [`embedding_generator.py`](embedding_generator.py): Computes dense, semantic sentence embeddings using a Sentence-Transformers model (`sentence-transformers/all-MiniLM-L6-v2`) to perform semantic similarity calculations.
*   [`clustering.py`](clustering.py): Performs text representation vectorization via TF-IDF and clusters unstructured posts via K-Means to identify and group trending topics.

---

## ⚙️ Installation & Setup

1. Navigate to this project directory:
   ```bash
   cd projects/cultural-enrichment-pipeline
   ```
2. Install dependencies:
   ```bash
   pip3 install -r requirements.txt
   ```

---

## 🚀 Execution & Verification

You can run individual modules directly to verify their operation with demonstration datasets:

### 1. Text NLP Pipeline (NER, Topic, Stance, Toxicity)
```bash
python3 text_enrichment.py
```
This runs text posts through the NLP classifiers, showing extracted entities, stance towards brand targets, sentiment polarity, and toxicity flags.

### 2. Multimodal CLIP Pipeline (Image/Video Safety)
```bash
python3 multimodal_enrichment.py
```
This downloads/generates demo images and a synthetic video, processes frames with OpenCV, and computes zero-shot safety scores using CLIP.

### 3. Text Embeddings
```bash
python3 embedding_generator.py
```
Generates 384-dimensional dense vectors for text items and outputs cosine similarity matrices.

### 4. Cultural Trend Clustering
```bash
python3 clustering.py
```
Runs K-Means clustering over a mock corpus of social records to discover and print trending keywords and clusters.

---

## 🧪 Running Unit Tests

This project includes a comprehensive test suite in `tests/` covering mock/real transformers runs, edge cases, and OpenCV frame extractors.

### Run from Workspace Root (Recommended)
```bash
PYTHONPATH=projects/cultural-enrichment-pipeline python3 -m pytest projects/cultural-enrichment-pipeline/tests/ -v
```

### Run from Project Directory
```bash
PYTHONPATH=. python3 -m pytest tests/ -v
```
