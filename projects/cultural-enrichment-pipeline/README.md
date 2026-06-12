# 🎨 Cultural Data & Multimodal Enrichment Pipeline

This project contains Python implementations for extracting entities, topics, stance, sentiment, and visual properties from social records, illustrating the core enrichment systems of the Brand Mentality® platform.

## 📁 File Structure

*   [`text_enrichment.py`](file:///Users/rahulsingh/.gemini/antigravity/scratch/ml-sandbox/projects/cultural-enrichment-pipeline/text_enrichment.py): Named Entity Recognition (NER), topic classification (using Zero-shot classifiers), sentiment scoring, and safety filters.
*   [`multimodal_enrichment.py`](file:///Users/rahulsingh/.gemini/antigravity/scratch/ml-sandbox/projects/cultural-enrichment-pipeline/multimodal_enrichment.py): Zero-shot image classification and brand safety screening using CLIP embeddings.
*   [`clustering.py`](file:///Users/rahulsingh/.gemini/antigravity/scratch/ml-sandbox/projects/cultural-enrichment-pipeline/clustering.py): Sentence-transformer text vectorization, dimension reduction, and trending theme discovery clustering.

## ⚙️ Installation & Setup

1.  Navigate to this project directory:
    ```bash
    cd projects/cultural-enrichment-pipeline
    ```
2.  Install requirements:
    ```bash
    pip3 install -r requirements.txt
    ```

## 🚀 Execution & Verification

Run the pipeline verification script:
```bash
python3 text_enrichment.py
```
This will run a set of test social posts through the NLP classifiers, showing extracted entities, topics, stance towards brands, and sentiment polarity scores.
