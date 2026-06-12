# 🧠 Concept: Multi-modal Enrichment

This guide covers computer vision and multi-modal enrichment models used to process image and video signals from social platforms, complementing text-heavy data pipelines.

---

## 1. Why Multi-modal?
Social media is heavily visual (memes, photos, video shorts). Text-only models miss vital context:
*   An image of a brand logo next to a controversial symbol.
*   A video featuring a scenic backdrop with audio discussing high-risk activities.
*   Contextual text that is sarcasm, where the image reveals the true meaning.

---

## 2. CLIP (Contrastive Language-Image Pretraining)
CLIP (by OpenAI) is a neural network trained on a variety of (image, text) pairs. It maps images and text to a shared embedding space, enabling zero-shot image classification and text-to-image search.

### Contrastive Embedding Space
CLIP consists of an **Image Encoder** (e.g. ViT) and a **Text Encoder** (e.g. Transformer).
*   Let $I$ be an image, and $T$ be a text description.
*   The image encoder outputs embedding $E_I = f_{\text{img}}(I) \in \mathbb{R}^d$.
*   The text encoder outputs embedding $E_T = f_{\text{txt}}(T) \in \mathbb{R}^d$.
*   Both embeddings are normalized:
    $$\hat{E}_I = \frac{E_I}{\|E_I\|}, \quad \hat{E}_T = \frac{E_T}{\|E_T\|}$$
*   The cosine similarity is the dot product:
    $$\text{sim}(I, T) = \hat{E}_I \cdot \hat{E}_T$$

By feeding CLIP a set of text labels (e.g., "a photo of a brand-safe environment", "a photo containing violence"), we can perform zero-shot image classification by selecting the label with the highest similarity score.

---

## 3. Video Enrichment & Temporal Pooling
Videos are sequences of frames. Processing every single frame through deep vision models is computationally prohibitive for pipelines handling millions of videos.

### Downsampling & Keyframe Extraction
1.  **Frame Sampling**: Sample frames at a fixed interval (e.g., 1 frame per second).
2.  **Keyframe Detection**: Use pixel-difference heuristics or motion vectors to extract frames where a scene change occurs.

### Temporal Feature Aggregation
Once frame embeddings $\{e_1, e_2, \dots, e_N\}$ are extracted, they are aggregated into a single video embedding $v$ using:
*   **Mean Pooling**:
    $$v = \frac{1}{N} \sum_{i=1}^N e_i$$
*   **Max Pooling**:
    $$v_j = \max_{i} (e_{i, j}) \quad \forall j \in \{1, \dots, d\}$$
*   **Attention-based Aggregation**: Learning a weighted average where weights are determined by an attention mechanism scoring frame relevance.

---

## 🛠️ Sandbox Implementation
Check out the CLIP implementation and multimodal pipeline in [`projects/cultural-enrichment-pipeline`](file:///Users/rahulsingh/.gemini/antigravity/scratch/ml-sandbox/projects/cultural-enrichment-pipeline).
