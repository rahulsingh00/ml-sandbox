import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
from PIL import Image

sys.path.append("projects/cultural-enrichment-pipeline")
from text_enrichment import TextEnrichmentPipeline
from multimodal_enrichment import MultimodalEnrichmentPipeline
from embedding_generator import EmbeddingGenerator

st.set_page_config(page_title="Cultural Data & NLP", page_icon="🎨", layout="wide")

st.markdown("# 🎨 Cultural Data & NLP Pipeline")
st.write("Extract entities, topics, sentiment, stance, and semantic embeddings from text and media assets.")
st.write("---")

# Initialize backends using cache to prevent reloading models on every rerun
@st.cache_resource
def get_nlp_pipeline():
    return TextEnrichmentPipeline()

@st.cache_resource
def get_multimodal_pipeline():
    return MultimodalEnrichmentPipeline()

@st.cache_resource
def get_embedding_generator():
    return EmbeddingGenerator()

nlp_pipeline = get_nlp_pipeline()
multimodal_pipeline = get_multimodal_pipeline()
embedding_gen = get_embedding_generator()

tab1, tab2, tab3 = st.tabs(["📝 Text Enrichment", "🖼️ Multimodal (CLIP)", "🔗 Semantic Similarity"])

with tab1:
    st.header("Social Post Enrichment")
    default_text = "Microsoft is releasing new AI features for Xbox today. I absolutely love playing games, but hate weapons and violence."
    text_input = st.text_area("Enter unstructured post text to enrich:", value=default_text)
    
    col1, col2 = st.columns(2)
    
    with col1:
        target_brand = st.text_input("Enter target brand for Stance Classification:", value="Xbox")
        
    if st.button("Run Text Enrichment"):
        with st.spinner("Processing text..."):
            # 1. Run entities
            entities = nlp_pipeline.extract_entities(text_input)
            
            # 2. Run sentiment
            sentiment_res = nlp_pipeline.analyze_sentiment(text_input)
            
            # 3. Run stance
            stance = nlp_pipeline.classify_stance(text_input, target_brand)
            
            # 4. Run topic
            topic_res = nlp_pipeline.classify_topic(text_input)
            
            # 5. Run safety
            safety_res = nlp_pipeline.check_brand_safety(text_input)
            
        st.subheader("💡 Enrichment Results")
        
        # Display key metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sentiment", sentiment_res["sentiment"], f"Score: {sentiment_res['score']:.2f}")
        c2.metric("Stance on " + target_brand, stance)
        c3.metric("Primary Topic", topic_res["primary_topic"])
        c4.metric("Safety Status", safety_res["status"], delta="SAFE" if safety_res["status"] == "SAFE" else "UNSAFE", delta_color="inverse")
        
        # Display detailed entities
        st.markdown("#### 🏷️ Extracted Entities & Brand Matches")
        if entities:
            ent_df = pd.DataFrame(entities)
            st.dataframe(ent_df, use_container_width=True)
        else:
            st.info("No entities detected.")
            
        # Display topic distribution
        st.markdown("#### 📊 Topic Taxonomy Distribution")
        dist_df = pd.DataFrame(list(topic_res["topic_distribution"].items()), columns=["Topic", "Probability"])
        st.bar_chart(dist_df.set_index("Topic"))

with tab2:
    st.header("CLIP Multimodal Image Safety")
    st.write("Uses CLIP zero-shot text-to-image similarity to categorize and flag visual assets.")
    
    # Selection of demo image or upload
    img_option = st.selectbox("Choose demo image:", ["Happy family eating breakfast", "Soldiers carrying weapons"])
    
    # Simulate image path based on option
    demo_images_dir = "projects/cultural-enrichment-pipeline/assets/images"
    os.makedirs(demo_images_dir, exist_ok=True)
    
    img_name = "happy_family.jpg" if img_option == "Happy family eating breakfast" else "weapons.jpg"
    img_path = os.path.join(demo_images_dir, img_name)
    
    # Generate mock PIL images for visual demo if they don't exist
    if not os.path.exists(img_path):
        if img_name == "happy_family.jpg":
            img = Image.new("RGB", (300, 300), color=(135, 206, 235)) # Sky blue
        else:
            img = Image.new("RGB", (300, 300), color=(105, 105, 105)) # Grey
        img.save(img_path)
        
    st.image(img_path, caption=img_option, width=300)
    
    if st.button("Evaluate Image Safety (CLIP)"):
        with st.spinner("Embedding image and text labels..."):
            safety_res = multimodal_pipeline.evaluate_image_safety(img_path)
            
        st.subheader("💡 Multimodal Results")
        st.metric("Visual Safety Status", "SAFE" if safety_res["is_safe"] else "UNSAFE")
        st.write(f"Primary Visual Concept: **{safety_res['primary_visual_concept']}**")
        
        # Chart similarities
        sim_df = pd.DataFrame(list(safety_res["similarity_scores"].items()), columns=["Concept", "Cosine Similarity"])
        st.bar_chart(sim_df.set_index("Concept"))

with tab3:
    st.header("Semantic Text Similarities")
    st.write("Generate dense vector embeddings (384 dimensions) and compare semantic similarity.")
    
    sent1 = st.text_input("Sentence A:", value="The advertisement was displayed on mobile devices.")
    sent2 = st.text_input("Sentence B:", value="An ad campaign was shown on smartphone screens.")
    
    if st.button("Calculate Similarity"):
        emb1 = embedding_gen.embed_texts([sent1])[0]
        emb2 = embedding_gen.embed_texts([sent2])[0]
        
        similarity = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
        
        st.subheader("💡 Similarity Score")
        st.metric("Cosine Similarity", f"{similarity:.4f}")
        
        if similarity > 0.7:
            st.success("High Semantic Similarity! (Very similar concepts)")
        elif similarity > 0.4:
            st.warning("Moderate Semantic Similarity.")
        else:
            st.error("Low Semantic Similarity. (Different concepts)")
