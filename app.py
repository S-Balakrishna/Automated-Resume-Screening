import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from src.resume_parser import extract_resume_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load trained model and TF-IDF vectorizer
MODEL_PATH = "models/trained_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"

# Load the trained model and vectorizer
with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)

with open(VECTORIZER_PATH, "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Streamlit UI
st.title("🚀 AI-Powered Resume Screening & Job Matching")
st.write(
    "Welcome to the **Automated Resume Screening System**! "
    "This AI-powered system helps HR professionals quickly filter and analyze resumes, "
    "matching candidates to job descriptions efficiently.")
st.write(  " **Features:** ")
st.write( "-NLP-based Resume Parsing ")
st.write(  "-AI-powered Candidate Matching")
st.write("-Bias-Free Resume Screening ")
st.sidebar.header("Upload Section")

# Upload resume
uploaded_resume = st.sidebar.file_uploader("Upload Resume (PDF or DOCX)", type=["pdf", "docx"])

# Upload job description
uploaded_jd = st.sidebar.text_area("Paste Job Description (JD)", placeholder="Enter the job requirements here...")

# Function to calculate ATS Score
def calculate_ats_score(resume_text, job_desc_text):
    """
    Compare resume and job description using TF-IDF & Cosine Similarity.
    Returns an ATS Score (0-100).
    """
    if not resume_text or not job_desc_text:
        return 0
    
    tfidf_matrix = vectorizer.transform([resume_text, job_desc_text])
    similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    
    return round(similarity * 100, 2)

# Process Resume
if uploaded_resume:
    st.subheader("📄 Extracted Resume Text")
    resume_text = extract_resume_text(uploaded_resume)
    st.write(resume_text[:1000] + "..." if len(resume_text) > 10000 else resume_text)

    # Predict Resume Category
    resume_vectorized = vectorizer.transform([resume_text])
    predicted_category = model.predict(resume_vectorized)[0]
    
    st.subheader("🧠 AI-Predicted Resume Category")
    st.success(f"Predicted Field: **{predicted_category}**")

    # ATS Score Calculation
    if uploaded_jd:
        ats_score = calculate_ats_score(resume_text, uploaded_jd)
        st.subheader("📊 ATS Score")
        st.metric(label="Match Score", value=f"{ats_score}%", delta=0)
        
        # Decision
        if ats_score >= 80:
            st.success("✅ Strong match! Highly recommended for the job.")
        elif ats_score >= 40:
            st.warning("⚠️ Moderate match. Some improvements needed.")
        else:
            st.error("❌ Weak match. Resume does not align well with the job description.")
