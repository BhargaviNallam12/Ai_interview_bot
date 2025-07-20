import streamlit as st
from transformers import pipeline

# Load the free Hugging Face text generation model
@st.cache_resource
def load_model():
    return pipeline("text-generation", model="distilgpt2")

generator = load_model()

st.title("🎯 AI Interview Bot")
st.subheader("Get job-specific AI-generated interview questions!")

# Dropdown for job roles
job_roles = ["Software Engineer", "Data Scientist", "HR Manager", "Marketing Executive", "Project Manager"]
selected_role = st.selectbox("Choose a job role:", job_roles)

if st.button("Generate Interview Question"):
    prompt = f"Interview question for a {selected_role}:"
    result = generator(prompt, max_length=50, num_return_sequences=1)
    st.success(result[0]["generated_text"])
