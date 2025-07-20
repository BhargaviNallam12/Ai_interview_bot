import streamlit as st
from transformers import pipeline

# Load the free Hugging Face text generation model
@st.cache_resource
def load_model():
    return pipeline("text-generation", model="distilgpt2")

generator = load_model()

st.title("AI Interview Bot")
st.subheader("Practice answering AI-generated interview questions!")

if st.button("Ask Me a Question"):
    prompt = "Interview question:"
    result = generator(prompt, max_length=50, num_return_sequences=1)
    st.success(result[0]["generated_text"])
