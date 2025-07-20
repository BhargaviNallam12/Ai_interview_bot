import streamlit as st
from transformers import pipeline

# Load question generation model (small and free)
@st.cache_resource
def load_pipeline():
    return pipeline("text2text-generation", model="iarfmoose/t5-base-question-generator")

question_generator = load_pipeline()

# Streamlit UI
st.title("🎤 AI Interview Question Generator")
st.subheader("Get interview questions to practice!")

# Default context
context = st.text_area("Enter context (e.g., job role, topic, or 'Tell me about yourself'):", 
                       "Tell me about yourself")

if st.button("Generate Question"):
    with st.spinner("Generating question..."):
        prompt = f"generate question: {context}"
        result = question_generator(prompt, max_length=64, do_sample=True)[0]['generated_text']
        st.success("Here's your interview question:")
        st.write(result)
