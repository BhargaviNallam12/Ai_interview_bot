import streamlit as st
from transformers import pipeline

# Caching the pipelines to avoid reloading every time
@st.cache_resource
def load_pipelines():
    # Better-supported models
    question_generator = pipeline("text2text-generation", model="google/flan-t5-small")
    answer_evaluator = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-sst2")
    return question_generator, answer_evaluator

q_gen, evaluator = load_pipelines()

# Streamlit UI
st.title("🎯 AI Interview Bot")
st.write("Practice with AI-generated questions and get feedback!")

# Input: Job Role
job_role = st.text_input("Enter a job role (e.g., Data Scientist, Software Engineer)")

if job_role:
    with st.spinner("Generating interview question..."):
        question = q_gen(f"Generate one interview question for a {job_role}")[0]['generated_text']
        st.subheader("🤖 Interview Question:")
        st.write(question)

        # Input: User answer
        answer = st.text_area("Your Answer", height=150)

        if st.button("Submit Answer"):
            with st.spinner("Evaluating your answer..."):
                evaluation = evaluator(answer)[0]
                st.subheader("📝 Evaluation Result:")
                st.write(f"Label: {evaluation['label']}")
                st.write(f"Confidence: {evaluation['score']:.2f}")
