import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_pipelines():
    question_generator = pipeline("text2text-generation", model="mrm8488/t5-base-finetuned-question-generation-ap")
    evaluator = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-emotion")
    return question_generator, evaluator

st.title("🎯 AI Interview Bot")
st.write("Practice with AI-generated questions and get feedback!")

q_gen, evaluator = load_pipelines()

if "question" not in st.session_state:
    st.session_state.question = ""

if st.button("🧠 Generate Interview Question"):
    question = q_gen("Generate an interview question about data structures")[0]['generated_text']
    st.session_state.question = question

if st.session_state.question:
    st.markdown(f"**Interview Question:** {st.session_state.question}")
    user_answer = st.text_area("💬 Your Answer", height=150)

    if st.button("✅ Evaluate Answer"):
        if user_answer.strip() != "":
            result = evaluator(user_answer)[0]
            st.markdown(f"**📝 Feedback:** {result['label']} (Confidence: {round(result['score'] * 100, 2)}%)")
        else:
            st.warning("Please enter an answer to evaluate.")
