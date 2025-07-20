import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_pipelines():
    question_generator = pipeline("text2text-generation", model="Vamsi/T5_Paraphrase_Paws")
    evaluator = pipeline("text-classification", model="lxyuan/distilbert-base-multilingual-cased-sentiments-student")
    return question_generator, evaluator

q_gen, evaluator = load_pipelines()

st.title("🎯 AI Interview Bot")
st.subheader("Practice with AI-generated questions and get feedback!")

if st.button("Generate Interview Question"):
    prompt = "Generate an interview question related to technology."
    result = q_gen(prompt, max_length=64, do_sample=True)
    question = result[0]['generated_text']
    st.session_state['question'] = question
    st.write("**Interview question:**", question)

if 'question' in st.session_state:
    answer = st.text_area("Your Answer:")
    if st.button("Evaluate Answer"):
        result = evaluator(answer)
        feedback = result[0]['label']
        st.write("**Feedback:**", feedback)
