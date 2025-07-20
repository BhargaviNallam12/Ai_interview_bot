import streamlit as st
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
import torch

@st.cache_resource
def load_models():
    # Question generation model (safe and small)
    q_model = AutoModelForSeq2SeqLM.from_pretrained("iarfmoose/t5-base-question-generator")
    q_tokenizer = AutoTokenizer.from_pretrained("iarfmoose/t5-base-question-generator")
    # Answer evaluation model (sentiment-based)
    sentiment_pipe = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    return q_model, q_tokenizer, sentiment_pipe

q_model, q_tokenizer, sentiment_pipe = load_models()

def generate_question(context="Tell me about yourself."):
    input_text = f"generate question: {context}"
    input_ids = q_tokenizer.encode(input_text, return_tensors="pt")
    outputs = q_model.generate(input_ids, max_length=64, num_return_sequences=1)
    question = q_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return question

def evaluate_answer(answer):
    result = sentiment_pipe(answer)[0]
    label = result['label']
    score = result['score']
    return f"Your answer was rated as **{label}** with confidence {score:.2f}."

# --- Streamlit App UI ---
st.title("🎯 AI Interview Bot")
st.subheader("Practice answering AI-generated interview questions!")

if st.button("Generate Interview Question"):
    question = generate_question()
    st.session_state.question = question

if "question" in st.session_state:
    st.markdown(f"**Interview Question:** {st.session_state.question}")
    user_answer = st.text_area("Your Answer:")
    if st.button("Evaluate Answer"):
        if user_answer.strip():
            result = evaluate_answer(user_answer)
            st.success(result)
        else:
            st.warning("Please enter your answer before evaluating.")
