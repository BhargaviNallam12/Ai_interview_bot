import streamlit as st
import openai

st.title("AI Interview Bot")

openai.api_key = st.secrets["OPENAI_API_KEY"]

# Get user input
question = st.text_input("Ask your interview question:")

# Ask OpenAI and get response
if st.button("Get Answer") and question:
    with st.spinner("Thinking..."):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful AI Interview Bot."},
                {"role": "user", "content": question}
            ]
        )
        st.success(response['choices'][0]['message']['content'])
