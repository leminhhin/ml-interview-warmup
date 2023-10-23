import streamlit as st
import requests

st.title("ML Interview Warmup")


def generate_qna(data):
    response = requests.get('https://leminhhin--ml-interview-warmup-lambda-funcs-generate-qna.modal.run/', json=data, timeout=30.0)
    return response.json()

def evaluate_answer(data):
    response = requests.get('https://leminhhin--ml-interview-warmup-lambda-funcs-evaluate-answer.modal.run', json=data, timeout=30.0)
    return response.json()

if 'topic_submitted' not in st.session_state:
    st.session_state.topic_submitted = False

def click_button():
    st.session_state.topic_submitted = True

with st.form("topic_choosing_form"):
    topic = st.text_input("Enter ML topic you want to learn:", placeholder="prompt engineering")
    topic_submitted = st.form_submit_button("Submit topic", on_click=click_button)
    if topic_submitted:
        topic_data = {'topic': topic}
        st.session_state.qna_response = generate_qna(topic_data)

if st.session_state.topic_submitted:
    with st.form("anwer_form"):
        user_answer = st.text_area(f"Question: {st.session_state.qna_response['question']}")
        answer_data = st.session_state.qna_response
        answer_data['user_answer'] = user_answer
        answer_submitted = st.form_submit_button("Submit answer")

        if answer_submitted:
            evaluation_response = evaluate_answer(answer_data)
            st.markdown(f"Score: {evaluation_response['score']}/5")
            st.markdown(f"Explain: {evaluation_response['justification']}")
