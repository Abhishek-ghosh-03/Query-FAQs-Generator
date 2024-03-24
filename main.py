import streamlit as st
from work import create_vector_db,get_qa_chain

st.title("Query")
btn =st.button("Create Knowledge")
if btn:
    pass



question = st.text_question("Question:")
if question:
    chain = get_qa_chain()
    response = chain(question)
    st.header("Answer:")
    st.response(response["result"])