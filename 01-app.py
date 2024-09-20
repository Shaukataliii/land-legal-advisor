import streamlit as st
from langchain_codebase.codebase import hit_chain_get_response_with_history, load_yaml_file
from src.streamlit_utils import initialize_session_vars
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config("Legal Guide", page_icon=":books:")
st.header("Land Legal Guide :books:")
print("Running...")

DATABASE_PATH = load_yaml_file("src/params.yaml")['database_path']
session = st.session_state
session = initialize_session_vars(session, DATABASE_PATH, 2)


query = st.chat_input("Enter your query here.")

if query:
    with st.spinner("Getting response.."):
        session['chat_history'] = hit_chain_get_response_with_history(session['rag_chain'], query, session['chat_history'])

        for message in session['chat_history']:
            if isinstance(message, HumanMessage):
                with st.chat_message("user"):
                    st.write(message.content)

            if isinstance(message, AIMessage):
                with st.chat_message("assistant"):
                    st.write(message.content)


