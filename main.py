import streamlit as st
from agent import run_agent, HumanMessage, AIMessage
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="LangGraph AI Agent", page_icon="🤖")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("🤖 AI Assistant with Tools")

user_query = st.text_input("Enter your query:", key="user_input")

if st.button("Submit"):
    if user_query:
        with st.spinner("Thinking..."):
            response, updated_history = run_agent(user_query, st.session_state.chat_history)
            st.session_state.chat_history = updated_history
            st.success("Done!")
    else:
        st.warning("Please enter a query.")

st.write("## Chat History")
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        st.write(f"**You:** {message.content}")
    elif isinstance(message, AIMessage):
        st.write(f"**AI:** {message.content}")
