import streamlit as st
from agent import run_agent, HumanMessage, AIMessage
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="LangGraph AI Agent", page_icon="🤖")

st.title("🤖 AI Assistant with Tools")

user_query = st.text_input("Enter your query:", key="user_input")

if st.button("Submit"):
    if user_query:
        with st.spinner("Thinking..."):
            # Only pass the current message, no history
            response, _ = run_agent(user_query, [])  # pass empty history
            st.success("Done!")
            st.write("### Response:")
            if isinstance(response, AIMessage):
                st.write(response.content)
            else:
                st.write(str(response))
    else:
        st.warning("Please enter a query.")
