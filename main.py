import streamlit as st
from agent import run_agent, HumanMessage, AIMessage
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="LangGraph AI Agent", page_icon="🤖")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("🤖 Zubair's AI Assistant (with Tools)")

user_query = st.text_input("Enter your query:", key="user_input")

if st.button("Submit"):
    if user_query:
        with st.spinner("Thinking..."):
            # Run agent with full history internally
            response, updated_history = run_agent(user_query, st.session_state.chat_history)
            st.session_state.chat_history = updated_history  # Store updated history
            st.success("Done!")
            st.write("### Response:")
            if isinstance(response, AIMessage):
                st.write(response.content)
            else:
                st.write(str(response))
    else:
        st.warning("Please enter a query.")

# Optional: Show chat history inside expandable markdown
with st.expander("📜 Show Chat History"):
    history_md = ""
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            history_md += f"**You:** {message.content}\n\n"
        elif isinstance(message, AIMessage):
            history_md += f"**AI:** {message.content}\n\n"
    st.markdown(history_md, unsafe_allow_html=True)

