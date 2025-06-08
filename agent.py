import os
from typing import TypedDict, Annotated, Sequence
# Annotated -> it provides additional information without affecting the type itself
# Sequence -> it automatically handles the state updates such as adding new messages to chat history
from langchain_core.messages import BaseMessage # the foundational class of all message types in langGraph
from langchain_core.messages import ToolMessage # passes data back to the llm after it calls tools
from langchain_core.messages import SystemMessage # Messages for providing instructions to the llm
from langchain_core.messages import HumanMessage # Messages for user input
from langchain_core.messages import AIMessage # Messages for the llm's response
from langchain_core.tools import tool #
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages # reducer function, 
# rule that controls how updates from nodes are combined to the existing sate, 
# tells us how to merge data into the current state
# Without a reducer updates would overide the previous sate and wipe them out
from langgraph.prebuilt import ToolNode #
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def add(a:int,b:int) -> int:
    """This function adds up two integers and returns and integer"""
    return a+b

@tool
def subtract(a:int,b:int) -> int:
    """This function subtracts two integers and returns an integer"""
    return a-b

@tool
def multiply(a:int,b:int) -> int:
    """This function multiplies two integers and returns an integer"""
    return a*b

@tool
def execute_python(code: str) -> str:
    """Executes Python code and returns the output or error."""
    import io
    import contextlib

    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            exec(code, {})
    except Exception as e:
        return f"Error: {str(e)}"
    return output.getvalue()


our_tools = [add,subtract,multiply, execute_python]

GROQ_API_KEY = os.getenv("groq_key")
model = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=GROQ_API_KEY, max_tokens=500).bind_tools(our_tools)

def agentNode(state:AgentState) -> AgentState:
    system_prompt = SystemMessage(content="You are an AI assistant. Help answering the queries with the best of your abilities. Keep your answer to the point.")
    response = model.invoke([system_prompt] + state['messages'])
    state['messages'].append(response) #state['messages'].append([AIMessage(content=response)])  # add the response to the messages
    return state # {"messages":[response]} # another way of updating the state

def should_continue(state:AgentState) -> AgentState:
    messages = state['messages']
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"
    
graph = StateGraph(AgentState)
graph.add_node("llm_caller", agentNode)
tool_node = ToolNode(tools=our_tools)
graph.add_node("tool", tool_node)
graph.add_edge(START, "llm_caller")
graph.add_conditional_edges(
    "llm_caller",
    should_continue,
    {
        "continue":"tool",
        "end":END
    }
)
graph.add_edge("tool","llm_caller")
app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s['messages'][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


chat_history = []
with open ('logs.txt','r') as file:
    lines = file.readlines()
    for line in lines:
        if line.startswith("You:"):
            chat_history.append(HumanMessage(content=line[4:].strip()))
        elif line.startswith("AI:"):
            chat_history.append(AIMessage(content=line[3:].strip()))


def run_agent(user_query: str, chat_history: list[BaseMessage]) -> tuple[str, list[BaseMessage]]:
    chat_history.append(HumanMessage(content=user_query))
    inputs = app.invoke({"messages": chat_history})
    new_history = inputs["messages"]
    last_response = new_history[-1].content
    return last_response, new_history

with open('logs.txt',"w") as file:
    file.write("Start of conversation:\n\n\n")

    for message in chat_history:
        if isinstance(message,HumanMessage):
            file.write(f"You: {message.content}\n\n")
        elif isinstance(message,AIMessage):
            file.write(f"AI: {message.content}\n\n\n")

    file.write("End of conversation:\n")

print("conversation saves to logs.txt")