import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

# LLM initialization
llm = ChatOllama(
    model="minimax-m2.5:cloud",
    temperature=0.7,
)

# Prompt template with conversation history placeholder
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI assistant."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

# Chain: prompt -> LLM -> parser
chain = prompt_template | llm | StrOutputParser()

# Session state to store chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of HumanMessage and AIMessage


def chat(question: str) -> str:
    """Send a question through the chain and update chat history."""
    response = chain.invoke({
        "question": question,
        "chat_history": st.session_state.chat_history,
    })

    st.session_state.chat_history.append(HumanMessage(content=question))
    st.session_state.chat_history.append(AIMessage(content=response))
    return response

# Page configuration
st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="wide")
st.title("🗣️ AI Chatbot Interface")

# Display conversation
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    else:
        st.chat_message("assistant").write(msg.content)

# Input area
input_text = st.chat_input("Ask me anything...")
if input_text:
    chat(input_text)

# Clear conversation button
if st.button("Clear conversation"):
    st.session_state.chat_history.clear()
