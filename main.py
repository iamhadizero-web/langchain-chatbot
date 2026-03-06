from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

import os
from dotenv import load_dotenv

load_dotenv()    #READ .ENV VARIABLES

MODEL_NAME = os.getenv("MODEL_NAME", "qwen2.5-coder:3b")
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7))
MAX_TURNS = int(os.getenv("MAX_TURNS", 5))

llm = ChatOllama(
    model=MODEL_NAME,
    temperature=TEMPERATURE
    )

prompt =ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human","{question}")
])

chain= prompt | llm | StrOutputParser()


chat_history = []    #python list to store the chat history


def chat(question):
    current_turns = len(chat_history) // 2   #each turn consists of a human message and

    if current_turns >= MAX_TURNS:
        return(
            "Context Window is full!"
            "The AI may not follow previous thread properly."
            "Please type 'clear' for a new chat."
        )
        
    response= chain.invoke(
        {
            "question": question,
            "chat_history": chat_history
        }
    )
    
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=response))

    remaining = MAX_TURNS - (current_turns + 1)
    if remaining <=2:
        response += f"Warning: Only {remaining} turn(s) left." 
       
    return response
def main():
    print("Langchain Chatbot Ready! Type 'quit' to exit or 'clear' to reset the chat history.")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
           continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "clear":
            chat_history.clear()
            print("History cleared, Starting fresh.")
            continue
        
        print(f"AI:{chat(user_input)}")


main()

