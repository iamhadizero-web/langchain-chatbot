from langchain_ollama import ChatOllama


llm = ChatOllama(
    model="minimax-m2.5:cloud",
    temperature=0.5
    )

response = llm.invoke("What is RAG?")
print(response.content)