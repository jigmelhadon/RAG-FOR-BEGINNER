from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

persist_directory = "db/chroma_db"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

query = "What is the national dish of Bhutan?"

retriever = db.as_retriever(search_kwargs={"k": 3})
relevant_docs = retriever.invoke(query)

context = "\n\n".join(doc.page_content for doc in relevant_docs)

messages = [
    SystemMessage(content="Answer only using the context provided. If the answer is not in the context, say you don't know."),
    HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"),
]

model = ChatOllama(model="llama3.2:3b")
result = model.invoke(messages)

print("Answer:")
print(result.content)