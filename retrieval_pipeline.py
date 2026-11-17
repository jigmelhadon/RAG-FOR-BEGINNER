from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

persist_directory = "db/chroma_db"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings,
    collection_metadata={"hnsw:space": "cosine"},
)

query = "What is the national dish of Bhutan?"

retriever = db.as_retriever(search_kwargs={"k": 3})
relevant_docs = retriever.invoke(query)

print(f"Question: {query}\n")
for i, doc in enumerate(relevant_docs, 1):
    print(f"--- Chunk {i} ---")
    print(doc.page_content)
    print()