import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def load_documents(docs_path="docs"):
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The folder '{docs_path}' does not exist.")

    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in '{docs_path}'.")

    print(f"Loaded {len(documents)} document(s)")
    return documents


def split_documents(documents, chunk_size=1000, chunk_overlap=100):
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunk(s)")
    return chunks


def create_vector_store(chunks, persist_directory="db/chroma_db"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"},
    )
    print(f"Saved vector store to {persist_directory}")
    return vector_store


def main():
    documents = load_documents()
    chunks = split_documents(documents)
    create_vector_store(chunks)


if __name__ == "__main__":
    main()