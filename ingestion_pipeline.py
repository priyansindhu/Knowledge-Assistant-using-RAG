import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def load_documents(docs_path="docs"):
    """
    Load TXT, PDF, and Markdown documents from the docs directory
    """
    print(f"Loading documents from {docs_path}...")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(
            f"The directory '{docs_path}' does not exist. Please create it and add documents."
        )

    documents = []

    for file in os.listdir(docs_path):
        file_path = os.path.join(docs_path, file)

        if file.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
            documents.extend(loader.load())
        elif file.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        elif file.endswith(".md"):
            loader = UnstructuredMarkdownLoader(file_path)
            documents.extend(loader.load())

    if len(documents) == 0:
        raise FileNotFoundError(
            f"No supported documents found in '{docs_path}'."
        )

    # preview 
    for i, doc in enumerate(documents[:2]):
        print(f"\nDocument {i+1}:")
        print(f"  Source: {doc.metadata.get('source')}")
        print(f"  Content length: {len(doc.page_content)} characters")
        print(f"  Preview: {doc.page_content[:100]}...")
        print(f"  Metadata: {doc.metadata}")

    return documents

def split_documents(documents, chunk_size=800, chunk_overlap=100):
    """Split documents into smaller chunks with overlap"""
    print("Splitting documents into chunks...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?"]
    )
    
    chunks = text_splitter.split_documents(documents)
    
    if chunks:
    
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content:")
            print(chunk.page_content)
            print("-" * 50)
        
        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks")
    
    return chunks

def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist ChromaDB vector store with open-source embeddings"""
    print("Creating embeddings and storing in ChromaDB...")

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Create ChromaDB vector store
    print("--- Creating vector store ---")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"} 
    )
    print("--- Finished creating vector store ---")
    print(f"Vector store created and saved to {persist_directory}")

    return vectorstore


def main():
    print("=== RAG Document Ingestion Pipeline ===\n")
    
    docs_path = "docs"
    persistent_directory = "db/chroma_db"

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # ✅ LOAD existing vector store
    if os.path.exists(persistent_directory):
        print("✅ Vector store already exists. Loading from disk...")

        vectorstore = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embedding_model
        )

        print(
            f"Loaded existing vector store with "
            f"{vectorstore._collection.count()} documents"
        )
        return vectorstore

    # ✅ CREATE new vector store
    print("Persistent directory does not exist. Initializing vector store...\n")

    documents = load_documents(docs_path)
    chunks = split_documents(documents)

    vectorstore = create_vector_store(
        chunks,
        persist_directory=persistent_directory
    )

    print("\n✅ Ingestion complete! Your documents are now ready for RAG queries.")
    return vectorstore


if __name__ == "__main__":
    main()

