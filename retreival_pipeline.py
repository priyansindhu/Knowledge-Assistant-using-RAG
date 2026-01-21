from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp

persistent_directory = "db/chroma_db"

# Embeddings (must MATCH ingestion embeddings!)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
       "k": 5,
       "score_threshold": 0.4
   }
)

# Load LLaMA
llm = LlamaCpp(
    model_path="models/tinyllama-1.1b-chat-v1.0.Q2_K.gguf",
    temperature=0.1,
    max_tokens=512,
    n_ctx=4096,
    verbose=False
)

def build_prompt(context, question):
    return f"""
You are a strict question-answering system.

Rules:
- Use ONLY the context provided for answering the question.
- If the context does not contain the answer, reply exactly: I don't know.
- Do NOT use prior knowledge.

Context:
{context}

Question:
{question}

Answer:
"""

def get_sources(docs):
    """Extract unique sources from retrieved documents"""
    sources = []
    for doc in docs:
        source = doc.metadata.get("source")
        if source and source not in sources:
            sources.append(source)
    return sources

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chat loop
while True:
    query = input("\nEnter your question (or type 'exit' to quit):\n> ")
    if query.lower() in ["exit", "quit"]:
        print("Bye!! See you again !!")
        break

    # Retrieve documents
    relevant_docs = retriever.invoke(query)

    if not relevant_docs:
        print("\n--- Answer ---")
        print("I don't know")
        print("\n--- Sources ---")
        print("No sources found")
        continue

    print(f"\nUser Query: {query}")
    """""
    print("\n--- Retrieved Context ---")
    for i, doc in enumerate(relevant_docs, 1):
        source = doc.metadata.get("source", "Unknown source")
        print(f"\nDocument {i} (Source: {source}):\n{doc.page_content}\n")
    """""

    # Build context + prompt
    context = format_docs(relevant_docs).strip()

    if not context:
        print("\n--- Answer ---")
        print("I don't know")
        continue

    prompt = build_prompt(context, query)
    answer = llm.invoke(prompt)

    print("\n--- Answer ---")
    print(answer)

    sources = get_sources(relevant_docs)
    print("\n--- Sources ---")
    if sources:
        for src in sources:
            print(f"- {src}")
    else:
        print("No sources found")

#Questions:
# 1.How does Google generate income
# 2.How much did Microsoft pay to acquire GitHub?
# 3.Who is the owner of Nvidia
# 4.What are the achievements of SpaceX?
# 5.Features of Tesla superchargers
# 6.Which direction Sun rises?
