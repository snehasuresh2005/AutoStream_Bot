import os
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Initialize Embeddings
# Assuming GOOGLE_API_KEY is in .env or environment
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "knowledge_base.md")
DB_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")

def init_knowledge_base():
    """Initializes the vector store with the knowledge base."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Knowledge base not found at {DATA_PATH}")

    loader = TextLoader(DATA_PATH)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = text_splitter.split_documents(documents)

    # Persist directory for valid local RAG
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="autostream_kb",
        persist_directory=DB_DIR
    )
    return vector_store

def get_retriever():
    """Returns the retriever for the RAG pipeline."""
    # Check if DB exists, if not create it (simplified logic for this assignment)
    # in a real app check if persist_dir has data
    if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
        vector_store = Chroma(
            embedding_function=embeddings,
            collection_name="autostream_kb",
            persist_directory=DB_DIR
        )
    else:
        vector_store = init_knowledge_base()
    
    return vector_store.as_retriever(search_kwargs={"k": 2})

if __name__ == "__main__":
    # Test run
    print("Initializing knowledge base...")
    try:
        retriever = get_retriever()
        docs = retriever.invoke("How much is the Pro plan?")
        print(f"Retrieved {len(docs)} docs. First doc content sample: {docs[0].page_content[:100]}")
    except Exception as e:
        print(f"Error: {e}")
