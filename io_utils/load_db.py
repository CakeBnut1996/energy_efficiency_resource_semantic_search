import chromadb, os
from sentence_transformers import SentenceTransformer

def get_or_create_collection(db_path: str, collection_name: str):
    """
    Connects to ChromaDB and gets OR creates the collection.
    Used for Ingestion.
    """
    full_path = os.path.abspath(db_path)
    print(f"🔌 Connecting to DB (Ingest Mode): {collection_name} at {db_path}")
    client = chromadb.PersistentClient(path=full_path)
    return client.get_or_create_collection(collection_name)

def get_db_collection(db_path: str, collection_name: str):
    """
    Connects to an EXISTING collection.
    Used for Retrieval.
    """
    full_path = os.path.abspath(db_path)
    print(f"🔌 Connecting to DB (Read Mode): {collection_name} at {db_path}")
    client = chromadb.PersistentClient(path=full_path)
    try:
        return client.get_collection(collection_name)
    except ValueError:
        print(f"⚠️ Collection '{collection_name}' not found. Did you run ingestion?")
        raise

def load_embedding_model(model_name: str):
    print(f"🔄 Loading Embedding Model: {model_name}")
    return SentenceTransformer(model_name)