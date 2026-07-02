import chromadb
from chromadb.utils import embedding_functions

client = chromadb.PersistentClient(path="./data/chroma_db")

embedding_fn = embedding_functions. SentenceTransformerEmbeddingFunction(model_name = "all-MiniLM-L6-v2")

# collection -vin chromadb is like a table in databse 
collection = client.get_or_create_collection(name="documents")
def add_chunks(doc_id: str, chunks: list[str]):
    """
    Adds a list of text chunks to the vector database collection.
    
    Args:
        doc_id (str): Unique identifier for the original document 
                     (e.g., filename or document name).
        chunks (list[str]): List of text chunks to be stored.
    
    Returns:
        int: Number of chunks successfully added.
    """
    
    # Create unique IDs for each chunk by combining document ID with chunk index.
    # Example: "report.pdf 0", "report.pdf 1", "report.pdf 2", ...
    # This ensures no duplicate IDs when adding multiple documents.
    ids = [f"{doc_id} {i}" for i in range(len(chunks))]
    
    # Create metadata for each chunk.
    # This metadata helps with filtering and traceability later.
    metadatas = [
        {
            "source": doc_id,           # Which document this chunk came from
            "chunk_index": i            # Position of this chunk in the original document
        }
        for i in range(len(chunks))
    ]
    
    # Add the chunks to the ChromaDB (or similar vector store) collection.
    # - documents: The actual text content of each chunk
    # - metadatas: Additional information stored with each embedding
    # - ids: Unique identifiers for each entry
    collection.add(
        documents=chunks, 
        metadatas=metadatas, 
        ids=ids
    )
    
    # Return the number of chunks added (useful for logging or confirmation)
    return len(chunks)

# search chunks ()- finding the close chunks to a question
def search_chunks(query: str, top_k: int = 4):
    return collection.query(query_texts=[query], n_results=3)