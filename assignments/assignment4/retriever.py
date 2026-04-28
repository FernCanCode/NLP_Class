import sys
import chromadb
from sentence_transformers import SentenceTransformer

# 1. Import needed constants from config.py
from config import (
    EMBEDDING_MODEL_NAME,
    CHROMA_PATH,
    COLLECTION_NAME,
    TOP_K,
)

# 3. Implement get_collection
def get_collection():
    """
    Creates a ChromaDB PersistentClient and loads the existing collection.
    """
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = client.get_collection(name=COLLECTION_NAME)
        return collection
    except Exception as e:
        raise RuntimeError(
            f"Error: Could not load the Chroma collection '{COLLECTION_NAME}' at '{CHROMA_PATH}'. "
            "Please run 'python build_index.py' first to build the index."
        ) from e


# 4. Implement embed_query
def embed_query(query: str, model: SentenceTransformer) -> list[float]:
    """
    Embeds a natural language query into a list of floats using the provided model.
    """
    # model.encode returns a numpy array, .tolist() converts it to a plain Python list
    embedding = model.encode(query).tolist()
    return embedding


# 6. Implement format_source (Defining before retrieve so it's logically above its use, though it can go anywhere)
def format_source(metadata: dict) -> str:
    """
    Formats the metadata into a citation string: repo/path::func_name
    """
    if not metadata:
        return "unknown_repo/unknown_path::unknown_func"
        
    repo = metadata.get("repo", "unknown_repo")
    path = metadata.get("path", "unknown_path")
    func_name = metadata.get("func_name", "unknown_func")
    
    return f"{repo}/{path}::{func_name}"


# 5. Implement the main retriever function
def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    Validates query, embeds it, queries Chroma, and returns top_k results.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")
        
    query = query.strip()
    
    # Load SentenceTransformer model
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    # Load Chroma collection
    collection = get_collection()
    
    # Embed query
    query_embedding = embed_query(query, model)
    
    # Query the collection
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    # Parse results
    retrieved_chunks = []
    
    # Chroma query results wrap the output in an external list because it handles batch queries natively.
    # Since we only query 1 embedding, we access index 0.
    if not results or not results.get("ids") or len(results["ids"]) == 0 or len(results["ids"][0]) == 0:
        return retrieved_chunks # return empty if nothing found
        
    for i in range(len(results["ids"][0])):
        # raw Chroma distance
        distance = results["distances"][0][i] if "distances" in results and results["distances"] else None
        
        # Calculate derived similarity if distance is available (cosine space: similarity = 1 - distance)
        similarity = 1.0 - distance if distance is not None else None
        
        doc_text = results["documents"][0][i] if "documents" in results and results["documents"] else ""
        metadata = results["metadatas"][0][i] if "metadatas" in results and results["metadatas"] else {}
        
        chunkinfo = {
            "rank": i + 1,
            "document": doc_text,
            "metadata": metadata,
            "distance": distance,
            "similarity": similarity
        }
        retrieved_chunks.append(chunkinfo)
        
    return retrieved_chunks


# 7. Implement a small CLI test mode
if __name__ == "__main__":
    test_query = "How can I parse or process a URL in Python?"
    print(f"Testing Retriever with query: '{test_query}'\n")
    
    try:
        top_results = retrieve(test_query)
        
        for result in top_results:
            source_citation = format_source(result['metadata'])
            
            print(f"=== Rank {result['rank']} ===")
            print(f"Source: {source_citation}")
            print(f"Distance: {result['distance']} | Similarity: {result['similarity']}")
            
            # Print first 500 characters of the document
            doc_snippet = result['document'][:500]
            if len(result['document']) > 500:
                doc_snippet += "..."
                
            print(f"Document snippet:\n{doc_snippet}\n")
            
    except Exception as e:
        print(f"Error during retrieval: {e}")
