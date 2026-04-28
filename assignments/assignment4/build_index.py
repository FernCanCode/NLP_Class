import sys
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import chromadb

# 1. Import constants from config.py
from config import (
    EMBEDDING_MODEL_NAME,
    CHROMA_PATH,
    COLLECTION_NAME,
    STARTER_FUNCTION_LIMIT,
)

def main():
    # 2. Load CodeSearchNet
    print("Dataset loading started...")
    try:
        dataset = load_dataset(
            "code_search_net",
            "python",
            split="train",
            streaming=True
        )
    except Exception as e:
        print(f"Error: Failed to load dataset: {e}")
        sys.exit(1)

    # 3. Load the embedding model
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # 4 & 5. Create Chroma persistent client and reset/create collection
    print(f"Initializing Chroma persistent client at '{CHROMA_PATH}'...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    print(f"Resetting and creating collection: '{COLLECTION_NAME}'...")
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass  # Collection might not exist yet, which is entirely expected
        
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    # 6. Iterate through CodeSearchNet and collect valid examples
    valid_examples = []
    
    print("Scanning for valid examples...")
    for row in dataset:
        # Check if all required fields exist and are not None or empty
        if (row.get("func_code_string") and 
            row.get("func_documentation_string") and
            row.get("func_name") and
            row.get("repository_name") and
            row.get("func_path_in_repository")):
            
            valid_examples.append(row)
            
        if len(valid_examples) >= STARTER_FUNCTION_LIMIT:
            break
            
    if len(valid_examples) < STARTER_FUNCTION_LIMIT:
        print(f"Warning: Only found {len(valid_examples)} valid examples, "
              f"which is fewer than the requested {STARTER_FUNCTION_LIMIT}.")
        
    print(f"Number of valid examples collected: {len(valid_examples)}")

    # 7-9. Embed and add elements to the collection in batches
    BATCH_SIZE = 128
    
    for i in range(0, len(valid_examples), BATCH_SIZE):
        batch = valid_examples[i:i + BATCH_SIZE]
        
        documents = []
        metadatas = []
        ids = []
        
        for j, row in enumerate(batch):
            # Format the document text
            doc_text = (
                f"Documentation:\n{row['func_documentation_string']}\n\n"
                f"Code:\n{row['func_code_string']}"
            )
            documents.append(doc_text)
            
            # Format the metadata dictionary
            metadata = {
                "source": "starter",
                "func_name": row["func_name"],
                "repo": row["repository_name"],
                "path": row["func_path_in_repository"]
            }
            metadatas.append(metadata)
            
            # Format the id string
            global_idx = i + j
            ids.append(f"starter_{global_idx}")
            
        # Generate embeddings
        print(f"Encoding batch of {len(batch)} documents (IDs {i} to {i + len(batch) - 1})...")
        embeddings = model.encode(documents, show_progress_bar=True).tolist()
        
        # Insert into ChromaDB
        print(f"Inserting batch into Chroma collection...")
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

    # 10. Print use final progress information
    final_count = collection.count()
    print(f"\nIndex built successfully! Final collection count: {final_count}")

if __name__ == "__main__":
    main()
