import ast
import os
import chromadb
from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_MODEL_NAME,
    CHROMA_PATH,
    COLLECTION_NAME,
    CUSTOM_FUNCTIONS_PATH
)


def read_source_file(path: str) -> str:
    """Read the source code file as a plain string."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Error: Required file '{path}' does not exist.")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def extract_functions(source_code: str) -> list[dict]:
    """Parse the source code and extract top-level ast.FunctionDef nodes."""
    parsed_ast = ast.parse(source_code)
    
    extracted = []
    for node in parsed_ast.body:
        # Ignore class methods or test blocks; only get top-level Python functions
        if isinstance(node, ast.FunctionDef):
            func_name = node.name
            source = ast.get_source_segment(source_code, node)
            docstring = ast.get_docstring(node) or ""
            
            extracted.append({
                "func_name": func_name,
                "source_code": source,
                "docstring": docstring
            })
            
    return extracted


def build_document(function_info: dict) -> str:
    """Format the raw node information into the expected Document standard for Semantic Retrieval."""
    return (
        f"Documentation:\n{function_info['docstring']}\n\n"
        f"Code:\n{function_info['source_code']}"
    )


def get_collection():
    """Attempt to load the existing ChromaDB connection."""
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = client.get_collection(name=COLLECTION_NAME)
        return collection
    except Exception as e:
        raise RuntimeError(
            f"Error: Could not load the Chroma collection '{COLLECTION_NAME}' at '{CHROMA_PATH}'. "
            "Please run 'python build_index.py' first to build the index."
        ) from e


def delete_existing_custom_items(collection):
    """
    Ensure idempotency by clearing out any previous runs of custom data using our distinct metadata filter.
    """
    existing_items = collection.get(where={"source": "custom"})
    ids_to_delete = existing_items.get("ids", [])
    
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
        print(f"Deleted {len(ids_to_delete)} existing 'custom' items from the collection to maintain reproducibility.")
    else:
        print("No existing custom items found to delete.")


def main():
    print("Custom function indexing started...\n")
    
    # 1. Read source
    print(f"Reading file '{CUSTOM_FUNCTIONS_PATH}'...")
    source_code = read_source_file(CUSTOM_FUNCTIONS_PATH)
    
    # 2. Extract components purely using native AST
    extracted_funcs = extract_functions(source_code)
    
    func_names = [f["func_name"] for f in extracted_funcs]
    print(f"Extracted {len(extracted_funcs)} functions: {', '.join(func_names)}")
    
    if len(extracted_funcs) < 5:
        print("Warning: Fewer than 5 custom functions were extracted.")
        
    # 3. Load embedding model and db instance
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    print(f"Loading ChromaDB collection from '{CHROMA_PATH}'...")
    collection = get_collection()
    
    initial_count = collection.count()
    print(f"Collection count before insertion: {initial_count}")
    
    # 4. Filter state to prevent overlapping inserts
    delete_existing_custom_items(collection)
    
    # 5. Extract & Template
    documents = []
    ids = []
    metadatas = []
    
    print("\nProcessing newly extracted custom functions...")
    for func in extracted_funcs:
        doc_text = build_document(func)
        documents.append(doc_text)
        
        ids.append(f"custom_{func['func_name']}")
        
        metadatas.append({
            "source": "custom",
            "func_name": func['func_name'],
            "repo": "local",
            "path": CUSTOM_FUNCTIONS_PATH
        })
        
    # 6. Encode batches & push
    print("Encoding function documents...")
    embeddings = model.encode(documents, show_progress_bar=True).tolist()
    
    print("Adding documents to Chroma...")
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )
    
    # 7. Print Output info
    final_count = collection.count()
    print(f"\nInsertion complete! Final collection count: {final_count}")


if __name__ == "__main__":
    main()
