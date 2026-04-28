import os
import pandas as pd

# 2. Import config
from config import PART2_RESULTS_PATH, TOP_K

# 3. Import generator logic
from generator import answer_query

# 4. Import retriever source formatter
from retriever import format_source


# 5. Define queries
PART2_QUERIES = [
    # Targeted
    {"query_id": "P2TQ01", "query_type": "targeted", "query_text": "How do I clean text by lowercasing and removing punctuation?"},
    {"query_id": "P2TQ02", "query_type": "targeted", "query_text": "How do I count word frequencies in a string?"},
    {"query_id": "P2TQ03", "query_type": "targeted", "query_text": "How can I extract hashtags from social media text?"},
    {"query_id": "P2TQ04", "query_type": "targeted", "query_text": "How can I classify simple positive or negative sentiment using word lists?"},
    {"query_id": "P2TQ05", "query_type": "targeted", "query_text": "How do I split a paragraph into sentences?"},
    
    # Cross-corpus
    {"query_id": "P2CQ01", "query_type": "cross_corpus", "query_text": "How do I process user-provided text before analysis?"},
    {"query_id": "P2CQ02", "query_type": "cross_corpus", "query_text": "How can I count meaningful tokens or items in Python?"},
    {"query_id": "P2CQ03", "query_type": "cross_corpus", "query_text": "How can I extract useful pieces of information from strings?"},
    {"query_id": "P2CQ04", "query_type": "cross_corpus", "query_text": "How can I classify text using simple rules?"},
    {"query_id": "P2CQ05", "query_type": "cross_corpus", "query_text": "How can I break text into smaller units for downstream processing?"},
]


# 6. Reimplement helpers
def first_two_sentences(text: str) -> str:
    """Returns the first two sentences of a string by naively splitting on '. '."""
    if not text:
        return ""
    sentences = text.split(". ")
    if len(sentences) <= 2:
        return text
    return sentences[0] + ". " + sentences[1] + "."

def sources_to_string(retrieved_chunks: list[dict]) -> str:
    """Formats the retrieved chunks into a semicolon-separated string of citations."""
    sources_str = []
    for chunk in retrieved_chunks:
        rank = chunk.get("rank", "?")
        metadata = chunk.get("metadata", {})
        source_cite = format_source(metadata)
        sources_str.append(f"[{rank}] {source_cite}")
    return "; ".join(sources_str)

def similarities_to_string(retrieved_chunks: list[dict]) -> str:
    """Formats the similarities into a semicolon-separated string."""
    sims_str = []
    for chunk in retrieved_chunks:
        rank = chunk.get("rank", "?")
        similarity = chunk.get("similarity")
        if similarity is not None:
            sims_str.append(f"[{rank}] {similarity:.3f}")
        else:
            sims_str.append(f"[{rank}] N/A")
    return "; ".join(sims_str)

def judge_grounded(answer: str, retrieved_chunks: list[dict]) -> str:
    """Simple heuristic to determine if the response explicitly cites one of the retrieved chunks."""
    if not answer or not retrieved_chunks:
        return "no"
    for chunk in retrieved_chunks:
        metadata = chunk.get("metadata", {})
        source_cite = format_source(metadata)
        if source_cite in answer:
            return "yes"
    return "no"


# 7. Implement corpus mix label logic
def corpus_mix_label(retrieved_chunks: list[dict]) -> str:
    """Classifies the overall source mix of the retrieved results based on metadata matching."""
    if not retrieved_chunks:
        return "unknown"
        
    has_custom = False
    has_starter = False
    
    for chunk in retrieved_chunks:
        src = chunk.get("metadata", {}).get("source", "")
        if src == "custom":
            has_custom = True
        elif src == "starter":
            has_starter = True
            
    if has_custom and has_starter:
        return "both"
    elif has_custom:
        return "custom_only"
    elif has_starter:
        return "starter_only"
    else:
        return "unknown"
        

# 8. Implement custom function identifier
def retrieved_custom_functions(retrieved_chunks: list[dict]) -> str:
    """Finds any custom function names among the retrieved chunks."""
    custom_funcs = []
    for chunk in retrieved_chunks:
        meta = chunk.get("metadata", {})
        if meta.get("source") == "custom":
            func_name = meta.get("func_name", "unknown_func")
            custom_funcs.append(func_name)
            
    return "; ".join(custom_funcs)


# 9. Implement main
def main():
    print("Starting Part 2 Queries evaluation...\n")
    results = []
    
    for q_idx, query_obj in enumerate(PART2_QUERIES):
        q_id = query_obj["query_id"]
        q_type = query_obj["query_type"]
        q_text = query_obj["query_text"]
        
        print(f"Running {q_id} {q_type}: {q_text}")
        
        try:
            # Query LLM and Retrieval
            response = answer_query(q_text, top_k=TOP_K)
            chunks = response["retrieved_chunks"]
            answer = response["answer"]
            
            # Map Table Formats
            row = {
                "query_id": q_id,
                "query_type": q_type,
                "query_text": q_text,
                "top_k_sources": sources_to_string(chunks),
                "similarity_scores": similarities_to_string(chunks),
                "corpus_mix": corpus_mix_label(chunks),
                "retrieved_custom_functions": retrieved_custom_functions(chunks),
                "generated_answer_first_two_sentences": first_two_sentences(answer),
                "grounded": judge_grounded(answer, chunks),
                "full_generated_answer": answer
            }
            results.append(row)
            
        except Exception as e:
            print(f"  -> Error encountered: {e}")
            row = {
                "query_id": q_id,
                "query_type": q_type,
                "query_text": q_text,
                "top_k_sources": "",
                "similarity_scores": "",
                "corpus_mix": "error",
                "retrieved_custom_functions": "",
                "generated_answer_first_two_sentences": "",
                "grounded": "error",
                "full_generated_answer": f"Error: {str(e)}"
            }
            results.append(row)
            
    # 10 & 11. Create directory and save DF
    results_dir = os.path.dirname(PART2_RESULTS_PATH)
    if results_dir and not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)
        
    df = pd.DataFrame(results)
    df.to_csv(PART2_RESULTS_PATH, index=False)
    
    # 12. Output print metrics
    print(f"\nExecution complete. Processed {len(df)} Part 2 queries.")
    print(f"Results successfully saved to {PART2_RESULTS_PATH}.\n")
    
    print("Corpus Mix Summary:")
    if "corpus_mix" in df.columns:
        print(df["corpus_mix"].value_counts().to_string())


if __name__ == "__main__":
    main()
