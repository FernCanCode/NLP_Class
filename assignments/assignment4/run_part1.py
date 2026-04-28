import os
import pandas as pd

# 2. Import config
from config import PART1_RESULTS_PATH, TOP_K

# 3. Import generator logic
from generator import answer_query

# 4. Import retriever source formatter
from retriever import format_source

# 5. Define baseline queries
BASELINE_QUERIES = [
    {"query_id": "P1Q01", "query_text": "How can I parse or process a URL in Python?"},
    {"query_id": "P1Q02", "query_text": "How can I read or open a file in Python?"},
    {"query_id": "P1Q03", "query_text": "How can I convert bytes into a string in Python?"},
    {"query_id": "P1Q04", "query_text": "How can I serialize or convert data to JSON in Python?"},
    {"query_id": "P1Q05", "query_text": "How can I validate or check input arguments before using them?"},
    {"query_id": "P1Q06", "query_text": "How can I retry an operation after a failure?"},
    {"query_id": "P1Q07", "query_text": "How can I extract a value from a nested data structure?"},
    {"query_id": "P1Q08", "query_text": "How can I handle or inspect HTTP headers in Python?"},
    {"query_id": "P1Q09", "query_text": "How can I process command line arguments in Python?"},
    {"query_id": "P1Q10", "query_text": "How can I create or format a timestamp in Python?"},
]

# 6. Implement first_two_sentences
def first_two_sentences(text: str) -> str:
    """
    Returns the first two sentences of a string by naively splitting on '. '.
    """
    if not text:
        return ""
    sentences = text.split(". ")
    if len(sentences) <= 2:
        return text
    # Re-join the first two, and add back the period that got split off
    return sentences[0] + ". " + sentences[1] + "."

# 7. Implement sources_to_string
def sources_to_string(retrieved_chunks: list[dict]) -> str:
    """
    Formats the retrieved chunks into a semicolon-separated string of citations.
    Example: [1] repo/path::func_name; [2] repo/path::func_name
    """
    sources_str = []
    for chunk in retrieved_chunks:
        rank = chunk.get("rank", "?")
        metadata = chunk.get("metadata", {})
        source_cite = format_source(metadata)
        sources_str.append(f"[{rank}] {source_cite}")
    return "; ".join(sources_str)

# 8. Implement similarities_to_string
def similarities_to_string(retrieved_chunks: list[dict]) -> str:
    """
    Formats the similarities into a semicolon-separated string.
    Example: [1] 0.612; [2] 0.581
    """
    sims_str = []
    for chunk in retrieved_chunks:
        rank = chunk.get("rank", "?")
        similarity = chunk.get("similarity")
        if similarity is not None:
            sims_str.append(f"[{rank}] {similarity:.3f}")
        else:
            sims_str.append(f"[{rank}] N/A")
    return "; ".join(sims_str)

# 9. Implement judge_grounded
def judge_grounded(answer: str, retrieved_chunks: list[dict]) -> str:
    """
    A simple heuristic to manually determine if the response explicitly cites 
    one of the retrieved chunks.
    """
    if not answer or not retrieved_chunks:
        return "no"
        
    # Check if any format_source representation appears exactly in the answer string
    for chunk in retrieved_chunks:
        metadata = chunk.get("metadata", {})
        source_cite = format_source(metadata)
        if source_cite in answer:
            return "yes"
            
    return "no"

# 10. Implement main
def main():
    print("Starting Part 1 Baseline Queries evaluation...\n")
    results = []
    
    for q_idx, query_obj in enumerate(BASELINE_QUERIES):
        q_id = query_obj["query_id"]
        q_text = query_obj["query_text"]
        
        print(f"Running {q_id}: {q_text}")
        
        try:
            # Generate the answer and retrieve the sources chunks
            response = answer_query(q_text, top_k=TOP_K)
            chunks = response["retrieved_chunks"]
            answer = response["answer"]
            
            # Format and gather table columns
            top_k_sources = sources_to_string(chunks)
            similarity_scores = similarities_to_string(chunks)
            gen_abstract = first_two_sentences(answer)
            grounded = judge_grounded(answer, chunks)
            
            row = {
                "query_id": q_id,
                "query_text": q_text,
                "top_k_sources": top_k_sources,
                "similarity_scores": similarity_scores,
                "generated_answer_first_two_sentences": gen_abstract,
                "grounded": grounded,
                "full_generated_answer": answer
            }
            results.append(row)
            
        except Exception as e:
            print(f"  -> Error encountered: {e}")
            row = {
                "query_id": q_id,
                "query_text": q_text,
                "top_k_sources": "",
                "similarity_scores": "",
                "generated_answer_first_two_sentences": "",
                "grounded": "error",
                "full_generated_answer": f"Error: {str(e)}"
            }
            results.append(row)
            
    # 11 & 12. Create directory and save DF
    results_dir = os.path.dirname(PART1_RESULTS_PATH)
    if results_dir and not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)
        
    df = pd.DataFrame(results)
    df.to_csv(PART1_RESULTS_PATH, index=False)
    
    # 13. Output print metrics
    print(f"\nExecution complete. Processed {len(results)} baseline queries.")
    print(f"Results successfully saved to {PART1_RESULTS_PATH}.")

# 14. Include main guard
if __name__ == "__main__":
    main()
