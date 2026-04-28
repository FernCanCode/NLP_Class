import os
import anthropic
from dotenv import load_dotenv

# 1 & 2. Load configurations and constants
from config import GENERATOR_MODEL_NAME
from retriever import retrieve, format_source

# 4. Load environment variables
# Note: This is safe at module level since we won't crash if the key is missing;
# we check explicitly inside generate_answer when it's absolutely needed.
load_dotenv()

# 5. Implement truncate_text
def truncate_text(text: str, max_chars: int = 2500) -> str:
    """
    Truncates the string to a maximum allowed length to save on context size and API costs.
    """
    if not text or len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[TRUNCATED_DUE_TO_LENGTH]..."


# 6. Implement build_context
def build_context(retrieved_chunks: list[dict], max_chars_per_chunk: int = 2500) -> str:
    """
    Formats all retrieved chunks into one large context string, truncating each piece of code.
    """
    context_str = ""
    for idx, chunk in enumerate(retrieved_chunks):
        rank = chunk.get("rank", idx + 1)
        similarity = chunk.get("similarity", "N/A")
        source = format_source(chunk.get("metadata", {}))
        
        doc_text = chunk.get("document", "")
        doc_truncated = truncate_text(doc_text, max_chars_per_chunk)
        
        context_str += f"[{rank}] source: {source}\n"
        context_str += f"similarity: {similarity}\n"
        context_str += f"code context:\n{doc_truncated}\n\n"
        
    return context_str.strip()


# 7. Implement build_prompt
def build_prompt(query: str, retrieved_chunks: list[dict]) -> str:
    """
    Assembles the grounded prompt for the LLM using the retrieved context.
    """
    context_str = build_context(retrieved_chunks)
    
    prompt = f"""You are a helpful programming assistant answering a question about Python code.
Use ONLY the following provided code context to answer the question. If the provided context does not contain the answer or does not support it, simply state that you do not know based on the retrieved code.
You must cite your sources using the format `repo/path::func_name` when referencing the code snippets.
Keep the answer concise and direct.

Provided code context:
{context_str}

User Question:
{query}
"""
    return prompt


# 8. Implement generate_answer
def generate_answer(query: str, retrieved_chunks: list[dict]) -> str:
    """
    Validates input, calls the Claude API with the generated prompt, and returns the response.
    """
    if not query or not query.strip():
        raise ValueError("Error: query cannot be empty.")
    
    if not retrieved_chunks:
        raise ValueError("Error: retrieved_chunks cannot be empty; no context to ground the answer.")
        
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Error: ANTHROPIC_API_KEY is not set in the environment or .env file.")
        
    client = anthropic.Anthropic(api_key=api_key)
    prompt = build_prompt(query, retrieved_chunks)
    
    try:
        response = client.messages.create(
            model=GENERATOR_MODEL_NAME,
            max_tokens=500,
            temperature=0,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text
    except Exception as e:
        raise RuntimeError(
            f"Error: API call to Anthropic failed: {e}\n"
            "Check that GENERATOR_MODEL_NAME in config.py is a currently available Anthropic model for your account."
        ) from e


# 9. Implement answer_query
def answer_query(query: str, top_k: int = 4) -> dict:
    """
    A single wrapper function that retrieves chunks and generates a grounded response.
    """
    retrieved_chunks = retrieve(query, top_k=top_k)
    answer = generate_answer(query, retrieved_chunks)
    
    return {
        "query": query,
        "retrieved_chunks": retrieved_chunks,
        "answer": answer
    }


# 10. Implement CLI test mode
if __name__ == "__main__":
    test_query = "How can I parse or process a URL in Python?"
    print(f"Testing Generator with query: '{test_query}'\n")
    
    try:
        result = answer_query(test_query)
        
        print("=== Test Query ===")
        print(result["query"])
        print("\n=== Retrieved Sources ===")
        for chunk in result["retrieved_chunks"]:
            print(format_source(chunk.get("metadata", {})))
            
        print("\n=== Generated Answer ===")
        print(result["answer"])
        print()
    except Exception as e:
        print(f"Error during generator testing: {e}")
