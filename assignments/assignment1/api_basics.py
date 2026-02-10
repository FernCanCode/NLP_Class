# Using OpenAI API
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI, APIConnectionError, RateLimitError, AuthenticationError, APITimeoutError, APIError

# Load environment variables from the root .env file
# This assumes the script is located in assignments/assignment1/ relative to the root
env_path = Path(__file__).resolve().parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("Warning: OPENAI_API_KEY not found in environment variables.")
    print("Please ensure you have a .env file in the project root with your API key.")
else:
    print(f"Successfully loaded API key: {api_key[:5]}... (masked)")

client = OpenAI(api_key=api_key)

def query_llm(prompt, model="gpt-5-nano", temperature=1.0, max_tokens=2000, retries=3, include_metrics=False, **kwargs):
    """
    Sends a prompt to the LLM and returns the response text.
    
    Args:
        prompt (str): The prompt to send to the LLM.
        model (str): The model to use. Defaults to "gpt-5-nano".
        temperature (float): Controls randomness (0-1). Defaults to 0.7.
        max_tokens (int): The maximum number of tokens to generate. Defaults to 100.
        retries (int): Number of retries for transient errors. Defaults to 3.
        include_metrics (bool): If True, returns a dictionary with content, token_usage, and execution_time.
        **kwargs: Additional arguments to pass to the API (e.g., top_p, frequency_penalty).
        
    Returns:
        str | dict: The generated response text, or a dictionary with metrics if include_metrics=True.
    """
    for attempt in range(retries):
        try:
            start_time = time.time()
            # Note: gpt-5-nano and newer models use max_completion_tokens instead of max_tokens
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_completion_tokens=max_tokens,
                **kwargs
            )
            end_time = time.time()
            content = response.choices[0].message.content
            
            if include_metrics:
                return {
                    "content": content,
                    "token_usage": response.usage.model_dump(),
                    "execution_time": end_time - start_time,
                    "model": response.model
                }
            return content
        except (RateLimitError, APIConnectionError, APITimeoutError) as e:
            if attempt < retries - 1:
                wait_time = 2 ** attempt
                print(f"Transient error occurred: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                msg = f"Error: Failed after {retries} attempts due to transient error: {str(e)}"
                if include_metrics: return {"content": msg, "error": str(e)}
                return msg
        except AuthenticationError:
            msg = "Error: Authentication failed. Please check your API key."
            if include_metrics: return {"content": msg, "error": "AuthenticationError"}
            return msg
        except APIError as e:
            msg = f"Error: An API error occurred: {str(e)}"
            if include_metrics: return {"content": msg, "error": str(e)}
            return msg
        except Exception as e:
            msg = f"Error: An unexpected error occurred: {str(e)}"
            if include_metrics: return {"content": msg, "error": str(e)}
            return msg
            if attempt < retries - 1:
                wait_time = 2 ** attempt
                print(f"Transient error occurred: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                return f"Error: Failed after {retries} attempts due to transient error: {str(e)}"
        except AuthenticationError:
            return "Error: Authentication failed. Please check your API key."
        except APIError as e:
            return f"Error: An API error occurred: {str(e)}"
        except Exception as e:
            return f"Error: An unexpected error occurred: {str(e)}"

def main():
    prompts = [
        "Explain the concept of NLP in one sentence.",
        "What are the three main components of a transformer architecture?",
        "Write a haiku about artificial intelligence."
    ]

    print("\n--- Starting LLM Query Demonstration ---\n")
    for i, prompt in enumerate(prompts, 1):
        print(f"Prompt {i}: {prompt}")
        response = query_llm(prompt)
        print(f"Response: {response}\n")
        print("-" * 50 + "\n")

if __name__ == "__main__":
    main()

