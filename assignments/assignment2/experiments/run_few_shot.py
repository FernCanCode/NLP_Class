import json
import os
import re
from dotenv import load_dotenv
from openai import OpenAI
from collections import Counter

load_dotenv()

def parse_verdict(text):
    if not text:
        return None
    text = text.lower()
    # Simple regex to catch standard formats, accounting for asterisks
    if "final answer: yes" in text or "verdict: yes" in text or "final verdict**: yes" in text or "final answer**: yes" in text:
        return "Yes"
    elif "final answer: no" in text or "verdict: no" in text or "final verdict**: no" in text or "final answer**: no" in text:
        return "No"
    else:
        return None

def main():
    # Load configuration
    with open("config/agent_config.json", "r") as f:
        config = json.load(f)
        
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    model_name = config.get("model_name", "gpt-5-nano")
    max_tokens = config.get("max_tokens", 1500)
    
    # Load dataset
    with open("data/strategy_qa_sample.json", "r") as f:
        full_dataset = json.load(f)
        
    # We will use the last 5 entries as our few-shot examples
    few_shot_examples = full_dataset[-5:]
    
    # We will use exactly 30 entries to generate statistical data quickly (same subset as zero-shot)
    dataset = full_dataset[:30]
    
    results = []
    
    # Build the few-shot examples context
    few_shot_context = ""
    for ex in few_shot_examples:
        ans_str = "Yes" if ex["answer"] is True else "No"
        few_shot_context += f"Question: {ex['question']}\n"
        few_shot_context += f"[Step-by-step reasoning goes here]\n"
        few_shot_context += f"Final answer: {ans_str}\n\n"

    system_prompt = (
        "You are an expert answering complex questions. Use a step-by-step logical analysis "
        "(Chain-of-Thought) to arrive at your answer. Based on your reasoning, you must definitively conclude with exactly "
        "'Final answer: Yes' or 'Final answer: No'.\n\n"
        "Here are some examples of the expected format:\n\n"
        f"{few_shot_context}"
    )
    
    # N=1 for Direct QA (baseline), N=20 for Self-Consistency
    direct_qa_N = 1
    self_consistency_N = 20
    
    print(f"Starting Few-Shot Evaluations on {len(dataset)} questions...")
    
    for i, entry in enumerate(dataset):
        question = entry["question"]
        ground_truth = entry["answer"]
        print(f"Processing {i+1}/{len(dataset)}: {question}")
        
        # 1. Direct QA Baseline
        direct_verdicts = []
        direct_texts = []
        for _ in range(direct_qa_N):
            try:
                direct_response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question}
                    ],
                    max_completion_tokens=max_tokens
                )
                direct_text = direct_response.choices[0].message.content.strip()
                direct_texts.append(direct_text)
                direct_verdicts.append(parse_verdict(direct_text))
            except Exception as e:
                print(f"Error in Direct QA: {e}")

        valid_direct_votes = [v for v in direct_verdicts if v in ["Yes", "No"]]
        if valid_direct_votes:
            counter = Counter(valid_direct_votes)
            direct_majority_verdict = counter.most_common(1)[0][0]
        else:
            direct_majority_verdict = None

        # 2. Self-Consistency Baseline
        sc_verdicts = []
        sc_texts = []
        for _ in range(self_consistency_N):
            try:
                sc_resp = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question}
                    ],
                    max_completion_tokens=max_tokens
                )
                sc_text = sc_resp.choices[0].message.content.strip()
                sc_texts.append(sc_text)
                sc_verdicts.append(parse_verdict(sc_text))
            except Exception as e:
                print(f"Error in SC: {e}")
                
        # Majority vote
        valid_votes = [v for v in sc_verdicts if v in ["Yes", "No"]]
        if valid_votes:
            counter = Counter(valid_votes)
            majority_verdict = counter.most_common(1)[0][0]
        else:
            majority_verdict = None
            
        result_payload = {
            "question": question,
            "ground_truth": ground_truth,
            "direct_qa": {
                "n_samples": direct_qa_N,
                "outputs": direct_texts,
                "verdicts": direct_verdicts,
                "majority_verdict": direct_majority_verdict
            },
            "self_consistency": {
                "n_samples": self_consistency_N,
                "outputs": sc_texts,
                "verdicts": sc_verdicts,
                "majority_verdict": majority_verdict
            }
        }
        
        results.append(result_payload)
        
        # Save incrementally
        with open("data/few_shot_results.json", "w") as f:
            json.dump(results, f, indent=4)
            
    print("\nFew-Shot experiments completed. Results saved to data/few_shot_results.json")

if __name__ == "__main__":
    main()
