import json
import os
import sys
from dotenv import load_dotenv

# Add src to python path for orchestrator import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from orchestrator import DebateManager
from agents import DebateAgent

load_dotenv()

def main():
    # Load configuration
    with open("config/agent_config.json", "r") as f:
        config = json.load(f)
        
    model_name = config.get("model_name", "gpt-5-nano")
    
    # Load dataset - use the same 30 question subset as the baselines
    with open("data/strategy_qa_sample.json", "r") as f:
        full_dataset = json.load(f)
    dataset = full_dataset[:30]
    
    ablation_results = {}
    
    # Test scaling from 1, to 3, to 5 max rounds
    rounds_to_test = [1, 3, 5]
    
    print(f"Starting Ablation Study on {len(dataset)} questions with rounds: {rounds_to_test}")
    
    for r in rounds_to_test:
        print(f"\n======================================")
        print(f"RUNNING EVALUATION WITH MAX_ROUNDS = {r}")
        print(f"======================================")
        
        round_results = []
        
        for i, entry in enumerate(dataset):
            question = entry["question"]
            ground_truth = entry["answer"]
            print(f"[{r} rounds] Processing {i+1}/{len(dataset)}")
            
            proponent = DebateAgent(
                role_name="Proponent",
                model_name=model_name,
                system_prompt_path="prompts/proponent_v1.txt",
                max_tokens=config["max_tokens"],
                temperature=config["temperature"]
            )
            
            opponent = DebateAgent(
                role_name="Opponent",
                model_name=model_name,
                system_prompt_path="prompts/opponent_v1.txt",
                max_tokens=config["max_tokens"],
                temperature=config["temperature"]
            )
            
            judge = DebateAgent(
                role_name="Judge",
                model_name=model_name,
                system_prompt_path="prompts/judge.txt",
                max_tokens=config["max_tokens"],
                temperature=config["judge_temperature"]
            )
            
            manager = DebateManager(proponent, opponent, judge)
            
            try:
                judge_output = manager.run_full_debate(question, max_rounds=r)
                verdict = manager.parse_verdict(judge_output)
                
                result_payload = {
                    "question": question,
                    "ground_truth": ground_truth,
                    "max_rounds_setting": r,
                    "actual_rounds_taken": len(manager.full_transcript) // 2,
                    "extracted_verdict": verdict
                }
                round_results.append(result_payload)
            except Exception as e:
                print(f"Error evaluating question: {e}")
                
        ablation_results[r] = round_results
        
        # Save incrementally
        with open("data/ablation_results.json", "w") as f:
            json.dump(ablation_results, f, indent=4)
            
    print("\nAblation study completed. Results saved to data/ablation_results.json")

if __name__ == "__main__":
    main()
