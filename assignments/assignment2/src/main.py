import json
import os
from dotenv import load_dotenv
from src.agents import DebateAgent
from src.orchestrator import DebateManager

load_dotenv()

def main():
    # Load configuration
    with open("config/agent_config.json", "r") as f:
        config = json.load(f)

    # 1. Initialize Debate Agents
    proponent = DebateAgent(
        role_name="Proponent",
        model_name=config.get("model_name", "gpt-5-nano"),
        system_prompt_path="prompts/proponent_v1.txt",
        max_tokens=config["max_tokens"],
        temperature=config["temperature"]
    )

    opponent = DebateAgent(
        role_name="Opponent",
        model_name=config.get("model_name", "gpt-5-nano"),
        system_prompt_path="prompts/opponent_v1.txt",
        max_tokens=config["max_tokens"],
        temperature=config["temperature"]
    )

    judge = DebateAgent(
        role_name="Judge",
        model_name=config.get("model_name", "gpt-5-nano"),
        system_prompt_path="prompts/judge.txt",
        max_tokens=config["max_tokens"],
        temperature=config["judge_temperature"]
    )

    # 2. Initialize the Orchestrator
    manager = DebateManager(proponent, opponent, judge)

    # 3. Load StrategyQA Dataset
    with open("data/strategy_qa_sample.json", "r") as f:
        dataset = json.load(f)
    
    # 4. Run the Pipeline
    results = []
    for entry in dataset:
        question = entry["question"]
        ground_truth = entry["answer"]

        print(f"\nEvaluating Question: {question}")

        # Execute the 3-phase process
        # Phase 1: Initial Check
        # Phase 2: Multi-Round Debate
        # Phase 3: Final Judgment

        max_rounds = config.get("max_rounds", 3)
        final_verdict_text = manager.run_full_debate(question, max_rounds=max_rounds)
        confidence_score = manager.extract_confidence(final_verdict_text)
        extracted_verdict = manager.parse_verdict(final_verdict_text)

        # 5. Log results for Phase 4 Evaluation
        results_payload={
            "question": question,
            "ground_truth": ground_truth,
            "judge_output": final_verdict_text,
            "extracted_verdict": extracted_verdict,
            "confidence_score": confidence_score,
            "transcript": manager.full_transcript
        }

        results.append(results_payload)

        # Reset manager for next question
        manager.reset()
    
        # 6. Export to JSON incrementally
        with open("data/results_log.json", "w") as f:
            json.dump(results, f, indent=4)
        
    print("\nDebate cycle complete. Results saved to data/results_log.json")

if __name__ == "__main__":
    main()