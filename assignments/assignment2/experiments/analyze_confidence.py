import json
import matplotlib.pyplot as plt
import os
import re

def load_json(filepath):
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found.")
        return []
    with open(filepath, 'r') as f:
        return json.load(f)

def normalize_bool_to_yes_no(val):
    if val is True: return "Yes"
    if val is False: return "No"
    if isinstance(val, str):
        if val.lower() == 'true': return "Yes"
        if val.lower() == 'false': return "No"
    return str(val).capitalize()

def parse_confidence(text):
    if not text: return None
    # Looking for a standalone number usually 1-5
    match = re.search(r'Confidence Score:\s*(\d)', text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None

def main():
    # Load the big 150-question run
    debate_results = load_json("data/results_log.json")
    
    if not debate_results:
        print("No debate results found.")
        return
        
    print(f"Loaded {len(debate_results)} debate records.")
    
    confidence_stats = {
        1: {"correct": 0, "total": 0},
        2: {"correct": 0, "total": 0},
        3: {"correct": 0, "total": 0},
        4: {"correct": 0, "total": 0},
        5: {"correct": 0, "total": 0}
    }
    
    unparsed_confidence = 0
    unparsed_verdict = 0

    for item in debate_results:
        gt = normalize_bool_to_yes_no(item["ground_truth"])
        verdict = item.get("extracted_verdict")
        
        if not verdict:
            unparsed_verdict += 1
            continue
            
        conf = item.get("confidence_score")
        if not conf and item.get("judge_output"):
            conf = parse_confidence(item["judge_output"])
            
        try:
            conf = int(conf)
            if conf < 1 or conf > 5:
                # clamp it or ignore, lets clamp to valid range just in case
                conf = max(1, min(5, conf))
                
            is_correct = (verdict == gt)
            if is_correct:
                confidence_stats[conf]["correct"] += 1
            confidence_stats[conf]["total"] += 1
        except (ValueError, TypeError):
            unparsed_confidence += 1
            
    print(f"\n--- CALIBRATION STATS ---")
    print(f"Valid Records Parsed: {sum(s['total'] for s in confidence_stats.values())}")
    print(f"Missing Verdicts: {unparsed_verdict}, Missing Confidence: {unparsed_confidence}")
    
    accuracies = []
    scores = []
    
    for score in range(1, 6):
        stats = confidence_stats[score]
        total = stats["total"]
        if total > 0:
            acc = (stats["correct"] / total) * 100
        else:
            acc = 0.0
            
        accuracies.append(acc)
        scores.append(str(score))
        print(f"Confidence {score}: {acc:.1f}% ({stats['correct']}/{total})")

    # Ensure experiments folder exists
    os.makedirs("experiments", exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(scores, accuracies, color='#87CEEB')
    
    plt.xlabel('Judge Reported Confidence Score (1-5)', fontsize=12)
    plt.ylabel('Actual Accuracy (%)', fontsize=12)
    plt.title('Confidence Calibration Analysis', fontsize=14, pad=20)
    plt.ylim(0, 100)
    
    # Add annotations on top of the bars
    for bar in bars:
        yval = bar.get_height()
        # Only annotate if there's data
        if yval > 0 or bar.get_x() > 0: # simplstic check 
            plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()

    # Save chart
    chart_path = "experiments/confidence_calibration.png"
    plt.savefig(chart_path, dpi=300)
    print(f"\nChart successfully saved to {chart_path}!")

if __name__ == "__main__":
    main()
