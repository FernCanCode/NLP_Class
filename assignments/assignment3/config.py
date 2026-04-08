# config.py
import re
import json

# --- MODEL SPECIFICATIONS ---
MODEL_ID = "microsoft/Phi-3.5-mini-instruct"
STAGE1_ADAPTER = "./adapters/checkpoint1_alpaca"
STAGE2_ADAPTER = "./adapters/checkpoint2_json"

# --- TRAINING HYPERPARAMETERS ---
LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "task_type": "CAUSAL_LM"
}

# --- PROMPT TEMPLATES ---

# Used in prepare_data.py for initially generating data
TEACHER_PROMPT_TEMPLATE = """Create a complex training example for the task: {task}.
Provide an 'instruction', an optional 'input' text, and a 'output' that MUST be a valid, sophisticated JSON object.
Return ONLY a JSON object with keys: instruction, input, output."""

# Used in fix_json.py to specifically stringify the output correctly
TEACHER_PROMPT_FIX = """Create a training example for an LLM for this task: {task}.
Return a JSON object with three keys: 'instruction' (the task description), 'input' (optional context/text, or empty string), 'output' (the result, which MUST be a valid JSON string).
The 'output' must be a JSON object inside a string, not a raw object."""

# Used in topup_json_repair.py
TEACHER_PROMPT_TOPUP = """Create a training example for an LLM for this task: {task}.
Return a JSON object with: 'instruction', 'input', 'output' (valid JSON string)."""

# Used in judge_eval.py
JUDGE_PROMPT_TEMPLATE = """Instruction: {prompt_text}

Response A: {response_a}

Response B: {response_b}

Which is better? Return JSON: {{'winner': 'A/B/Tie', 'reason': '...'}}"""


# --- UTILITIES ---

def robust_json_parser(resp_text):
    """
    Attempts to successfully extract and parse a JSON string,
    even if the model wrapped it in markdown codeblocks.
    """
    try:
        json.loads(resp_text)
        return True
    except:
        pass
        
    try:
        # Regex to find json within markdown backticks
        match = re.search(r"```json\n(.*?)\n```", resp_text, re.DOTALL)
        if match:
            json.loads(match.group(1))
            return True
            
        # Try generic backtick extraction
        match = re.search(r"```(.*?)```", resp_text, re.DOTALL)
        if match:
            json.loads(match.group(1))
            return True
            
    except:
        pass
        
    return False
