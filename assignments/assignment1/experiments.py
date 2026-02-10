import pandas as pd
import time
from api_basics import query_llm

"""
Assignment 1 - Part 2: Prompt Engineering Experiments
-----------------------------------------------------
This script conducts systematic experiments comparing different prompting strategies
on two NLP tasks: Text Summarization and Question Answering.

It compares four strategies:
1. Zero-shot prompting
2. Few-shot prompting
3. Chain-of-thought prompting
4. Custom variation (Persona-based)

The results, including the model's response, token usage, and execution time,
are saved to a CSV file ('experiment_results.csv').
"""

# ==========================================
# DATASETS
# ==========================================

# Task 1: Text Summarization
# Input: A short text. Output: A 2-3 sentence summary.
summarization_data = [
    "The James Webb Space Telescope (JWST) has captured its first direct image of a planet outside our solar system. The exoplanet, HIP 65426 b, is a gas giant about six to 12 times the mass of Jupiter. It is young, about 15 to 20 million years old, compared to our 4.5-billion-year-old Earth. Astronomers used four different light filters on JWST’s Near-Infrared Camera (NIRCam) and Mid-Infrared Instrument (MIRI) to capture the planet.",
    "Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy that, through cellular respiration, can later be released to fuel the organism's activities. This chemical energy is stored in carbohydrate molecules, such as sugars, which are synthesized from carbon dioxide and water – hence the name photosynthesis, from the Greek phēs, 'light', and sunthesis, 'putting together'.",
    "The Great Wall of China is a series of fortifications that were built across the historical northern borders of ancient Chinese states and Imperial China as protection against various nomadic groups from the Eurasian Steppe. Several walls were built from as early as the 7th century BC, with select stretches later joined together by Qin Shi Huang (220–206 BC), the first emperor of China. Little of the Qin wall remains.",
    "Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to the natural intelligence displayed by animals including humans. AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals.",
    "The internet is a global system of interconnected computer networks that uses the Internet protocol suite (TCP/IP) to communicate between networks and devices. It is a network of networks that consists of private, public, academic, business, and government networks of local to global scope, linked by a broad array of electronic, wireless, and optical networking technologies.",
    "Climate change includes both global warming driven by human-induced emissions of greenhouse gases and the resulting large-scale shifts in weather patterns. Though there have been previous periods of climatic change, since the mid-20th century humans have had an unprecedented impact on Earth's climate system and caused change on a global scale.",
    "The human brain is the central organ of the human nervous system, and with the spinal cord makes up the central nervous system. The brain consists of the cerebrum, the brainstem and the cerebellum. It controls most of the activities of the body, processing, integrating, and coordinating the information it receives from the sense organs, and making decisions as to the instructions sent to the rest of the body.",
    "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation. Python is dynamically typed and garbage-collected. It supports multiple programming paradigms, including structured (particularly procedural), object-oriented and functional programming.",
    "Democracy is a form of government in which the people have the authority to deliberate and decide legislation, or to choose governing officials to do so. Who is considered part of 'the people' and how authority is shared among or delegated by the people has changed over time and at different rates in different countries.",
    "Quantum mechanics is a fundamental theory in physics that provides a description of the physical properties of nature at the scale of atoms and subatomic particles. It is the foundation of all quantum physics including quantum chemistry, quantum field theory, quantum technology, and quantum information science."
]

# Task 2: Question Answering
# Input: A question (and optionally context). Output: The answer.
qa_data = [
    "What is the capital of France?",
    "Who wrote 'Romeo and Juliet'?",
    "What is the chemical symbol for Gold?",
    "How many continents are there on Earth?",
    "What is the speed of light in vacuum (approximate km/s)?",
    "Who painted the Mona Lisa?",
    "What is the largest mammal in the world?",
    "In what year did the Titanic sink?",
    "What comes after the number 99?",
    "What is the boiling point of water at sea level?"
]

# ==========================================
# STRATEGIES
# ==========================================

def run_zero_shot(task_name, inputs):
    """
    Executes the Zero-shot prompting strategy.
    
    In Zero-shot, the model is given a direct instruction without any examples.
    
    Args:
        task_name (str): The name of the task (e.g., "Summarization", "QA").
        inputs (list): A list of input strings to test.
        
    Returns:
        list: A list of dictionaries containing the results and metrics for each input.
    """
    results = []
    print(f"\n--- Running Zero-Shot for {task_name} ---")
    for item in inputs:
        # Construct the prompt based on the task type
        prompt = ""
        if task_name == "Summarization":
            prompt = f"Summarize the following text in 2-3 sentences.\n\nText:\n{item}"
        elif task_name == "QA":
            prompt = f"Answer the following question correctly.\n\nQuestion:\n{item}"
            
        # Helper to run the query and catch partial errors (like missing metrics)
        response = query_llm(prompt, include_metrics=True)
        
        # Handle errors
        if "error" in response:
            print(f"Error in Zero-shot: {response['error']}")
            tokens = 0
            exec_time = 0
        else:
            tokens = response["token_usage"]["total_tokens"]
            exec_time = response["execution_time"]

        results.append({
            "task": task_name,
            "strategy": "Zero-shot",
            "input": item,
            "prompt": prompt,
            "response": response["content"],
            "tokens_total": tokens,
            "execution_time": exec_time
        })
    return results

def run_few_shot(task_name, inputs):
    """
    Executes the Few-shot prompting strategy.
    
    In Few-shot, the model is provided with 2-3 examples (input-output pairs)
    before the actual test input to guide its behavior.
    """
    results = []
    print(f"\n--- Running Few-Shot for {task_name} ---")
    
    # Define examples (shots) for each task to verify the pattern
    examples = ""
    if task_name == "Summarization":
        examples = (
            "Text: Apple Inc. is an American multinational technology company headquartered in Cupertino, California. Apple is the world's largest technology company by revenue.\n"
            "Summary: Apple is a major US tech company based in Cupertino. It is the largest tech company by revenue globally.\n\n"
            "Text: The koala is an arboreal herbivorous marsupial native to Australia. It is the only extant representative of the family Phascolarctidae.\n"
            "Summary: The koala is a tree-dwelling plant-eating marsupial from Australia. It represents the only living member of its family.\n\n"
        )
    elif task_name == "QA":
        examples = (
            "Question: What is 2 + 2?\nAnswer: 4\n\n"
            "Question: What is the capital of Japan?\nAnswer: Tokyo\n\n"
        )

    for item in inputs:
        prompt = ""
        if task_name == "Summarization":
            prompt = f"{examples}Text: {item}\nSummary:"
        elif task_name == "QA":
            prompt = f"{examples}Question: {item}\nAnswer:"
            
        response = query_llm(prompt, include_metrics=True)
        
        if "error" in response:
            print(f"Error in Few-shot: {response['error']}")
            tokens = 0
            exec_time = 0
        else:
            tokens = response["token_usage"]["total_tokens"]
            exec_time = response["execution_time"]

        results.append({
            "task": task_name,
            "strategy": "Few-shot",
            "input": item,
            "prompt": prompt,
            "response": response["content"],
            "tokens_total": tokens,
            "execution_time": exec_time
        })
    return results

def run_chain_of_thought(task_name, inputs):
    """
    Executes the Chain-of-Thought (CoT) prompting strategy.
    
    CoT encourages the model to generate intermediate reasoning steps
    before providing the final answer, improving performance on complex tasks.
    """
    results = []
    print(f"\n--- Running Chain-of-Thought for {task_name} ---")
    for item in inputs:
        prompt = ""
        # We explicitly ask for "Step-by-step thinking" or "Reasoning"
        if task_name == "Summarization":
            prompt = f"Summarize the text by first identifying the main points, then synthesizing them into 2-3 sentences.\nText: {item}\nStep-by-step thinking:\n"
        elif task_name == "QA":
            prompt = f"Answer the question by thinking through it step-by-step.\nQuestion: {item}\nReasoning and Answer:\n"
            
        response = query_llm(prompt, include_metrics=True)
        
        if "error" in response:
            print(f"Error in Chain-of-Thought: {response['error']}")
            tokens = 0
            exec_time = 0
        else:
            tokens = response["token_usage"]["total_tokens"]
            exec_time = response["execution_time"]

        results.append({
            "task": task_name,
            "strategy": "Chain-of-Thought",
            "input": item,
            "prompt": prompt,
            "response": response["content"],
            "tokens_total": tokens,
            "execution_time": exec_time
        })
    return results

def run_custom_variation(task_name, inputs):
    """
    Executes a Custom prompting strategy (Persona-based).
    
    This strategy assigns a specific role or persona to the AI (e.g., "Expert Editor")
    to influence the tone and quality of the output.
    """
    results = []
    print(f"\n--- Running Custom (Persona) for {task_name} ---")
    for item in inputs:
        prompt = ""
        if task_name == "Summarization":
            # Persona: Expert Editor
            prompt = f"You are an expert editor for a newspaper. Summarize this text for a general audience in exactly 2 sentences.\nText: {item}"
        elif task_name == "QA":
            # Persona: Helpful Professor
            prompt = f"You are a helpful professor. Answer this question clearly and concisely.\nQuestion: {item}"
            
        response = query_llm(prompt, include_metrics=True)

        if "error" in response:
            print(f"Error in Custom: {response['error']}")
            tokens = 0
            exec_time = 0
        else:
            tokens = response["token_usage"]["total_tokens"]
            exec_time = response["execution_time"]
            
        results.append({
            "task": task_name,
            "strategy": "Custom (Persona)",
            "input": item,
            "prompt": prompt,
            "response": response["content"],
            "tokens_total": tokens,
            "execution_time": exec_time
        })
    return results

# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    all_results = []
    
    # Run Summarization
    for strategy in [run_zero_shot, run_few_shot, run_chain_of_thought, run_custom_variation]:
        all_results.extend(strategy("Summarization", summarization_data))
        
    # Run QA
    for strategy in [run_zero_shot, run_few_shot, run_chain_of_thought, run_custom_variation]:
        all_results.extend(strategy("QA", qa_data))
        
    # Validation / Reporting
    df = pd.DataFrame(all_results)
    print("\n\n=== Experiment Results Summary ===")
    print(df.groupby(['task', 'strategy'])[['tokens_total', 'execution_time']].mean())
    
    # Save to CSV
    df.to_csv("experiment_results.csv", index=False)
    print("\nFull results saved to 'experiment_results.csv'")

if __name__ == "__main__":
    main()
