# Multi-Agent Debate for QA (Assignment 2)

This repository implements a multi-agent debate pipeline designed to answer questions from the StrategyQA dataset. Two LLM agents ("Proponent" and "Opponent") argue a question back-and-forth for a set number of rounds before a standalone "Judge" agent evaluates the transcript to render a final Yes/No verdict along with a confidence score. 

This repository fulfills the Assignment 2 requirements including modular agent design, dynamic prompt templates, evaluation scripts for baseline comparisons, and a Streamlit-based web interface ("Vibe Coding" UI).

## 🚀 Setup & Installation

**Prerequisites:** Python 3.10+

1. **Clone the repository:**
   ```bash
   git clone https://github.com/FernCanCode/NLP_Class.git
   cd NLP_Class/assignments/assignment2
   ```

2. **Set up a Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

3. **Install Dependencies:**
   Install all required libraries including `openai`, `streamlit`, and `matplotlib`:
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables:**
   Create a `.env` file in the root `assignment2/` directory and add your OpenAI API Key:
   ```env
   OPENAI_API_KEY=your_api_key_here
   ```

## ⚙️ Project Structure & Modularity

The codebase is strictly modularized into distinct operational layers:

- `src/agents.py`: Contains the core `DebateAgent` class which manages chat history, API calls, and prompt templating.
- `src/orchestrator.py`: Contains the `DebateManager` which handles the adaptive stopping criterion, round-robin transcript passing, and phase transitions.
- `src/main.py`: The main execution script to run the batch debate experiment on the dataset.
- `app.py`: A fully functional Streamlit Web UI enabling custom question input, live round-by-round replay, and verdict panels.
- `config/agent_config.json`: All hyperparameters are strictly disjointed from the code here. Edit this file to change `model_name`, `max_rounds`, `temperature`, and `max_tokens`.
- `prompts/`: Contains `.txt` files (`proponent_v1.txt`, etc.). These are fully editable templates utilizing `{{QUESTION}}` and `{{TRANSCRIPT}}` placeholders handled dynamically at runtime.

## 📊 Running the Core Debate Pipeline

To run the batch debate evaluation against the StrategyQA dataset snippet and generate the required logs:

```bash
python -m src.main
```
This script will produce `data/results_log.json`, which logs the complete JSON transcription for every run, including the question, per-round arguments, Judge reasoning, final verdict, confidence score, and ground truth.

## 🕸️ Running the UI

To run the interactive UI for testing custom questions and rendering real-time debates:

```bash
streamlit run app.py
```

## 🔬 Reproducing Experiments

All evaluation scripts are located in the `experiments/` directory. Running these will write output statistics to the console and generate `.png` charts used in the final write-up.

### 1. Baselines (Zero-Shot & Few-Shot)
To reproduce the Direct QA (N=1) and Self-Consistency (N=20) baseline data:
```bash
python experiments/run_zero_shot.py
python experiments/run_few_shot.py
```
*Run the analysis script to generate the comparison chart (`experiments/accuracy_comparison.png`):*
```bash
python experiments/analyze_results.py
```

### 2. Ablation Study: Debate Round Scaling
To test scaling performance across `max_rounds` of [1, 3, 5]:
```bash
python experiments/run_ablation.py
python experiments/analyze_ablation.py
```
*This outputs `experiments/ablation_scaling.png`.*

### 3. Judge Confidence Calibration
To parse the `results_log.json` and map the Judge's self-reported confidence against actual accuracy:
```bash
python experiments/analyze_confidence.py
```
*This outputs `experiments/confidence_calibration.png`.*
