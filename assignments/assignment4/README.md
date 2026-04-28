# Assignment 4: Retrieval Augmented Generation

**Track Declaration**: Track B, Code RAG
**Dataset**: `code_search_net`, `python` config
**Embedding model**: `sentence-transformers/all-MiniLM-L6-v2`
**Vector database**: Chroma persistent client
**Generator**: Anthropic Claude API (`claude-haiku-4-5`)

## Current Status
- [x] **Step 1:** Project setup complete.
- [x] **Step 2:** Starter index built. The starter Chroma index contains 1,000 CodeSearchNet Python functions. The vector database is stored locally at `./chroma_code` and can be rebuilt by running `python build_index.py`.
- [x] **Step 3:** Retriever implemented and tested. You can run `python retriever.py` to test retrieval. The retriever returns the top k=4 chunks, embedded with their cosine similarity, distance, and `repo/path::func_name` citation metadata.
- [x] **Step 4:** Generator implemented and tested. You can run `python generator.py` to test the full retrieve + generate loop with Claude. The generator uses `ANTHROPIC_API_KEY` from `.env` to make authenticated calls.
- [x] **Step 5:** Baseline execution logic (`run_part1.py`) complete. The results are saved to `results/part1_results.csv` containing 10 baseline CodeSearchNet queries.
- [x] **Step 6:** Custom function creation complete. `data/custom_functions.py` contains 5 custom NLP utility functions.
- [x] **Step 7:** Custom function indexing complete. `index_custom_functions.py` successfully inserts 5 custom NLP functions and raises the collection size to 1005.
- [x] **Step 8:** Part 2 execution logic (`run_part2.py`) complete. Processed all 10 queries.
- [x] **Step 9:** Reporting finalized.

## How to Reproduce
1. `python build_index.py`
2. `python retriever.py`
3. `python generator.py`
4. `python run_part1.py`
5. `python data/custom_functions.py`
6. `python index_custom_functions.py`
7. `python run_part2.py`

**Note:** The `.env` file must contain your `ANTHROPIC_API_KEY` for `generator.py`, `run_part1.py`, and `run_part2.py`.
**Note:** Final results are written to `results/part1_results.csv` and `results/part2_results.csv`.
**Note:** The local vector database is stored in `chroma_code/` and can be cleanly rebuilt.

## Setup Instructions

1. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables:**
   Create a `.env` file in the root of the project directory and insert your Anthropic api key.
   ```bash
   echo 'ANTHROPIC_API_KEY="your_api_key_here"' > .env
   ```
