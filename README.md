# LLMBenchMark

This project benchmarks local LLMs (Large Language Models) on tasks such as question answering, code generation, reasoning, and summarization. It produces structured evaluation results with visualizations and markdown reports.

---

## Installation and Environment Setup

### 1. Clone the project and navigate to the directory
```bash
git clone https://github.com/LuckyJH2024/LLMBenchMark.git your_file_name
```

### 2. Create and activate a virtual environment (optional)
```bash
python3 -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Start Ollama and Load Models

### 4. Start Ollama (Local inference service)
Make sure [Ollama](https://ollama.com/) is installed and running:
```bash
ollama serve
```

### 5. Pull required models (first-time only)
```bash
ollama run phi
ollama run mistral
ollama run llama3:8b
```
If you want to see all models downloaded on your machine, use this in your terminal (outside the prompt):
```bash
ollama list
```
The one you ran most recently is typically the one you're interacting with unless you switched.

---

## Run the Benchmark

### 6. Execute the main script
```bash
python main.py
```

This will run benchmarks and generate outputs in the `results/` directory:

- JSON results per model-task combination (e.g. `phi_qa.json`)
- Visualizations:
  - `average_duration.png`
  - `average_memory_usage.png`
  - `performance_dashboard.png`
- Summary Markdown Report:
  - e.g. `benchmark_report_20250412_211738.md`

---

## Task Datasets

The following input files are used for benchmarking tasks:

- `qa_benchmark.json` – Question answering
- `code_benchmark.json` – Code generation
- `reasoning_benchmark.json` – Logical reasoning
- `summarization_benchmark.json` – Text summarization

You can modify these JSON files to add your own benchmark tasks.

---

## View Results

- Markdown reports can be viewed with VSCode, Typora, or any markdown viewer.
- PNG charts provide visual summaries of model performance.

---

## Supported Models

Ollama-compatible local models, such as:

- `phi`
- `mistral`
- `llama3:8b`

To add other models, modify the model list in `main.py`.

---

## Project Structure

```
├── main.py                 # Entry point
├── benchmark.py           # Core logic: model calls, metrics, runtime
├── tasks.py               # Task loading functions
├── visualization.py       # Plotting performance
├── report.py              # Markdown report generation
├── requirements.txt
├── *.json                 # Task input data
└── results/               # All benchmark outputs
```

---

## Example Output

You will see results like the following chart:

![Sample](results/average_duration.png)

---

For further assistance, explore the source code and in-line documentation. Contributions welcome!

#############################################Reasoning_benchmark################################

### 🧠 Reasoning Benchmark Overview

This benchmark is designed to **evaluate the reasoning capabilities** of Large Language Models (LLMs) across 7 core types of reasoning. Each type is tested with **3 carefully constructed examples**, yielding a total of **21 tasks**.

#### 📁 Dataset File

```
data/sample_reasoning_eval.json
```

Each entry in the file includes:

```json
{
  "context": "...",
  "question": "...",
  "response": "...",
  "ground_truth": "...",
  "reasoning_steps": [...],
  "reference_steps": [...],
  "paraphrased_response": "..."
}
```

---

### 🔍 Covered Reasoning Types

| Reasoning Type                | Description                                     | # Samples |
|------------------------------|--------------------------------------------------|-----------|
| Multi-hop Reasoning          | Requires combining 2+ facts (e.g., A > B > C)    | 3         |
| Syllogistic Reasoning        | Deductive logic from categorical statements      | 3         |
| Causal Reasoning             | Cause-effect relations, avoiding fallacies       | 3         |
| Numerical / Symbolic         | Arithmetic, quantity comparison                  | 3         |
| Boolean / Logical            | Modus Tollens, contradictions, set logic         | 3         |
| Counterfactual Reasoning     | Hypotheticals, reasoning about what-if           | 3         |
| Planning / Procedural        | Steps to solve or achieve a goal                 | 3         |

---

### ⚙️ How It Works

1. **Model Inference**: Our benchmark system sends prompts (context + question) to an LLM (e.g., via Ollama API).
2. **Scoring Metrics**: For each sample, we compute:
   - `answer_score`: semantic similarity to ground truth
   - `chain_score`: similarity of reasoning steps
   - `consistency_score`: paraphrase stability (optional)
   - `final_score`: weighted average (for radar/summary)
3. **Visualization**:
   - Radar chart of 3 reasoning dimensions
   - Bar chart comparison across models
   - Performance heatmaps
4. **Output**: Results saved to:
   ```
   results/{model_name}_reasoning.json
   results/radar_chart.png
   results/benchmark_report_*.md
   ```

---

### 📌 Example Prompt Format

```
Context: All mammals are warm-blooded. Whales are mammals.
Question: Are whales warm-blooded?
Answer:
```

Expected model output:
> Yes, because whales are mammals and mammals are warm-blooded.

---

### 📤 To Extend the Dataset

- Add more examples to `sample_reasoning_eval.json`
- Maintain the same format and append new `type` fields
- Consider adding adversarial variants to test robustness

---

