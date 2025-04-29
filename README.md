# 🧠 LLMBenchMark

This project benchmarks local Large Language Models (LLMs) on tasks such as **question answering**, **code generation**, **logical reasoning**, and **summarization**. It provides **structured evaluation results**, rich **visualizations**, and a detailed **markdown report**.

---

## 🚀 Installation and Environment Setup

### 1. Clone the project
```bash
git clone https://github.com/LuckyJH2024/LLMBenchMark.git your_file_name
cd your_file_name
```

### 2. Create and activate a virtual environment (optional)
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🧩 Start Ollama and Load Models

### 4. Start Ollama
Ensure [Ollama](https://ollama.com/) is installed and running:
```bash
ollama serve
```

### 5. Pull required models
```bash
ollama run phi
ollama run mistral
ollama run llama3:8b
```

To check available models:
```bash
ollama list
```

---

## 🔑 Setting Up API Keys

To use API models (OpenAI, DeepSeek, Anthropic), you need to set up API keys:

### Method 1: Environment Variables (Recommended)
```bash
# Linux/macOS
export OPENAI_API_KEY="your_openai_key"
export DEEPSEEK_API_KEY="your_deepseek_key"
export ANTHROPIC_API_KEY="your_anthropic_key"

# Windows (Command Prompt)
set OPENAI_API_KEY=your_openai_key
set DEEPSEEK_API_KEY=your_deepseek_key
set ANTHROPIC_API_KEY=your_anthropic_key

# Windows (PowerShell)
$env:OPENAI_API_KEY="your_openai_key"
$env:DEEPSEEK_API_KEY="your_deepseek_key"
$env:ANTHROPIC_API_KEY="your_anthropic_key"
```

### Method 2: Create api_keys.json file
Create a file named `api_keys.json` in the project root:
```json
{
  "openai": "your_openai_key",
  "deepseek": "your_deepseek_key",
  "anthropic": "your_anthropic_key"
}
```

Note: `api_keys.json` is included in `.gitignore` to avoid accidentally committing your keys.

---

## ⚙️ Configuring Benchmark Parameters

The project uses `config.yaml` for all benchmark settings. Here's how to configure it:

### Model Selection
```yaml
models:
  # API Models
  api:
    - openai:gpt-4o
    - anthropic:claude3.7
    # - deepseek:deepseek-coder  # Uncomment to enable
  
  # Local Models (using Ollama)
  local: 
    - phi
    - mistral
    - llama3:8b
```

### Test Configuration
```yaml
tests:
  # Reasoning tests
  reasoning:
    enabled: true    # Set to false to skip
    samples: 10      # Number of samples to test
  
  # Code generation tests (APPS dataset)
  coding:
    enabled: true    # Set to false to skip
    difficulties:    # Choose difficulty levels
      - interview
      - competition
    problems_per_difficulty: 3  # Problems per level
    data_path: "data/APPS"      # Dataset path
  
  # Question-Answer tests
  qa:
    enabled: true    # Set to false to skip
    samples: 10      # Number of samples to test
```

### Output Settings
```yaml
output:
  results_dir: "results"  # Output directory
  visualize: true         # Generate visualizations
  save_details: true      # Save detailed results
```

### Runtime Parameters
```yaml
run:
  timeout: 120   # API request timeout (seconds)
  workers: 4     # Parallel workers
  retries: 3     # Retry attempts for API calls
  seed: 0        # Random seed (0 = use time)
```

---

## 🧪 Running the Benchmark

### Using the Unified Script
Run all configured tests with a single command:
```bash
python run_benchmark.py --config config.yaml
```

The script will:
1. Read configuration from `config.yaml`
2. Run enabled tests (reasoning, coding, QA)
3. Generate visualizations and reports
4. Save results to the specified directory

### Running Specific Tests
For individual test types:

```bash
# Run only reasoning tests
python run_api_benchmark.py --models openai:gpt-4o --task-types reasoning

# Run only QA tests
python run_api_benchmark.py --models anthropic:claude3.7 --task-types qa

# Test specific models on the APPS dataset
python benchmark_framework/apps_eval/apps_benchmark.py --models phi mistral --difficulties interview --problems 5
```

---

## 📚 Task Datasets

| Task Type      | File Name                  | Description                     |
|----------------|----------------------------|---------------------------------|
| QA             | `qa_benchmark.json`        | Question answering              |
| Code           | `code_benchmark.json`      | Code completion & generation    |
| Reasoning      | `sample_reasoning_eval.json` | Multi-type logical reasoning    |
| Summarization  | `summarization_benchmark.json` | Text summarization tasks    |

Modify or extend any dataset to suit your testing needs.

---

## 📊 View Results

After running the benchmark:

- **HTML Report**: `results/report/benchmark_report.html`
- **Visualizations**: 
  - `results/report/*.png` - Bar charts, heatmaps
  - Task-specific visualizations in respective folders
- **Raw Data**: 
  - `results/{task_type}/{model_name}.json`
  - `results/{task_type}/summary.json`

---

## 🧱 Project Structure

```
├── run_benchmark.py        # Unified benchmark runner
├── run_api_benchmark.py    # API models benchmark script
├── config.yaml             # Main configuration file
├── benchmark_framework/
│   ├── benchmark.py        # Core logic & evaluation
│   ├── api_models.py       # API model integration
│   ├── tasks.py            # Load and format task data
│   ├── visualization.py    # Create plots & dashboards
│   ├── report.py           # Report generation
│   └── apps_eval/          # APPS benchmark components
├── data/                   # Task datasets
│   ├── qa_benchmark.json
│   ├── reasoning_benchmark.json
│   ├── APPS/               # APPS coding dataset
│   └── ...
├── requirements.txt
└── results/                # Output results and visualizations
```

---

# 🧠 Reasoning Benchmark Overview

### 🧩 File:
```
data/sample_reasoning_eval.json
```

### 🧪 Structure:
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

### 🔍 Covered Reasoning Types

| Type                       | Description                                 | #
|---------------------------|---------------------------------------------|----|
| Multi-hop Reasoning       | Chain of facts (e.g., A > B > C)             | 3  |
| Syllogistic Reasoning     | Classic logic from category relationships   | 3  |
| Causal Reasoning          | Fallacy avoidance, cause-effect logic       | 3  |
| Numerical / Symbolic      | Basic arithmetic, quantities, logic math    | 3  |
| Boolean Reasoning         | Truth, contradiction, set membership        | 3  |
| Counterfactual Reasoning  | What-if reasoning                           | 3  |
| Planning / Procedural     | Goal-driven task planning                   | 3  |

---

### ⚙️ Evaluation Metrics

Each sample is automatically scored across:

| Metric             | Meaning                                              |
|--------------------|------------------------------------------------------|
| `answer_score`     | Response vs. ground truth (BERTScore / SBERT)       |
| `chain_score`      | Reasoning steps vs. reference reasoning             |
| `consistency_score`| Paraphrased response consistency                    |
| `score`            | Final weighted score (used in radar/visuals)        |

---

## 🔬 Three Evaluation Directions

### 1. ✅ Answer Accuracy Only
Evaluate whether the model gave the **correct final answer**.

- Compare `response` vs `ground_truth`
- Use BERTScore / SBERT
- Metric: `answer_score`

### 2. 🔗 Reasoning Process Matching
Compare **step-by-step** logic to `reference_steps`.

- Use `reasoning_steps` and `reference_steps`
- Metric: `chain_score` via average semantic match

### 3. ♻️ Paraphrase Consistency
Check **robustness** against paraphrasing.

- Compare `response` and `paraphrased_response`
- Metric: `consistency_score`

---

## 📌 Example Prompt Format

```
Context: All mammals are warm-blooded. Whales are mammals.
Question: Are whales warm-blooded?
Answer:
```

Expected output:
> Yes, because whales are mammals and all mammals are warm-blooded.

---

## 📤 Extend the Benchmark

- Add new examples to `sample_reasoning_eval.json`
- Keep same field structure, append `"type"` as needed
- You may add adversarial variants for robustness testing