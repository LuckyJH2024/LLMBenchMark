import os
import json
import numpy as np
from benchmark_framework.tasks import create_qa_benchmark
from benchmark_framework.benchmark import LLMBenchmark
from benchmark_framework.report import generate_report

os.chdir(os.path.dirname(os.path.abspath(__file__)))


RESULTS_DIR = "results"
MODELS = ["gpt-4o", "deepseek-chat", "claude-3-sonnet-20240229"]


# Step 1: Load new QA tasks
qa_tasks = create_qa_benchmark("data/qa_benchmark.json")

# Step 2: Run QA tasks for selected models
benchmark = LLMBenchmark(models=MODELS, tasks={"qa": qa_tasks})
qa_results = benchmark.run_benchmarks()

# Step 3: Save QA results (each model to its own file, safe filename)
for model, task_results in qa_results.items():
    safe_model_name = model.replace(":", "-")
    with open(os.path.join(RESULTS_DIR, f"{safe_model_name}_qa.json"), "w") as f:
        json.dump(task_results["qa"], f, indent=4)

# Step 4: Load all existing results (including old non-QA ones)
def load_all_results():
    all_results = {}
    for filename in os.listdir(RESULTS_DIR):
        if filename.endswith(".json"):
            model_task = filename.replace(".json", "")
            model_part, task_type = model_task.split("_", 1)
            model = model_part.replace("-", ":")  # Convert back to proper model name
            if model not in all_results:
                all_results[model] = {}
            with open(os.path.join(RESULTS_DIR, filename), "r") as f:
                all_results[model][task_type] = json.load(f)
    return all_results

# Step 5: Compute summary statistics
def compute_summary(all_results):
    summary = {}
    for model, tasks in all_results.items():
        summary[model] = {}
        for task_type, entries in tasks.items():
            durations = [x["duration"] for x in entries if "duration" in x]
            memories = [x["memory_usage"] for x in entries if "memory_usage" in x]
            scores = [x["score"] for x in entries if x.get("score") is not None]

            summary[model][task_type] = {
                "average_duration": np.mean(durations) if durations else 0,
                "average_memory_usage": np.mean(memories) if memories else 0,
                "average_score": np.mean(scores) if scores else 0,
                "total_tasks": len(entries),
            }
    return summary

# Step 6: Generate updated report
if __name__ == "__main__":
    all_results = load_all_results()
    summary = compute_summary(all_results)
    generate_report(summary, all_results, output_dir=RESULTS_DIR)
