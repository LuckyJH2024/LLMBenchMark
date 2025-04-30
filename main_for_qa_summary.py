import os
import json
import numpy as np
from benchmark_framework.tasks import create_qa_benchmark, create_summarization_benchmark
from benchmark_framework.benchmark import LLMBenchmark
from benchmark_framework.visualization_qa_summary import generate_report as generate_visual_report

# Set working directory to script location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

RESULTS_DIR = "results"
MODELS = ["gpt-4o", "deepseek-chat", "claude-3-7-sonnet-20250219"]

# Load QA and summarization benchmark tasks
qa_tasks = create_qa_benchmark("data/qa_benchmark.json")
summ_tasks = create_summarization_benchmark("data/summarization_benchmark.json")
tasks = {
    "qa": qa_tasks,
    "summarization": summ_tasks
}

# Initialize benchmark and run both QA and summarization
benchmark = LLMBenchmark(models=MODELS, tasks=tasks)
results = benchmark.run_benchmarks()

# Compute summary statistics
summary = benchmark.get_summary_statistics()

# Generate visualizations and markdown report
generate_visual_report(summary, results, output_dir=RESULTS_DIR)

print("QA and Summarization benchmarks complete. Results and visualizations saved.")
