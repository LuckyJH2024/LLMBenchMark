#!/usr/bin/env python3

import os
import json
import time
import argparse
from benchmark_framework.api_models import APIModelBenchmark
from benchmark_framework.tasks import load_all_benchmarks
from benchmark_framework.visualization import create_visualizations, create_performance_dashboard
from benchmark_framework.report import generate_report
from benchmark_framework.benchmark import LLMBenchmark
from benchmark_framework.summary import calculate_api_summary_statistics
import pandas as pd


def main():
    """
    Run benchmarks on API-based LLM models like OpenAI/ChatGPT and DeepSeek.
    """
    parser = argparse.ArgumentParser(description="Benchmark API-based LLM models")
    parser.add_argument("--models", nargs="+", required=True,
                       help="List of API models to benchmark in format provider:model (e.g., openai:gpt-4)")
    parser.add_argument("--task-types", nargs="+", default=["qa", "reasoning", "code", "summarization"],
                       help="Task types to run (default: all)")
    parser.add_argument("--results-dir", default="results", help="Directory to save results")
    
    args = parser.parse_args()
    models = args.models
    task_types = args.task_types
    results_dir = args.results_dir
    
    print(f"Starting API LLM benchmarking for models: {models}")
    print(f"Task types: {task_types}")
    start_time = time.time()
    
    # Load benchmark tasks
    all_tasks = load_all_benchmarks()
    
    # Filter tasks based on specified task types
    tasks = {task_type: all_tasks[task_type] for task_type in task_types if task_type in all_tasks}
    
    # Initialize API benchmark engine
    api_benchmark = APIModelBenchmark()
    
    # Check API keys
    api_key_status = api_benchmark.check_api_keys()
    for provider, status in api_key_status.items():
        if not status:
            print(f"⚠️ Warning: {provider.upper()} API key not found. Set {provider.upper()}_API_KEY environment variable.")
    
    # Create empty results dictionary
    results = {}
    
    # Run benchmarks for each model
    for model in models:
        provider = model.split(":", 1)[0]
        if not api_key_status.get(provider, False):
            print(f"⚠️ Skipping {model} due to missing API key")
            continue
            
        results[model] = {}
        
        for task_type, task_list in tasks.items():
            print(f"Running {task_type} tasks for model {model}")
            
            # Run the benchmark
            task_results = api_benchmark.benchmark_api_model(model, task_list, task_type)
            
            # Create a temporary LLMBenchmark instance for evaluation
            temp_benchmark = LLMBenchmark(["temp"], {})
            
            # Evaluate each result using the existing evaluation logic
            for task in task_results:
                if "ground_truth" in task:
                    if task_type == "reasoning":
                        score_detail = temp_benchmark.evaluate_reasoning_all(task)
                        task.update(score_detail)
                        task["score"] = score_detail["answer_score"]
                    else:
                        task["score"] = temp_benchmark.evaluate(task["response"], task["ground_truth"], task_type)
                else:
                    task["score"] = None
            
            results[model][task_type] = task_results
            
            # Save individual results
            safe_model_name = model.replace(":", "-")
            os.makedirs(results_dir, exist_ok=True)
            with open(f"{results_dir}/{safe_model_name}_{task_type}.json", 'w', encoding='utf-8') as file:
                json.dump(task_results, file, indent=4)
                
    # Calculate summary statistics using the new module
    summary_stats = calculate_api_summary_statistics(results)
    
    # Generate visualizations
    create_visualizations(summary_stats, results_dir)
    df = pd.DataFrame.from_dict({(i, j): summary_stats[i][j] 
                               for i in summary_stats.keys() 
                               for j in summary_stats[i].keys()},
                              orient='index')
    df.reset_index(inplace=True)
    df.columns = ['model', 'task_type', 'average_duration', 'average_memory_usage', 'average_score', 'total_tasks']

    create_performance_dashboard(df, results_dir)

    # Generate report
    generate_report(summary_stats, results, results_dir, prefix="api_")

    # Print completion message and time taken
    elapsed_time = time.time() - start_time
    print(f"API LLM benchmarking completed in {elapsed_time:.2f} seconds. Results saved in {results_dir}.")


if __name__ == "__main__":
    main() 