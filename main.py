#!/usr/bin/env python3

import os
import json
import time
from benchmark_framework.benchmark import LLMBenchmark
from benchmark_framework.tasks import load_all_benchmarks
from benchmark_framework.visualization import create_visualizations, create_performance_dashboard
from benchmark_framework.report import generate_report
from benchmark_framework.summary import calculate_summary_statistics
import pandas as pd


def main():
    """Run the LLM benchmark and generate results."""
    print("Starting LLM benchmarking")
    start_time = time.time()

    # Define models to benchmark
    models = ['phi', 'mistral', 'llama3:8b']
    
    # Load benchmark tasks
    tasks = load_all_benchmarks()

    # Initialize and run benchmark
    benchmark = LLMBenchmark(models, tasks)
    results = benchmark.run_benchmarks()

    # Calculate summary statistics using the new module
    summary = calculate_summary_statistics(results)

    # Generate visualizations
    results_dir = 'results'
    create_visualizations(summary, results_dir)
    df = pd.DataFrame.from_dict({(i, j): summary[i][j] 
                                 for i in summary.keys() 
                                 for j in summary[i].keys()},
                                orient='index')
    df.reset_index(inplace=True)
    df.columns = ['model', 'task_type', 'average_duration', 'average_memory_usage', 'average_score', 'total_tasks']

    create_performance_dashboard(df, results_dir)

    # Generate report
    generate_report(summary, results, results_dir)

    # Print completion message and time taken
    elapsed_time = time.time() - start_time
    print(f"LLM benchmarking completed in {elapsed_time:.2f} seconds. Results saved in {results_dir}.")

if __name__ == "__main__":
    main()




