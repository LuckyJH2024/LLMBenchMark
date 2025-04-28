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
from tqdm import tqdm


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
    
    print(f"开始API模型基准测试...")
    print(f"测试模型: {', '.join(models)}")
    print(f"测试任务类型: {', '.join(task_types)}")
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
            print(f"⚠️ 警告: 未找到{provider.upper()}的API密钥。请设置{provider.upper()}_API_KEY环境变量。")
    
    # Create empty results dictionary
    results = {}
    
    # 使用tqdm显示模型测试进度
    for model in tqdm(models, desc="测试模型进度", ncols=100):
        provider = model.split(":", 1)[0]
        if not api_key_status.get(provider, False):
            print(f"⚠️ 跳过 {model} (API密钥缺失)")
            continue
            
        results[model] = {}
        
        # 使用tqdm显示任务类型进度
        task_types_list = list(tasks.keys())
        for task_type in tqdm(task_types_list, desc=f"{model}的任务类型", leave=False, ncols=100):
            task_list = tasks[task_type]
            
            # 运行基准测试 (已经在api_models.py中添加了进度条)
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
    print("生成可视化结果...")
    create_visualizations(summary_stats, results_dir)
    df = pd.DataFrame.from_dict({(i, j): summary_stats[i][j] 
                               for i in summary_stats.keys() 
                               for j in summary_stats[i].keys()},
                              orient='index')
    df.reset_index(inplace=True)
    df.columns = ['model', 'task_type', 'average_duration', 'average_memory_usage', 'average_score', 'total_tasks']

    create_performance_dashboard(df, results_dir)

    # Generate report
    print("生成报告...")
    generate_report(summary_stats, results, results_dir, prefix="api_")

    # Print completion message and time taken
    elapsed_time = time.time() - start_time
    print(f"基准测试完成! 用时 {elapsed_time:.2f} 秒。结果已保存到 {results_dir} 目录.")


if __name__ == "__main__":
    main() 