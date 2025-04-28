#!/usr/bin/env python3

import os
import json
import pandas as pd
from benchmark_framework.summary import calculate_api_summary_statistics
from benchmark_framework.visualization import create_visualizations, create_performance_dashboard

def load_results(results_dir):
    """
    Load all result JSON files from the results directory and reconstruct the results structure.
    
    Returns:
        Dictionary with structure {model: {task_type: [task_results]}}
    """
    results = {}
    json_files = [f for f in os.listdir(results_dir) if f.endswith('.json') and not f.startswith('api_benchmark')]
    
    for filename in json_files:
        parts = filename.replace('.json', '').split('_')
        if len(parts) >= 2:
            model_name = parts[0]
            # Handle models with ":" in their name (converted to "-" in filenames)
            if "-" in model_name:
                model_name = model_name.replace("-", ":", 1)
                
            task_type = '_'.join(parts[1:])  # Combine remaining parts as task_type
            
            file_path = os.path.join(results_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    task_results = json.load(file)
                    
                if model_name not in results:
                    results[model_name] = {}
                    
                results[model_name][task_type] = task_results
                print(f"Loaded {len(task_results)} results from {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    return results

def main():
    """Generate all visualizations from existing results."""
    results_dir = 'results'
    
    # Check if results directory exists
    if not os.path.exists(results_dir):
        print(f"⚠️ Results directory '{results_dir}' not found.")
        return
    
    # Load results
    print(f"Loading results from '{results_dir}'...")
    results = load_results(results_dir)
    
    if not results:
        print("⚠️ No results found.")
        return
    
    # Calculate summary statistics
    print("Calculating summary statistics...")
    summary = calculate_api_summary_statistics(results)
    
    # Generate visualizations
    print("Generating all visualizations...")
    
    # Create standard visualizations (duration, memory, heatmap)
    create_visualizations(summary, results_dir)
    
    # Create performance dashboard
    df = pd.DataFrame.from_dict({(i, j): summary[i][j] 
                               for i in summary.keys() 
                               for j in summary[i].keys()},
                              orient='index')
    df.reset_index(inplace=True)
    df.columns = ['model', 'task_type', 'average_duration', 'average_memory_usage', 'average_score', 'total_tasks']
    
    create_performance_dashboard(df, results_dir)
    
    print(f"✅ All visualizations generated successfully!")
    print(f"📊 Check the '{results_dir}' directory for the following files:")
    print("   - average_duration.png")
    print("   - average_memory_usage.png")
    print("   - performance_heatmap.png")
    print("   - performance_dashboard.png")
    print("   - radar_chart.png (if reasoning results available)")
    print("   - reasoning_bar_comparison.png (if reasoning results available)")

if __name__ == "__main__":
    main() 