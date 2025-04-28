#!/usr/bin/env python3

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from benchmark_framework.summary import calculate_api_summary_statistics
from benchmark_framework.charts import set_plotting_style

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

def create_enhanced_heatmap(summary, results_dir):
    """Create a heatmap of model performance metrics."""
    try:
        # Extract average_score values from the nested dictionary structure
        score_data = {}
        for model, tasks in summary.items():
            score_data[model] = {}
            for task_type, metrics in tasks.items():
                if isinstance(metrics, dict) and 'average_score' in metrics:
                    score_data[model][task_type] = metrics['average_score']
        
        # Convert to DataFrame with models as rows and task types as columns
        df = pd.DataFrame(score_data).transpose()  # Transpose to make models as rows
        
        # Create the heatmap
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(df, annot=True, cmap='coolwarm', fmt=".2f")
        
        # Improve readability
        plt.title('Model Performance Scores by Task Type')
        plt.ylabel('Model')
        plt.xlabel('Task Type')
        
        # Rotate x-axis labels if needed
        plt.xticks(rotation=45, ha='right')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the figure
        heatmap_path = os.path.join(results_dir, "performance_heatmap.png")
        plt.savefig(heatmap_path)
        plt.close()
        print(f"✅ Performance heatmap saved to: {heatmap_path}")
        
        return heatmap_path
    except Exception as e:
        print(f"⚠️ Error creating heatmap: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Generate heatmap from existing results."""
    results_dir = 'results'
    
    # Check if results directory exists
    if not os.path.exists(results_dir):
        print(f"⚠️ Results directory '{results_dir}' not found.")
        return
    
    # Set plotting style
    set_plotting_style()
    
    # Load results
    print(f"Loading results from '{results_dir}'...")
    results = load_results(results_dir)
    
    if not results:
        print("⚠️ No results found.")
        return
    
    # Calculate summary statistics
    print("Calculating summary statistics...")
    summary = calculate_api_summary_statistics(results)
    
    # Create heatmap
    print("Generating heatmap...")
    heatmap_path = create_enhanced_heatmap(summary, results_dir)
    
    if heatmap_path:
        print(f"✅ Heatmap generation complete! File saved at: {heatmap_path}")

if __name__ == "__main__":
    main() 