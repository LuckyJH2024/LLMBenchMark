#!/usr/bin/env python3

"""
Visualization module that serves as a facade for all chart generation functions.
This module imports from charts.py which contains the actual implementation.
"""

import os
import pandas as pd
from benchmark_framework.charts import (
    create_duration_chart,
    create_memory_chart,
    create_performance_dashboard as _create_performance_dashboard,
    create_enhanced_radar_chart,
    create_enhanced_heatmap,
    create_reasoning_score_comparison
)

def create_visualizations(summary, results_dir='results'):
    """
    Create all standard visualizations from benchmark results.
    
    Args:
        summary: Dictionary with summary statistics
        results_dir: Directory to save visualization files
    """
    os.makedirs(results_dir, exist_ok=True)

    # Convert summary to dataframe for visualization
    df = pd.DataFrame.from_dict({(i, j): summary[i][j] 
                                 for i in summary.keys() 
                                 for j in summary[i].keys()},
                                orient='index')
    df.reset_index(inplace=True)
    df.columns = ['model', 'task_type', 'average_duration', 'average_memory_usage', 'average_score', 'total_tasks']

    # Create individual charts
    create_duration_chart(df, results_dir)
    create_memory_chart(df, results_dir)
    create_enhanced_heatmap(summary, results_dir)
    
    # Create reasoning-specific visualizations
    create_reasoning_score_comparison(results_dir)

# Re-export this function since it's called directly by main.py and run_api_benchmark.py
def create_performance_dashboard(df, results_dir):
    """
    Create a performance dashboard from the provided DataFrame.
    This is a wrapper around the implementation in charts.py.
    
    Args:
        df: DataFrame with performance metrics
        results_dir: Directory to save the dashboard
    """
    return _create_performance_dashboard(df, results_dir)