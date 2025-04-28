#!/usr/bin/env python3

import numpy as np
from typing import Dict, List, Any, Optional

def calculate_summary_statistics(results: Dict[str, Dict[str, List[Dict[str, Any]]]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Calculate summary statistics from benchmark results.
    
    Args:
        results: Nested dictionary with structure {model: {task_type: [task_results]}}
        
    Returns:
        Dictionary with summary statistics for each model and task type
    """
    summary_stats = {}
    
    for model, task_types in results.items():
        summary_stats[model] = {}
        for task_type, task_results in task_types.items():
            durations = [result['duration'] for result in task_results if 'duration' in result]
            memory_usages = [result['memory_usage'] for result in task_results if 'memory_usage' in result]
            scores = [result['score'] for result in task_results if 'score' is not None]

            average_duration = np.mean(durations) if durations else 0
            average_memory_usage = np.mean(memory_usages) if memory_usages else 0
            average_score = np.mean(scores) if scores else None

            summary_stats[model][task_type] = {
                'average_duration': average_duration,
                'average_memory_usage': average_memory_usage,
                'average_score': average_score,
                'total_tasks': len(task_results)
            }
    
    return summary_stats

def calculate_api_summary_statistics(results: Dict[str, Dict[str, List[Dict[str, Any]]]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Calculate summary statistics from API benchmark results.
    Similar to calculate_summary_statistics but excludes memory usage for API models.
    
    Args:
        results: Nested dictionary with structure {model: {task_type: [task_results]}}
        
    Returns:
        Dictionary with summary statistics for each API model and task type
    """
    summary_stats = {}
    
    for model, task_types in results.items():
        summary_stats[model] = {}
        for task_type, task_results in task_types.items():
            durations = [result['duration'] for result in task_results if 'duration' in result]
            scores = [result['score'] for result in task_results if result['score'] is not None]

            average_duration = np.mean(durations) if durations else 0
            average_score = np.mean(scores) if scores else None

            summary_stats[model][task_type] = {
                'average_duration': average_duration,
                'average_memory_usage': 0,  # Not applicable for API models
                'average_score': average_score,
                'total_tasks': len(task_results)
            }
    
    return summary_stats 