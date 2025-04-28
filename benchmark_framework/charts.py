#!/usr/bin/env python3

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi

def set_plotting_style():
    """Set consistent style for all plots."""
    plt.style.use('ggplot')
    sns.set_context("talk")
    sns.set_palette("pastel")

def create_duration_chart(df, results_dir):
    """Create bar chart of average durations by model and task type."""
    plt.figure(figsize=(10, 6))
    sns.barplot(x='model', y='average_duration', hue='task_type', data=df)
    plt.title('Average Duration by Model and Task')
    plt.ylabel('Duration (s)')
    plt.savefig(f"{results_dir}/average_duration.png")
    plt.close()
    print(f"✅ Duration chart saved to: {results_dir}/average_duration.png")

def create_memory_chart(df, results_dir):
    """Create bar chart of average memory usage by model and task type."""
    plt.figure(figsize=(10, 6))
    sns.barplot(x='model', y='average_memory_usage', hue='task_type', data=df)
    plt.title('Average Memory Usage by Model and Task')
    plt.ylabel('Memory Usage (bytes)')
    plt.savefig(f"{results_dir}/average_memory_usage.png")
    plt.close()
    print(f"✅ Memory usage chart saved to: {results_dir}/average_memory_usage.png")

def create_performance_dashboard(df, results_dir):
    """Create a pairplot dashboard of all performance metrics."""
    sns.pairplot(df, hue="model", markers=["o", "s", "D"])
    plt.savefig(f"{results_dir}/performance_dashboard.png")
    plt.close()
    print(f"✅ Performance dashboard saved to: {results_dir}/performance_dashboard.png")

def create_enhanced_radar_chart(df, results_dir):
    """Create a radar chart for reasoning capabilities comparison."""
    if df.empty:
        print("⚠️ No reasoning score data available for radar chart.")
        return

    categories = list(df.columns[1:])
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    plt.xticks(angles[:-1], categories, color='grey', size=10)
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=8)
    plt.ylim(0, 1)

    for i in range(len(df)):
        values = df.iloc[i].drop('model').values.flatten().tolist()
        values += values[:1]
        label = df.iloc[i]['model']
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=label)
        ax.fill(angles, values, alpha=0.1)

    plt.title("LLM Reasoning Capability Comparison", size=14, y=1.08)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    save_path = os.path.join(results_dir, "radar_chart.png")
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Radar chart saved to: {save_path}")

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
        df = pd.DataFrame(score_data)
        
        # Create the heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(df, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Heatmap of Model Performance Scores')
        plt.savefig(f"{results_dir}/performance_heatmap.png")
        plt.close()
        print(f"✅ Performance heatmap saved to: {results_dir}/performance_heatmap.png")
    except Exception as e:
        print(f"⚠️ Error creating heatmap: {e}")
        print("Continuing with other visualizations...")

def create_reasoning_bar_chart(df_mean, results_dir):
    """Create a bar chart comparing reasoning subscores by model."""
    df_bar = df_mean.set_index("model")[["answer_score", "chain_score", "consistency_score"]]
    ax = df_bar.plot(kind="bar", figsize=(10, 6))
    plt.title("Reasoning Subscores by Model")
    plt.ylabel("Score")
    plt.ylim(0, 1.0)
    plt.xticks(rotation=0)
    plt.legend(title="Subscore", bbox_to_anchor=(1.02, 1), loc="upper left")
    bar_path = os.path.join(results_dir, "reasoning_bar_comparison.png")
    plt.tight_layout()
    plt.savefig(bar_path)
    plt.close()
    print(f"✅ Reasoning bar chart saved to: {bar_path}")

def create_reasoning_score_comparison(results_dir="results"):
    """Create comparison charts for reasoning task scores."""
    reasoning_files = [f for f in os.listdir(results_dir) if f.endswith("_reasoning.json")]
    records = []

    for file in reasoning_files:
        path = os.path.join(results_dir, file)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            model_name = file.replace("_reasoning.json", "")
            for item in data:
                if all(k in item for k in ["answer_score", "chain_score", "consistency_score"]):
                    records.append({
                        "model": model_name,
                        "answer_score": item["answer_score"],
                        "chain_score": item["chain_score"],
                        "consistency_score": item["consistency_score"]
                    })
        except Exception as e:
            print(f"⚠️ Failed to load {file}: {e}")

    df = pd.DataFrame(records)
    if df.empty:
        print("⚠️ No complete reasoning scores found in any file.")
        return

    df_mean = df.groupby("model").mean().reset_index()
    print("✅ Reasoning data loaded. Generating radar chart...")
    create_enhanced_radar_chart(df_mean, results_dir)
    create_reasoning_bar_chart(df_mean, results_dir)

# Initialize plotting style
set_plotting_style() 