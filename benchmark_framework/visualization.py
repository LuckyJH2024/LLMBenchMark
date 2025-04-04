import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
def set_plotting_style():
    plt.style.use('ggplot')
    sns.set_context("talk")
    sns.set_palette("pastel")


def create_visualizations(summary, results_dir='results'):
    os.makedirs(results_dir, exist_ok=True)

    df = pd.DataFrame.from_dict({(i, j): summary[i][j] 
                                 for i in summary.keys() 
                                 for j in summary[i].keys()},
                                orient='index')
    df.reset_index(inplace=True)
    df.columns = ['model', 'task_type', 'average_duration', 'average_memory_usage', 'average_score', 'total_tasks']

    plt.figure(figsize=(10, 6))
    sns.barplot(x='model', y='average_duration', hue='task_type', data=df)
    plt.title('Average Duration by Model and Task')
    plt.ylabel('Duration (s)')
    plt.savefig(f"{results_dir}/average_duration.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.barplot(x='model', y='average_memory_usage', hue='task_type', data=df)
    plt.title('Average Memory Usage by Model and Task')
    plt.ylabel('Memory Usage (bytes)')
    plt.savefig(f"{results_dir}/average_memory_usage.png")
    plt.close()


def create_performance_dashboard(df, results_dir):
    sns.pairplot(df, hue="model", markers=["o", "s", "D"])
    plt.savefig(f"{results_dir}/performance_dashboard.png")
    plt.close()


def create_enhanced_radar_chart(df, results_dir):
    categories = list(df)[1:]
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    plt.xticks(angles[:-1], categories, color='grey', size=8)
    ax.set_rlabel_position(0)
    plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5], ["0.1", "0.2", "0.3", "0.4", "0.5"], color="grey", size=7)
    plt.ylim(0, 0.5)

    for i in range(len(df)):
        values = df.iloc[i].drop('model').values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=df.iloc[i]['model'])
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.savefig(f"{results_dir}/radar_chart.png")
    plt.close()


def create_enhanced_heatmap(summary, results_dir):
    df = pd.DataFrame(summary).transpose()
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Heatmap of Model Performance and Latency')
    plt.savefig(f"{results_dir}/performance_heatmap.png")
    plt.close()


set_plotting_style()
