import os
import re
import json
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def clean_code_response(text, task_type=None):
    if not isinstance(text, str):
        return text

    if task_type == "code":
        text = re.sub(r"```(?:python)?\n", "", text)
        text = re.sub(r"```", "", text)
        return "<br>".join(text.strip().splitlines())

    return "<br>".join(text.strip().splitlines())


def generate_report(summary, all_results, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)

    # Plot settings
    sns.set(style="whitegrid")

    # Extract model-task-score for plotting
    plot_data = []
    for model, tasks in all_results.items():
        for task_type, results in tasks.items():
            for record in results:
                score = record.get("score", None)
                if score is not None:
                    plot_data.append({
                        "Model": model,
                        "Task Type": task_type,
                        "Score": score
                    })

    plot_df = pd.DataFrame(plot_data)

    # Create bar plots
    if not plot_df.empty:
        # Barplot: Average score per model for QA tasks
        qa_df = plot_df[plot_df['Task Type'] == 'qa']
        if not qa_df.empty:
            plt.figure(figsize=(8, 6))
            sns.barplot(data=qa_df, x="Model", y="Score", ci="sd")
            plt.title("QA Task - Average Scores by Model")
            plt.ylim(0, 1)
            plt.xticks(rotation=45)
            plt.tight_layout()
            qa_plot_path = os.path.join(output_dir, f"qa_scores_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.savefig(qa_plot_path)
            plt.close()
            print(f"QA Score plot saved to {qa_plot_path}")

        # Barplot: Average score per model for Summarization tasks
        summarization_df = plot_df[plot_df['Task Type'] == 'summarization']
        if not summarization_df.empty:
            plt.figure(figsize=(8, 6))
            sns.barplot(data=summarization_df, x="Model", y="Score", ci="sd")
            plt.title("Summarization Task - Average Scores by Model")
            plt.ylim(0, 1)
            plt.xticks(rotation=45)
            plt.tight_layout()
            summarization_plot_path = os.path.join(output_dir, f"summarization_scores_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.savefig(summarization_plot_path)
            plt.close()
            print(f"Summarization Score plot saved to {summarization_plot_path}")

    # Save a minimal report
    report_sections = [
        "# Benchmark Report",
        "## Note",
        "Bar plots for QA and Summarization scores have been generated and saved separately."
    ]

    full_report = "\n\n".join(report_sections)
    report_filename = os.path.join(output_dir, f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
    with open(report_filename, 'w') as file:
        file.write(full_report)
    print(f"Report saved to {report_filename}")
