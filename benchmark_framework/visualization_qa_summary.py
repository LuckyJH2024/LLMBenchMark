import os
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def generate_report(summary, all_results, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)

    sns.set(style="whitegrid", palette="pastel")


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

    # QA plot
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

    # Summarization plot
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
