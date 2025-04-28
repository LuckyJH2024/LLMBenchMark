import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi

def generate_report(summary, all_results, output_dir='results', prefix=''):
    os.makedirs(output_dir, exist_ok=True)

    summary_df = pd.DataFrame.from_dict({(i, j): summary[i][j] 
                                         for i in summary.keys() 
                                         for j in summary[i].keys()},
                                        orient='index')
    summary_df.reset_index(inplace=True)
    summary_df.columns = ['Model', 'Task Type', 'Average Duration', 'Average Memory Usage', 'Average Score', 'Total Tasks']

    report_sections = [
        "# Benchmark Report",
        "## Summary Statistics\n",
        summary_df.to_markdown(index=False),
        "## Detailed Results\n"
    ]

    for model, tasks in all_results.items():
        report_sections.append(f"### Model: {model}\n")
        for task, results in tasks.items():
            df = pd.DataFrame(results)
            report_sections.append(f"#### Task Type: {task}\n")
            if task == "reasoning":
                cols = ["prompt", "response", "answer_score", "chain_score", "consistency_score", "score"]
                filtered = df[[col for col in cols if col in df.columns]].copy()
                report_sections.append(filtered.to_markdown(index=False))
            else:
                report_sections.append(df.to_markdown(index=False))

    full_report = "\n\n".join(report_sections)
    report_filename = os.path.join(output_dir, f"{prefix}benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
    with open(report_filename, 'w') as file:
        file.write(full_report)
    print(f"Report saved to {report_filename}")
