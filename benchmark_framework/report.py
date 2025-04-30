import os
import re
import json
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi

def clean_code_response(text, task_type=None):
    """For code tasks: keep all content but remove fenced code blocks, replacing them with <br>. 
    For non-code tasks: replace newlines with <br>."""
    if not isinstance(text, str):
        return text

    if task_type == "code":
        # Remove fenced code block markers like ```python or ```
        text = re.sub(r"```(?:python)?\n", "", text)
        text = re.sub(r"```", "", text)
        # Use <br> to preserve visual line breaks
        return "<br>".join(text.strip().splitlines())

    # Handling for non-code tasks
    return "<br>".join(text.strip().splitlines())



def generate_report(summary, all_results, output_dir='results'):
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

    '''for model, tasks in all_results.items():
        report_sections.append(f"### Model: {model}\n")
        for task, results in tasks.items():
            df = pd.DataFrame(results)

            df["prompt"] = df["prompt"].apply(lambda x: clean_code_response(x, task_type=task))
            df["response"] = df["response"].apply(lambda x: clean_code_response(x, task_type=task))


            report_sections.append(f"#### Task Type: {task}\n")
            report_sections.append(df.to_markdown(index=False))'''

    full_report = "\n\n".join(report_sections)
    report_filename = os.path.join(output_dir, f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
    with open(report_filename, 'w') as file:
        file.write(full_report)
    print(f"Report saved to {report_filename}")
