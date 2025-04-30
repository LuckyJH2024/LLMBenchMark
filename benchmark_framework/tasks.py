import json
import os

def create_qa_benchmark(file_path):
    """
    Load and format question-answering benchmark tasks.

    Args:
        file_path (str): Path to the JSON file containing QA pairs

    Returns:
        list: List of formatted benchmark tasks
    """
    # Step 1: Load JSON file
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No such file: '{file_path}'")   

    with open(file_path, 'r', encoding="utf-8") as file:
        data = json.load(file)

    # Step 2: Format each question into a prompt
    tasks = []
    for item in data:
        question = item.get("question", "")
        answer = item.get("answer", "")
        prompt = f"Question: {question}\nAnswer:"
        tasks.append({'prompt': prompt, 'ground_truth': answer})

    # Step 3: Return list of tasks with prompts and ground truth
    return tasks


def create_code_benchmark(file_path):
    """
    Load and format code generation benchmark tasks.

    Args:
        file_path (str): Path to the JSON file containing coding problems

    Returns:
        list: List of formatted benchmark tasks
    """
    # Step 1: Load JSON file
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No such file: '{file_path}'")

    with open(file_path, 'r') as file:
        data = json.load(file)

    # Step 2: Format each coding problem into a task
    tasks = []
    for item in data:
        problem_description = item.get("description", "")
        starter_code = item.get("starter_code", "")
        prompt = f"Problem: {problem_description}\nStarter Code: {starter_code}\nComplete the code:"
        tasks.append({'prompt': prompt, 'ground_truth': item.get("solution", "")})

    # Step 3: Return list of tasks with prompts and ground truth
    return tasks


def create_reasoning_benchmark(file_path):
    """
    Load and format reasoning benchmark tasks.

    Args:
        file_path (str): Path to the JSON file containing reasoning problems

    Returns:
        list: List of formatted benchmark tasks
    """
    # Step 1: Load JSON file
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No such file: '{file_path}'")

    with open(file_path, 'r') as file:
        data = json.load(file)

    # Step 2: Format each reasoning problem into a task
    tasks = []
    for item in data:
        context = item.get("context", "")
        question = item.get("question", "")
        answer = item.get("answer", "")
        prompt = f"Context: {context}\nQuestion: {question}\nWhat is the reasoning?"
        tasks.append({'prompt': prompt, 'ground_truth': answer})

    # Step 3: Return list of tasks with prompts and ground truth
    return tasks


def create_summarization_benchmark(file_path):
    """
    Load and format text summarization benchmark tasks.

    Args:
        file_path (str): Path to the JSON file containing summarization tasks

    Returns:
        list: List of formatted benchmark tasks
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No such file: '{file_path}'")

    with open(file_path, 'r', encoding="utf-8") as file:
        data = json.load(file)

    tasks = []
    for item in data:
        text = item.get("text", "")
        summary = item.get("summary", "")
        prompt = f"Summarize the following text:\n{text}\nSummary:"
        tasks.append({'prompt': prompt, 'ground_truth': summary})

    return tasks

def load_all_benchmarks(data_dir="data"):
    """
    Load all benchmark tasks from the data directory.

    Args:
        data_dir (str): Path to directory containing benchmark data files

    Returns:
        dict: Dictionary mapping task types to lists of benchmark tasks
    """
    benchmarks = {}
    task_types = {
        "qa": "qa_benchmark.json",
        "code": "code_benchmark.json",
        "summarization": "summarization_benchmark.json",
        "reasoning": "reasoning_benchmark.json"
    }

    for task_type, file_name in task_types.items():
        file_path = os.path.join(data_dir, file_name)
        if os.path.exists(file_path):
            if task_type == "qa":
                benchmarks[task_type] = create_qa_benchmark(file_path)
            elif task_type == "code":
                benchmarks[task_type] = create_code_benchmark(file_path)
            elif task_type == "summarization":
                benchmarks[task_type] = create_summarization_benchmark(file_path)
            elif task_type == "reasoning":
                benchmarks[task_type] = create_reasoning_benchmark(file_path)
        else:
            print(f"Warning: No data file found for {task_type} at {file_path}")

    return benchmarks
