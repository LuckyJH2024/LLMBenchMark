import json
import os

def create_qa_benchmark(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No such file: '{file_path}'")   

    with open(file_path, 'r') as file:
        data = json.load(file)

    tasks = []
    for item in data:
        question = item.get("question", "")
        answer = item.get("answer", "")
        prompt = f"Question: {question}\nAnswer:"
        tasks.append({'prompt': prompt, 'ground_truth': answer, 'task_type': 'qa'})

    return tasks

def create_code_benchmark(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No such file: '{file_path}'")

    with open(file_path, 'r') as file:
        data = json.load(file)

    tasks = []
    for item in data:
        problem_description = item.get("description", "")
        starter_code = item.get("starter_code", "")
        prompt = f"Problem: {problem_description}\nStarter Code: {starter_code}\nComplete the code:"
        tasks.append({'prompt': prompt, 'ground_truth': item.get("solution", ""), 'task_type': 'code'})

    return tasks

def create_summarization_benchmark(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No such file: '{file_path}'")

    with open(file_path, 'r') as file:
        data = json.load(file)

    tasks = []
    for item in data:
        text = item.get("text", "")
        summary = item.get("summary", "")
        prompt = f"Summarize the following text:\n{text}\nSummary:"
        tasks.append({'prompt': prompt, 'ground_truth': summary, 'task_type': 'summarization'})

    return tasks

def create_reasoning_full_benchmark(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No such file: '{file_path}'")

    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    tasks = []
    for item in data:
        context = item.get("context", "")
        question = item.get("question", "")
        ground_truth = item.get("ground_truth", item.get("answer", ""))
        response = item.get("response", "")
        steps = item.get("reasoning_steps", [])
        ref_steps = item.get("reference_steps", [])
        paraphrase = item.get("paraphrased_response", "")

        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"

        tasks.append({
            'prompt': prompt,
            'ground_truth': ground_truth,
            'context': context,
            'question': question,
            'response': response,
            'reasoning_steps': steps,
            'reference_steps': ref_steps,
            'paraphrased_response': paraphrase,
            'task_type': 'reasoning'
        })

    return tasks

def load_all_benchmarks(data_dir="data"):
    benchmarks = {}
    task_types = {
        # "qa": "qa_benchmark.json",
        # "code": "code_benchmark.json",
        # "summarization": "summarization_benchmark.json",
        "reasoning": "sample_reasoning_eval.json"
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
                benchmarks[task_type] = create_reasoning_full_benchmark(file_path)
        else:
            print(f"Warning: No data file found for {task_type} at {file_path}")

    return benchmarks
