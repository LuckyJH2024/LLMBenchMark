import time
import threading
import ollama
import psutil
import os
import json
import numpy as np
import requests
from sentence_transformers import SentenceTransformer, util
from transformers import RobertaTokenizer, RobertaModel
import torch
import re

class LLMBenchmark:
    def __init__(self, models, tasks):
        self.models = models
        self.tasks = tasks
        self.results = {}
        self._stop_thinking = False
        self.sbert = SentenceTransformer("all-MiniLM-L6-v2")

        os.makedirs('results', exist_ok=True)

    def run_benchmarks(self):
        for model in self.models:
            self.results[model] = {}
            for task_type, tasks in self.tasks.items():
                print(f"Running {task_type} tasks for model {model}")
                self.results[model][task_type] = self.benchmark_task(model, tasks)
                safe_model_name = model.replace(":", "-")  # 处理非法文件名字符
                self.save_results(f"results/{safe_model_name}_{task_type}.json", self.results[model][task_type])
        return self.results

    def benchmark_task(self, model, task_data):
        results = []
        for task in task_data:
            print(f"Running task with prompt: {task['prompt']}")
            start_time = time.time()

            try:
                api_url = "http://localhost:11434/api/generate"
                payload = {
                    "model": model,
                    "prompt": task["prompt"],
                    "stream": False
                }
                response = requests.post(api_url, json=payload)
                model_response = response.json().get("response", "")
            except Exception as e:
                model_response = f"Error: {e}"

            duration = time.time() - start_time
            memory_usage = psutil.Process(os.getpid()).memory_info().rss

            if "ground_truth" in task:
                score = self.evaluate(model_response, task["ground_truth"], task.get("task_type"))
            else:
                score = None

            results.append({
                "prompt": task["prompt"],
                "response": model_response,
                "duration": duration,
                "memory_usage": memory_usage,
                "score": score
            })

        return results

    def _thinking_animation(self):
        print("Thinking: ", end="")
        animation_chars = ['|', '/', '-', '\\']
        idx = 0
        while not self._stop_thinking:
            print(animation_chars[idx % 4], end="\r")
            idx += 1
            time.sleep(0.2)

    def evaluate(self, response, ground_truth, task_type):
        response = response.strip().lower()
        ground_truth = ground_truth.strip().lower()

        if not response or not ground_truth:
            return 0.0

        if task_type == "qa":
            return self.evaluate_qa_style(response, ground_truth)
        elif task_type == "summarization":
            return self.text_overlap(response, ground_truth)
        elif task_type == "code":
            return self.code_similarity(response, ground_truth)
        elif task_type == "reasoning":
            return self.evaluate_reasoning_style(response, ground_truth)
        else:
            return self.text_similarity(response, ground_truth)

    def evaluate_qa_style(self, response, ground_truth):
        response_clean = re.sub(r'[^\w\s]', '', response.lower())
        truth_clean = re.sub(r'[^\w\s]', '', ground_truth.lower())

        keyword_score = 1.0 if truth_clean in response_clean else \
                        0.5 if any(word in response_clean for word in truth_clean.split()) else 0.0

        similarity_score = self.text_similarity(response, ground_truth)
        final_score = 0.4 * keyword_score + 0.6 * similarity_score

        print(f"[QA DEBUG] HIT: {keyword_score}, SIM: {similarity_score}, FINAL: {final_score}")
        return round(final_score, 4)

    def evaluate_reasoning_style(self, response, ground_truth):
        response_clean = re.sub(r'[^\w\s]', '', response.lower())
        truth_clean = re.sub(r'[^\w\s]', '', ground_truth.lower())

        # 简化关键词匹配逻辑
        hit_score = 1.0 if truth_clean in response_clean else \
                    0.5 if any(word in response_clean for word in truth_clean.split()) else 0.0

        semantic_score = self.text_similarity(response, ground_truth)
        final_score = 0.3 * hit_score + 0.7 * semantic_score

        print(f"[REASONING DEBUG] HIT: {hit_score}, SIM: {semantic_score}, FINAL: {final_score}")
        return round(final_score, 4)

    def text_similarity(self, text1, text2):
        emb1 = self.sbert.encode(text1, convert_to_tensor=True)
        emb2 = self.sbert.encode(text2, convert_to_tensor=True)
        cosine_score = util.cos_sim(emb1, emb2)
        return round(cosine_score.item(), 4)

    def text_overlap(self, summary, reference):
        summary_words = set(summary.split())
        ref_words = set(reference.split())
        if not ref_words:
            return 0.0
        return len(summary_words.intersection(ref_words)) / len(ref_words)

    def init_codebert(self):
        print("Loading CodeBERT model...")
        self.codebert_tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        self.codebert_model = RobertaModel.from_pretrained("microsoft/codebert-base")

    def code_similarity(self, code1, code2):
        if not hasattr(self, "codebert_model"):
            self.init_codebert()

        def embed(code):
            inputs = self.codebert_tokenizer(code, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.codebert_model(**inputs)
            return outputs.last_hidden_state.mean(dim=1)

        emb1 = embed(code1)
        emb2 = embed(code2)

        cosine_score = torch.nn.functional.cosine_similarity(emb1, emb2).item()
        return round(cosine_score, 4)

    def get_summary_statistics(self):
        summary_stats = {}
        for model, task_types in self.results.items():
            summary_stats[model] = {}
            for task_type, results in task_types.items():
                durations = [result['duration'] for result in results if 'duration' in result]
                memory_usages = [result['memory_usage'] for result in results if 'memory_usage' in result]
                scores = [result['score'] for result in results if result['score'] is not None]

                average_duration = np.mean(durations) if durations else 0
                average_memory_usage = np.mean(memory_usages) if memory_usages else 0
                average_score = np.mean(scores) if scores else None

                summary_stats[model][task_type] = {
                    'average_duration': average_duration,
                    'average_memory_usage': average_memory_usage,
                    'average_score': average_score,
                    'total_tasks': len(results)
                }

        return summary_stats

    def save_results(self, filename, data):
        with open(filename, 'w') as file:
            json.dump(data, file, indent=4)
