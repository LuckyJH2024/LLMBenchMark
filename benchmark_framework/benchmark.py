import time
import threading
import ollama
import psutil
from bert_score import score
import os
import json
import numpy as np
import requests
from typing import List, Dict, Any
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
                self.results[model][task_type] = self.benchmark_task(model, tasks, task_type)
                safe_model_name = model.replace(":", "-")  # 处理非法文件名字符
                self.save_results(f"results/{safe_model_name}_{task_type}.json", self.results[model][task_type])
        return self.results

    def benchmark_task(self, model, task_data, task_type):
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

            task["response"] = model_response  # ⬅️ 非常关键！传入给 reasoning scorer

            if "ground_truth" in task:
                if task_type == "reasoning":
                    score_detail = self.evaluate_reasoning_all(task)
                    task.update(score_detail)  # ⬅️ 一次性保留全部字段
                    task["score"] = score_detail["answer_score"]  # legacy 兼容性字段
                else:
                    task["score"] = self.evaluate(model_response, task["ground_truth"], task_type)
            else:
                task["score"] = None

            task.update({
                "duration": duration,
                "memory_usage": memory_usage,
            })

            results.append(task)  # ⬅️ 保存完整字段

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
        # Semantic alignment
        try:
            P, R, F1 = score([response], [ground_truth], lang='en', verbose=False)
            semantic_score = F1.item()
        except Exception:
            semantic_score = self.text_similarity(response, ground_truth)

        # Keyword containment
        truth_words = set(ground_truth.lower().split())
        response_words = set(response.lower().split())
        hit_score = 1.0 if truth_words <= response_words else 0.5 if truth_words & response_words else 0.0

        final_score = 0.3 * hit_score + 0.7 * semantic_score
        print(f"[REASONING DEBUG] HIT: {hit_score}, SIM: {semantic_score}, FINAL: {final_score}")
        return round(final_score, 4)
    
    def evaluate_reasoning_all(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        response = sample.get("response", "")
        ground_truth = sample.get("ground_truth", "")
        steps = sample.get("reasoning_steps", [])
        ref_steps = sample.get("reference_steps", [])
        paraphrased = sample.get("paraphrased_response", "")

        response_clean = response.strip().lower()
        ground_clean = ground_truth.strip().lower()
        if response_clean == ground_clean:
            answer_score = 1.0
        else:
            try:
                P, R, F1 = score([response], [ground_truth], lang='en', verbose=False)
                answer_score = round(F1.item(), 4)
            except:
                answer_score = self.text_similarity(response, ground_truth)

        if steps and ref_steps:
            total = 0
            for pred, ref in zip(steps, ref_steps):
                total += self.text_similarity(pred, ref)
            chain_score = round(total / len(ref_steps), 4)
        else:
            chain_score = None

        consistency_score = self.text_similarity(response, paraphrased) if paraphrased else None

        return {
            "answer_score": answer_score,
            "chain_score": chain_score,
            "consistency_score": consistency_score
        }
    
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

    def save_results(self, filename, data):
        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4)
