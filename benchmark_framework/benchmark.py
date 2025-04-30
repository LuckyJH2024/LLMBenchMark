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
from rouge import Rouge
from bert_score import score as bert_score_lib



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
    
    def call_model_api(self, model_name, prompt):
        # 统一格式化 prompt（Claude 必需，其他模型兼容）
        formatted_prompt = f"Human: {prompt}\n\nAssistant:"

        if "gpt" in model_name.lower():
            api_url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
                "Content-Type": "application/json"
            }
            data = {
                "model": model_name,
                "messages": [{"role": "user", "content": formatted_prompt}],
                "temperature": 0.7
            }
            response = requests.post(api_url, headers=headers, json=data)
            return response.json()["choices"][0]["message"]["content"]

        elif "deepseek" in model_name.lower():
            api_url = "https://api.deepseek.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}",
                "Content-Type": "application/json"
            }
            data = {
                "model": model_name,
                "messages": [{"role": "user", "content": formatted_prompt}],
                "temperature": 0.7
            }
            response = requests.post(api_url, headers=headers, json=data)
            return response.json()["choices"][0]["message"]["content"]

        elif "claude" in model_name.lower():
            api_url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": os.getenv("ANTHROPIC_API_KEY"),
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json"
            }
            data = {
                "model": model_name,
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": formatted_prompt}]
            }
            response = requests.post(api_url, headers=headers, json=data)
            return response.json()["content"][0]["text"]

        else:
            raise ValueError(f"Unsupported model: {model_name}")



    def benchmark_task(self, model, task_data):
        results = []
        for task in task_data:
            print(f"Running task with prompt: {task['prompt']}")
            start_time = time.time()

            try:
                model_response = self.call_model_api(model, task["prompt"])
                print(f"Model replied: {model_response.strip()}")
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
            return self.hybrid_summarization_score(response, ground_truth)
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

        # similarity_score = self.text_similarity(response, ground_truth)
        # final_score = 0.4 * keyword_score + 0.6 * similarity_score

        # print(f"[QA DEBUG] HIT: {keyword_score}, SIM: {similarity_score}, FINAL: {final_score}")
        return keyword_score

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


    def hybrid_summarization_score(self, prediction: str, reference: str) -> float:
        # Step 1: 计算 ROUGE-L F1
        try:
            rouge = Rouge()
            rouge_f1 = rouge.get_scores(prediction, reference)[0]['rouge-l']['f']
        except Exception as e:
            print(f"[ROUGE error] {e}")
            rouge_f1 = 0.0

        # Step 2: 计算 BERTScore F1
        try:
            P, R, F1 = bert_score_lib([prediction], [reference], lang="en", verbose=False)
            bert_f1 = F1[0].item()
        except Exception as e:
            print(f"[BERTScore error] {e}")
            bert_f1 = 0.0

        # Step 3: 加权平均
        final_score = 0.5 * rouge_f1 + 0.5 * bert_f1
        return round(final_score, 4)


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
