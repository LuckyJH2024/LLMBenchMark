#!/usr/bin/env python3

import time
import json
import requests
import os
from typing import Dict, Any, List, Optional, Union
from tqdm import tqdm

class APIModelBenchmark:
    """
    A class for benchmarking cloud-based LLM APIs like OpenAI/ChatGPT and DeepSeek.
    This complements the local LLM benchmarking with Ollama.
    """
    
    def __init__(self):
        # Load API keys from environment variables
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        self.deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
        self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        
        # Base URLs for different API services
        self.openai_base_url = "https://api.openai.com/v1/chat/completions"
        self.deepseek_base_url = "https://api.deepseek.com/v1/chat/completions"
        self.anthropic_base_url = "https://api.anthropic.com/v1/messages"
        
        # Available API models
        self.available_models = {
            "openai": ["gpt-3.5-turbo", "gpt-4o", "gpt-4-turbo"],
            "deepseek": ["deepseek-chat", "deepseek-coder"],
            "anthropic": [
                "claude-3-opus", 
                "claude-3-sonnet", 
                "claude-3-5-sonnet", 
                "claude-3-5-sonnet-20240620",
                "claude-3-7-sonnet-20250219",
                "claude3.5",  # 简化别名
                "claude3.7",  # 简化别名
            ]
        }
        
        # Default timeout for API calls in seconds
        self.timeout = 60
    
    def set_timeout(self, timeout: int):
        """
        Set the timeout for API calls.
        
        Args:
            timeout: Timeout in seconds
        """
        self.timeout = timeout
        print(f"API timeout set to {timeout} seconds")
    
    def get_available_api_models(self) -> Dict[str, List[str]]:
        """Returns a dictionary of available API models by provider."""
        return self.available_models
    
    def check_api_keys(self) -> Dict[str, bool]:
        """Checks if API keys are properly configured."""
        return {
            "openai": self.openai_api_key is not None,
            "deepseek": self.deepseek_api_key is not None,
            "anthropic": self.anthropic_api_key is not None
        }
    
    def format_prompt(self, prompt: str, model_type: str) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Format prompt according to the API requirements of different providers."""
        if model_type.startswith("gpt"):
            return [{"role": "user", "content": prompt}]
        elif model_type.startswith("deepseek"):
            return [{"role": "user", "content": prompt}]
        elif model_type.startswith("claude"):
            return {"role": "user", "content": prompt}
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def call_api_model(self, model: str, prompt: str) -> Dict[str, Any]:
        """
        Call an API-based model and return the response with metadata.
        
        Args:
            model: Full model name including provider prefix (e.g., "openai:gpt-4")
            prompt: The input prompt to send to the model
            
        Returns:
            Dictionary with response text and metadata (latency, etc.)
        """
        start_time = time.time()
        provider, model_name = model.split(":", 1)
        
        try:
            if provider == "openai":
                if not self.openai_api_key:
                    raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
                
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.openai_api_key}"
                }
                data = {
                    "model": model_name,
                    "messages": self.format_prompt(prompt, model_name)
                }
                response = requests.post(self.openai_base_url, headers=headers, json=data, timeout=self.timeout)
                response.raise_for_status()
                result = response.json()
                response_text = result["choices"][0]["message"]["content"]
                
            elif provider == "deepseek":
                if not self.deepseek_api_key:
                    raise ValueError("DeepSeek API key not found. Set DEEPSEEK_API_KEY environment variable.")
                
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.deepseek_api_key}"
                }
                data = {
                    "model": model_name,
                    "messages": self.format_prompt(prompt, model_name)
                }
                response = requests.post(self.deepseek_base_url, headers=headers, json=data, timeout=self.timeout)
                response.raise_for_status()
                result = response.json()
                response_text = result["choices"][0]["message"]["content"]
                
            elif provider == "anthropic":
                if not self.anthropic_api_key:
                    raise ValueError("Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable.")
                
                # 使用最新的Anthropic API格式
                headers = {
                    "Content-Type": "application/json",
                    "x-api-key": self.anthropic_api_key,
                    "anthropic-version": "2023-06-01"
                }
                
                # 根据用户输入的模型名映射到Anthropic API正确的模型名
                # 处理各种简化输入格式
                if model_name == "claude3.7" or model_name == "claude-3.7" or model_name == "claude3-7":
                    api_model_name = "claude-3-7-sonnet-20250219"
                elif model_name == "claude3.5" or model_name == "claude-3.5" or model_name == "claude3-5":
                    api_model_name = "claude-3-5-sonnet-20240620"
                # 处理标准格式名称
                elif model_name == "claude-3.7-sonnet" or model_name == "claude-3-7-sonnet":
                    api_model_name = "claude-3-7-sonnet-20250219"
                elif model_name == "claude-3.5-sonnet" or model_name == "claude-3-5-sonnet":
                    api_model_name = "claude-3-5-sonnet-20240620"
                # 如果用户已经提供了完整版本号，则直接使用
                elif model_name == "claude-3-7-sonnet-20250219" or model_name == "claude-3-5-sonnet-20240620":
                    api_model_name = model_name
                elif model_name == "claude-3-opus":
                    api_model_name = "claude-3-opus-20240229"
                elif model_name == "claude-3-sonnet":
                    api_model_name = "claude-3-sonnet-20240229"
                elif model_name == "claude-3-haiku":
                    api_model_name = "claude-3-haiku-20240307"
                else:
                    api_model_name = model_name

                # 准备请求数据
                data = {
                    "model": api_model_name,
                    "messages": [
                        self.format_prompt(prompt, model_name)
                    ],
                    "max_tokens": 4000
                }
                
                response = requests.post(self.anthropic_base_url, headers=headers, json=data, timeout=self.timeout)
                
                if response.status_code != 200:
                    print(f"API error (HTTP {response.status_code})")
                    response.raise_for_status()
                
                result = response.json()
                if "content" in result and len(result["content"]) > 0 and "text" in result["content"][0]:
                    response_text = result["content"][0]["text"]
                else:
                    print(f"Unexpected API response format")
                    response_text = "Error: Unexpected API response format"
                
            else:
                raise ValueError(f"Unsupported provider: {provider}")
                
        except Exception as e:
            print(f"API call error: {str(e)}")
            response_text = f"Error: {str(e)}"
            
        duration = time.time() - start_time
        
        return {
            "text": response_text,
            "duration": duration,
            "provider": provider,
            "model": model_name
        }
    
    def benchmark_api_model(self, model: str, tasks: List[Dict[str, Any]], task_type: str) -> List[Dict[str, Any]]:
        """
        Run benchmark tasks on an API-based model.
        
        Args:
            model: Full model name including provider prefix (e.g., "openai:gpt-4")
            tasks: List of task dictionaries with prompts and ground truths
            task_type: Type of task (qa, reasoning, etc.)
            
        Returns:
            List of task results with scores and metrics
        """
        results = []
        
        print(f"正在运行{len(tasks)}个{task_type}任务...")
        for task in tqdm(tasks, desc=f"测试 {model} 的{task_type}任务", ncols=100):
            # Call the API model
            api_response = self.call_api_model(model, task['prompt'])
            model_response = api_response["text"]
            duration = api_response["duration"]
            
            # Store the results (similar structure to LLMBenchmark results)
            task_result = task.copy()
            task_result.update({
                "response": model_response,
                "duration": duration,
                "memory_usage": 0,  # Not applicable for API models
            })
            
            # Note: This assumes scoring will be handled separately
            # by the existing LLMBenchmark.evaluate methods
            
            results.append(task_result)
        
        print(f"{model}的{task_type}测试已完成!")
        return results 