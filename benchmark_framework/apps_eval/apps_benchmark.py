#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
APPS Benchmark for Language Models
---------------------------------
This module runs benchmarks on programming problems from the APPS dataset,
focusing on interview and competition level problems.

APPS (Automated Programming Progress Standard) is a benchmark of coding problems
with varying difficulty levels: introductory, interview, and competition.
"""

# First check for required dependencies
import sys
import importlib
from typing import List
import os
# 添加项目根目录到sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))


def check_dependencies(dependencies: List[str]) -> None:
    """Check if required dependencies are installed and provide clear error messages if not."""
    missing_packages = []
    for package in dependencies:
        try:
            importlib.import_module(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("错误: 缺少必要的依赖库。请使用以下命令安装:")
        print(f"pip install {' '.join(missing_packages)}")
        sys.exit(1)


# List of required dependencies (excluding standard library modules)
DEPENDENCIES = [
    "os", "json", "requests", "pandas", "numpy", "matplotlib", 
    "seaborn", "tqdm", "concurrent.futures"
]

# Check for dependencies first
check_dependencies([dep for dep in DEPENDENCIES if dep not in ["os", "json", "concurrent.futures"]])

# Import remaining libraries after dependency check
import json
import time
import random
import argparse
import requests
import subprocess
import tempfile
import traceback
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from typing import Dict, List, Any, Optional, Union, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from benchmark_framework.api_models import APIModelBenchmark
from benchmark_framework.summary import calculate_api_summary_statistics
# 从当前目录导入模块
from benchmark_framework.apps_eval import testing_util
from benchmark_framework.apps_eval import pyext

# Constants
TIMEOUT = 10  # Seconds for code execution timeout
MAX_RETRIES = 3  # Maximum number of retries for API calls
RETRY_DELAY = 5  # Seconds to wait between retries

# 添加一个函数用于并行处理单个问题
def process_problem(api_benchmark, model, problem, problem_index, total_problems):
    """
    Process a single problem for a model in parallel.
    
    Args:
        api_benchmark: API benchmark instance
        model: Model to test
        problem: Problem dictionary
        problem_index: Index of the problem
        total_problems: Total number of problems
        
    Returns:
        Result dictionary
    """
    try:
        # Format prompt and get model response
        prompt = APPSBenchmark.format_prompt_static(problem)
        
        print(f"Testing [{problem_index+1}/{total_problems}] {model} on problem {problem['problem_id']} ({problem['difficulty']})")
        
        # Add retry logic for API calls
        response_data = None
        retries = 0
        last_error = None
        
        while retries <= MAX_RETRIES:
            try:
                response_data = api_benchmark.call_api_model(model, prompt)
                break  # Exit the retry loop if successful
            except Exception as e:
                last_error = e
                retries += 1
                if retries <= MAX_RETRIES:
                    print(f"Attempt {retries}/{MAX_RETRIES} failed for problem {problem['problem_id']}: {e}")
                    print(f"Retrying in {RETRY_DELAY} seconds...")
                    time.sleep(RETRY_DELAY)
                else:
                    print(f"All {MAX_RETRIES} retry attempts failed for problem {problem['problem_id']}")
                    raise
        
        if response_data is None:
            # This should not happen if the while loop exits normally, but just in case
            raise Exception(f"Failed to get API response after {MAX_RETRIES} attempts: {last_error}")
            
        solution = response_data["text"]
        
        # 使用testing_util.py中的run_test函数评估解决方案
        evaluation = APPSBenchmark.evaluate_using_testing_util(solution, problem)
        
        result = {
            "problem_id": problem["problem_id"],
            "category": problem["category"],
            "difficulty": problem["difficulty"],
            "prompt": prompt,
            "solution": solution,
            "duration": response_data["duration"],
            "execution_success": evaluation["execution_success"],
            "correct_output": evaluation["correct_output"],
            "passed_tests": evaluation["passed_tests"],
            "total_tests": evaluation["total_tests"],
            "pass_rate": evaluation["pass_rate"],
            "apps_style_result": evaluation["apps_style_result"],
            "test_results": evaluation["test_results"],
            "error": evaluation["error"]
        }
        
        return result
    except Exception as e:
        print(f"Error processing problem {problem['problem_id']}: {e}")
        traceback.print_exc()
        # Instead of returning None, return a result with error information
        # This ensures we track API failures in results
        return {
            "problem_id": problem["problem_id"],
            "category": problem["category"],
            "difficulty": problem["difficulty"] if "difficulty" in locals() and problem.get("difficulty") else "unknown",
            "prompt": prompt if 'prompt' in locals() else "",
            "solution": f"Error: {str(e)}",
            "duration": 0,
            "execution_success": False,
            "correct_output": False,
            "passed_tests": 0,
            "total_tests": 0,
            "pass_rate": 0,
            "apps_style_result": [-2],  # Mark as compile error for tracking
            "test_results": [],
            "error": str(e)
        }

class APPSBenchmark:
    """Benchmark for testing models on APPS dataset problems"""
    
    def __init__(self, data_path: str = "data/APPS", api_timeout: int = 60):
        """
        Initialize the APPS benchmark.
        
        Args:
            data_path: Path to the APPS dataset
            api_timeout: Timeout in seconds for API calls
        """
        self.data_path = data_path
        self.api_timeout = api_timeout
        self.api_benchmark = APIModelBenchmark()
        
        # Set timeout for API calls
        self.api_benchmark.set_timeout(api_timeout)
        
        # Check if APPS directory exists
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"APPS dataset directory '{self.data_path}' does not exist. Please ensure it's downloaded.")
        
        print(f"Found APPS dataset at: {self.data_path}")
        print(f"API timeout set to {api_timeout} seconds")
        
        # Verify dataset structure
        self._verify_dataset_structure()
    
    def _verify_dataset_structure(self):
        """Verify the APPS dataset structure and count problems by category"""
        # We still need to check test/train directories as that's the physical structure
        categories = ["test", "train"]
        self.problem_counts = {}
        
        # Difficulty levels to track
        self.difficulty_counts = {
            "introductory": 0,
            "interview": 0,
            "competition": 0,
            "unknown": 0
        }
        
        for category in categories:
            category_path = os.path.join(self.data_path, category)
            if not os.path.exists(category_path):
                print(f"Warning: {category} directory not found in APPS dataset.")
                self.problem_counts[category] = 0
            else:
                problem_ids = [d for d in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, d))]
                self.problem_counts[category] = len(problem_ids)
                print(f"Found {len(problem_ids)} problems in {category} category")
                
                # Check difficulties
                for problem_id in problem_ids:
                    difficulty = self._get_problem_difficulty(category, problem_id)
                    self.difficulty_counts[difficulty] += 1
        
        # Print difficulty distribution
        print("\nProblem difficulty distribution:")
        for difficulty, count in self.difficulty_counts.items():
            print(f"  {difficulty}: {count} problems")
    
    def _get_problem_difficulty(self, category: str, problem_id: str) -> str:
        """
        Get the difficulty level of a problem from its metadata.json file.
        
        Args:
            category: The category folder (test or train)
            problem_id: The problem ID (directory name)
            
        Returns:
            Difficulty level (introductory, interview, competition, or unknown)
        """
        metadata_path = os.path.join(self.data_path, category, problem_id, "metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                difficulty = metadata.get("difficulty", "unknown")
                return difficulty
            except Exception as e:
                print(f"Error reading metadata for {category}/{problem_id}: {e}")
                return "unknown"
        else:
            return "unknown"
    
    def get_problems_by_difficulty(self, difficulty: str) -> List[Dict[str, str]]:
        """
        Get all problems of a specific difficulty level across test and train sets.
        
        Args:
            difficulty: Difficulty level ('introductory', 'interview', or 'competition')
            
        Returns:
            List of problem info dictionaries with category and problem_id
        """
        if difficulty not in ["introductory", "interview", "competition"]:
            raise ValueError(f"Invalid difficulty: {difficulty}. Must be 'introductory', 'interview', or 'competition'.")
        
        problems = []
        
        # Check both test and train directories
        for category in ["test", "train"]:
            category_path = os.path.join(self.data_path, category)
            if not os.path.exists(category_path):
                continue
            
            # Get all problem directories in this category
            problem_ids = [d for d in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, d))]
            
            # Check the difficulty of each problem
            for problem_id in problem_ids:
                problem_difficulty = self._get_problem_difficulty(category, problem_id)
                if problem_difficulty == difficulty:
                    problems.append({
                        "category": category,
                        "problem_id": problem_id
                    })
        
        return problems
    
    def load_problems_by_difficulty(self, difficulty: str, limit: int = 10, random_select: bool = False) -> List[Dict[str, Any]]:
        """
        Load problems from the APPS dataset with a specific difficulty level.
        
        Args:
            difficulty: Difficulty level ('introductory', 'interview', or 'competition')
            limit: Maximum number of problems to load
            random_select: Whether to randomly select problems
            
        Returns:
            List of problem dictionaries
        """
        # Get all problems of the specified difficulty
        all_problems = self.get_problems_by_difficulty(difficulty)
        
        if not all_problems:
            print(f"No problems found with difficulty '{difficulty}'")
            return []
            
        print(f"Found {len(all_problems)} problems with difficulty '{difficulty}'")
        
        # Select problems based on specified criteria
        if random_select:
            # 使用基于时间的随机种子
            random.seed(int(time.time() * 1000000) % 2**32)
            
            # Randomly select up to 'limit' problems
            if limit > len(all_problems):
                limit = len(all_problems)
                print(f"Warning: Requested more problems than available. Using all {limit} problems.")
            selected_problems = random.sample(all_problems, limit)
            print(f"Randomly selected {len(selected_problems)} {difficulty} problems using time-based seed")
        else:
            # Use the first 'limit' problems
            selected_problems = all_problems[:limit]
        
        # Load the full problem details
        loaded_problems = []
        for problem_info in selected_problems:
            category = problem_info["category"]
            problem_id = problem_info["problem_id"]
            problem_dir = os.path.join(self.data_path, category, problem_id)
            
            try:
                # Read problem description
                with open(os.path.join(problem_dir, "question.txt"), "r", encoding="utf-8") as f:
                    description = f.read().strip()
                
                # Read test cases
                test_cases = {}
                input_path = os.path.join(problem_dir, "input_output.json")
                if os.path.exists(input_path):
                    with open(input_path, "r", encoding="utf-8") as f:
                        test_cases = json.load(f)
                
                # If no test cases found in JSON, look for input/output txt files
                if not test_cases:
                    inputs = []
                    outputs = []
                    i = 1
                    while os.path.exists(os.path.join(problem_dir, f"input_{i}.txt")):
                        with open(os.path.join(problem_dir, f"input_{i}.txt"), "r", encoding="utf-8") as f:
                            inputs.append(f.read().strip())
                        
                        with open(os.path.join(problem_dir, f"output_{i}.txt"), "r", encoding="utf-8") as f:
                            outputs.append(f.read().strip())
                        i += 1
                    
                    if inputs and outputs:
                        test_cases = {"inputs": inputs, "outputs": outputs}
                
                # Read solutions if available
                solutions = []
                sol_path = os.path.join(problem_dir, "solutions.json")
                if os.path.exists(sol_path):
                    with open(sol_path, "r", encoding="utf-8") as f:
                        solutions = json.load(f)
                
                # Add problem to the list
                problem = {
                    "problem_id": problem_id,
                    "category": category,  # Original category (test/train)
                    "difficulty": difficulty,  # Actual difficulty level
                    "description": description,
                    "test_cases": test_cases,
                    "solutions": solutions
                }
                
                # Add first test case for easy access
                if test_cases and "inputs" in test_cases and test_cases["inputs"]:
                    problem["input"] = test_cases["inputs"][0]
                else:
                    problem["input"] = ""
                    
                if test_cases and "outputs" in test_cases and test_cases["outputs"]:
                    problem["output"] = test_cases["outputs"][0]
                else:
                    problem["output"] = ""
                
                loaded_problems.append(problem)
                
            except Exception as e:
                print(f"Error loading problem {category}/{problem_id}: {e}")
        
        print(f"Loaded {len(loaded_problems)} problems with difficulty '{difficulty}'")
        return loaded_problems
    
    def run_benchmark(self, models: List[str], 
                     difficulties: List[str] = ["introductory", "interview", "competition"], 
                     problems_per_difficulty: int = 5, random_select: bool = False,
                     specific_problems: Dict[str, List[Dict[str, str]]] = None,
                     max_workers: int = 4) -> Dict[str, Any]:
        """
        Run the APPS benchmark on the specified models and problem difficulties.
        
        Args:
            models: List of models to test (in format "provider:model")
            difficulties: Problem difficulty levels to test
            problems_per_difficulty: Number of problems to test per difficulty level
            random_select: Whether to randomly select problems
            specific_problems: Dict of specific problem IDs to test by difficulty
            max_workers: Maximum number of workers for parallel processing
            
        Returns:
            Benchmark results dictionary
        """
        results = {}
        
        # Check API keys
        api_key_status = self.api_benchmark.check_api_keys()
        for model in models:
            provider = model.split(":", 1)[0]
            if not api_key_status.get(provider, False):
                print(f"⚠️ Skipping {model} due to missing API key")
                continue
                
            results[model] = {}
            
            # Collect all problems first
            all_problems = []
            for difficulty in difficulties:
                # Load problems based on selection criteria
                if specific_problems and difficulty in specific_problems:
                    # Here we'd need to load each specific problem
                    problems = []
                    for problem_info in specific_problems[difficulty]:
                        category = problem_info["category"]
                        problem_id = problem_info["problem_id"]
                        problem_dir = os.path.join(self.data_path, category, problem_id)
                        # Load problem details (this is simplified, should handle errors)
                        # ... (load problem details code)
                else:
                    problems = self.load_problems_by_difficulty(
                        difficulty, 
                        limit=problems_per_difficulty, 
                        random_select=random_select
                    )
                
                # Track all problems
                for problem in problems:
                    all_problems.append(problem)
            
            # Process problems in parallel
            difficulty_results_map = {difficulty: [] for difficulty in difficulties}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_problem = {
                    executor.submit(process_problem, self.api_benchmark, model, problem, i, len(all_problems)): problem
                    for i, problem in enumerate(all_problems)
                }
                
                # Process results as they complete
                for future in as_completed(future_to_problem):
                    problem = future_to_problem[future]
                    try:
                        result = future.result()
                        if result:
                            # Use difficulty instead of category for grouping results
                            difficulty = problem["difficulty"]
                            difficulty_results_map[difficulty].append(result)
                    except Exception as e:
                        print(f"Error processing problem {problem['problem_id']}: {e}")
                        traceback.print_exc()
            
            # Assign results to the model by difficulty
            for difficulty, difficulty_results in difficulty_results_map.items():
                if difficulty_results:
                    results[model][difficulty] = difficulty_results
                    
                    # Save results for this model and difficulty
                    safe_model_name = model.replace(":", "-")
                    os.makedirs("results/apps", exist_ok=True)
                    with open(f"results/apps/{safe_model_name}_{difficulty}.json", "w") as f:
                        json.dump(difficulty_results, f, indent=2)
        
        # Save combined results
        with open(f"results/apps/all_codes.json", "w") as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def calculate_metrics(self, results: Dict[str, Dict[str, List[Dict[str, Any]]]]) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Calculate metrics from benchmark results.
        
        Args:
            results: Benchmark results dictionary
            
        Returns:
            Metrics dictionary
        """
        metrics = {}
        
        for model, difficulties in results.items():
            metrics[model] = {}
            
            for difficulty, problems in difficulties.items():
                if not problems:
                    continue
                
                # Calculate APPS-style metrics
                compile_errors = sum(1 for p in problems if p["apps_style_result"] == [-2])
                runtime_errors = sum(1 for p in problems if p["apps_style_result"] == [-1])
                
                # Calculate success rates
                execution_successes = [p["execution_success"] for p in problems]
                correct_outputs = [p["correct_output"] for p in problems]
                pass_rates = [p["pass_rate"] for p in problems]
                durations = [p["duration"] for p in problems]
                
                # Success rate ignoring compile/runtime errors
                valid_problems = [p for p in problems if p["apps_style_result"] not in ([-2], [-1])]
                test_case_avg = np.mean([np.mean([r for r in p["apps_style_result"] if isinstance(r, bool)]) 
                                        for p in valid_problems]) if valid_problems else 0
                
                # Strict accuracy - all test cases must pass
                strict_accuracy = np.mean([p["correct_output"] for p in problems])
                
                # Average success rates
                exec_success_rate = np.mean(execution_successes) if execution_successes else 0
                correct_output_rate = np.mean(correct_outputs) if correct_outputs else 0
                avg_pass_rate = np.mean(pass_rates) if pass_rates else 0
                avg_duration = np.mean(durations) if durations else 0
                
                metrics[model][difficulty] = {
                    "compile_error_rate": compile_errors / len(problems),
                    "runtime_error_rate": runtime_errors / len(problems),
                    "execution_success_rate": exec_success_rate,
                    "correct_output_rate": correct_output_rate,
                    "test_case_average": test_case_avg,  # APPS original metric
                    "strict_accuracy": strict_accuracy,  # APPS original metric
                    "average_pass_rate": avg_pass_rate,
                    "average_duration": avg_duration,
                    "total_problems": len(problems)
                }
        
        return metrics
    
    def visualize_results(self, metrics: Dict[str, Dict[str, Dict[str, float]]], output_dir: str = "results/apps"):
        """
        Create visualizations from benchmark results.
        
        Args:
            metrics: Benchmark metrics dictionary
            output_dir: Directory to save visualizations
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare data for visualization
        records = []
        for model, difficulties in metrics.items():
            for difficulty, stats in difficulties.items():
                records.append({
                    "model": model,
                    "difficulty": difficulty,  # Changed from 'category' to 'difficulty'
                    "compile_error_rate": stats["compile_error_rate"],
                    "runtime_error_rate": stats["runtime_error_rate"],
                    "execution_success_rate": stats["execution_success_rate"],
                    "correct_output_rate": stats["correct_output_rate"],
                    "test_case_average": stats["test_case_average"],
                    "strict_accuracy": stats["strict_accuracy"],
                    "average_pass_rate": stats["average_pass_rate"],
                    "average_duration": stats["average_duration"],
                    "total_problems": stats["total_problems"]
                })
        
        df = pd.DataFrame(records)
        
        if df.empty:
            print("No data to visualize")
            return
        
        # 1. Create strict accuracy bar chart (APPS style metric)
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x="model", y="strict_accuracy", hue="difficulty", data=df)
        plt.title("APPS Benchmark: Strict Accuracy by Model and Difficulty")
        plt.ylabel("Strict Accuracy (all tests passed)")
        plt.xlabel("Model")
        plt.ylim(0, 1.0)
        
        # Add values on top of bars
        for i, p in enumerate(ax.patches):
            ax.annotate(f"{p.get_height():.2f}", 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha = 'center', va = 'bottom', 
                       xytext = (0, 5), textcoords = 'offset points')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "apps_strict_accuracy.png"))
        plt.close()
        
        # 2. Create test case average bar chart (APPS style metric)
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x="model", y="test_case_average", hue="difficulty", data=df)
        plt.title("APPS Benchmark: Test Case Average by Model and Difficulty")
        plt.ylabel("Test Case Average")
        plt.xlabel("Model")
        plt.ylim(0, 1.0)
        
        # Add values on top of bars
        for i, p in enumerate(ax.patches):
            ax.annotate(f"{p.get_height():.2f}", 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha = 'center', va = 'bottom', 
                       xytext = (0, 5), textcoords = 'offset points')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "apps_test_case_average.png"))
        plt.close()
        
        # 3. Create error rate comparison
        plt.figure(figsize=(14, 7))
        
        # Melt the dataframe for error rates
        error_df_melt = df.melt(id_vars=["model", "difficulty"], 
                              value_vars=["compile_error_rate", "runtime_error_rate"],
                              var_name="error_type", value_name="rate")
        
        # Make error type labels prettier
        error_df_melt["error_type"] = error_df_melt["error_type"].map({
            "compile_error_rate": "Compile Errors",
            "runtime_error_rate": "Runtime Errors"
        })
        
        sns.catplot(x="model", y="rate", hue="error_type", col="difficulty", 
                   data=error_df_melt, kind="bar", height=5, aspect=1.2)
        
        plt.savefig(os.path.join(output_dir, "apps_error_rates.png"))
        plt.close()
        
        # 4. Create metrics comparison
        plt.figure(figsize=(14, 7))
        
        # Melt the dataframe for success metrics
        success_df_melt = df.melt(id_vars=["model", "difficulty"], 
                                value_vars=["strict_accuracy", "test_case_average", "average_pass_rate"],
                                var_name="metric", value_name="rate")
        
        # Make metric names prettier
        success_df_melt["metric"] = success_df_melt["metric"].map({
            "strict_accuracy": "Strict Accuracy",
            "test_case_average": "Test Case Average",
            "average_pass_rate": "Average Pass Rate"
        })
        
        sns.catplot(x="model", y="rate", hue="metric", col="difficulty", 
                   data=success_df_melt, kind="bar", height=5, aspect=1.2)
        
        plt.savefig(os.path.join(output_dir, "apps_metrics_comparison.png"))
        plt.close()
        
        # 5. Create heatmap of strict accuracy
        pivot_df = df.pivot(index="model", columns="difficulty", values="strict_accuracy")
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", fmt=".2f", vmin=0, vmax=1)
        plt.title("APPS Benchmark: Strict Accuracy Heatmap by Difficulty")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "apps_strict_accuracy_heatmap.png"))
        plt.close()
        
        # 6. Save metrics as JSON
        with open(os.path.join(output_dir, "apps_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Visualizations saved to {output_dir}")

    @staticmethod
    def format_prompt_static(problem: Dict[str, Any]) -> str:
        """
        Static method to format a problem into a prompt for the language model.
        
        Args:
            problem: Problem dictionary
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""
Write a Python function to solve the following problem:

{problem['description']}

Example:
Input: {problem['input']}
Expected Output: {problem['output']}

Write your complete solution:
```python
"""
        return prompt.strip()
    
    def format_prompt(self, problem: Dict[str, Any]) -> str:
        """
        Format a problem into a prompt for the language model.
        
        Args:
            problem: Problem dictionary
            
        Returns:
            Formatted prompt string
        """
        return self.format_prompt_static(problem)
    
    @staticmethod
    def execute_code_static(code: str, test_input: str) -> Tuple[bool, str, str]:
        """
        Static method to execute the generated code and check if it produces the expected output.
        
        Args:
            code: Python code to execute
            test_input: Input to provide to the program
            
        Returns:
            Tuple of (success, output, error_message)
        """
        # Extract the code from markdown if needed
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0].strip()
        elif "```" in code:
            code = code.split("```")[1].split("```")[0].strip()
            
        # Create a temporary file for the code
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(code)
            temp_filename = f.name
        
        try:
            # Run the code with the provided input
            process = subprocess.Popen(
                ["python", temp_filename],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate(input=test_input, timeout=TIMEOUT)
            
            if process.returncode != 0:
                return False, "", stderr
            
            return True, stdout.strip(), ""
            
        except subprocess.TimeoutExpired:
            process.kill()
            return False, "", "Execution timed out"
        except Exception as e:
            return False, "", str(e)
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)
    
    def execute_code(self, code: str, test_input: str) -> Tuple[bool, str, str]:
        """
        Execute the generated code and check if it produces the expected output.
        
        Args:
            code: Python code to execute
            test_input: Input to provide to the program
            
        Returns:
            Tuple of (success, output, error_message)
        """
        return self.execute_code_static(code, test_input)
    
    @staticmethod
    def evaluate_using_testing_util(solution: str, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        使用testing_util.py评估代码
        
        Args:
            solution: 模型生成的代码
            problem: 包含测试用例的问题
            
        Returns:
            评估结果字典
        """
        try:
            # 提取代码部分
            if "```python" in solution:
                solution_code = solution.split("```python")[1].split("```")[0].strip()
            elif "```" in solution:
                solution_code = solution.split("```")[1].split("```")[0].strip()
            else:
                solution_code = solution
            
            # 调用testing_util.py中的run_test函数
            apps_results = testing_util.run_test(problem=problem, test=solution_code, debug=False)
            
            # 计算测试通过情况
            if isinstance(apps_results, list):
                # 检查是否有编译错误或运行时错误
                has_compile_error = -2 in apps_results
                has_runtime_error = -1 in apps_results
                
                # 过滤掉错误代码，只保留布尔值结果
                valid_results = [r for r in apps_results if isinstance(r, bool)]
                
                passed_tests = sum(1 for r in valid_results if r)
                total_tests = len(valid_results) if valid_results else 0
                
                if total_tests == 0:
                    if has_compile_error:
                        # 编译错误
                        return {
                            "execution_success": False,
                            "correct_output": False,
                            "passed_tests": 0,
                            "total_tests": 1,  # 至少有一个测试
                            "pass_rate": 0.0,
                            "apps_style_result": [-2],
                            "test_results": [],
                            "error": "Compilation error"
                        }
                    elif has_runtime_error:
                        # 运行时错误
                        return {
                            "execution_success": False,
                            "correct_output": False,
                            "passed_tests": 0,
                            "total_tests": 1,  # 至少有一个测试
                            "pass_rate": 0.0,
                            "apps_style_result": [-1],
                            "test_results": [],
                            "error": "Runtime error"
                        }
                    else:
                        # 没有有效结果但也没有错误(不应该发生)
                        return {
                            "execution_success": False,
                            "correct_output": False,
                            "passed_tests": 0,
                            "total_tests": 0,
                            "pass_rate": 0.0,
                            "apps_style_result": [],
                            "test_results": [],
                            "error": "Unknown error: No valid test results"
                        }
                else:
                    # 计算通过率
                    pass_rate = passed_tests / total_tests
                    
                    return {
                        "execution_success": not (has_compile_error or has_runtime_error),
                        "correct_output": passed_tests == total_tests,  # 全部通过才算正确
                        "passed_tests": passed_tests,
                        "total_tests": total_tests,
                        "pass_rate": pass_rate,
                        "apps_style_result": apps_results,
                        "test_results": [{"correct": r} for r in valid_results],
                        "error": ""
                    }
            else:
                # 如果返回的不是列表(不应该发生)
                return {
                    "execution_success": False,
                    "correct_output": False,
                    "passed_tests": 0,
                    "total_tests": 1,
                    "pass_rate": 0.0,
                    "apps_style_result": [-2],  # 假设为编译错误
                    "test_results": [],
                    "error": f"Unexpected result format: {apps_results}"
                }
        except Exception as e:
            # 处理评估过程中的错误
            return {
                "execution_success": False,
                "correct_output": False,
                "passed_tests": 0,
                "total_tests": 1,
                "pass_rate": 0.0,
                "apps_style_result": [-2],  # 假设为编译错误
                "test_results": [],
                "error": f"Evaluation error: {str(e)}"
            }
    
    @staticmethod
    def evaluate_solution_static(solution: str, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Static method to evaluate a solution against all problem's test cases.
        This method is kept for backward compatibility.
        
        Args:
            solution: Model-generated solution code
            problem: Problem dictionary with test cases
            
        Returns:
            Evaluation results dictionary
        """
        return APPSBenchmark.evaluate_using_testing_util(solution, problem)
    
    def evaluate_solution(self, solution: str, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a solution against all problem's test cases.
        This method is kept for backward compatibility.
        
        Args:
            solution: Model-generated solution code
            problem: Problem dictionary with test cases
            
        Returns:
            Evaluation results dictionary
        """
        return self.evaluate_using_testing_util(solution, problem)

def main():
    parser = argparse.ArgumentParser(description="Benchmark API language models on APPS dataset")
    parser.add_argument("--models", nargs="+", required=True,
                       help="List of API models to benchmark in format provider:model (e.g., openai:gpt-4)")
    parser.add_argument("--difficulties", nargs="+", default=["introductory", "interview", "competition"],
                       help="Problem difficulties to test (default: introductory, interview, competition)")
    parser.add_argument("--problems", type=int, default=5,
                       help="Number of problems per difficulty (default: 5)")
    parser.add_argument("--random", action="store_true",
                       help="Randomly select problems instead of using the first N")
    parser.add_argument("--specific-problems", type=str, default=None,
                       help="JSON file containing specific problem IDs to test by difficulty")
    parser.add_argument("--seed", type=int, default=0,
                       help="Random seed for problem selection (0 for time-based seed, default: 0)")
    parser.add_argument("--output-dir", default="results/apps",
                       help="Directory to save results (default: results/apps)")
    parser.add_argument("--data-path", default="data/APPS",
                       help="Path to the APPS dataset directory (default: data/APPS)")
    parser.add_argument("--workers", type=int, default=4,
                       help="Number of workers for parallel processing (default: 4)")
    parser.add_argument("--timeout", type=int, default=120,
                       help="Timeout in seconds for API calls (default: 120)")
    parser.add_argument("--retries", type=int, default=3,
                       help="Maximum number of retry attempts for API calls (default: 3)")
    
    args = parser.parse_args()
    
    # Update global constants based on args
    global MAX_RETRIES
    MAX_RETRIES = args.retries
    
    # 仅当用户明确指定非零种子值时才使用固定种子
    if args.seed != 0:
        random.seed(args.seed)
        print(f"Using fixed random seed: {args.seed}")
    else:
        # 否则使用基于时间的随机种子
        random_seed = int(time.time() * 1000000) % 2**32
        random.seed(random_seed)
        print(f"Using time-based random seed: {random_seed}")
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load specific problems if provided
    specific_problems = None
    if args.specific_problems:
        if os.path.exists(args.specific_problems):
            with open(args.specific_problems, 'r') as f:
                specific_problems = json.load(f)
            print(f"Loaded specific problems from {args.specific_problems}")
        else:
            print(f"Warning: Specific problems file {args.specific_problems} not found. Ignoring.")
    
    # Initialize benchmark
    benchmark = APPSBenchmark(data_path=args.data_path, api_timeout=args.timeout)
    
    # Run benchmark
    print(f"Running APPS benchmark on models: {', '.join(args.models)}")
    print(f"Using {args.workers} worker threads for parallel processing")
    print(f"API timeout: {args.timeout}s, Max retries: {MAX_RETRIES}")
    
    results = benchmark.run_benchmark(
        models=args.models, 
        difficulties=args.difficulties,
        problems_per_difficulty=args.problems,
        random_select=args.random,
        specific_problems=specific_problems,
        max_workers=args.workers
    )
    
    # Calculate metrics
    metrics = benchmark.calculate_metrics(results)
    
    # Visualize results
    benchmark.visualize_results(metrics, args.output_dir)
    
    print(f"APPS benchmark completed. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 