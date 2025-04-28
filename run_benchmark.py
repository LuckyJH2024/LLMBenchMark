#!/usr/bin/env python3

"""
LLM基准测试统一运行脚本
---------------------------------
此脚本读取配置文件并运行所有指定的测试。
"""

import os
import sys
import time
import yaml
import json
import argparse
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_config(config_path):
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"加载配置文件出错: {e}")
        sys.exit(1)

def run_reasoning_test(models, config):
    """运行推理测试"""
    output_dir = config['output']['results_dir']
    
    print("\n=== 开始推理测试 ===")
    
    # 构建命令 - 根据run_api_benchmark.py实际支持的参数格式
    cmd = [
        "python3", "run_api_benchmark.py",
        "--models", *models,
        "--task-types", "reasoning",
        "--results-dir", os.path.join(output_dir, "reasoning")
    ]
    
    # 运行命令
    try:
        subprocess.run(cmd, check=True)
        print(f"推理测试完成")
    except subprocess.CalledProcessError as e:
        print(f"推理测试失败: {e}")

def run_coding_test(models, config):
    """运行代码测试"""
    difficulties = config['tests']['coding']['difficulties']
    problems_per_difficulty = config['tests']['coding']['problems_per_difficulty']
    data_path = config['tests']['coding']['data_path']
    output_dir = config['output']['results_dir']
    workers = config['run']['workers']
    timeout = config['run']['timeout']
    retries = config['run']['retries']
    
    print("\n=== 开始代码测试 ===")
    
    for model in models:
        print(f"测试模型: {model}")
        
        # 构建命令
        cmd = [
            "python3", "apps_benchmark.py",
            "--models", model,
            "--difficulties", *difficulties,
            "--problems", str(problems_per_difficulty),
            "--random",
            "--data-path", data_path,
            "--output-dir", os.path.join(output_dir, "apps"),
            "--workers", str(workers),
            "--timeout", str(timeout),
            "--retries", str(retries)
        ]
        
        # 运行命令
        try:
            subprocess.run(cmd, check=True)
            print(f"模型 {model} 的代码测试完成")
        except subprocess.CalledProcessError as e:
            print(f"模型 {model} 的代码测试失败: {e}")

def run_qa_test(models, config):
    """运行问答测试"""
    output_dir = config['output']['results_dir']
    
    print("\n=== 开始问答测试 ===")
    
    # 构建命令 - 根据run_api_benchmark.py实际支持的参数格式
    cmd = [
        "python3", "run_api_benchmark.py",
        "--models", *models,
        "--task-types", "qa",
        "--results-dir", os.path.join(output_dir, "qa")
    ]
    
    # 运行命令
    try:
        subprocess.run(cmd, check=True)
        print(f"问答测试完成")
    except subprocess.CalledProcessError as e:
        print(f"问答测试失败: {e}")

def generate_unified_report(config):
    """生成统一的测试报告"""
    if not config['output']['visualize']:
        return
    
    output_dir = config['output']['results_dir']
    report_path = os.path.join(output_dir, "report")
    os.makedirs(report_path, exist_ok=True)
    
    # 收集所有结果
    results = {}
    
    # 收集推理测试结果
    if config['tests']['reasoning']['enabled']:
        reasoning_dir = os.path.join(output_dir, "reasoning")
        if os.path.exists(reasoning_dir):
            try:
                # 加载推理测试结果
                results['reasoning'] = load_results(reasoning_dir)
            except Exception as e:
                print(f"加载推理测试结果失败: {e}")
    
    # 收集代码测试结果
    if config['tests']['coding']['enabled']:
        coding_dir = os.path.join(output_dir, "apps")
        if os.path.exists(coding_dir):
            try:
                # 加载代码测试结果
                results['coding'] = load_results(coding_dir)
            except Exception as e:
                print(f"加载代码测试结果失败: {e}")
    
    # 收集问答测试结果
    if config['tests']['qa']['enabled']:
        qa_dir = os.path.join(output_dir, "qa")
        if os.path.exists(qa_dir):
            try:
                # 加载问答测试结果
                results['qa'] = load_results(qa_dir)
            except Exception as e:
                print(f"加载问答测试结果失败: {e}")
    
    # 生成综合报告
    generate_report(results, report_path)
    
    print(f"\n报告已生成在: {report_path}")

def load_results(directory):
    """加载目录中的结果文件"""
    results = {}
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                # 将文件名作为键(去掉.json扩展名)
                key = os.path.splitext(filename)[0]
                results[key] = data
            except Exception as e:
                print(f"加载文件 {file_path} 失败: {e}")
    return results

def generate_report(results, report_path):
    """生成综合报告"""
    # 写入简单的HTML报告
    html_report = os.path.join(report_path, "benchmark_report.html")
    
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    
    with open(html_report, 'w') as f:
        # 使用普通字符串而不是格式化字符串，避免%字符被误解为格式化字符
        f.write("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>LLM基准测试报告</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
                h1, h2, h3 { color: #333; }
                table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                th, td { text-align: left; padding: 8px; border: 1px solid #ddd; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                .section { margin-bottom: 30px; }
                .chart { margin: 20px 0; }
            </style>
        </head>
        <body>
            <h1>LLM基准测试报告</h1>
        """)
        
        # 分别添加时间戳，避免在格式字符串中使用%
        f.write(f'<p>生成时间: {current_time}</p>\n')
        
        # 添加测试结果部分
        for test_type, test_results in results.items():
            f.write(f'<div class="section">\n')
            f.write(f'<h2>{test_type.capitalize()} 测试结果</h2>\n')
            
            # 为每个测试类型生成结果表格
            if test_results:
                f.write('<table>\n')
                f.write('<tr><th>模型</th><th>指标</th><th>值</th></tr>\n')
                
                for model_key, model_data in test_results.items():
                    # 简化数据表示
                    if isinstance(model_data, dict):
                        for metric, value in model_data.items():
                            if isinstance(value, dict):
                                for sub_metric, sub_value in value.items():
                                    f.write(f'<tr><td>{model_key}</td><td>{metric}.{sub_metric}</td><td>{sub_value}</td></tr>\n')
                            else:
                                f.write(f'<tr><td>{model_key}</td><td>{metric}</td><td>{value}</td></tr>\n')
                
                f.write('</table>\n')
            else:
                f.write('<p>无可用结果</p>\n')
            
            f.write('</div>\n')
        
        # 添加图表的引用(如果有生成图表)
        chart_files = [f for f in os.listdir(report_path) if f.endswith('.png')]
        if chart_files:
            f.write('<div class="section">\n')
            f.write('<h2>结果可视化</h2>\n')
            
            for chart_file in chart_files:
                f.write(f'<div class="chart">\n')
                f.write(f'<h3>{os.path.splitext(chart_file)[0]}</h3>\n')
                f.write(f'<img src="{chart_file}" alt="{chart_file}" width="800">\n')
                f.write('</div>\n')
            
            f.write('</div>\n')
        
        f.write("""
        </body>
        </html>
        """)
    
    # 在这里可以添加生成图表的代码
    # 可视化各测试的相对性能

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="LLM基准测试统一运行脚本")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 创建输出目录
    os.makedirs(config['output']['results_dir'], exist_ok=True)
    
    # 获取要测试的模型
    api_models = config['models'].get('api', [])
    local_models = config['models'].get('local', [])  # 使用get方法处理缺少local键的情况
    
    # 合并API模型和本地模型列表
    all_models = api_models + local_models
    
    # 如果没有指定任何模型，打印警告并退出
    if not all_models:
        print("警告: 配置文件中没有指定任何模型。请在config.yaml的models部分添加至少一个模型。")
        sys.exit(1)
    
    # 运行测试
    if config['tests']['reasoning']['enabled']:
        run_reasoning_test(all_models, config)
    
    if config['tests']['coding']['enabled']:
        run_coding_test(all_models, config)
    
    if config['tests']['qa']['enabled']:
        run_qa_test(all_models, config)
    
    # 生成统一报告
    generate_unified_report(config)
    
    print("\n所有基准测试完成!")

if __name__ == "__main__":
    main() 