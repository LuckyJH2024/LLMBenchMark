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
import seaborn as sns

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
        
        # 构建命令 - 更新为新路径
        cmd = [
            "python3", "benchmark_framework/apps_eval/apps_benchmark.py",
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
    """运行问答测试 - 使用直接加载QA任务的方式"""
    output_dir = config['output']['results_dir']
    qa_dir = os.path.join(output_dir, "qa")
    os.makedirs(qa_dir, exist_ok=True)
    
    print("\n=== 开始问答测试 ===")
    
    try:
        # 从benchmark_framework.tasks导入必要的函数
        from benchmark_framework.tasks import create_qa_benchmark
        from benchmark_framework.benchmark import LLMBenchmark
        
        # 加载QA任务
        qa_file_path = "data/qa_benchmark.json"
        print(f"正在加载QA任务从: {qa_file_path}")
        
        # 确保文件存在且格式正确
        try:
            with open(qa_file_path, 'r', encoding='utf-8') as f:
                # 验证JSON格式
                json_data = json.load(f)
                print(f"成功读取QA数据: {len(json_data)}个问题")
        except json.JSONDecodeError as e:
            print(f"QA文件{qa_file_path}不是有效的JSON格式: {e}")
            print("尝试修复或重新创建QA文件...")
            repair_qa_file(qa_file_path)
        except FileNotFoundError:
            print(f"QA文件{qa_file_path}不存在，创建示例文件...")
            create_example_qa_file(qa_file_path)
            
        # 使用benchmark_framework中的函数加载任务
        qa_tasks = create_qa_benchmark(qa_file_path)
        
        # 创建benchmark对象
        benchmark = LLMBenchmark(models=models, tasks={"qa": qa_tasks})
        
        # 运行基准测试
        print(f"开始为{len(models)}个模型运行QA测试...")
        qa_results = benchmark.run_benchmarks()
        
        # 将每个模型的结果保存到单独的文件
        for model, task_results in qa_results.items():
            safe_model_name = model.replace(":", "-")
            result_file = os.path.join(qa_dir, f"{safe_model_name}.json")
            with open(result_file, "w") as f:
                json.dump(task_results["qa"], f, indent=4)
            print(f"模型 {model} 的问答结果已保存到 {result_file}")
            
        # 计算和保存摘要统计信息
        compute_qa_summary(qa_results, qa_dir)
        
        print(f"问答测试完成")
    except Exception as e:
        import traceback
        print(f"问答测试失败: {e}")
        traceback.print_exc()

def repair_qa_file(file_path):
    """修复QA基准测试文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 尝试简单的JSON语法修复
        if content.strip().startswith('<!DOCTYPE html>') or content.strip().startswith('<html>'):
            print("检测到HTML内容而非JSON，替换为示例QA数据...")
            create_example_qa_file(file_path)
        else:
            # 尝试修复JSON格式
            try:
                # 尝试修复常见的JSON错误
                content = content.replace("'", '"')  # 单引号替换为双引号
                content = content.strip()
                if not content.startswith('['):
                    content = '[' + content
                if not content.endswith(']'):
                    content = content + ']'
                    
                # 验证是否为有效JSON
                json.loads(content)
                
                # 如果有效，写回文件
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"已修复QA文件: {file_path}")
            except:
                print("无法修复JSON格式，创建示例QA数据...")
                create_example_qa_file(file_path)
    except Exception as e:
        print(f"修复QA文件失败: {e}")
        create_example_qa_file(file_path)

def create_example_qa_file(file_path):
    """创建示例QA数据文件"""
    example_data = [
        {"question": "机器学习是什么？", "answer": "机器学习是人工智能的一个子领域，它使用统计技术让计算机系统利用数据学习和改进，而无需被明确编程。"},
        {"question": "深度学习与传统机器学习有何不同？", "answer": "深度学习是机器学习的一个子集，它使用多层神经网络处理数据，可以自动学习特征，而传统机器学习通常需要手动特征工程。"},
        {"question": "什么是大型语言模型？", "answer": "大型语言模型是一种基于深度学习的AI系统，通过大规模文本数据训练，能够理解和生成人类语言，执行各种语言相关任务。"}
    ]
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(example_data, f, indent=2, ensure_ascii=False)
        print(f"已创建示例QA文件: {file_path}")
    except Exception as e:
        print(f"创建示例QA文件失败: {e}")

def compute_qa_summary(qa_results, output_dir):
    """计算QA结果的摘要统计信息"""
    summary = {}
    
    for model, tasks in qa_results.items():
        summary[model] = {}
        entries = tasks.get("qa", [])
        
        if entries:
            durations = [x.get("duration", 0) for x in entries]
            memories = [x.get("memory_usage", 0) for x in entries]
            scores = [x.get("score", 0) for x in entries if x.get("score") is not None]
            
            summary[model]["qa"] = {
                "average_duration": np.mean(durations) if durations else 0,
                "average_memory_usage": np.mean(memories) if memories else 0,
                "average_score": np.mean(scores) if scores else 0,
                "total_tasks": len(entries),
            }
    
    # 保存摘要到文件
    summary_file = os.path.join(output_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"QA测试摘要已保存到: {summary_file}")
    
    return summary

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
    from datetime import datetime
    import pandas as pd
    import seaborn as sns
    
    # 写入简单的HTML报告
    html_report = os.path.join(report_path, "benchmark_report.html")
    
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # 创建可视化图表
    print("正在生成可视化图表...")
    
    # 设置Seaborn样式
    sns.set(style="whitegrid")
    
    # 为QA和摘要任务创建专门的条形图
    plot_data = []
    
    # 收集所有任务的评分数据
    for test_type, test_results in results.items():
        for model_key, model_data in test_results.items():
            if isinstance(model_data, dict) and not model_key.endswith('summary'):
                # 尝试从结果中提取分数
                for record in model_data:
                    if isinstance(record, dict):
                        score = record.get("score")
                        if score is not None:
                            plot_data.append({
                                "Model": model_key,
                                "Task Type": test_type,
                                "Score": score
                            })
    
    # 创建DataFrame
    if plot_data:
        plot_df = pd.DataFrame(plot_data)
        
        # 生成QA任务的条形图
        qa_df = plot_df[plot_df['Task Type'] == 'qa']
        if not qa_df.empty:
            plt.figure(figsize=(10, 6))
            sns.barplot(data=qa_df, x="Model", y="Score", ci="sd")
            plt.title("QA任务 - 模型平均得分")
            plt.ylim(0, 1)
            plt.xticks(rotation=45)
            plt.tight_layout()
            qa_plot_path = os.path.join(report_path, "qa_scores.png")
            plt.savefig(qa_plot_path)
            plt.close()
            print(f"QA得分图表已保存到 {qa_plot_path}")
        
        # 生成摘要任务的条形图
        summarization_df = plot_df[plot_df['Task Type'] == 'summarization']
        if not summarization_df.empty:
            plt.figure(figsize=(10, 6))
            sns.barplot(data=summarization_df, x="Model", y="Score", ci="sd")
            plt.title("摘要任务 - 模型平均得分")
            plt.ylim(0, 1)
            plt.xticks(rotation=45)
            plt.tight_layout()
            summarization_plot_path = os.path.join(report_path, "summarization_scores.png")
            plt.savefig(summarization_plot_path)
            plt.close()
            print(f"摘要得分图表已保存到 {summarization_plot_path}")
        
        # 生成推理任务的条形图
        reasoning_df = plot_df[plot_df['Task Type'] == 'reasoning']
        if not reasoning_df.empty:
            plt.figure(figsize=(10, 6))
            sns.barplot(data=reasoning_df, x="Model", y="Score", ci="sd")
            plt.title("推理任务 - 模型平均得分")
            plt.ylim(0, 1)
            plt.xticks(rotation=45)
            plt.tight_layout()
            reasoning_plot_path = os.path.join(report_path, "reasoning_scores.png")
            plt.savefig(reasoning_plot_path)
            plt.close()
            print(f"推理得分图表已保存到 {reasoning_plot_path}")
    
    # 写入HTML报告
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
    
    print(f"HTML报告已生成: {html_report}")

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