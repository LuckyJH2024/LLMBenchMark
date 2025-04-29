#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLM基准测试主入口文件
---------------------------------
这是项目的主入口点，包含默认配置并调用benchmark测试功能。
"""

import os
import sys
import yaml
import argparse
import importlib.util
from pathlib import Path

# 默认配置 - 可以根据需要直接在这里修改
DEFAULT_CONFIG = {
    # 测试模型配置
    "models": {
        # API模型
        "api": [
            # OpenAI模型
            "openai:gpt-4o",
            # Anthropic模型
            "anthropic:claude3.7",
            # DeepSeek模型
            "deepseek:deepseek-coder"
        ],
        # 本地Ollama模型 (即使不使用也需要保留此键)
        "local": []
    },
    
    # 测试类型配置
    "tests": {
        # 推理测试
        "reasoning": {
            "enabled": True,
            "samples": 10  # 测试样本数量
        },
        
        # 代码测试
        "coding": {
            "enabled": True,
            "difficulties": ["interview", "competition"],
            "problems_per_difficulty": 3,  # 每种难度的问题数量
            "data_path": "data/APPS"  # APPS数据集路径，请更新为实际路径
        },
        
        # 问答测试
        "qa": {
            "enabled": True,
            "samples": 10
        }
    },
    
    # 输出配置
    "output": {
        "results_dir": "results",  # 结果输出目录
        "visualize": True,         # 是否生成可视化
        "save_details": True       # 是否保存详细结果
    },
    
    # 运行配置
    "run": {
        "timeout": 120,  # API请求超时时间(秒)
        "workers": 4,    # 并行工作线程数量
        "retries": 3,    # 重试次数
        "seed": 0        # 随机种子(0表示使用时间种子)
    }
} 

def load_config(config_path):
    """从文件加载配置，如果文件不存在则使用默认配置"""
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"加载配置文件出错: {e}")
            print("使用默认配置继续...")
    
    return DEFAULT_CONFIG

def ensure_run_benchmark_available():
    """确保run_benchmark.py可用，如果存在导入它，否则提示错误"""
    if not os.path.exists("run_benchmark.py"):
        print("错误: 未找到run_benchmark.py文件，无法继续执行。")
        print("请确保您在正确的目录中运行此脚本。")
        sys.exit(1)
        
    # 从run_benchmark.py导入所需函数
    try:
        # 使用importlib.util动态导入模块
        spec = importlib.util.spec_from_file_location("run_benchmark", "run_benchmark.py")
        run_benchmark = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run_benchmark)
        return run_benchmark
    except Exception as e:
        print(f"导入run_benchmark.py出错: {e}")
        sys.exit(1)

def save_temp_config(config):
    """将配置保存到临时YAML文件"""
    temp_config_path = "_temp_config.yaml"
    try:
        with open(temp_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        return temp_config_path
    except Exception as e:
        print(f"保存临时配置文件出错: {e}")
        sys.exit(1)

def main():
    """主函数 - 解析命令行参数并启动测试"""
    parser = argparse.ArgumentParser(description="LLM基准测试主入口")
    parser.add_argument("--config", help="配置文件路径（可选，不提供则使用内置默认配置）")
    
    # 添加配置覆盖选项
    parser.add_argument("--models", nargs="+", help="要测试的模型列表，以空格分隔")
    parser.add_argument("--enable-reasoning", action="store_true", help="启用推理测试")
    parser.add_argument("--enable-coding", action="store_true", help="启用代码测试")
    parser.add_argument("--enable-qa", action="store_true", help="启用问答测试")
    parser.add_argument("--apps-path", help="APPS数据集路径")
    parser.add_argument("--results-dir", help="结果输出目录")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    
    args = parser.parse_args()
    
    # 输出诊断信息
    print("🔍 开始诊断信息 🔍")
    print(f"命令行参数: {args}")
    
    # 加载配置（从文件或使用默认配置）
    try:
        print(f"正在加载配置文件: {args.config if args.config else '(使用默认配置)'}")
        config = load_config(args.config)
        print("✅ 配置文件加载成功")
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
        sys.exit(1)
    
    # 应用命令行参数覆盖配置
    if args.models:
        print(f"从命令行覆盖模型设置: {args.models}")
        config["models"]["api"] = args.models
    
    # 如果指定了任何enable标志，则只启用指定的测试
    if args.enable_reasoning or args.enable_coding or args.enable_qa:
        print("从命令行覆盖测试类型设置")
        config["tests"]["reasoning"]["enabled"] = args.enable_reasoning
        config["tests"]["coding"]["enabled"] = args.enable_coding
        config["tests"]["qa"]["enabled"] = args.enable_qa
    
    if args.apps_path:
        print(f"从命令行覆盖APPS路径: {args.apps_path}")
        config["tests"]["coding"]["data_path"] = args.apps_path
    
    if args.results_dir:
        print(f"从命令行覆盖结果目录: {args.results_dir}")
        config["output"]["results_dir"] = args.results_dir
    
    # 打印配置信息
    print("\n=== LLM基准测试配置 ===")
    print(f"模型: {config['models']['api']}")
    print(f"测试类型: 推理({'启用' if config['tests']['reasoning']['enabled'] else '禁用'}), "
          f"代码({'启用' if config['tests']['coding']['enabled'] else '禁用'}), "
          f"问答({'启用' if config['tests']['qa']['enabled'] else '禁用'})")
    print(f"APPS数据路径: {config['tests']['coding']['data_path']}")
    print(f"结果目录: {config['output']['results_dir']}")
    print("========================\n")
    
    # 检查是否至少启用了一种测试
    if not (config['tests']['reasoning']['enabled'] or 
            config['tests']['coding']['enabled'] or 
            config['tests']['qa']['enabled']):
        print("❌ 错误: 没有启用任何测试类型。请在配置文件中启用至少一种测试类型。")
        sys.exit(1)
    
    # 检查模型列表是否为空
    if not config['models']['api'] and not config['models']['local']:
        print("❌ 错误: 没有指定任何模型。请在配置文件中添加至少一个模型。")
        sys.exit(1)
    
    # 确保run_benchmark.py可用
    try:
        print("正在加载run_benchmark模块...")
        run_benchmark = ensure_run_benchmark_available()
        print("✅ run_benchmark模块加载成功")
    except Exception as e:
        print(f"❌ 加载run_benchmark模块失败: {e}")
        sys.exit(1)
    
    # 将配置保存到临时文件并调用run_benchmark
    try:
        print("正在保存临时配置文件...")
        temp_config_path = save_temp_config(config)
        print(f"✅ 临时配置文件保存成功: {temp_config_path}")
    except Exception as e:
        print(f"❌ 保存临时配置文件失败: {e}")
        sys.exit(1)
    
    try:
        print("正在创建输出目录...")
        # 初始化必要的目录
        os.makedirs(config['output']['results_dir'], exist_ok=True)
        print(f"✅ 输出目录创建成功: {config['output']['results_dir']}")
        
        print("正在加载配置到run_benchmark...")
        # 加载run_benchmark中的函数
        if hasattr(run_benchmark, 'load_config'):
            loaded_config = run_benchmark.load_config(temp_config_path)
            print("✅ 配置已加载到run_benchmark")
        else:
            print("⚠️ run_benchmark没有load_config方法，使用原始配置")
            loaded_config = config
        
        # 调用run_benchmark中的各个函数
        api_models = loaded_config['models'].get('api', [])
        local_models = loaded_config['models'].get('local', [])
        all_models = api_models + local_models
        print(f"将测试的模型: {all_models}")
        
        if loaded_config['tests']['reasoning']['enabled'] and hasattr(run_benchmark, 'run_reasoning_test'):
            print("\n📊 开始运行推理测试...")
            run_benchmark.run_reasoning_test(all_models, loaded_config)
            print("✅ 推理测试完成")
        
        if loaded_config['tests']['coding']['enabled'] and hasattr(run_benchmark, 'run_coding_test'):
            print("\n💻 开始运行代码测试...")
            run_benchmark.run_coding_test(all_models, loaded_config)
            print("✅ 代码测试完成")
        
        if loaded_config['tests']['qa']['enabled'] and hasattr(run_benchmark, 'run_qa_test'):
            print("\n❓ 开始运行问答测试...")
            run_benchmark.run_qa_test(all_models, loaded_config)
            print("✅ 问答测试完成")
        
        # 生成报告
        if hasattr(run_benchmark, 'generate_unified_report'):
            print("\n📝 开始生成统一报告...")
            run_benchmark.generate_unified_report(loaded_config)
            print("✅ 报告生成完成")
        
        print("\n🎉 所有基准测试完成!")
        
    except Exception as e:
        print(f"\n❌ 执行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # 清理临时配置文件
        if os.path.exists(temp_config_path):
            print(f"正在清理临时配置文件: {temp_config_path}")
            os.remove(temp_config_path)
            print("✅ 临时文件清理完成")

if __name__ == "__main__":
    main() 