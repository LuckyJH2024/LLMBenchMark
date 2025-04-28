#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
配置脚本: 用于设置各种API提供商的密钥和端点
"""

import os
import json
import getpass
from typing import Dict, Any, Optional

# 默认配置结构
DEFAULT_CONFIG = {
    "api_keys": {
        "openai": {"api_key": ""},
        "anthropic": {"api_key": ""},
        "anthropic_bedrock": {
            "aws_access_key": "",
            "aws_secret_key": "",
            "aws_region": "us-west-2"
        },
        "anthropic_vertex": {
            "project_id": "",
            "region": "us-central1"
        },
        "google": {"api_key": ""},
        "baidu": {"api_key": "", "secret_key": ""},
        "zhipu": {"api_key": ""}
    },
    "api_base": {
        "openai": "https://api.openai.com/v1",
        "anthropic": "https://api.anthropic.com",
        "google": "https://generativelanguage.googleapis.com/v1",
        "baidu": "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop",
        "zhipu": "https://open.bigmodel.cn/api/paas"
    }
}

# 配置文件路径
CONFIG_PATH = "config.json"

def load_config() -> Dict[str, Any]:
    """加载现有配置文件或返回默认配置"""
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"警告: 配置文件{CONFIG_PATH}格式错误，将使用默认配置")
            return DEFAULT_CONFIG.copy()
    return DEFAULT_CONFIG.copy()

def save_config(config: Dict[str, Any]) -> None:
    """保存配置到文件"""
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
    print(f"配置已保存到{CONFIG_PATH}")
    # 设置文件权限为仅当前用户可读写
    os.chmod(CONFIG_PATH, 0o600)

def setup_openai(config: Dict[str, Any]) -> None:
    """设置OpenAI API密钥"""
    api_key = getpass.getpass("请输入OpenAI API密钥: ")
    if api_key:
        config["api_keys"]["openai"]["api_key"] = api_key
        print("OpenAI API密钥已保存!")
    else:
        print("API密钥为空，跳过保存")

def setup_anthropic(config: Dict[str, Any]) -> None:
    """设置Anthropic API密钥"""
    api_key = getpass.getpass("请输入Anthropic API密钥: ")
    if api_key:
        config["api_keys"]["anthropic"]["api_key"] = api_key
        print("Anthropic API密钥已保存!")
    else:
        print("API密钥为空，跳过保存")

def setup_anthropic_bedrock(config: Dict[str, Any]) -> None:
    """设置Anthropic Bedrock配置"""
    aws_access_key = getpass.getpass("请输入AWS Access Key: ")
    aws_secret_key = getpass.getpass("请输入AWS Secret Key: ")
    aws_region = input("请输入AWS区域(默认为us-west-2): ") or "us-west-2"
    
    if aws_access_key and aws_secret_key:
        config["api_keys"]["anthropic_bedrock"]["aws_access_key"] = aws_access_key
        config["api_keys"]["anthropic_bedrock"]["aws_secret_key"] = aws_secret_key
        config["api_keys"]["anthropic_bedrock"]["aws_region"] = aws_region
        print("AWS Bedrock配置已保存!")
    else:
        print("部分关键配置为空，跳过保存")

def setup_anthropic_vertex(config: Dict[str, Any]) -> None:
    """设置Anthropic Vertex AI配置"""
    project_id = input("请输入Google Cloud项目ID: ")
    region = input("请输入Vertex AI区域(默认为us-central1): ") or "us-central1"
    
    if project_id:
        config["api_keys"]["anthropic_vertex"]["project_id"] = project_id
        config["api_keys"]["anthropic_vertex"]["region"] = region
        print("Google Vertex AI配置已保存!")
    else:
        print("项目ID为空，跳过保存")

def setup_google(config: Dict[str, Any]) -> None:
    """设置Google API密钥"""
    api_key = getpass.getpass("请输入Google API密钥: ")
    if api_key:
        config["api_keys"]["google"]["api_key"] = api_key
        print("Google API密钥已保存!")
    else:
        print("API密钥为空，跳过保存")

def setup_baidu(config: Dict[str, Any]) -> None:
    """设置百度API密钥"""
    api_key = getpass.getpass("请输入百度API密钥: ")
    secret_key = getpass.getpass("请输入百度Secret Key: ")
    
    if api_key and secret_key:
        config["api_keys"]["baidu"]["api_key"] = api_key
        config["api_keys"]["baidu"]["secret_key"] = secret_key
        print("百度API配置已保存!")
    else:
        print("部分关键配置为空，跳过保存")

def setup_zhipu(config: Dict[str, Any]) -> None:
    """设置智谱API密钥"""
    api_key = getpass.getpass("请输入智谱API密钥: ")
    if api_key:
        config["api_keys"]["zhipu"]["api_key"] = api_key
        print("智谱API密钥已保存!")
    else:
        print("API密钥为空，跳过保存")

def setup_all(config: Dict[str, Any]) -> None:
    """设置所有API密钥"""
    print("\n--- 设置OpenAI ---")
    setup_openai(config)
    
    print("\n--- 设置Anthropic ---")
    setup_anthropic(config)
    
    print("\n--- 设置Anthropic (Bedrock) ---")
    setup_anthropic_bedrock(config)
    
    print("\n--- 设置Anthropic (Vertex AI) ---")
    setup_anthropic_vertex(config)
    
    print("\n--- 设置Google ---")
    setup_google(config)
    
    print("\n--- 设置百度 ---")
    setup_baidu(config)
    
    print("\n--- 设置智谱 ---")
    setup_zhipu(config)

def main():
    """主函数"""
    print("欢迎使用API配置助手")
    
    # 加载配置
    config = load_config()
    
    # 如果存在配置文件，通知用户
    if os.path.exists(CONFIG_PATH):
        print(f"检测到现有配置文件: {CONFIG_PATH}")
        
    while True:
        print("\n选择要配置的API提供商:")
        print("1. OpenAI")
        print("2. Anthropic")
        print("3. Anthropic (Bedrock)")
        print("4. Anthropic (Vertex)")
        print("5. Google")
        print("6. Baidu")
        print("7. Zhipu")
        print("8. 全部配置")
        print("9. 退出")
        
        choice = input("\n请输入选项(1-9): ")
        
        if choice == "1":
            setup_openai(config)
        elif choice == "2":
            setup_anthropic(config)
        elif choice == "3":
            setup_anthropic_bedrock(config)
        elif choice == "4":
            setup_anthropic_vertex(config)
        elif choice == "5":
            setup_google(config)
        elif choice == "6":
            setup_baidu(config)
        elif choice == "7":
            setup_zhipu(config)
        elif choice == "8":
            setup_all(config)
        elif choice == "9":
            break
        else:
            print("无效选项，请重新选择")
            continue
        
        # 保存配置
        save_config(config)
    
    print("\n配置完成! 您可以开始使用API基准测试工具。")

if __name__ == "__main__":
    main() 