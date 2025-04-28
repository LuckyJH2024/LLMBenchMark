#!/bin/bash

# API Keys for LLM Benchmark Testing
# Replace the placeholder values with your actual API keys

# OpenAI API Key (for GPT models)
export OPENAI_API_KEY=""

# DeepSeek API Key
export DEEPSEEK_API_KEY=""

# Anthropic API Key (for Claude models)
export ANTHROPIC_API_KEY=""

# Print confirmation message
echo "✅ API keys have been set as environment variables"
echo "To confirm, you can run: env | grep _API_KEY"

# Instructions for using the script
echo ""
echo "Important: This script must be 'sourced' to work properly:"
echo "$ source set_api_keys.sh"
echo ""
echo "If you just executed it with bash or sh, the API keys won't be available:"
echo "$ bash set_api_keys.sh  (❌ WRONG - keys only available in sub-process)"
echo "$ sh set_api_keys.sh    (❌ WRONG - keys only available in sub-process)" 