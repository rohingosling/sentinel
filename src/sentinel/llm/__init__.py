#-----------------------------------------------------------------------------------------------------------------------
# Package: sentinel.llm
# Project: Sentinel
# Version: 0.1.0
# Date:    2025
# Author:  Rohin Gosling
# Note:    Import-only package; not executable directly.
#
# Description:
#
#   LLM adapter abstraction.
#
#   The agentic loop never imports a provider SDK. It talks to LlmAdapter, which Claude and Ollama implement, so adding
#   a provider does not touch the loop and mocking a provider in tests does not mock the loop.
#-----------------------------------------------------------------------------------------------------------------------
