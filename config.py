from __future__ import annotations

import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM
    llm_provider: str = "anthropic"
    llm_model: str = "claude-sonnet-4-20250514"
    anthropic_api_key: str = ""
    openai_api_key: str = ""

    # GitHub
    github_token: str = ""

    # Knowledge Base (RAG API)
    rag_api_url: str = "http://localhost:8000"
    rag_api_key: str = ""

    # Agent
    max_agent_steps: int = 20

    # Logging
    log_level: str = "INFO"
    log_json: bool = False

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
