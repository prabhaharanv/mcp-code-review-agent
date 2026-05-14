from __future__ import annotations

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
    max_input_tokens: int = 100_000
    max_output_tokens: int = 20_000
    max_total_tokens: int = 120_000

    # Webhook
    webhook_secret: str = ""

    # Logging
    log_level: str = "INFO"
    log_json: bool = False

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
