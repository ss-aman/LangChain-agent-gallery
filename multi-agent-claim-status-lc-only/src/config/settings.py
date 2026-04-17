from __future__ import annotations
from functools import lru_cache
from typing import List, Optional
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    LLM_PROVIDER: str = "openai"
    LLM_API_KEY: str = "sk-placeholder"
    LLM_BASE_URL: Optional[str] = None
    LLM_MODEL: str = "gpt-4o"
    LLM_TEMPERATURE: float = 0.0
    LLM_MAX_TOKENS: int = 2048

    DATABASE_URL: str = "sqlite+aiosqlite:///./claims.db"
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    DB_ECHO: bool = False

    REDIS_URL: str = "redis://localhost:6379/0"
    SESSION_TTL_SECONDS: int = 3600
    CACHE_TTL_SECONDS: int = 300

    JWT_SECRET_KEY: str = "CHANGE_ME_IN_PRODUCTION"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    RATE_LIMIT_REQUESTS_PER_MINUTE: int = 60
    RATE_LIMIT_BURST: int = 10

    APP_ENV: str = "development"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    ALLOWED_ORIGINS: str = "http://localhost:3000"

    LOG_LEVEL: str = "INFO"

    @property
    def allowed_origins_list(self) -> List[str]:
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",")]

    @property
    def is_production(self) -> bool:
        return self.APP_ENV == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
