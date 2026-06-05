"""
Модуль конфигурации приложения.
Отвечает за загрузку переменных окружения через pydantic-settings
и глобальную инициализацию клиентов языковых моделей (LLM).
"""
import logging
from pathlib import Path
from langchain_openai import ChatOpenAI
from pydantic import Field, BaseModel, model_validator, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent.parent

class LLMSettings(BaseModel):
    base_url: str
    api_key: SecretStr
    model_name: str

class ResearcherSettings(BaseModel):
    recursion_limit: int = Field(default=40)

class TelegramSettings(BaseModel):
    token: SecretStr

class Settings(BaseSettings):
    use_local_llm: bool = False
    llm: Optional[LLMSettings] = None
    local_llm: Optional[LLMSettings] = None
    cheap_llm: Optional[LLMSettings] = None

    researchers: ResearcherSettings



    telegram: Optional[TelegramSettings] = None

    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env",
                                      env_nested_delimiter="__")

    @model_validator(mode='after')
    def check_llm_config(self) -> 'Settings':
        if self.use_local_llm:
            if self.local_llm is None:
                raise ValueError("local_llm is required when use_local_llm is True")
        else:
            if self.llm is None:
                raise ValueError("llm is required when use_local_llm is False")
        return self

settings = Settings()

if settings.use_local_llm:
    llm = ChatOpenAI(
        base_url=settings.local_llm.base_url,
        api_key=settings.local_llm.api_key,
        model=settings.local_llm.model_name,
        temperature=0.0)
    if not settings.cheap_llm:
        cheap_llm = llm

else:
    extra_body = {"thinking": {"type": "disabled"}} if "deepseek" in settings.llm.model_name.lower() else None
    llm = ChatOpenAI(
        base_url=settings.llm.base_url,
        api_key=settings.llm.api_key,
        model=settings.llm.model_name,
        temperature=0.0,
        extra_body=extra_body
    )
    if not settings.cheap_llm:
        logging.warning("USING PRICEY NON-LOCAL LLM AS A CHEAP ONE FOR TEXT COMPRESSION")
        cheap_llm = llm

if settings.cheap_llm:
    cheap_llm = ChatOpenAI(
        base_url=settings.cheap_llm.base_url,
        api_key=settings.cheap_llm.api_key,
        model=settings.cheap_llm.model_name,
    )
