from pydantic import Field, BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseModel):
    base_url: str
    api_key: str
    model_name: str

class SearchSettings(BaseModel):
    api_key: str

class Settings(BaseSettings):
    llm: LLMSettings
    search: SearchSettings
    model_config = SettingsConfigDict(env_file=".env", env_nested_delimiter="__")


settings = Settings()
