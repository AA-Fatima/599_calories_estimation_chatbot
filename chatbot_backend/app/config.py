from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    # OpenAI
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"  # or "gpt-4" for better results
    
    # DeepSeek (optional)
    DEEPSEEK_API_KEY: Optional[str] = None
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    
    # Data paths
    USDA_FOUNDATION_PATH: str = "app/data/raw/USDA_foundation.json"
    USDA_SR_LEGACY_PATH: str = "app/data/raw/USDA_sr_legacy.json"
    DISHES_PATH: str = "app/data/raw/dishes.xlsx"
    
    # NLP settings
    SIMILARITY_THRESHOLD: float = 0.7
    
    # Server settings
    PORT: int = 8000
    HOST: str = "0.0.0.0"
    RELOAD: bool = False
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "allow"  # Allow extra fields


settings = Settings()

# Validate critical settings
if not settings.OPENAI_API_KEY:
    import warnings
    warnings.warn("⚠️  OPENAI_API_KEY not set!  Fallback responses will not work.")