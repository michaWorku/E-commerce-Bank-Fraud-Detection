"""
Configuration settings for the application.
Load environment variables using dotenv for sensitive information.
"""

import os
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

class Settings:
    # Example setting:
    # DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./test.db")
    
    # Add your application-specific settings here
    APP_NAME: str = "Your Project Name"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")

settings = Settings()
