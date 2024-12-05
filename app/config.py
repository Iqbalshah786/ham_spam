from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Retrieve MongoDB URL from environment variable for security
    MONGODB_URL: str = os.getenv("MONGODB_URL")
    
    # Database name
    DATABASE_NAME: str = os.getenv("DATABASE_NAME")

    # Path where the model will be saved
    MODEL_PATH: str = os.getenv("MODEL_PATH")

settings = Settings()
