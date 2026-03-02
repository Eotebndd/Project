"""
配置文件 - 使用dotenv加载环境变量
"""
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", None)
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

DATA_PATH = os.getenv("DATA_PATH", "titanic_cleaned.csv")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output_plots")
