import os
from dotenv import load_dotenv

# 会读取.env文件中的配置项，如果没有则使用环境变量/默认值
load_dotenv()


class Config:
    # Flask 基础配置
    DEBUG = os.getenv("DEBUG", False)
    SECRET_KEY = os.getenv("SECRET_KEY", "default_secret_key")

    # sqlite配置
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    SQLALCHEMY_DATABASE_FILE_NAME = os.getenv("SQLALCHEMY_DATABASE_FILE_NAME", "power_db.db")
    SQLALCHEMY_DATABASE_URI = f"sqlite:///{os.path.join(BASE_DIR, 'database', SQLALCHEMY_DATABASE_FILE_NAME)}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # 模型相关配置
    MODEL1_PATH = os.getenv("MODEL1_PATH", "path/to/model1")
    MODEL2_PATH = os.getenv("MODEL2_PATH", "path/to/model2")

    # 日志相关配置
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE_PATH = os.getenv("LOG_FILE_PATH", "logs/app.log")

    # 其他配置项可以根据项目需求添加
