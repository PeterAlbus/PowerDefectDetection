# models/__init__.py
from flask_sqlalchemy import SQLAlchemy

# 初始化 SQLAlchemy 实例
db = SQLAlchemy()


def init_db(app):
    db.init_app(app)
