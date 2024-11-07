from flask import Flask
from flask_cors import CORS
from config import Config
from models import init_db, db
from controllers.roubustness_eval_controller import robustness_bp
from controllers.user_controller import user_bp
from common.utils.json_response import JsonResponse
from common.utils.json_flask import JsonFlask


def create_app():
    # 初始化 Flask 应用
    app = JsonFlask(__name__)
    CORS(app, supports_credentials=True)
    # 加载配置
    app.config.from_object(Config)
    # 注册蓝图
    app.register_blueprint(robustness_bp, url_prefix='/robustness')
    app.register_blueprint(user_bp, url_prefix='/user')
    # 更多的蓝图可以在此处注册

    # 创建数据库表
    with app.app_context():
        init_db(app)
        db.create_all()

    @app.errorhandler(Exception)
    def error_handler(e):
        return JsonResponse.error(msg=str(e),code=500)

    return app


# 运行 Flask 应用
if __name__ == "__main__":
    app = create_app()
    app.run(host='0.0.0.0', port=5000)
