import os

from flask import Flask

from .config import config
from .main import bp as main_blueprint


def create_app():
    """Flaskアプリケーションインスタンスを作成する"""
    app = Flask(__name__)

    # 環境変数から設定を読み込む
    config_name = os.getenv("FLASK_CONFIG", "default")
    app.config.from_object(config[config_name])

    # Blueprintを登録
    app.register_blueprint(main_blueprint)

    return app
