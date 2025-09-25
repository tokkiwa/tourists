import os

from flask import Flask
from flask_cors import CORS

from .config import config
from .main import bp as main_blueprint


def create_app():
    """Flaskアプリケーションインスタンスを作成する"""
    app = Flask(__name__)

    # 環境変数から設定を読み込む
    config_name = os.getenv("FLASK_CONFIG", "default")
    app.config.from_object(config[config_name])

    # CORSを設定（開発環境ではReactアプリからのアクセスを許可）
    CORS(app, resources={
        r"/api/*": {
            "origins": ["http://localhost:3000", "http://127.0.0.1:3000"],
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"],
            "supports_credentials": True
        }
    })

    # Blueprintを登録
    app.register_blueprint(main_blueprint, url_prefix='/api')

    return app
