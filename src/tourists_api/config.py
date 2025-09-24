import os
from dotenv import load_dotenv

# .envファイルから環境変数を読み込む
load_dotenv()


class Config:
    """ベースとなる設定クラス"""

    SECRET_KEY = os.getenv("SECRET_KEY", "a-default-secret-key")
    # 他の共通設定...


class DevelopmentConfig(Config):
    """開発環境用の設定"""

    DEBUG = True


class ProductionConfig(Config):
    """本番環境用の設定"""

    DEBUG = False
    # 本番用の設定...


# 文字列とクラスをマッピング
config = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig,
}
