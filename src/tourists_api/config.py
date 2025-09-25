import os

from dotenv import load_dotenv

# .envファイルから環境変数を読み込む
load_dotenv()


class Config:
    """ベースとなる設定クラス"""

    SECRET_KEY = os.getenv("SECRET_KEY", "a-default-secret-key")
    
    # Supabaseの設定
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")


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
