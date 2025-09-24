from flask import Blueprint

# Blueprintオブジェクトを作成
bp = Blueprint("main", __name__)

# routes.pyをインポートしてエンドポイントを登録
from . import routes
