# Blueprintオブジェクトを作成
from flask import Blueprint

bp = Blueprint("main", __name__)

# Blueprintの定義後にroutesをインポート
from . import routes
