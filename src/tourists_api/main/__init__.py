from flask import Blueprint

from . import routes

# Blueprintオブジェクトを作成
bp = Blueprint("main", __name__)

