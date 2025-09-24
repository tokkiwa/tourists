from flask import jsonify
from . import bp


@bp.route("/")
def health_check():
    """サーバーの状態を確認するためのエンドポイント"""
    return jsonify({"status": "ok"})


@bp.route("/api/items", methods=["GET"])
def get_items():
    """アイテムのリストを返すサンプルAPI"""
    items = [
        {"id": 1, "name": "Item 1"},
        {"id": 2, "name": "Item 2"},
    ]
    return jsonify(items)
