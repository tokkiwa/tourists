from flask import Response, jsonify
from postgrest import APIResponse

from tourists_api.supabase import supabase

from . import bp


@bp.route("/")
def health_check() -> Response:
    """サーバーの状態を確認するためのエンドポイント"""
    return jsonify({"status": "ok"})


@bp.route('/api/users', methods=['GET'])
def get_users() -> Response | tuple[Response, int]:
    """
    userのリストを取得する。
    成功時はResponse、失敗時は現状(Response, status_code)のタプルを返す。
    """
    try:
        if supabase is None:
            return jsonify({"error": "Supabase client is not initialized"}), 500
        response: APIResponse = supabase.table('user').select('*').execute()
        
        return jsonify(response.data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500