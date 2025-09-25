from flask import Response, jsonify
from postgrest import APIResponse

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
        # 循環インポートを避けるため関数内でインポート
        from tourists_api.supabase import get_supabase_client

        # リクエストごとにSupabaseクライアントを取得
        supabase = get_supabase_client()
        response: APIResponse = supabase.table('user').select('*').execute()
        
        return jsonify(response.data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500