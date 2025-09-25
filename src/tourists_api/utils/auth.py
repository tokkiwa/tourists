# auth_utils.py (修正版)
import re
from functools import wraps

from flask import g, jsonify, request
from gotrue.errors import AuthApiError

from tourists_api.supabase_client import supabase


def validate_email(email: str) -> bool:
    """メールアドレスの形式を検証"""
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(email_pattern, email) is not None

def validate_password(password: str) -> bool:
    """パスワードの強度を検証（最低8文字、英数字を含む）"""
    if len(password) < 8:
        return False
    if not re.search(r'[A-Za-z]', password):
        return False
    if not re.search(r'\d', password):
        return False
    return True

def require_auth(f):
    """
    Supabaseが発行したJWTを検証し、認証済みユーザーをリクエストコンテキストに格納するデコレータ。
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Authorization header is missing or invalid'}), 401

        token = auth_header.split(' ')[1]
        
        try:
            user_response = supabase.auth.get_user(token)
            g.user = user_response.user
        except AuthApiError as e:
            return jsonify({'error': 'Invalid token', 'details': e.message}), 401
        except Exception as e:
            return jsonify({'error': 'An unexpected error occurred', 'details': str(e)}), 500

        return f(*args, **kwargs)
    
    return decorated