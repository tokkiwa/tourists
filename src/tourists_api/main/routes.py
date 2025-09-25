from datetime import datetime

from flask import Response, g, jsonify, request
from gotrue.errors import AuthError
from postgrest import APIResponse

from src.tourists_api.supabase_client import supabase

from ..utils.auth import require_auth, validate_email, validate_password
from . import bp


@bp.route("/")
def health_check() -> Response:
    """サーバーの状態を確認するためのエンドポイント"""
    return jsonify({"status": "ok"})


@bp.route('/api/users', methods=['GET'])
def get_users() -> Response | tuple[Response, int]:
    """
    userのリストを取得する。
    """
    try:
        response: APIResponse = supabase.table('user').select('*').execute()
        return jsonify(response.data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/api/auth/register', methods=['POST'])
def register_user() -> Response | tuple[Response, int]:
    """
    Supabase Authを使用してユーザーを登録する。
    """
    try:
        data = request.get_json()
        
        required_fields = ['name', 'email', 'password']
        if not data or not all(field in data for field in required_fields):
            return jsonify({"error": "Name, email and password are required"}), 400
        
        name = data['name'].strip()
        email = data['email'].lower().strip()
        password = data['password']
        
        if not name or not validate_email(email) or not validate_password(password):
            return jsonify({"error": "Invalid input provided"}), 400

        auth_response = supabase.auth.sign_up({
            "email": email,
            "password": password,
        })
        
        if auth_response.user is None:
            return jsonify({"error": "Failed to create user account"}), 500

        return jsonify({
            "message": "User registered successfully. Please check your email for confirmation.",
            "user": {
                "user_id": auth_response.user.id,
                "email": auth_response.user.email,
            }
        }), 201
            
    except AuthError as e:
        return jsonify({"error": e.message}), 409
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@bp.route('/api/auth/login', methods=['POST'])
def login_user() -> Response | tuple[Response, int]:
    """
    Supabase Authを使用してログインする
    """
    try:
        data = request.get_json()
        
        if not data or 'email' not in data or 'password' not in data:
            return jsonify({"error": "Email and password are required"}), 400
        
        email = data['email'].lower().strip()
        password = data['password']
        
        auth_response = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })
        
        if auth_response.user is None or auth_response.session is None:
            return jsonify({"error": "Invalid email or password"}), 401
        
        profile_response = supabase.table('profiles').select('*').eq('user_id', auth_response.user.id).execute()
        
        user_data = {
            "user_id": auth_response.user.id,
            "email": auth_response.user.email
        }
        
        if profile_response.data:
            profile = profile_response.data[0]
            user_data.update({
                "name": profile.get('name'),
                "birth_date": profile.get('birth_date'),
                "occupation": profile.get('occupation'),
                "family_structure": profile.get('family_structure'),
                "number_of_children": profile.get('number_of_children')
            })
        
        return jsonify({
            "message": "Login successful",
            "access_token": auth_response.session.access_token,
            "refresh_token": auth_response.session.refresh_token,
            "user": user_data
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 401
    

@bp.route('/api/children', methods=['POST'])
@require_auth
def add_child() -> Response | tuple[Response, int]:
    """
    子供情報を追加
    """
    try:
        data = request.get_json()
        user_id = g.user.id
        
        if not data or 'birth_year' not in data:
            return jsonify({"error": "birth_year is required"}), 400
        
        birth_year = data['birth_year']
        
        current_year = datetime.now().year
        if not isinstance(birth_year, int) or birth_year < 1900 or birth_year > current_year:
            return jsonify({"error": f"birth_year must be between 1900 and {current_year}"}), 400
        
        
        
        child_data = {
            'user_id': user_id,
            'birth_year': birth_year
        }
        
        response = supabase.table('children').insert(child_data).execute()
        
        if response.data:
            return jsonify({
                "message": "Child information added successfully",
                "child": response.data[0]
            }), 201
        else:
            return jsonify({"error": "Failed to add child information"}), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@bp.route('/api/children', methods=['GET'])
@require_auth
def get_children() -> Response | tuple[Response, int]:
    """
    子供情報一覧を取得
    """
    try:
        user_id = g.user.id
        
        
        
        response = supabase.table('children').select('*').eq('user_id', user_id).execute()
        
        return jsonify({
            "children": response.data
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@bp.route('/api/children/<int:child_id>', methods=['DELETE'])
@require_auth
def delete_child(child_id: int) -> Response | tuple[Response, int]:
    """
    子供情報を削除
    """
    try:
        user_id = g.user.id
        
        child_response = supabase.table('children').select('*').eq('child_id', child_id).eq('user_id', user_id).execute()
        
        if not child_response.data:
            return jsonify({"error": "Child information not found"}), 404

        delete_response = supabase.table('children').delete().eq('child_id', child_id).execute()
        
        if delete_response.data:
            return jsonify({
                "message": "Child information deleted successfully"
            }), 200
        else:
            return jsonify({"error": "Failed to delete child information"}), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@bp.route('/api/profiles/me', methods=['GET'])
@require_auth
def get_my_profile() -> Response | tuple[Response, int]:
    """
    現在認証されているユーザーのプロフィールを取得する。
    """
    try:
        user_id = g.user.id
        
        # 認証トークンを取得
        auth_header = request.headers.get('Authorization')
        token = auth_header.split(' ')[1] if auth_header else None
        
        if not token:
            return jsonify({"error": "Authentication token is required"}), 401
        
        # 認証されたSupabaseクライアントを作成
        from ..supabase_client import supabase as base_supabase
        authenticated_supabase = base_supabase
        
        # リクエストヘッダーでアクセストークンを設定
        authenticated_supabase.postgrest.auth(token)
        
        response = authenticated_supabase.table('profiles').select('*').eq('user_id', user_id).single().execute()
        
        return jsonify(response.data), 200
        
    except Exception as e:
        if "JSON object requested, multiple (or no) rows returned" in str(e):
            return jsonify({"error": "Profile not found"}), 404
        return jsonify({"error": str(e)}), 500
    

@bp.route('/api/profiles/me', methods=['PUT'])
@require_auth
def upsert_my_profile() -> Response | tuple[Response, int]:
    """
    プロフィール情報を作成または更新する (UPSERT)。
    `user_id` が一致する行があれば更新、なければ新規作成する。
    """
    try:
        user_id = g.user.id
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "Profile data is required"}), 400
        
        # 認証トークンを取得
        auth_header = request.headers.get('Authorization')
        token = auth_header.split(' ')[1] if auth_header else None
        
        if not token:
            return jsonify({"error": "Authentication token is required"}), 401
        
        allowed_fields = [
            "name",
            "birth_date",
            "occupation",
            "family_structure",
            "number_of_children"
        ]
        
        profile_data = {key: data[key] for key in allowed_fields if key in data}

        if 'family_structure' in profile_data and profile_data['family_structure'] not in ['独身', '既婚']:
            return jsonify({"error": "family_structure must be '独身' or '既婚'"}), 400
        
        profile_data['user_id'] = user_id
        
        # 認証されたSupabaseクライアントを作成
        from ..supabase_client import supabase as base_supabase
        authenticated_supabase = base_supabase
        
        # リクエストヘッダーでアクセストークンを設定
        authenticated_supabase.postgrest.auth(token)
        
        response = authenticated_supabase.table('profiles').upsert(profile_data, on_conflict='user_id').execute()

        return jsonify(response.data[0]), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/api/profile', methods=['GET'])
@require_auth
def get_profile() -> Response | tuple[Response, int]:
    """
    個人データを取得
    """
    try:
        user_id = g.user.id
        response = supabase.table('user_profile').select('*').eq('user_id', user_id).execute()
        
        if response.data:
            return jsonify({
                "profile": response.data[0]
            }), 200
        else:
            return jsonify({"error": "Profile not found"}), 404
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/api/transactions', methods=['POST'])
@require_auth
def add_transaction() -> Response | tuple[Response, int]:
    """
    取引情報を追加
    """
    try:
        data = request.get_json()
        user_id = g.user.id
        
        required_fields = ['category', 'transaction_date', 'store_name', 'amount', 'source']
        if not data or not all(field in data for field in required_fields):
            return jsonify({"error": f"Required fields: {', '.join(required_fields)}"}), 400

        valid_categories = ['食費', '交通費', '住居費', '光熱費', '通信費', '娯楽費', '医療費', 'その他']
        if data['category'] not in valid_categories:
            return jsonify({"error": f"category must be one of: {', '.join(valid_categories)}"}), 400

        try:
            amount = int(data['amount'])
            if amount < 0:
                return jsonify({"error": "amount must be non-negative"}), 400
        except (ValueError, TypeError):
            return jsonify({"error": "amount must be a valid integer"}), 400

        transaction_data = {
            'user_id': user_id,
            'category': data['category'],
            'transaction_date': data['transaction_date'],  # YYYY-MM-DD形式
            'store_name': data['store_name'],
            'amount': amount,
            'source': data['source']
        }
        
        response = supabase.table('transactions').insert(transaction_data).execute()
        
        if response.data:
            return jsonify({
                "message": "Transaction added successfully",
                "transaction": response.data[0]
            }), 201
        else:
            return jsonify({"error": "Failed to add transaction"}), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@bp.route('/api/transactions', methods=['GET'])
@require_auth
def get_transactions() -> Response | tuple[Response, int]:
    """
    取引情報一覧を取得
    """
    try:
        user_id = g.user.id

        category = request.args.get('category')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        limit = request.args.get('limit', 100)
        
        try:
            limit = int(limit)
            if limit > 1000:
                limit = 1000
        except ValueError:
            limit = 100
        
        
        
        query = supabase.table('transactions').select('*').eq('user_id', user_id)

        if category:
            query = query.eq('category', category)
        if start_date:
            query = query.gte('transaction_date', start_date)
        if end_date:
            query = query.lte('transaction_date', end_date)
            
        query = query.order('transaction_date', desc=True).limit(limit)
        
        response = query.execute()
        
        return jsonify({
            "transactions": response.data,
            "count": len(response.data)
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@bp.route('/api/transactions/<int:transaction_id>', methods=['PUT'])
@require_auth
def update_transaction(transaction_id: int) -> Response | tuple[Response, int]:
    """
    取引情報を更新
    """
    try:
        data = request.get_json()
        user_id = g.user.id
        
        if not data:
            return jsonify({"error": "Update data is required"}), 400

        existing_transaction = supabase.table('transactions').select('*').eq('transaction_id', transaction_id).eq('user_id', user_id).execute()
        
        if not existing_transaction.data:
            return jsonify({"error": "Transaction not found"}), 404
        
        allowed_fields = ['category', 'transaction_date', 'store_name', 'amount', 'source']
        update_data = {}
        
        for field in allowed_fields:
            if field in data:
                update_data[field] = data[field]
        
        if not update_data:
            return jsonify({"error": "No valid fields to update"}), 400
        
        if 'category' in update_data:
            valid_categories = ['食費', '交通費', '住居費', '光熱費', '通信費', '娯楽費', '医療費', 'その他']
            if update_data['category'] not in valid_categories:
                return jsonify({"error": f"category must be one of: {', '.join(valid_categories)}"}), 400
        
        if 'amount' in update_data:
            try:
                amount = int(update_data['amount'])
                if amount < 0:
                    return jsonify({"error": "amount must be non-negative"}), 400
                update_data['amount'] = amount
            except (ValueError, TypeError):
                return jsonify({"error": "amount must be a valid integer"}), 400
        
        response = supabase.table('transactions').update(update_data).eq('transaction_id', transaction_id).execute()
        
        if response.data:
            return jsonify({
                "message": "Transaction updated successfully",
                "transaction": response.data[0]
            }), 200
        else:
            return jsonify({"error": "Failed to update transaction"}), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/api/transactions/<int:transaction_id>', methods=['DELETE'])
@require_auth
def delete_transaction(transaction_id: int) -> Response | tuple[Response, int]:
    """
    取引情報を削除
    """
    try:
        user_id = g.user.id
        
        existing_transaction = supabase.table('transactions').select('*').eq('transaction_id', transaction_id).eq('user_id', user_id).execute()
        
        if not existing_transaction.data:
            return jsonify({"error": "Transaction not found"}), 404

        delete_response = supabase.table('transactions').delete().eq('transaction_id', transaction_id).execute()
        
        if delete_response.data:
            return jsonify({
                "message": "Transaction deleted successfully"
            }), 200
        else:
            return jsonify({"error": "Failed to delete transaction"}), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@bp.route('/api/long-term-plans', methods=['POST'])
@require_auth
def create_long_term_plan() -> Response | tuple[Response, int]:
    """
    長期計画を作成
    """
    try:
        data = request.get_json()
        user_id = g.user.id
        
        required_fields = ['plan_name', 'target_amount', 'target_date']
        if not data or not all(field in data for field in required_fields):
            return jsonify({"error": f"Required fields: {', '.join(required_fields)}"}), 400

        try:
            target_amount = int(data['target_amount'])
            if target_amount <= 0:
                return jsonify({"error": "target_amount must be positive"}), 400
        except (ValueError, TypeError):
            return jsonify({"error": "target_amount must be a valid integer"}), 400
        
        
        
        plan_data = {
            'user_id': user_id,
            'plan_name': data['plan_name'],
            'target_amount': target_amount,
            'target_date': data['target_date']  # YYYY-MM-DD形式
        }
        
        response = supabase.table('long_term_plans').insert(plan_data).execute()
        
        if response.data:
            return jsonify({
                "message": "Long-term plan created successfully",
                "plan": response.data[0]
            }), 201
        else:
            return jsonify({"error": "Failed to create long-term plan"}), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/api/long-term-plans', methods=['GET'])
@require_auth
def get_long_term_plans() -> Response | tuple[Response, int]:
    """
    長期計画一覧を取得
    """
    try:
        user_id = g.user.id
        
        
        
        response = supabase.table('long_term_plans').select('*').eq('user_id', user_id).order('target_date').execute()
        
        return jsonify({
            "plans": response.data
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/api/short-term-plans', methods=['POST'])
@require_auth
def create_short_term_plan() -> Response | tuple[Response, int]:
    """
    短期計画を作成
    """
    try:
        data = request.get_json()
        user_id = g.user.id
        
        required_fields = ['period', 'monthly_saving_goal']
        if not data or not all(field in data for field in required_fields):
            return jsonify({"error": f"Required fields: {', '.join(required_fields)}"}), 400
        
        try:
            monthly_saving_goal = int(data['monthly_saving_goal'])
            if monthly_saving_goal < 0:
                return jsonify({"error": "monthly_saving_goal must be non-negative"}), 400
        except (ValueError, TypeError):
            return jsonify({"error": "monthly_saving_goal must be a valid integer"}), 400
        
        existing_plan = supabase.table('short_term_plans').select('short_plan_id').eq('user_id', user_id).eq('period', data['period']).execute()
        if existing_plan.data:
            return jsonify({"error": "Plan for this period already exists. Use PUT to update."}), 409
        
        plan_data = {
            'user_id': user_id,
            'period': data['period'],  # 例: "2025-09"
            'monthly_saving_goal': monthly_saving_goal,
            'memo': data.get('memo', '')
        }
        
        response = supabase.table('short_term_plans').insert(plan_data).execute()
        
        if response.data:
            return jsonify({
                "message": "Short-term plan created successfully",
                "plan": response.data[0]
            }), 201
        else:
            return jsonify({"error": "Failed to create short-term plan"}), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@bp.route('/api/short-term-plans', methods=['GET'])
@require_auth
def get_short_term_plans() -> Response | tuple[Response, int]:
    """
    短期計画一覧を取得
    """
    try:
        user_id = g.user.id
        
        
        
        response = supabase.table('short_term_plans').select('*').eq('user_id', user_id).order('period', desc=True).execute()
        
        return jsonify({
            "plans": response.data
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/api/user-policies', methods=['POST'])
@require_auth
def create_user_policy() -> Response | tuple[Response, int]:
    """
    ユーザーポリシーを作成（認証が必要）
    """
    try:
        data = request.get_json()
        user_id = g.user.id
        
        if not data or 'policy_text' not in data:
            return jsonify({"error": "policy_text is required"}), 400
        
        
        
        policy_data = {
            'user_id': user_id,
            'policy_text': data['policy_text']
        }
        
        response = supabase.table('user_policies').insert(policy_data).execute()
        
        if response.data:
            return jsonify({
                "message": "User policy created successfully",
                "policy": response.data[0]
            }), 201
        else:
            return jsonify({"error": "Failed to create user policy"}), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@bp.route('/api/user-policies', methods=['GET'])
@require_auth
def get_user_policies() -> Response | tuple[Response, int]:
    """
    ユーザーポリシー一覧を取得
    """
    try:
        user_id = g.user.id

        response = supabase.table('user_policies').select('*').eq('user_id', user_id).execute()
        
        return jsonify({
            "policies": response.data
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ========= 新しいAPI: 叱り通知とチャットボット =========

@bp.route('/api/scolding-notifications', methods=['POST'])
@require_auth
def create_scolding_notification() -> Response | tuple[Response, int]:
    """
    メールから購入を抽出して叱り通知を生成
    """
    try:
        data = request.get_json()
        user_id = g.user.id
        
        if not data or 'email_content' not in data:
            return jsonify({"error": "email_content is required"}), 400
        
        email_content = data['email_content']
        
        # ユーザーの目標を取得
        goals_response = supabase.table('long_term_plans').select('*').eq('user_id', user_id).execute()
        user_goals = []
        for goal in goals_response.data:
            user_goals.append({
                "purpose": goal.get('plan_name'),
                "target_amount": goal.get('target_amount'),
                "by": goal.get('target_date')
            })
        
        # 現在の日時でメールを処理
        from datetime import datetime
        from zoneinfo import ZoneInfo
        JST = ZoneInfo("Asia/Tokyo")
        ts_ymd = datetime.now(JST).strftime("%Y-%m-%d")
        
        dated_emails = [{
            "text": email_content,
            "ts_ymd": ts_ymd,
        }]
        
        # agentを使って処理
        from . import agent
        result = agent.run_agent_once(dated_emails=dated_emails, user_goals=user_goals)
        
        # 結果をデータベースに保存（オプション）
        notification_data = {
            'user_id': user_id,
            'message': result.get('message', ''),
            'should_scold': result.get('should_scold', False),
            'pace_ratio': result.get('pace_ratio', 0),
            'spent_amount': result.get('month_spent_jpy_updated', 0),
            'budget_amount': result.get('month_plan_jpy', 0),
            'scolding_strategy': result.get('scolding_strategy', 'general_financial'),
            'created_at': datetime.now(JST).isoformat()
        }
        
        # 通知テーブルに保存（テーブルが存在する場合）
        try:
            supabase.table('scolding_notifications').insert(notification_data).execute()
        except:
            pass  # テーブルが存在しない場合はスキップ
        
        return jsonify({
            "message": result.get('message', ''),
            "should_scold": result.get('should_scold', False),
            "details": {
                "pace_ratio": result.get('pace_ratio', 0),
                "spent_amount": int(result.get('month_spent_jpy_updated', 0)),
                "budget_amount": int(result.get('month_plan_jpy', 0)),
                "scolding_strategy": result.get('scolding_strategy', 'general_financial'),
                "return_candidates": len(result.get('return_candidates', [])),
                "should_open_orders": result.get('should_open_orders', False)
            }
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/api/financial-chat', methods=['POST'])
@require_auth
def financial_chat() -> Response | tuple[Response, int]:
    """
    家計管理に関する相談チャットボット
    """
    try:
        data = request.get_json()
        user_id = g.user.id
        
        if not data or 'message' not in data:
            return jsonify({"error": "message is required"}), 400
        
        user_message = data['message']
        
        # ユーザーの情報を取得
        user_response = supabase.table('user').select('*').eq('id', user_id).execute()
        goals_response = supabase.table('long_term_plans').select('*').eq('user_id', user_id).execute()
        policies_response = supabase.table('user_policies').select('*').eq('user_id', user_id).execute()
        
        # ユーザーコンテキストを構築
        user_context = {}
        if user_response.data:
            user_data = user_response.data[0]
            user_context.update({
                'occupation': user_data.get('occupation'),
                'birth_date': user_data.get('birth_date'),
                'family_structure': user_data.get('family_structure'),
                'number_of_children': user_data.get('number_of_children', 0)
            })
        
        goals_text = "\n".join([f"- {g.get('plan_name')}: {g.get('target_amount')}円 ({g.get('target_date')})" 
                               for g in goals_response.data])
        
        policies_text = "\n".join([f"- {p.get('policy_text')}" 
                                  for p in policies_response.data])
        
        # チャットボット用のプロンプトを作成
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """あなたは親しみやすい家計管理アドバイザーです。
ユーザーの個人的な状況を理解し、実用的で優しいアドバイスを提供してください。

ユーザー情報:
- 職業: {occupation}
- 家族構成: {family_structure}
- 子供の数: {number_of_children}人

長期目標:
{goals}

支出ポリシー:
{policies}

以下の点を心がけてください:
1. 判断的にならず、共感的に対応する
2. 具体的で実行可能なアドバイスを提供する
3. ユーザーの目標と現状を考慮した提案をする
4. 必要に応じて質問で深掘りする
5. 200-300文字程度で簡潔に回答する"""),
            
            ("human", "{user_message}")
        ])
        
        response = llm.invoke(prompt.format_messages(
            occupation=user_context.get('occupation', '未設定'),
            family_structure=user_context.get('family_structure', '未設定'),
            number_of_children=user_context.get('number_of_children', 0),
            goals=goals_text or "設定されていません",
            policies=policies_text or "設定されていません",
            user_message=user_message
        ))
        
        # チャット履歴を保存（オプション）
        try:
            chat_data = {
                'user_id': user_id,
                'user_message': user_message,
                'bot_response': response.content,
                'created_at': datetime.now().isoformat()
            }
            supabase.table('chat_history').insert(chat_data).execute()
        except:
            pass  # テーブルが存在しない場合はスキップ
        
        return jsonify({
            "response": response.content,
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/api/scolding-notifications', methods=['GET'])
@require_auth
def get_scolding_notifications() -> Response | tuple[Response, int]:
    """
    ユーザーの叱り通知履歴を取得
    """
    try:
        user_id = g.user.id
        
        response = supabase.table('scolding_notifications').select('*').eq('user_id', user_id).order('created_at', desc=True).limit(20).execute()
        
        return jsonify({
            "notifications": response.data
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/api/chat-history', methods=['GET'])
@require_auth
def get_chat_history() -> Response | tuple[Response, int]:
    """
    ユーザーのチャット履歴を取得
    """
    try:
        user_id = g.user.id
        
        response = supabase.table('chat_history').select('*').eq('user_id', user_id).order('created_at', desc=True).limit(50).execute()
        
        return jsonify({
            "chat_history": response.data
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
