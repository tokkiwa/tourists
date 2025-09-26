# mailAgent.py
import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import List, Dict
import os
import sys
import pathlib

# プロジェクトルートを sys.path に追加
project_root = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tourists_api.main.listen_gmail import start_listening  # 既存ファイル
from tourists_api.main import agent  # あなたの LangGraph 実装ファイル（run_agent_once を追記済み）

JST = ZoneInfo("Asia/Tokyo")

# ---- あなたの目標はここに ----
USER_GOALS: List[Dict] = [
    {"purpose": "来春の台湾旅行", "target_amount": 200000, "by": "2026-03"},
    {"purpose": "非常用資金",   "target_amount": 300000, "by": "2026-09"},
]



async def consume_and_run(email_queue: asyncio.Queue):
    """メール1通につきLangGraphを1回実行"""
    print("[mailAgent] consumer task started, waiting for emails...")

    while True:
        email_info = await email_queue.get()  # dict: sender_email, subject, body, ...
        try:
            print("\n[mailAgent] === New email pulled from queue ===")
            print(f"[mailAgent] From   : {email_info.get('sender_email')}")
            print(f"[mailAgent] Subject: {email_info.get('subject')}")
            print(f"[mailAgent] Body len: {len(email_info.get('body',''))} chars")
            
            # 環境変数からuser_idを取得（main関数で設定済み）
            user_id = os.getenv("USER_ID")

            # 受信した瞬間の JST 日付を ts_ymd にする
            ts_ymd = datetime.now(JST).strftime("%Y-%m-%d")

            dated_emails = [{
                "text": email_info.get("body", ""),
                "ts_ymd": ts_ymd,
            }]

            print("[mailAgent] invoking agent.run_agent_once ...")
            result = agent.run_agent_once(dated_emails=dated_emails, user_goals=USER_GOALS)

            # --- LangGraph の実行結果ログ ---
            print("\n[mailAgent] === AGENT RESULT ===")
            print("month_id         :", result.get("month_id"))
            print("auto month_plan  :", int(result.get("month_plan_jpy", 0)))
            print("spent (this month):", int(result.get("month_spent_jpy_updated", 0)))
            print("pace_ratio       :", f"{result.get('pace_ratio', 0):.3f}")
            print("should_scold     :", result.get("should_scold"))
            print("message          :")
            print(result.get("message", ""))
            print("[mailAgent] =====================\n")

            # --- データベースに結果を保存 ---
            try:
                from tourists_api.supabase_client import supabase
                
                # 結果をデータベースに保存（created_atはデフォルト値を使用）
                notification_data = {
                    'user_id': user_id,
                    'content': result.get('message', '')
                }
                
                # 認証済みSupabaseクライアントを使用してデータを挿入
                supabase.table('scolding_notifications').insert(notification_data).execute()
                print("[mailAgent] Result saved to database successfully")
                
            except Exception as db_error:
                print(f"[mailAgent] ERROR saving to database: {db_error}")
                # 重複エラーの場合は、UPSERTを試行
                if "duplicate key" in str(db_error).lower():
                    try:
                        from tourists_api.supabase_client import supabase
                        print("[mailAgent] Attempting upsert instead...")
                        # タイムスタンプを含むユニークなデータで再試行
                        notification_data_with_timestamp = {
                            'user_id': user_id,
                            'content': result.get('message', ''),
                            'created_at': datetime.now(JST).isoformat()
                        }
                        # upsertを使用（既存レコードがある場合は更新）
                        supabase.table('scolding_notifications').upsert(notification_data_with_timestamp).execute()
                        print("[mailAgent] Result saved via upsert successfully")
                    except Exception as upsert_error:
                        print(f"[mailAgent] ERROR with upsert: {upsert_error}")

        except Exception as e:
            print(f"[mailAgent] ERROR while running agent: {e}")
        finally:
            email_queue.task_done()


async def main():
    #ログインしてアクセストークンを取得・保存
    #環境変数にUSER_EMAIL, USER_PASSWORDがあるとする
    user_email = os.getenv("USER_EMAIL")
    user_password = os.getenv("USER_PASSWORD")
    
    if not user_email or not user_password:
        print("[mailAgent] ERROR: USER_EMAIL and USER_PASSWORD must be set in environment variables")
        return
    
    try:
        from tourists_api.supabase_client import supabase
        
        # Supabaseでログイン（routes.pyと同じ方法）
        auth_response = supabase.auth.sign_in_with_password({
            "email": user_email,
            "password": user_password
        })
        
        if auth_response.user is None or auth_response.session is None:
            print("[mailAgent] ERROR: login failed, invalid credentials")
            return
            
        # アクセストークンとユーザーIDを取得
        token = auth_response.session.access_token
        user_id = auth_response.user.id
        
        # 環境変数に設定
        os.environ["SUPABASE_ACCESS_TOKEN"] = token
        os.environ["USER_ID"] = user_id
        
        print(f"[mailAgent] logged in successfully")
        print(f"[mailAgent] user_id: {user_id}")
        print(f"[mailAgent] token: {token[:20]}...")  # トークンの一部を表示
        
        # Supabaseクライアントを認証済みトークンで設定
        supabase.postgrest.auth(token)
        print("[mailAgent] Supabase client authenticated successfully")
        
    except Exception as e:
        print(f"[mailAgent] ERROR during login: {e}")
        return
    
    print("[mailAgent] starting main...")
    email_queue = asyncio.Queue()

    consumer = asyncio.create_task(consume_and_run(email_queue))
    print("[mailAgent] consumer task created")

    try:
        print("[mailAgent] starting Gmail listener...")
        await start_listening(email_queue)
    except KeyboardInterrupt:
        print("[mailAgent] KeyboardInterrupt: shutting down...")
    finally:
        consumer.cancel()
        try:
            await consumer
        except asyncio.CancelledError:
            print("[mailAgent] consumer task cancelled")


if __name__ == "__main__":
    print("[mailAgent] running asyncio main...")
    asyncio.run(main())
