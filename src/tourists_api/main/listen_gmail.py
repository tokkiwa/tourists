import os.path
import base64
import json
import time
import re
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from dotenv import load_dotenv, find_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.cloud import pubsub_v1

# .envファイルのパスを検索し、そのディレクトリを基準にする
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
dotenv_dir = os.path.dirname(dotenv_path)
SENDER_LIST = [email.strip() for email in os.getenv("SENDER_LIST", "").split(",") if email.strip()]

# --- 設定項目 ---
# Gmail APIのスコープ (変更不要)
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly", "https://www.googleapis.com/auth/pubsub"]
# GCPプロジェクトID (環境変数から取得)
PROJECT_ID = os.getenv("GCP_PROJECT_ID")
# Pub/SubのトピックID (GCPで設定したもの)
PUB_SUB_TOPIC = os.getenv("GCP_PUB_SUB_TOPIC")
# Pub/SubのサブスクリプションID (GCPで設定したもの)
PUB_SUB_SUBSCRIPTION = os.getenv("GCP_PUB_SUB_SUBSCRIPTION")
# 認証情報ファイルのパス (.envファイルからの相対パス)
CREDENTIALS_PATH = os.path.join(dotenv_dir, os.getenv("GCP_CREDENTIALS_PATH", "credentials.json"))
TOKEN_PATH = os.path.join(dotenv_dir, os.getenv("GCP_TOKEN_PATH", "token.json"))
# ----------------

def get_email_body(msg):
    """Gmailメッセージから本文を抽出する"""
    if "parts" in msg["payload"]:
        parts = msg["payload"]["parts"]
        part = next((p for p in parts if p["mimeType"] == "text/plain"), None)
        if part:
            return part["body"]["data"]
        part = next((p for p in parts if p["mimeType"] == "text/html"), None)
        if part:
            return part["body"]["data"]
    elif "data" in msg["payload"]["body"]:
        return msg["payload"]["body"]["data"]
    return None

def extract_email_info(msg):
    """メールから必要な情報を抽出し、tracking対象の場合は辞書を返す"""
    headers = msg["payload"]["headers"]
    sender_raw = next((h["value"] for h in headers if h["name"] == "From"), "Unknown Sender")
    subject = next((h["value"] for h in headers if h["name"] == "Subject"), "No Subject")

    # 送信者メールアドレスを抽出
    match = re.search(r'<(.+?)>', sender_raw)
    sender_email = match.group(1) if match else sender_raw

    # SENDER_LISTに含まれているかチェック
    if not any(tracked_sender in sender_email for tracked_sender in SENDER_LIST):
        print(f"\nSender not tracked: {sender_email}")
        return None

    print("\n" + "=" * 30)
    print(f"NEW EMAIL from: {sender_raw}")

    body_data = get_email_body(msg)
    body = ""
    if body_data:
        body = base64.urlsafe_b64decode(body_data).decode("utf-8", errors="ignore")

    return {
        "sender_raw": sender_raw,
        "sender_email": sender_email,
        "subject": subject,
        "body": body,
        "message_id": msg["id"]
    }

def process_notification(message: pubsub_v1.subscriber.message.Message, creds, start_history_id_ref, email_queue) -> None:
    """Pub/Subから通知を受け取ったときの処理"""
    message.ack()  # メッセージを正常に受信したことをPub/Subに伝える
    data = json.loads(message.data)
    email_address = data["emailAddress"]
    # 通知のhistoryIdは使わないが、ログのために表示
    print(f"\n--- Notification received for {email_address} (Notification History ID: {data['historyId']}) ---")
    print(f"Using startHistoryId: {start_history_id_ref['id']}")

    try:
        # スレッドセーフのため、コールバック内でサービスオブジェクトを生成する
        service = build("gmail", "v1", credentials=creds)

        # 前回のhistoryIdを使って、新しいメッセージを取得する
        history = (
            service.users()
            .history()
            .list(userId="me", startHistoryId=start_history_id_ref['id'])
            .execute()
        )
        
        # 次回のためにhistoryIdを更新
        new_history_id = history.get("historyId")
        if new_history_id:
            print(f"Updating historyId from {start_history_id_ref['id']} to {new_history_id}")
            start_history_id_ref['id'] = new_history_id

        # 履歴に新しいメッセージがあれば処理
        if "history" not in history:
            print("No new history found.")
            return
            
        messages_added = []
        for h in history["history"]:
            messages_added.extend(h.get("messagesAdded", []))

        if not messages_added:
            print("No new messages found in history.")
            return

        for added_msg in messages_added:
            msg_id = added_msg["message"]["id"]
            msg = (
                service.users()
                .messages()
                .get(userId="me", id=msg_id)
                .execute()
            )

            # メール情報を抽出
            email_info = extract_email_info(msg)
            if email_info:
                print(f"Subject: {email_info['subject']}")
                print("Body:")
                print(email_info['body'][:500] + "..." if len(email_info['body']) > 500 else email_info['body'])
                print("=" * 30)
                
                # キューに追加（thread-safeな操作）
                try:
                    email_queue.put_nowait(email_info)
                    print(f"Email added to queue: {email_info['subject']}")
                except:
                    print("Failed to add email to queue")

    except HttpError as error:
        print(f"An error occurred while fetching email details: {error}")


async def start_listening(email_queue: Optional[asyncio.Queue] = None):
    """Gmail APIの認証と監視を開始する（async版）"""
    if email_queue is None:
        email_queue = asyncio.Queue()
    
    # 認証処理
    creds = _authenticate()
    
    try:
        # 1. Gmailにプッシュ通知の監視をリクエスト
        gmail_service = build("gmail", "v1", credentials=creds)

        request = {
            "labelIds": ["INBOX"],
            "topicName": f"projects/{PROJECT_ID}/topics/{PUB_SUB_TOPIC}",
        }
        
        response = gmail_service.users().watch(userId="me", body=request).execute()
        
        print(f"Watch request successful. Monitoring started.")
        print(f"History ID: {response['historyId']}")

        # 監視開始時のhistoryIdをミュータブルなオブジェクト（辞書）に保存して参照渡しする
        start_history_id_ref = {'id': response['historyId']}

        # 2. Pub/Subサブスクリプションでメッセージを待機
        subscriber = pubsub_v1.SubscriberClient(credentials=creds)
        subscription_path = subscriber.subscription_path(PROJECT_ID, PUB_SUB_SUBSCRIPTION)

        # thread-safeなキューを作成
        import queue
        thread_safe_queue = queue.Queue()

        # コールバック関数にhistoryIdとキューへの参照を渡す
        callback = lambda message: process_notification(message, creds, start_history_id_ref, thread_safe_queue)
        
        streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback)
        print(f"Listening for messages on {subscription_path}...")

        # キューの監視タスクを開始
        queue_task = asyncio.create_task(_monitor_queue(thread_safe_queue, email_queue))

        # プログラムが終了しないように待機
        try:
            # streaming_pull_futureを別のタスクで実行
            pull_task = asyncio.create_task(_run_streaming_pull(streaming_pull_future))
            await pull_task
        except (TimeoutError, KeyboardInterrupt):
            streaming_pull_future.cancel()
            streaming_pull_future.result()
        finally:
            queue_task.cancel()
            try:
                await queue_task
            except asyncio.CancelledError:
                pass

    except HttpError as error:
        print(f"An error occurred: {error}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
    return email_queue

async def _run_streaming_pull(streaming_pull_future):
    """streaming pull futureを非同期で実行"""
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        await loop.run_in_executor(executor, streaming_pull_future.result)

def _authenticate():
    """認証処理を同期関数として分離"""
    creds = None
    # token.jsonファイルは、ユーザーのアクセストークンとリフレッシュトークンを保存します。
    # 最初の認証フローが完了すると自動的に作成されます。
    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
    # 有効な認証情報がない場合は、ユーザーにログインを促します。
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                CREDENTIALS_PATH, SCOPES
            )
            creds = flow.run_local_server(port=0)
        # 次回のために認証情報を保存します。
        with open(TOKEN_PATH, "w") as token:
            token.write(creds.to_json())
    return creds

async def _monitor_queue(thread_safe_queue, async_queue):
    """thread-safeなキューからasyncioキューにメッセージを転送"""
    while True:
        try:
            # 0.1秒ごとにキューをチェック
            await asyncio.sleep(0.1)
            while not thread_safe_queue.empty():
                try:
                    email_info = thread_safe_queue.get_nowait()
                    await async_queue.put(email_info)
                    print(f"Email transferred to async queue: {email_info['subject']}")
                except:
                    break
        except asyncio.CancelledError:
            break


async def process_emails_from_queue(email_queue: asyncio.Queue):
    """キューからメールを取得して処理する関数の例"""
    while True:
        try:
            # キューからメールを取得（タイムアウト付き）
            email_info = await asyncio.wait_for(email_queue.get(), timeout=1.0)
            
            # ここで取得したメールを処理
            print(f"\n[PROCESSING] Email from: {email_info['sender_email']}")
            print(f"[PROCESSING] Subject: {email_info['subject']}")
            print(f"[PROCESSING] Body length: {len(email_info['body'])} characters")
            
            # 他の処理に使う例
            # await some_other_function(email_info)
            
            # キューのタスク完了を通知
            email_queue.task_done()
            
        except asyncio.TimeoutError:
            # タイムアウト時は継続
            continue
        except Exception as e:
            print(f"Error processing email: {e}")
            continue

async def main():
    """メイン関数"""
    # GCPプロジェクトIDが設定されているか確認
    if not PROJECT_ID or PROJECT_ID == "your-gcp-project-id":
        print("Error: Please set your GCP PROJECT_ID in the script.")
        return
    
    # メール用のキューを作成
    email_queue = asyncio.Queue()
    
    # メール処理タスクを開始
    process_task = asyncio.create_task(process_emails_from_queue(email_queue))
    
    try:
        # Gmail監視を開始
        await start_listening(email_queue)
    except KeyboardInterrupt:
        print("Shutting down...")
        process_task.cancel()
        try:
            await process_task
        except asyncio.CancelledError:
            pass

# 他のプロセスから使用する場合の例:
# 
# import asyncio
# from test_gmail import start_listening
# 
# async def your_main_function():
#     email_queue = asyncio.Queue()
#     
#     # Gmail監視を並行して開始
#     gmail_task = asyncio.create_task(start_listening(email_queue))
#     
#     # メール処理ループ
#     while True:
#         email_info = await email_queue.get()
#         # メール内容を使った処理
#         await process_email(email_info)
#         email_queue.task_done()

if __name__ == "__main__":
    asyncio.run(main())

