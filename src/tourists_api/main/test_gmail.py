import os.path
import base64
import json
import time
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.cloud import pubsub_v1

# --- 設定項目 ---
# Gmail APIのスコープ (変更不要)
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly", "https://www.googleapis.com/auth/pubsub"]
# GCPプロジェクトID (自分のプロジェクトIDに書き換えてください)
PROJECT_ID = "fair-sandbox-388708"  # ← ここを書き換える！
# Pub/SubのトピックID (GCPで設定したもの)
PUB_SUB_TOPIC = "gmail-push-notification"
# Pub/SubのサブスクリプションID (GCPで設定したもの)
PUB_SUB_SUBSCRIPTION = "gmail-push-subscription"
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

def process_notification(message: pubsub_v1.subscriber.message.Message, creds, start_history_id_ref) -> None:
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

            headers = msg["payload"]["headers"]
            sender = next((h["value"] for h in headers if h["name"] == "From"), "Unknown Sender")
            subject = next((h["value"] for h in headers if h["name"] == "Subject"), "No Subject")

            print("\n" + "=" * 30)
            print(f"NEW EMAIL from: {sender}")
            print(f"Subject: {subject}")

            body_data = get_email_body(msg)
            if body_data:
                body = base64.urlsafe_b64decode(body_data).decode("utf-8", errors="ignore")
                print("Body:")
                print(body[:500] + "..." if len(body) > 500 else body)
            else:
                print("Body: Not found.")
            print("=" * 30)

    except HttpError as error:
        print(f"An error occurred while fetching email details: {error}")


def main():
    """Gmail APIの認証と監視を開始する"""
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json", SCOPES
            )
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(creds.to_json())

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

        # コールバック関数にhistoryIdへの参照を渡す
        callback = lambda message: process_notification(message, creds, start_history_id_ref)
        
        streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback)
        print(f"Listening for messages on {subscription_path}...")

        # プログラムが終了しないように待機
        try:
            streaming_pull_future.result()
        except TimeoutError:
            streaming_pull_future.cancel()
            streaming_pull_future.result()
        except KeyboardInterrupt:
            streaming_pull_future.cancel()
            streaming_pull_future.result()

    except HttpError as error:
        print(f"An error occurred: {error}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    # GCPプロジェクトIDが設定されているか確認
    if PROJECT_ID == "your-gcp-project-id":
        print("Error: Please set your GCP PROJECT_ID in the script.")
    else:
        main()

