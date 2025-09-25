# mailAgent.py
import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import List, Dict

from listen_gmail import start_listening  # 既存ファイル
import agent  # あなたの LangGraph 実装ファイル（run_agent_once を追記済み）

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

        except Exception as e:
            print(f"[mailAgent] ERROR while running agent: {e}")
        finally:
            email_queue.task_done()


async def main():
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
