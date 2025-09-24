from __future__ import annotations
from typing import List, Optional, TypedDict
from datetime import datetime, timezone
import hashlib

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from functools import wraps

def traced(name: str):
    def deco(fn):
        @wraps(fn)
        def wrapper(state: GraphState) -> GraphState:
            now = datetime.now().strftime("%H:%M:%S")
            print(f"[{now}] ▶ START {name}")
            out = fn(state)
            now2 = datetime.now().strftime("%H:%M:%S")
            # 主要メトリクスがあれば併せて表示
            if name == "update_budget":
                print(f"    spent -> {out.get('month_spent_jpy_updated')}")
            if name == "judge":
                print(f"    pace_ratio={out.get('pace_ratio'):.3f} should_scold={out.get('should_scold')}")
            print(f"[{now2}] ◀ END   {name}\n")
            return out
        return wrapper
    return deco


# ========= 1) State =========
class Purchase(BaseModel):
    vendor: Optional[str] = Field(None, description="Merchant or platform, e.g., Amazon, Rakuten")
    item_name: str = Field(..., description="Purchased product name")
    price: float = Field(..., description="Total paid price as number")
    currency: str = Field(..., description="Currency code like JPY, USD")

class Purchases(BaseModel):
    purchases: List[Purchase]

class GraphState(TypedDict, total=False):
    # inputs
    raw_emails: List[str]                  # data1: Gmail-like raw text
    user_goals: dict                       # data2: goals {purpose, target_amount, by_yyyy_mm}
    month_plan_jpy: float                  # data2: 月の支出計画(円)
    month_spent_jpy: float                 # data2: 現在までの実支出(円)
    month_id: str                          # "YYYY-MM", 冪等性管理に使う
    
    # internals
    extracted: List[Purchase]
    dedup_hashes: set
    month_spent_jpy_updated: float
    pace_ratio: float                      # 実支出 / 線形想定支出
    should_scold: bool
    message: str

# ========= 2) Models =========
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 構造化出力（商品名・価格・通貨）  ─ LangChainの with_structured_output が推奨。:contentReference[oaicite:2]{index=2}
extractor = llm.with_structured_output(Purchases)

# ========= 3) Nodes =========
@traced("extract")
def node_extract(state: GraphState) -> GraphState:
    """(1) LLMで購入品を構造化抽出"""
    extracted: List[Purchase] = []
    for raw in state["raw_emails"]:
        res = extractor.invoke(
            f"""From the following text, extract all purchases. 
Return product name, price (number), currency code, and, if you can, vendor.
Text:
{raw}
""")
        extracted.extend(res.purchases)
    state["extracted"] = extracted
    # 初期化
    state["dedup_hashes"] = set()
    return state

@traced("update_budget")
def node_update_budget(state: GraphState) -> GraphState:
    """(2) Python関数で実支出を月次に加算（冪等性・重複排除）"""
    spent = state.get("month_spent_jpy", 0.0)
    seen = state["dedup_hashes"]
    for p in state.get("extracted", []):
        # メール本文×商品名×金額×通貨×月でハッシュ → 同一月の二重計上を排除
        key = f'{state["month_id"]}:{p.item_name}:{p.price}:{p.currency}'
        h = hashlib.sha256(key.encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        # 通貨は今回はJPYのみ加算（実運用では為替レート変換を挟む）
        if p.currency.upper() == "JPY":
            spent += float(p.price)
    state["month_spent_jpy_updated"] = spent
    return state

def _month_linear_quota(month_plan_jpy: float, month_id: str) -> float:
    """当月線形想定支出（今日時点）"""
    year, month = map(int, month_id.split("-"))
    # 月の日数をざっくり（28〜31を動的に）:
    import calendar
    days_in_month = calendar.monthrange(year, month)[1]
    # 今日が同じ月でない可能性があるデモ環境のため、月末日を越えないよう min
    today = datetime.now(timezone.utc)
    # 同月なら日数ベース、異なる月なら最終日扱い
    day = min(days_in_month, today.day if today.month == month and today.year == year else days_in_month)
    return month_plan_jpy * (day / days_in_month)

@traced("judge")
def node_should_scold(state: GraphState) -> GraphState:
    """(3) 線形ペース超過で叱る。ヒステリシス3%で安定化"""
    plan = state["month_plan_jpy"]
    spent = state["month_spent_jpy_updated"]
    quota = _month_linear_quota(plan, state["month_id"])
    ratio = (spent / quota) if quota > 0 else 1.0
    state["pace_ratio"] = ratio
    # 3%しきい値（チラつき防止）
    state["should_scold"] = ratio > 1.03
    return state

# def node_scold_message(state: GraphState) -> GraphState:
#     """(4-1) 叱るメッセージをLLMで生成"""
#     purchases = "\n".join([f"- {p.item_name}: {int(p.price)} {p.currency}" for p in state.get("extracted", [])])
#     prompt = ChatPromptTemplate.from_messages([
#         ("system",
#          "You are a friendly but strict financial coach. Be supportive but direct. "
#          "Give 3 concrete actions to get back on track. Keep it within 200 Japanese characters per section."),
#         ("human",
#          """ユーザー情報:
# - 今月の支出計画: {plan} 円
# - 現在の実支出: {spent} 円
# - 線形ペース比: {ratio:.2f}x
# - 目標: {goal}

# 今月の最近の購入:
# {purchases}

# これらを踏まえ、どの商品と金額が効いているかを指摘しつつ、やや厳しめに、しかしモチベが上がる文面で日本語出力してください。""")
#     ])
#     msg = llm.invoke(prompt.format_messages(
#         plan=int(state["month_plan_jpy"]),
#         spent=int(state["month_spent_jpy_updated"]),
#         ratio=state["pace_ratio"],
#         goal=state.get("user_goals"),
#         purchases=purchases or "（抽出なし）",
#     ))
#     state["message"] = msg.content
#     return state

@traced("scold_msg")
def node_scold_message(state: GraphState) -> GraphState:
    """(4-1) 厳しく、理詰めで叱るメッセージをLLMで生成"""
    # ----- 数値前処理（叱責に使う定量値） -----
    plan = float(state["month_plan_jpy"])
    spent = float(state["month_spent_jpy_updated"])
    month_id = state["month_id"]

    # 当月の線形想定支出（今日時点）
    quota = _month_linear_quota(plan, month_id)

    # 進捗率（今日までの経過日数）
    from datetime import datetime, timezone
    import calendar
    y, m = map(int, month_id.split("-"))
    days_in_month = calendar.monthrange(y, m)[1]
    today = datetime.now(timezone.utc)
    day = min(days_in_month, today.day if (today.year == y and today.month == m) else days_in_month)
    frac = max(1e-9, day / days_in_month)  # 0割回避

    # 月末時点の単純予測（このペースを継続した場合）
    projected_eom = spent / frac

    # 逸脱量（現在 / 月末予測）
    over_abs_now = max(0.0, spent - quota)
    over_pct_now = (spent / quota - 1.0) * 100.0 if quota > 0 else 0.0
    over_abs_eom = max(0.0, projected_eom - plan)
    over_pct_eom = (projected_eom / plan - 1.0) * 100.0 if plan > 0 else 0.0

    # 高額購入TOP3（価格降順）
    extracted = state.get("extracted", [])
    top3 = sorted(extracted, key=lambda p: float(p.price), reverse=True)[:3]
    top_lines = "\n".join([f"- {p.item_name}: {int(p.price)} {p.currency}" for p in top3]) or "（高額購入は抽出なし）"

    # ----- プロンプト -----
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         """あなたは節約太郎です。目標のためにユーザーを厳しく叱るファイナンシャルコーチです。
         ユーザーを後悔させるような表現を用いてください。
         必ず俺だったらこうする、と自分の意見を入れてください。"""),
        ("human",
         """【警告】予算逸脱
事実:
- 予算（今月）: {plan:,} 円
- 現在の実支出: {spent:,} 円
- 今日までの線形想定: {quota:,} 円
- 乖離: +{over_abs_now:,} 円（+{over_pct_now:.1f}%）
- このペース継続なら月末予測: {proj:,} 円 → 予算比 +{over_abs_eom:,} 円（+{over_pct_eom:.1f}%）
- 線形ペース比: {ratio:.2f}x

高額購入TOP3:
{top3}

方針:
1) 何が原因かを明確化した上で叱ること（上記TOP3に言及）。
2) できる限りユーザーに無駄な支出をしたことを後悔させるような表現を用いること。

ユーザーの目標:
{goal}

これらを踏まえ、日本語でユーザーを叱ってください""")
    ])

    msg = llm.invoke(prompt.format_messages(
        plan=int(plan),
        spent=int(spent),
        quota=int(quota),
        over_abs_now=int(over_abs_now),
        over_pct_now=over_pct_now,
        proj=int(projected_eom),
        over_abs_eom=int(over_abs_eom),
        over_pct_eom=over_pct_eom,
        ratio=state.get("pace_ratio", (spent / max(1.0, quota))),
        top3=top_lines,
        goal=state.get("user_goals"),
    ))
    state["message"] = msg.content
    return state

@traced("enc_msg")
def node_encourage_message(state: GraphState) -> GraphState:
    """(4-2) 褒める・継続を促すメッセージ"""
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a motivational financial coach. Encourage and give 2 small tips to keep momentum. "
         "Output in Japanese, warm tone, max ~250 characters."),
        ("human",
         """今月の支出計画 {plan} 円に対して順調（線形以下）。目標: {goal}。
短い応援メッセージを。""")
    ])
    msg = llm.invoke(prompt.format_messages(
        plan=int(state["month_plan_jpy"]),
        goal=state.get("user_goals"),
    ))
    state["message"] = msg.content
    return state

def route_scold(state: GraphState) -> str:
    route = "SCOLD" if state.get("should_scold") else "ENCOURAGE"
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] ➜ ROUTE judge -> {route}")
    return route


# ========= 4) Graph =========
graph = StateGraph(GraphState)
graph.add_node("extract", node_extract)
graph.add_node("update_budget", node_update_budget)
graph.add_node("judge", node_should_scold)
graph.add_node("scold_msg", node_scold_message)
graph.add_node("enc_msg", node_encourage_message)

graph.add_edge(START, "extract")
graph.add_edge("extract", "update_budget")
graph.add_edge("update_budget", "judge")

# 条件分岐（最新のGraph APIでのやり方）:contentReference[oaicite:3]{index=3}
graph.add_conditional_edges(
    "judge",
    route_scold,
    {
        "SCOLD": "scold_msg",
        "ENCOURAGE": "enc_msg",
    },
)
graph.add_edge("scold_msg", END)
graph.add_edge("enc_msg", END)

app = graph.compile(checkpointer=InMemorySaver())  # 簡易に永続化（スレッド再開等）。:contentReference[oaicite:4]{index=4}

# ========= 5) 例：実行 =========
if __name__ == "__main__":
    init: GraphState = {
        "raw_emails": [
            "ご購入ありがとうございます。Amazon.co.jpでご注文の商品『Echo Pop スマートスピーカー』合計金額 ¥5,480（税込）。",
            "【楽天市場】ご注文内容の確認：商品名：ワイヤレスイヤホン 税込 63,980円 送料0円 合計63,980円"
        ],
        "user_goals": {"purpose": "来春の台湾旅行", "target_amount": 200000, "by": "2026-03"},
        "month_plan_jpy": 50000,
        "month_spent_jpy": 12000,
        "month_id": "2025-09",
    }
    result = app.invoke(init, config={"configurable": {"thread_id": "demo-thread-1"}}) # result["message"] に最終メッセージ

    print("=== FINAL ===")
    print("should_scold:", result["should_scold"])
    print("pace_ratio :", f"{result['pace_ratio']:.3f}")
    print(result["message"])
