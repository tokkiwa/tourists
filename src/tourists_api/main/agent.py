from __future__ import annotations
from typing import List, Optional, TypedDict, Dict
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import math
import calendar
from collections import defaultdict

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from functools import wraps

# ========= 定数（ポリシー） =========
JST = ZoneInfo("Asia/Tokyo")

DEFAULT_BASELINE_SPEND_JPY = 200_000     # 履歴が無いときの平常月の仮置き
MIN_BUDGET_JPY             = 30_000      # 生活下限
SAFETY_BUFFER_JPY          = 5_000       # 予備費（保守バッファ）
CAP_MIN_RATIO              = 0.5         # 直近平均の50%未満に下げない
CAP_MAX_RATIO              = 1.2         # 直近平均の120%を超えない
BASELINE_LOOKBACK_MONTHS   = 3           # 直近Nヶ月平均

# ========= 共通ユーティリティ =========
def traced(name: str):
    def deco(fn):
        @wraps(fn)
        def wrapper(state: GraphState) -> GraphState:
            now = datetime.now(JST).strftime("%H:%M:%S")
            print(f"[{now}] ▶ START {name}")
            out = fn(state)
            now2 = datetime.now(JST).strftime("%H:%M:%S")
            if name == "update_budget":
                print(f"    spent -> {out.get('month_spent_jpy_updated')}")
            if name == "judge":
                print(f"    pace_ratio={out.get('pace_ratio'):.3f} should_scold={out.get('should_scold')}")
            if name == "decide_month_plan":
                print(f"    month_plan_jpy -> {out.get('month_plan_jpy')}")
            if name == "estimate_monthly_saving_need":
                print(f"    required_monthly_saving_jpy -> {out.get('required_monthly_saving_jpy')}")
            if name == "estimate_baseline_spend":
                print(f"    baseline_spend_jpy -> {out.get('baseline_spend_jpy')}")
            print(f"[{now2}] ◀ END   {name}\n")
            return out
        return wrapper
    return deco

def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def _safe_float(x) -> Optional[float]:
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None

def months_between_jst(today: datetime, by_ym: str) -> int:
    """today(JST)からYYYY-MM締切までの月数。最低1。"""
    y, m = map(int, by_ym.split("-"))
    months = (y - today.year) * 12 + (m - today.month)
    return max(1, months)

def ym_from_ymd(ymd: str) -> str:
    """YYYY-MM-DD -> YYYY-MM（壊れた文字列は当月）"""
    try:
        dt = datetime.strptime(ymd, "%Y-%m-%d")
        return dt.strftime("%Y-%m")
    except Exception:
        return datetime.now(JST).strftime("%Y-%m")

# ========= 1) State =========
class Purchase(BaseModel):
    vendor: Optional[str] = Field(None, description="Merchant or platform, e.g., Amazon, Rakuten")
    item_name: str = Field(..., description="Purchased product name")
    price: float = Field(..., description="Total paid price as number")
    currency: str = Field(..., description="Currency code like JPY, USD")
    raw_hash: Optional[str] = None
    ym: Optional[str] = None  # この購入の計上対象年月（YYYY-MM）

class Purchases(BaseModel):
    purchases: List[Purchase]

class GraphState(TypedDict, total=False):
    # inputs
    # 「日付つきメール」推奨。なければ raw_emails を使って今月扱い。
    dated_emails: List[Dict]     # {"text": str, "ts_ymd": "YYYY-MM-DD"}
    raw_emails: List[str]        # 互換：古い形式
    user_goals: List[dict]       # 複数の目標 [{"purpose":..., "target_amount":..., "by":"YYYY-MM"}, ...]

    # inferred inputs (auto)
    month_id: str                # "YYYY-MM"（当月、自動決定）

    # internals
    extracted: List[Purchase]
    dedup_hashes: set
    month_spent_jpy_updated: float
    baseline_spend_jpy: float
    required_monthly_saving_jpy: float
    month_plan_jpy: float
    pace_ratio: float
    should_scold: bool
    message: str

# ========= 2) Models =========
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
extractor = llm.with_structured_output(Purchases)

# ========= 抽出補助 =========
def _normalize_purchase_with_date(p: Purchase, raw_hash: str, ym: str) -> Optional[Purchase]:
    price = _safe_float(p.price)
    if price is None:
        return None
    item = (p.item_name or "").strip()
    if not item:
        return None
    currency = (p.currency or "").strip().upper()
    if not currency:
        return None
    vendor = (p.vendor or "").strip() or None
    return Purchase(vendor=vendor, item_name=item, price=price, currency=currency, raw_hash=raw_hash, ym=ym)

# ========= 3) Nodes =========

@traced("init_month")
def node_init_month(state: GraphState) -> GraphState:
    """当月YYYY-MMをセット（JST）"""
    state["month_id"] = datetime.now(JST).strftime("%Y-%m")
    return state

@traced("extract")
def node_extract(state: GraphState) -> GraphState:
    """日付つき/なしメールから購入抽出 + ym付与 + 原文ハッシュ付与（例外安全）"""
    extracted: List[Purchase] = []

    # 1) dated_emails 優先
    for rec in state.get("dated_emails", []) or []:
        raw = rec.get("text", "")
        ymd = rec.get("ts_ymd", "")
        ym = ym_from_ymd(ymd)
        raw_hash = _sha256(f"{ym}:{raw}")
        try:
            res = extractor.invoke(
                f"""From the following text, extract all purchases. 
Return product name, price (number), currency code, and, if you can, vendor.
Text:
{raw}
""")
            found = 0
            for p in getattr(res, "purchases", []) or []:
                norm = _normalize_purchase_with_date(p, raw_hash, ym)
                if norm is not None:
                    extracted.append(norm)
                    found += 1
            if found == 0:
                print(f"[extract] INFO: no valid purchases (ym={ym}, raw={raw_hash[:8]}...)")
        except Exception as e:
            print(f"[extract] WARN: extractor failed (ym={ym}, raw={raw_hash[:8]}...): {e}")

    # 2) 後方互換: raw_emails は「当月」扱い
    default_ym = state["month_id"]
    for raw in state.get("raw_emails", []) or []:
        raw_hash = _sha256(f"{default_ym}:{raw}")
        try:
            res = extractor.invoke(
                f"""From the following text, extract all purchases. 
Return product name, price (number), currency code, and, if you can, vendor.
Text:
{raw}
""")
            found = 0
            for p in getattr(res, "purchases", []) or []:
                norm = _normalize_purchase_with_date(p, raw_hash, default_ym)
                if norm is not None:
                    extracted.append(norm)
                    found += 1
            if found == 0:
                print(f"[extract] INFO: no valid purchases (ym={default_ym}, raw={raw_hash[:8]}...)")
        except Exception as e:
            print(f"[extract] WARN: extractor failed (ym={default_ym}, raw={raw_hash[:8]}...): {e}")

    state["extracted"] = extracted
    state["dedup_hashes"] = state.get("dedup_hashes", set())
    return state

def _month_linear_quota(month_plan_jpy: float, month_id: str) -> float:
    year, month = map(int, month_id.split("-"))
    days_in_month = calendar.monthrange(year, month)[1]
    today = datetime.now(JST)
    day = min(days_in_month, today.day if today.month == month and today.year == year else days_in_month)
    return month_plan_jpy * (day / days_in_month)

@traced("estimate_monthly_saving_need")
def node_estimate_monthly_saving_need(state: GraphState) -> GraphState:
    """複数ゴールから「今月必要な貯蓄額」を推定"""
    today = datetime.now(JST)
    total_need = 0.0
    for g in state.get("user_goals", []) or []:
        target = _safe_float(g.get("target_amount"))
        by = g.get("by")
        if target is None or not by:
            continue
        months = months_between_jst(today, by)
        total_need += target / months
    state["required_monthly_saving_jpy"] = float(total_need)
    return state

def sum_monthly_totals(extracted: List[Purchase]) -> Dict[str, float]:
    """YYYY-MM毎のJPY合計を返す（他通貨は除外）"""
    monthly = defaultdict(float)
    for p in extracted or []:
        if (p.currency or "").upper() != "JPY":
            continue
        ym = p.ym or ""
        if ym:
            monthly[ym] += float(p.price)
    return dict(monthly)

def estimate_baseline_from_history(month_totals: Dict[str, float], current_ym: str) -> float:
    """current_ym から遡る BASELINE_LOOKBACK_MONTHS ヶ月の平均。足りなければ既定値。"""
    y, m = map(int, current_ym.split("-"))
    yms = []
    yy, mm = y, m
    for _ in range(BASELINE_LOOKBACK_MONTHS):
        yms.append(f"{yy:04d}-{mm:02d}")
        mm -= 1
        if mm == 0:
            mm = 12
            yy -= 1
    vals = [month_totals.get(x, None) for x in yms]
    vals = [v for v in vals if v is not None]
    if len(vals) >= max(1, BASELINE_LOOKBACK_MONTHS // 2):  # 半分以上あれば平均採用
        return float(sum(vals) / len(vals))
    return float(DEFAULT_BASELINE_SPEND_JPY)

@traced("estimate_baseline_spend")
def node_estimate_baseline_spend(state: GraphState) -> GraphState:
    """履歴から平常月の支出ベースラインを推定（なければ既定値）"""
    month_totals = sum_monthly_totals(state.get("extracted", []))
    baseline = estimate_baseline_from_history(month_totals, state["month_id"])
    state["baseline_spend_jpy"] = baseline
    return state

@traced("decide_month_plan")
def node_decide_month_plan(state: GraphState) -> GraphState:
    """baselineと必要貯蓄から当月予算を自動決定（下限・上限キャップ）"""
    baseline = float(state.get("baseline_spend_jpy", DEFAULT_BASELINE_SPEND_JPY))
    need = float(state.get("required_monthly_saving_jpy", 0.0))

    raw_plan = baseline - need - SAFETY_BUFFER_JPY
    # 生活下限
    raw_plan = max(MIN_BUDGET_JPY, raw_plan)

    # 直近平均に対するキャップ
    capped_min = baseline * CAP_MIN_RATIO
    capped_max = baseline * CAP_MAX_RATIO
    month_plan = min(max(raw_plan, capped_min), capped_max)

    state["month_plan_jpy"] = float(month_plan)
    return state

@traced("update_budget")
def node_update_budget(state: GraphState) -> GraphState:
    """当月のJPY支出を加算（原文ハッシュ込み重複排除）"""
    spent = 0.0  # 自動推定のため「今月の累計」を抽出から再算出する
    seen = state.get("dedup_hashes", set())
    month_id = state["month_id"]

    for p in state.get("extracted", []) or []:
        if p.ym != month_id:
            continue  # 当月のみ計上
        raw_hash = p.raw_hash or "nohash"
        vendor = p.vendor or "unknown"
        key = f"{month_id}:{raw_hash}:{vendor}:{p.item_name}:{p.price}:{p.currency}"
        h = hashlib.sha256(key.encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        if (p.currency or "").upper() == "JPY":
            spent += float(p.price)

    state["dedup_hashes"] = seen
    state["month_spent_jpy_updated"] = spent
    return state

@traced("judge")
def node_should_scold(state: GraphState) -> GraphState:
    """線形ペース超過で叱る（ヒステリシス3%）"""
    plan = float(state["month_plan_jpy"])
    spent = float(state.get("month_spent_jpy_updated", 0.0))
    quota = _month_linear_quota(plan, state["month_id"])
    ratio = (spent / quota) if quota > 0 else 1.0
    state["pace_ratio"] = ratio
    state["should_scold"] = ratio > 1.03
    return state

@traced("scold_msg")
def node_scold_message(state: GraphState) -> GraphState:
    """厳しく、理詰めで叱るメッセージ"""
    plan = float(state["month_plan_jpy"])
    spent = float(state["month_spent_jpy_updated"])
    month_id = state["month_id"]

    quota = _month_linear_quota(plan, month_id)
    y, m = map(int, month_id.split("-"))
    days_in_month = calendar.monthrange(y, m)[1]
    today = datetime.now(JST)
    day = min(days_in_month, today.day if (today.year == y and today.month == m) else days_in_month)
    frac = max(1e-9, day / days_in_month)
    projected_eom = spent / frac

    over_abs_now = max(0.0, spent - quota)
    over_pct_now = (spent / quota - 1.0) * 100.0 if quota > 0 else 0.0
    over_abs_eom = max(0.0, projected_eom - plan)
    over_pct_eom = (projected_eom / plan - 1.0) * 100.0 if plan > 0 else 0.0

    extracted = [p for p in state.get("extracted", []) or [] if p.ym == month_id]
    top3 = sorted(extracted, key=lambda p: float(p.price), reverse=True)[:3]
    top_lines = "\n".join([f"- {p.item_name}: {int(p.price)} {p.currency}" for p in top3]) or "（高額購入は抽出なし）"

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         """あなたは節約太郎です。皮肉屋で目標のためにユーザーを厳しく叱るファイナンシャルコーチです。
         ユーザーを後悔させるような表現を用いてください。
         例：「ワイヤレスイヤホンに63,980円も使ったのはどういうつもりだ？そんな高額なものを買う余裕があるなら、旅行資金に回せたはずだろうが。Echo Pop スマートスピーカーも5480円だって？そのお金があれば、台湾での美味しい食事や観光に使えたかもしれないのに、何を考えているんだ？」
         必ず俺だったらこうする、と自分の意見を入れてください。"""),
        ("human",
         """【警告】予算逸脱
設定予算（自動推定）:
- 当月予算: {plan:,} 円（平常月×キャップ・必要貯蓄控除適用）

事実:
- 現在の実支出: {spent:,} 円
- 今日までの線形想定: {quota:,} 円
- 乖離: +{over_abs_now:,} 円（+{over_pct_now:.1f}%）
- このペース継続なら月末予測: {proj:,} 円 → 予算比 +{over_abs_eom:,} 円（+{over_pct_eom:.1f}%）
- 線形ペース比: {ratio:.2f}x

高額購入TOP3:
{top3}

ユーザーの目標群（抜粋）:
{goals}

以上を踏まえ、日本語で厳しく叱責してください。""")
    ])

    goals_lines = []
    for g in state.get("user_goals", []) or []:
        goals_lines.append(f"- {g.get('purpose','?')}：{g.get('target_amount','?')}円 / 期限 {g.get('by','?')}")
    goals_text = "\n".join(goals_lines) or "（目標未設定）"

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
        goals=goals_text,
    ))
    state["message"] = msg.content
    return state

@traced("enc_msg")
def node_encourage_message(state: GraphState) -> GraphState:
    """順調時の短い応援メッセージ"""
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a motivational financial coach. Encourage and give 2 small tips to keep momentum. "
         "Output in Japanese, warm tone, max ~250 characters."),
        ("human",
         """当月予算（自動推定）{plan} 円に対して順調（線形以下）。ユーザーの目標群: 
{goals}
短い応援メッセージを。""")
    ])

    goals_lines = []
    for g in state.get("user_goals", []) or []:
        goals_lines.append(f"- {g.get('purpose','?')}：{g.get('target_amount','?')}円 / 期限 {g.get('by','?')}")
    goals_text = "\n".join(goals_lines) or "（目標未設定）"

    msg = llm.invoke(prompt.format_messages(
        plan=int(state["month_plan_jpy"]),
        goals=goals_text,
    ))
    state["message"] = msg.content
    return state

def route_scold(state: GraphState) -> str:
    route = "SCOLD" if state.get("should_scold") else "ENCOURAGE"
    now = datetime.now(JST).strftime("%H:%M:%S")
    print(f"[{now}] ➜ ROUTE judge -> {route}")
    return route

# ========= 4) Graph =========
graph = StateGraph(GraphState)
graph.add_node("init_month", node_init_month)
graph.add_node("extract", node_extract)
graph.add_node("estimate_monthly_saving_need", node_estimate_monthly_saving_need)
graph.add_node("estimate_baseline_spend", node_estimate_baseline_spend)
graph.add_node("decide_month_plan", node_decide_month_plan)
graph.add_node("update_budget", node_update_budget)
graph.add_node("judge", node_should_scold)
graph.add_node("scold_msg", node_scold_message)
graph.add_node("enc_msg", node_encourage_message)

graph.add_edge(START, "init_month")
graph.add_edge("init_month", "extract")
graph.add_edge("extract", "estimate_monthly_saving_need")
graph.add_edge("estimate_monthly_saving_need", "estimate_baseline_spend")
graph.add_edge("estimate_baseline_spend", "decide_month_plan")
graph.add_edge("decide_month_plan", "update_budget")
graph.add_edge("update_budget", "judge")

graph.add_conditional_edges(
    "judge",
    route_scold,
    {"SCOLD": "scold_msg", "ENCOURAGE": "enc_msg"},
)
graph.add_edge("scold_msg", END)
graph.add_edge("enc_msg", END)

app = graph.compile(checkpointer=InMemorySaver())

# ========= 5) 例：実行 =========
if __name__ == "__main__":
    init: GraphState = {
        "dated_emails": [
            # 8月の購入（履歴学習用）
            {"text": "【Amazon】メカニカルキーボード 13,980円（税込）", "ts_ymd": "2025-08-20"},
            {"text": "【楽天】ドライヤー 7,680円（税込）", "ts_ymd": "2025-08-05"},
            # 9月（当月）の購入（評価対象）
            {"text": "ご購入ありがとうございます。Amazon.co.jp『Echo Pop スマートスピーカー』合計金額 ¥5,480（税込）。", "ts_ymd": "2025-09-02"},
            {"text": "【楽天市場】ご注文内容：ワイヤレスイヤホン 税込 63,980円 送料0円 合計63,980円", "ts_ymd": "2025-09-10"},
            # 壊れた例（数値でない）→ スキップ
            {"text": "おはようございます", "ts_ymd": "2025-09-15"},
        ],
        "user_goals": [
            {"purpose": "来春の台湾旅行", "target_amount": 200000, "by": "2026-03"},
            {"purpose": "ノートパソコンの買い替え費用", "target_amount": 300000, "by": "2026-09"},
        ],
        # raw_emails は省略可（dated_emailsを推奨）
    }

    result = app.invoke(init, config={"configurable": {"thread_id": "demo-thread-2"}})

    print("=== FINAL ===")
    print("month_id         :", result["month_id"])
    print("baseline_spend   :", int(result["baseline_spend_jpy"]))
    print("req_save/month   :", int(result["required_monthly_saving_jpy"]))
    print("auto month_plan  :", int(result["month_plan_jpy"]))
    print("spent (this month):", int(result["month_spent_jpy_updated"]))
    print("pace_ratio       :", f"{result['pace_ratio']:.3f}")
    print("should_scold     :", result["should_scold"])
    print(result["message"])
