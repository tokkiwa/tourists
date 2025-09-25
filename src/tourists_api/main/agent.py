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

    # 返品関連
    return_candidates: List[Purchase]   # Amazon等、返品検討対象
    should_open_orders: bool           # 返品ページを開くべきか
    open_url: str                      # 既定: 'https://www.amazon.co.jp/gp/css/order-history'
    
    # 叱り方決定関連
    user_context: Dict                 # DBから取得したユーザー情報
    scolding_strategy: str             # 叱り方の戦略

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
    """戦略に基づいて個別化された叱りメッセージを生成"""
    plan = float(state["month_plan_jpy"])
    spent = float(state["month_spent_jpy_updated"])
    month_id = state["month_id"]
    strategy = state.get("scolding_strategy", "general_financial")
    user_context = state.get("user_context", {})

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

    # 戦略別のシステムプロンプト
    strategy_prompts = {
        "family_focused": """あなたは家族思いの節約コーチです。家族や子供への影響を重視した叱り方をしてください。
例：「子供たちの将来のために貯蓄すべきお金を、こんなものに使ってしまって大丈夫ですか？子供たちが大学に行きたいと言った時に、『お父さん（お母さん）がワイヤレスイヤホンを買ったから学費がない』と言うつもりですか？」""",
        
        "career_focused": """あなたはキャリア重視の節約コーチです。仕事や職業上の成長への影響を強調した叱り方をしてください。
例：「エンジニアとして本当に必要な投資を見極められないのですか？このお金があれば、もっとスキルアップに使えたはずです。」""",
        
        "goal_focused": """あなたは目標達成重視の節約コーチです。設定した目標への影響を強調した叱り方をしてください。
例：「台湾旅行の夢は諦めるということですか？このペースでは永遠に行けませんよ。」""",
        
        "policy_violation": """あなたは一貫性重視の節約コーチです。ユーザー自身が決めたポリシー違反を厳しく指摘してください。
例：「あなた自身が『趣味は月5万円まで』と決めたんじゃないですか？自分で決めたルールも守れないのですか？」""",
        
        "pattern_focused": """あなたは行動分析重視の節約コーチです。支出パターンの問題を冷静に分析して指摘してください。
例：「直近のデータを見ると、電子機器への支出が異常に多いですね。この傾向が続けば確実に破綻します。」""",
        
        "general_financial": """あなたは節約太郎です。皮肉屋で目標のためにユーザーを厳しく叱るファイナンシャルコーチです。
ユーザーを後悔させるような表現を用いてください。"""
    }

    system_prompt = strategy_prompts.get(strategy, strategy_prompts["general_financial"])
    system_prompt += """あなたは節約太郎です。皮肉屋で目標のためにユーザーを厳しく叱るファイナンシャルコーチです。 
    ユーザーを後悔させるような表現を用いてください。 
    例：「ワイヤレスイヤホンに63,980円も使ったのはどういうつもりだ？
    そんな高額なものを買う余裕があるなら、旅行資金に回せたはずだろうが。
    Echo Pop スマートスピーカーも5480円だって？そのお金があれば、
    台湾での美味しい食事や観光に使えたかもしれないのに、何を考えているんだ？」 必ず俺だったらこうする、と自分の意見を入れてください。"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", """【警告】予算逸脱

ユーザー情報:
- 家族構成: {family_structure}
- 子供の数: {number_of_children}人
- 職業: {occupation}
- 支出ポリシー: {user_policies}

設定予算（自動推定）:
- 当月予算: {plan:,} 円

事実:
- 現在の実支出: {spent:,} 円
- 今日までの線形想定: {quota:,} 円
- 乖離: +{over_abs_now:,} 円（+{over_pct_now:.1f}%）
- このペース継続なら月末予測: {proj:,} 円 → 予算比 +{over_abs_eom:,} 円（+{over_pct_eom:.1f}%）

高額購入TOP3:
{top3}

ユーザーの目標:
{goals}

選択された叱り戦略: {strategy}

以上の個人情報と戦略に基づき、最も効果的な叱責メッセージを日本語で生成してください。""")
    ])

    goals_lines = []
    for g in state.get("user_goals", []) or []:
        goals_lines.append(f"- {g.get('purpose','?')}：{g.get('target_amount','?')}円 / 期限 {g.get('by','?')}")
    goals_text = "\n".join(goals_lines) or "（目標未設定）"

    policies_text = " / ".join(user_context.get("user_policies", ["設定なし"]))

    msg = llm.invoke(prompt.format_messages(
        family_structure=user_context.get("family_structure", "unknown"),
        number_of_children=user_context.get("number_of_children", 0),
        occupation=user_context.get("occupation", "unknown"),
        user_policies=policies_text,
        plan=int(plan),
        spent=int(spent),
        quota=int(quota),
        over_abs_now=int(over_abs_now),
        over_pct_now=over_pct_now,
        proj=int(projected_eom),
        over_abs_eom=int(over_abs_eom),
        over_pct_eom=over_pct_eom,
        top3=top_lines,
        goals=goals_text,
        strategy=strategy
    ))
    state["message"] = msg.content
    return state

# ========= ユーザー情報取得とScoldingStrategy決定 =========

@traced("fetch_user_context")
def node_fetch_user_context(state: GraphState) -> GraphState:
    """Supabaseからユーザー情報を取得"""
    try:
        from src.tourists_api.supabase_client import supabase
        from datetime import datetime
        
        # ユーザーIDを取得（実際の実装では認証情報から取得）
        # ここでは仮のユーザーIDを使用（必要に応じて state から取得）
        user_id = state.get('user_id', '00000000-0000-0000-0000-000000000000')
        
        # プロフィール情報を取得
        profile_response = supabase.table('profiles').select('*').eq('user_id', user_id).execute()
        
        # 長期目標を取得
        goals_response = supabase.table('long_term_plans').select('*').eq('user_id', user_id).execute()
        
        # ユーザーポリシーを取得
        policies_response = supabase.table('user_policies').select('*').eq('user_id', user_id).execute()
        
        # 子供情報を取得
        children_response = supabase.table('children').select('*').eq('user_id', user_id).execute()
        
        # 取引履歴を取得（最近1ヶ月）
        transactions_response = supabase.table('transactions').select('*').eq('user_id', user_id).order('transaction_date', desc=True).limit(20).execute()
        
        # プロフィール情報を処理
        user_context = {}
        if profile_response.data:
            profile = profile_response.data[0]
            user_context.update({
                'occupation': profile.get('occupation', 'unknown'),
                'family_structure': 'married_with_children' if profile.get('family_structure') == '既婚' and profile.get('number_of_children', 0) > 0 
                                 else 'married' if profile.get('family_structure') == '既婚'
                                 else 'single',
                'number_of_children': profile.get('number_of_children', 0),
                'birth_date': profile.get('birth_date', '1990-01-01'),
                'name': profile.get('name', 'ユーザー')
            })
        else:
            # デフォルト値
            user_context = {
                'occupation': 'software_engineer',
                'family_structure': 'single',
                'number_of_children': 0,
                'birth_date': '1990-01-01',
                'name': 'ユーザー'
            }
        
        # 長期目標を処理
        long_term_plans = []
        for goal in goals_response.data:
            long_term_plans.append({
                'plan_name': goal.get('plan_name', ''),
                'target_amount': goal.get('target_amount', 0),
                'target_date': goal.get('target_date', '')
            })
        
        if not long_term_plans:
            # デフォルトの目標
            long_term_plans = [
                {"plan_name": "緊急資金", "target_amount": 1000000, "target_date": "2025-12-31"},
                {"plan_name": "旅行資金", "target_amount": 500000, "target_date": "2026-06-30"}
            ]
        
        user_context['long_term_plans'] = long_term_plans
        
        # ユーザーポリシーを処理
        user_policies = []
        for policy in policies_response.data:
            if policy.get('policy_text'):
                user_policies.append(policy['policy_text'])
        
        if not user_policies:
            # デフォルトのポリシー
            user_policies = [
                "計画的な支出を心がける",
                "無駄な買い物は避ける",
                "健康と学習への投資は優先する"
            ]
        
        user_context['user_policies'] = user_policies
        
        # 取引履歴を処理
        transactions = []
        for transaction in transactions_response.data:
            transactions.append({
                'category': transaction.get('category', 'その他'),
                'transaction_date': transaction.get('transaction_date', ''),
                'store_name': transaction.get('store_name', ''),
                'amount': transaction.get('amount', 0)
            })
        
        user_context['transactions'] = transactions
        
        print(f"  fetched user context from Supabase: {len(user_context)} fields")
        print(f"  - goals: {len(long_term_plans)}, policies: {len(user_policies)}, transactions: {len(transactions)}")
        
    except Exception as e:
        print(f"Supabaseからの情報取得に失敗: {e}")
        # エラー時はデフォルト値を使用
        user_context = {
            "family_structure": "single",
            "number_of_children": 0,
            "occupation": "software_engineer",
            "birth_date": "1990-05-15",
            "long_term_plans": [
                {"plan_name": "緊急資金", "target_amount": 1000000, "target_date": "2025-12-31"},
                {"plan_name": "旅行資金", "target_amount": 500000, "target_date": "2026-06-30"}
            ],
            "user_policies": [
                "計画的な支出を心がける",
                "無駄な買い物は避ける", 
                "健康と学習への投資は優先する"
            ],
            "transactions": []
        }
        print(f"  using default user context: {len(user_context)} fields")
    
    state["user_context"] = user_context
    return state

@traced("decide_scolding_strategy")
def node_decide_scolding_strategy(state: GraphState) -> GraphState:
    """ユーザー情報と購入内容に基づいて叱り方を決める"""
    user_context = state.get("user_context", {})
    extracted = state.get("extracted", [])
    month_id = state.get("month_id", "")
    
    # 今月の購入のみフィルタ
    current_purchases = [p for p in extracted if p.ym == month_id]
    
    # LLMに叱り戦略を決定させる
    purchases_text = "\n".join([f"- {p.item_name}: {int(p.price)}円 ({p.vendor or 'unknown'})" 
                               for p in current_purchases])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         """あなたは個人に最適化された叱り方を決めるストラテジストです。
ユーザーの個人情報、家族構成、職業、目標、支出ポリシー、購入履歴を基に、
最も効果的な叱り方のアプローチを決定してください。

戦略オプション:
1. family_focused - 家族・子供への影響を強調
2. career_focused - 仕事・キャリアへの影響を強調
3. goal_focused - 設定した目標への影響を強調
4. policy_violation - ユーザー自身のポリシー違反を指摘
5. pattern_focused - 支出パターンの問題を指摘
6. general_financial - 一般的な家計管理の観点"""),
        
        ("human", """
ユーザー情報:
- 家族構成: {family_structure}
- 子供の数: {number_of_children}
- 職業: {occupation}
- 生年月日: {birth_date}

長期目標:
{long_term_plans}

支出ポリシー:
{user_policies}

直近の取引履歴:
{transactions}

今月の問題のある購入:
{purchases}

上記を踏まえ、最も効果的な叱り戦略を1つ選んで、理由とともにJSONで回答:
{{"strategy": "family_focused", "reason": "理由説明", "key_points": ["重要ポイント1", "重要ポイント2"]}}
""")
    ])
    
    # フォーマット用データ準備
    plans_text = "\n".join([f"- {p['plan_name']}: {p['target_amount']}円 ({p['target_date']})" 
                           for p in user_context.get("long_term_plans", [])])
    policies_text = "\n".join([f"- {p}" for p in user_context.get("user_policies", [])])
    transactions_text = "\n".join([f"- {t['category']}: {t['amount']}円 @ {t['store_name']} ({t['transaction_date']})" 
                                  for t in user_context.get("transactions", [])])
    
    response = llm.invoke(prompt.format_messages(
        family_structure=user_context.get("family_structure", "unknown"),
        number_of_children=user_context.get("number_of_children", 0),
        occupation=user_context.get("occupation", "unknown"),
        birth_date=user_context.get("birth_date", "unknown"),
        long_term_plans=plans_text,
        user_policies=policies_text,
        transactions=transactions_text,
        purchases=purchases_text
    )).content
    
    print(f"  strategy response: {response}")
    
    # JSON解析
    import json
    import re
    try:
        strategy_data = json.loads(response.strip())
        state["scolding_strategy"] = strategy_data.get("strategy", "general_financial")
        print(f"  selected strategy: {strategy_data.get('strategy')}")
        print(f"  reason: {strategy_data.get('reason', '')}")
    except:
        # フォールバック
        state["scolding_strategy"] = "general_financial"
        print("  fallback to general_financial strategy")
    
    return state

# 返品候補抽出: Amazon系で非消耗っぽい・十分高額などを軽規則で候補に
def is_non_consumable(name: str) -> bool:
    """簡易ヒューリスティック（必要に応じて拡張）"""
    ng = ["食品", "生鮮", "飲料", "ギフトカード", "デジタルコード", "ダウンロード", "ポイント"]
    return not any(tok in name for tok in ng)

@traced("pick_return_candidates")
def node_pick_return_candidates(state: GraphState) -> GraphState:
    """返品候補抽出：Amazon系で非消耗品・高額商品を候補に"""
    cands = []
    for p in state.get("extracted", []):
        vendor = (p.vendor or "").lower()
        # Amazon系の判定を緩くする
        if ("amazon" in vendor or "amazon.co.jp" in vendor or vendor == "" and "amazon" in p.item_name.lower()):
            if is_non_consumable(p.item_name):
                # 閾値を下げる: 3,000円以上
                if float(p.price) >= 3000 and p.currency.upper() == "JPY":
                    cands.append(p)
                    print(f"  candidate added: {p.item_name} / {p.price} {p.currency}")
    state["return_candidates"] = cands
    return state

@traced("decide_returnability")
def node_decide_returnability(state: GraphState) -> GraphState:
    """返品可否をLLMに判断させる（ツール選択の裁量ポイント）"""
    cands = state.get("return_candidates", [])
    if not cands:
        state["should_open_orders"] = False
        return state

    # LLMに候補一覧を渡し、「開く価値があるか？」を総合判断させる
    from langchain_core.prompts import ChatPromptTemplate
    plist = "\n".join([f"- {p.item_name} / {int(p.price)} {p.currency}" for p in cands])
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a returns strategist. Based on items and general Amazon JP policies, "
         "decide if opening the order history page to attempt a return is worthwhile now. "
         "Be generous in your judgment - if items seem returnable, recommend opening the page. "
         "Consider non-consumable items, typical 30-day windows, and total potential savings. "
         "IMPORTANT: Respond ONLY with valid JSON, no other text."),
        ("human",
         """候補一覧:
{plist}

判断基準:
- 非消耗品（電子機器、家電など）は返品可能性が高い
- 購入から30日以内なら通常返品可能
- 合計金額が5,000円以上なら検討価値あり
- 少しでも返品の可能性があれば「開く価値あり」とする

必須：以下の形式でJSONのみ回答:
{{"open": true, "reason": "短文理由"}}""")
    ])
    out = llm.invoke(prompt.format_messages(plist=plist)).content
    
    print(f"  LLM response: {out}")  # デバッグ用に全レスポンスを表示

    # JSONパースを堅牢に：テキストからJSON部分を抽出
    import json
    import re
    try:
        # まず、レスポンス全体をJSONとしてパースしてみる
        data = json.loads(out.strip())
        state["should_open_orders"] = bool(data.get("open", False))
        print("  decision reason:", data.get("reason", ""))
    except json.JSONDecodeError:
        try:
            # JSON部分を正規表現で抽出
            json_match = re.search(r'\{[^}]*"open"[^}]*\}', out)
            if json_match:
                json_str = json_match.group(0)
                print(f"  extracted JSON: {json_str}")
                data = json.loads(json_str)
                state["should_open_orders"] = bool(data.get("open", False))
                print("  decision reason:", data.get("reason", ""))
            else:
                # 最後の手段：テキスト中の"open": trueのパターンを検索
                if '"open": true' in out or '"open":true' in out:
                    state["should_open_orders"] = True
                    print("  decision: found 'open: true' in text")
                elif '"open": false' in out or '"open":false' in out:
                    state["should_open_orders"] = False
                    print("  decision: found 'open: false' in text")
                else:
                    print("  fallback: defaulting to false")
                    state["should_open_orders"] = False
        except Exception as e:
            print(f"  fallback parse failed: {e}")
            state["should_open_orders"] = False    # URL設定
    state["open_url"] = "https://www.amazon.co.jp/gp/css/order-history"
    return state

@traced("open_orders_page")
def node_open_orders_page(state: GraphState) -> GraphState:
    """実行：返品ページを開く"""
    import webbrowser
    url = state.get("open_url") or "https://www.amazon.co.jp/gp/css/order-history"
    webbrowser.open(url)
    print(f"  opened URL: {url}")
    return state

def route_return_open(state: GraphState) -> str:
    """decide_returnabilityの結果で分岐"""
    route = "OPEN" if state.get("should_open_orders") else "SKIP"
    now = datetime.now(JST).strftime("%H:%M:%S")
    print(f"[{now}] ➜ ROUTE return -> {route}")
    return route

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

# 新しい叱り戦略関連ノード
graph.add_node("fetch_user_context", node_fetch_user_context)
graph.add_node("decide_scolding_strategy", node_decide_scolding_strategy)
graph.add_node("scold_msg", node_scold_message)
graph.add_node("enc_msg", node_encourage_message)

# 返品関連ノードを追加
graph.add_node("pick_return_candidates", node_pick_return_candidates)
graph.add_node("decide_returnability", node_decide_returnability)
graph.add_node("open_orders_page", node_open_orders_page)

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
    {"SCOLD": "fetch_user_context", "ENCOURAGE": "enc_msg"},
)

# 新しい叱りフロー：judge → fetch_user_context → decide_scolding_strategy → scold_msg → 返品フロー
graph.add_edge("fetch_user_context", "decide_scolding_strategy")
graph.add_edge("decide_scolding_strategy", "scold_msg")

# 叱りルートに返品フローを追加
graph.add_edge("scold_msg", "pick_return_candidates")
graph.add_edge("pick_return_candidates", "decide_returnability")

graph.add_conditional_edges(
    "decide_returnability",
    route_return_open,
    {
        "OPEN": "open_orders_page",
        "SKIP": END,
    },
)
graph.add_edge("open_orders_page", END)

# 励ましはそのまま
graph.add_edge("enc_msg", END)

app = graph.compile(checkpointer=InMemorySaver())

# agent.py の末尾などに追記
from typing import List, Dict

def run_agent_once(dated_emails: List[Dict], user_goals: List[Dict]):
    """
    受け取ったメール群（本関数呼び出し時点での受信日をts_ymdとして渡す前提）で
    LangGraphを1回だけ実行し、最終stateを返す。
    """
    init_state = {
        "dated_emails": dated_emails,  # [{"text": str, "ts_ymd": "YYYY-MM-DD"}, ...]
        "user_goals": user_goals,      # [{"purpose":..., "target_amount":..., "by":"YYYY-MM"}, ...]
        # raw_emails は未使用（dated_emails 推奨）
    }
    # thread_id は用途に応じてユニーク化してもOK
    result = app.invoke(init_state, config={"configurable": {"thread_id": "mail-agent-thread"}})
    return result


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
    if result.get("user_context"):
        print("user_context     : fetched")
    if result.get("scolding_strategy"):
        print("scolding_strategy:", result.get("scolding_strategy"))
    print("return_candidates:", len(result.get("return_candidates", [])))
    if result.get("return_candidates"):
        for c in result["return_candidates"]:
            print(f"  - {c.item_name}: {int(c.price)} {c.currency}")
    print("should_open_orders:", result.get("should_open_orders", False))
    print(result["message"])
