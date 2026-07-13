#!/usr/bin/env python3
"""
weekly_report.py — 일요일 발송용 '주간 자산배분 리포트' (STRATEGY.md §4 — 2026-07 개편).

구조(일일 리포트와 같은 철학):
  · 자산 배분 비율은 안정형/공격형 두 가지 표준안으로 고정. AI는 해설만.
  · 차익실현/저점매수 = '리밸런싱 밴드'(목표의 1.2배 초과 → 초과분 매도, 0.8배 미만 → 매수).
    (기존 '1주 ±5% 등락' 규칙은 소음 기반·미검증이라 폐기 — Daryanani 2008 밴드 방식 채택)
  · 방어 컷(추세 오버레이): 자산이 레짐 OFF(200일선 히스테리시스) + 12개월 수익률 음수
    → 목표 비중의 절반만 유지, 컷분은 채권·현금. 일일 신호 엔진(market_signals)과 동일 규칙.
  · AI 실패 시 deterministic 해설로 무조건 발송(누락 방지).

자산군: 미국 주식(SPY) · 한국 주식(^KS11) · 코인(BTC-USD) · 미국 국채(IEF) · 금(GLD) · 현금성
참고 지역(언급용): 유럽(VGK) · 일본(EWJ) · 중국(MCHI) + 환율(USD/KRW)

표준 배분(통념·연구 부합 — 기존 '금 25~30%' 배분은 특정 구간 과최적화로 폐기):
  안정형: 미국 30 · 한국 10 · 코인 2 · 채권 40 · 금 10 · 현금 8
  공격형: 미국 50 · 한국 15 · 코인 5 · 채권 15 · 금 10 · 현금 5

실행:  python weekly_report.py             # 발송
       python weekly_report.py --no-email  # 미리보기만(output/weekly_report.html)
"""
from __future__ import annotations
import os, sys, io, json, argparse, datetime as dt

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm

import sp500_daily_report as R
import ai_report as AR
try:
    from ai_commentary import _extract_json
except Exception:
    def _extract_json(t):
        try: return json.loads(t)
        except Exception: return None

# 한글 폰트(있으면)
_KFONT = None
for _p in (r"C:\Windows\Fonts\malgun.ttf", "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"):
    if os.path.exists(_p):
        try:
            _fm.fontManager.addfont(_p)
            _KFONT = _fm.FontProperties(fname=_p).get_name()
            plt.rcParams["font.family"] = _KFONT
            break
        except Exception:
            pass
plt.rcParams["axes.unicode_minus"] = False


def _log(m): print(f"[주간] {m}", file=sys.stderr)


# ------------------------- 자산군 정의 -------------------------
ASSETS = [
    # key, 이름, 야후 티커(None=시세 없음), 접근 방법(국내 투자자용 예시)
    ("US_STOCK", "미국 주식",   "SPY",     "S&P500 — 예: SPY, KODEX 미국S&P500"),
    ("KR_STOCK", "한국 주식",   "^KS11",
     "코어-새틀라이트 권장(STRATEGY.md §3, 2026-07-14): 코어 65% KODEX 200 등 + "
     "새틀라이트 35% 코스피200 개별종목(국장 메일 추천 — valuediv 밸류×배당 팩터)"),
    ("COIN",     "코인",        "BTC-USD", "비트코인 — 예: BTC, 현물 ETF(IBIT 등)"),
    ("BOND",     "채권",        "IEF",     "미국채 7-10년 — 예: IEF, TIGER 미국채10년"),
    ("GOLD",     "금",          "GLD",     "금 현물 — 예: GLD, ACE KRX금현물"),
    ("CASH",     "현금성",      None,      "CMA · 파킹통장 · 달러 RP 등 (리밸런싱 실탄)"),
]
GLOBAL_SUB = [("EU", "유럽", "VGK"), ("JP", "일본", "EWJ"), ("CN", "중국", "MCHI")]
FX_TICKER = "KRW=X"   # USD/KRW

# STRATEGY.md §4 — 통념·연구 부합 표준 배분 (금 5~15% 권장 범위, 코인 1~5% 상한)
STABLE_WEIGHTS = {"US_STOCK": 30, "KR_STOCK": 10, "COIN": 2, "BOND": 40, "GOLD": 10, "CASH": 8}
AGGRESSIVE_WEIGHTS = {"US_STOCK": 50, "KR_STOCK": 15, "COIN": 5, "BOND": 15, "GOLD": 10, "CASH": 5}
PORTFOLIOS = {
    "STABLE": {"name": "안정형", "weights": STABLE_WEIGHTS},
    "AGGRESSIVE": {"name": "공격형", "weights": AGGRESSIVE_WEIGHTS},
}
PROFILE_ORDER = ["STABLE", "AGGRESSIVE"]

# 리밸런싱 밴드(Daryanani 2008): 보유 비중이 목표의 1.2배 초과 → 차익실현, 0.8배 미만 → 저점매수
REBALANCE_BAND = 0.20


def _weights_line(weights: dict) -> str:
    names = {key: name for key, name, _, _ in ASSETS}
    return " · ".join(f"{names.get(key, key)} {weights[key]}%" for key, *_ in ASSETS if key in weights)


# ------------------------- 데이터 -------------------------
def _fetch_closes(tickers: list[str]) -> dict[str, list[float]]:
    R._require_yf()
    df = R.yf.download(tickers, period="2y", interval="1d",
                       auto_adjust=True, progress=False, threads=True)
    close = df["Close"] if "Close" in getattr(df, "columns", []) else df
    out = {}
    for t in tickers:
        try:
            s = close[t].dropna()
            if len(s) >= 30:
                out[t] = [round(float(v), 4) for v in s.tolist()]
        except Exception:
            pass
    return out


def _verdict_price_lookup_batch(entries: list[dict]) -> dict:
    """AI verdict 로그에 있는 모든 종목의 현재가를 한 번에 조회 — {(symbol,market): price}.
    한국(market='kr')은 6자리 코드에 .KS 접미사를 붙여 조회."""
    tickers, key_map = [], {}
    for e in entries:
        sym, mkt = e.get("symbol"), e.get("market", "us")
        if not sym:
            continue
        t = f"{sym}.KS" if mkt == "kr" else sym
        tickers.append(t)
        key_map[t] = (sym, mkt)
    if not tickers:
        return {}
    closes = _fetch_closes(sorted(set(tickers)))
    return {key_map[t]: c[-1] for t, c in closes.items() if c}


def _verdict_summary_html() -> str:
    """AI 검증(매수유지 vs 관찰강등·제외) 사후추적 — 표본(그룹당 MIN_N) 부족하면 빈 문자열
    (섹션 자체 생략, score_calibration.py 게이트와 동일 철학: 표본 쌓이면 자동으로 켜짐)."""
    try:
        import ai_verdict_log as AVL
        entries = AVL.load()
        if not entries:
            return ""
        price_map = _verdict_price_lookup_batch(entries)
        summary = AVL.forward_return_summary(lambda s, m: price_map.get((s, m)))
        if not summary:
            return ""
        rows = "".join(
            f'<tr><td style="padding:3px 8px">{_esc(g)}</td>'
            f'<td style="padding:3px 8px;font-weight:700;color:{"#15803d" if v["avg_ret_pct"] >= 0 else "#b91c1c"}">'
            f'{v["avg_ret_pct"]:+.1f}%</td><td style="padding:3px 8px;color:#6b7280">{v["n"]}건</td></tr>'
            for g, v in summary.items())
        return (
            '<h3 style="margin:18px 0 6px">🔍 AI 검증 사후추적 <span style="color:#9ca3af;font-size:12px">'
            '(매수유지 vs 관찰강등·제외 — 판정 후 4주+ 경과분 평균 수익률)</span></h3>'
            '<table role="presentation" style="border-collapse:collapse;font-size:12px;width:100%;max-width:500px">'
            '<tr style="color:#6b7280;text-align:left"><th style="padding:3px 8px">판정</th>'
            '<th style="padding:3px 8px">평균 수익률</th><th style="padding:3px 8px">표본</th></tr>'
            + rows + '</table>')
    except Exception as e:
        _log(f"AI verdict 사후추적 집계 생략({type(e).__name__}: {e})")
        return ""


def _ret(closes, days):
    if len(closes) <= days:
        return None
    p0, p1 = closes[-days - 1], closes[-1]
    return round((p1 / p0 - 1) * 100, 1) if p0 else None


def _gap_ma(closes, w):
    if len(closes) < w:
        return None
    ma = sum(closes[-w:]) / w
    return round((closes[-1] / ma - 1) * 100, 1) if ma else None


def _trend_label(gap200, gap20):
    if gap200 is None:
        return "데이터 부족"
    if gap200 >= 0:
        return "상승 추세" if (gap20 is None or gap20 >= 0) else "추세 양호·단기 조정"
    return "바닥 다지기·반등 시도" if (gap20 is not None and gap20 >= 0) else "하락 추세"


def _signals(closes, trend_n: int = 200):
    g200, g20 = _gap_ma(closes, 200), _gap_ma(closes, 20)
    gtr = g200 if trend_n == 200 else _gap_ma(closes, trend_n)
    return {"price": round(closes[-1], 2), "gap200": g200, "gap20": g20,
            "gap_tr": gtr, "trend_n": trend_n,
            "ret_1w": _ret(closes, 5), "ret_1m": _ret(closes, 21), "ret_3m": _ret(closes, 63),
            "ret_6m": _ret(closes, 126), "ret_1y": _ret(closes, 252),
            "trend": _trend_label(gtr, g20)}


def gather() -> dict:
    import market_signals as MS
    trend_n = 200
    tickers = [t for _, _, t, _ in ASSETS if t] + [t for _, _, t in GLOBAL_SUB] + [FX_TICKER]
    closes = _fetch_closes(tickers)
    assets = {}
    for key, name, tic, howto in ASSETS:
        if tic is None:   # 현금성 — 시세 없음
            assets[key] = {"key": key, "name": name, "ticker": "", "howto": howto,
                           "closes": [], "trend": "해당 없음", "regime": None, "mom12": None,
                           "signal": None, "signal_kr": None}
            continue
        c = closes.get(tic)
        if c:
            kind = "crypto" if key == "COIN" else "equity"
            ms = MS.analyze(c, kind)   # 일일 신호 엔진과 동일 규칙(레짐 히스테리시스+모멘텀)
            meta = MS.STATE_META.get(ms["signal"], ("", "", "#6b7280", ""))
            assets[key] = {"key": key, "name": name, "ticker": tic, "howto": howto,
                           "closes": c, **_signals(c, trend_n),
                           "regime": ms["regime"], "mom12": ms["mom"],
                           "signal": ms["signal"], "signal_kr": meta[1], "signal_action": meta[3]}
    gsub = {}
    for key, name, tic in GLOBAL_SUB:
        c = closes.get(tic)
        if c:
            s = _signals(c)
            # 확실한 매수 신호: 200일선 위 + 3개월 +5% 이상
            s["buy_signal"] = bool(s["gap200"] is not None and s["gap200"] > 0
                                   and (s["ret_3m"] or 0) >= 5)
            gsub[key] = {"key": key, "name": name, "ticker": tic, **s}
    fx = None
    if closes.get(FX_TICKER):
        c = closes[FX_TICKER]
        fx = {"usdkrw": round(c[-1], 1), "chg_1w": _ret(c, 5), "chg_1m": _ret(c, 21)}
    as_of = dt.date.today().isoformat()
    portfolios = {k: {"name": v["name"], "weights": dict(v["weights"])} for k, v in PORTFOLIOS.items()}
    return {"as_of": as_of, "assets": assets, "global_sub": gsub, "fx": fx,
            "portfolios": portfolios,
            "stable_weights": dict(STABLE_WEIGHTS),
            "aggressive_weights": dict(AGGRESSIVE_WEIGHTS)}


# ------------------------- AI 해설 -------------------------
_W_SYSTEM = (
    "당신은 한국 개인투자자(투자에 관심 있는 아버지)에게 매주 일요일 보내는 '주간 자산배분 리포트'를 쓰는 "
    "애널리스트다. 규칙:\n"
    "1) 수치는 제공된 JSON 값만 인용. 지어내지 않는다.\n"
    "2) 자산 배분은 안정형/공격형 두 가지 표준안으로 이미 정해져 있다. 비중 숫자를 바꾸거나 새 비중을 제안하지 않는다. 배분표는 설명만 하고, 대응 전략은 차익실현 후보와 저점매수 후보만 짧게 제시한다.\n"
    "3) web_search 로 이번 주 주요 매크로 이벤트(금리·FOMC·고용지표 등)를 확인해 반영. 확인 안 되면 생략.\n"
    "4) 단정·수익 보장 금지. 이건 '조언·참고용'임을 전제로, 일반인이 이해할 쉬운 말로 간결하게.\n"
    "5) 한자 절대 금지. 이동평균은 '20일선/200일선' 표기.\n"
    "6) 최종 출력은 지정된 JSON 하나만(코드블록·군더더기 없이)."
)

_W_SCHEMA = (
    '{\n'
    '  "overview": "이번 주 글로벌 시장 전체 요약 2문장",\n'
    '  "strategy": "방어 컷: ... / 눌림목 분할 매수: ... / 나머지는 밴드 규칙 처럼 짧은 한 줄",\n'
    '  "assets": [ {"key":"US_STOCK","comment":"이번 주 흐름+배경 1-2문장","action":"대응 한 줄"} ],\n'
    '  "global_notes": [ {"key":"EU","comment":"이번 주 흐름 한 줄(buy_signal=true면 매수 신호 근거 언급)"} ],\n'
    '  "fx_note": "환율(USD/KRW) 흐름과 의미 한 줄",\n'
    '  "risks": "공통 유의사항 한 줄"\n'
    '}'
)


def _lean_ctx(ctx):
    assets = [{k: v for k, v in a.items() if k != "closes"} for a in ctx["assets"].values()]
    return {"as_of": ctx["as_of"], "assets": assets,
            "global_sub": list(ctx["global_sub"].values()), "fx": ctx["fx"],
            "portfolios": ctx["portfolios"]}


def build_ai(ctx: dict) -> dict:
    if not AR._enabled():
        _log("AI 비활성 → 지표 기반 해설.")
        return {}
    instr = (
        "아래 데이터로 주간 자산배분 리포트의 해설을 작성하라. 배분은 개인별 추천이 아니라 공용 표준안으로만 제시한다.\n"
        "- portfolios: 안정형과 공격형의 표준 배분이다. 숫자는 절대 바꾸지 않고, 개인별 맞춤안처럼 쓰지 않는다.\n"
        "- assets: 자산군 각각 comment(이번 주 흐름·배경)와 action(대응 한 줄). "
        "trend·ret_1w·gap200·signal_kr(코드가 확정한 추세 신호)을 근거로, web_search 로 배경 뉴스 보강. "
        "action은 신호와 일치시킨다: 방어 컷 대상(레짐 OFF+12개월 음수)은 '비중 절반 유지', "
        "레짐 ON+눌림은 '분할 매수 후보', 그 외는 '목표 비중 유지(밴드 이탈 시에만 매매)'. "
        "특정 주의 등락만으로 사고팔라는 표현 금지. 배분이 매주 바뀌는 것처럼 보이는 표현도 금지.\n"
        "- global_notes: 유럽(EU)·일본(JP)·중국(CN) 각 한 줄. buy_signal=true 인 지역은 "
        "'추세·모멘텀상 매수 신호'임을 언급.\n"
        "- strategy: '방어 컷: A / 눌림목 분할 매수: B / 나머지는 밴드 규칙' 형태의 짧은 한 줄.\n\n"
        f"출력 스키마(JSON):\n{_W_SCHEMA}\n\n"
        f"CONTEXT = {json.dumps(_lean_ctx(ctx), ensure_ascii=False)}\n"
    )
    try:
        try:
            text = AR._call_cli(instr, AR.REPORT_WEB, system=_W_SYSTEM) if AR.AI_BACKEND == "cli" \
                else AR._call_api(instr, AR.REPORT_WEB, system=_W_SYSTEM)
        except Exception as e1:
            _log(f"1차 실패({type(e1).__name__}) → 웹검색 없이 재시도")
            text = AR._call_cli(instr, False, system=_W_SYSTEM) if AR.AI_BACKEND == "cli" \
                else AR._call_api(instr, False, system=_W_SYSTEM)
        parsed = _extract_json(text or "")
        if not isinstance(parsed, dict):
            _log("JSON 파싱 실패 → 지표 기반 해설.")
            return {}
        return parsed
    except Exception as e:
        _log(f"호출 실패({type(e).__name__}: {e}) → 지표 기반 해설.")
        return {}



def defense_cuts(ctx: dict) -> list[dict]:
    """방어 컷(STRATEGY.md §4): 레짐 OFF + 12개월 수익률 음수 → 목표 비중 절반 (컷분은 현금·채권).
    일일 신호 엔진과 동일 규칙 — 레짐은 200일선(코인 120일선) 히스테리시스."""
    cuts = []
    for key, name, tic, _ in ASSETS:
        if key in ("CASH", "BOND") or tic is None:
            continue   # 도피처 자산은 컷 대상 아님
        a = ctx.get("assets", {}).get(key) or {}
        y = a.get("ret_1y")
        if a.get("regime") == "OFF" and y is not None and y < 0:
            cuts.append({"key": key, "name": name, "ret_1y": y})
    return cuts


def adjusted_weights(weights: dict, cuts: list[dict]) -> dict:
    """방어 컷 적용 배분: 컷 자산은 절반, 컷분은 현금성으로."""
    adj = dict(weights)
    freed = 0.0
    for c in cuts:
        k = c["key"]
        if k in adj:
            half = adj[k] / 2.0
            freed += adj[k] - half
            adj[k] = half
    adj["CASH"] = adj.get("CASH", 0) + freed
    return adj


def dip_buy_candidates(ctx: dict) -> list[str]:
    """레짐 ON + 눌림(20일선 아래) — 상승 추세 속 조정, 분할 매수 후보."""
    out = []
    for key, name, tic, _ in ASSETS:
        if tic is None:
            continue
        a = ctx.get("assets", {}).get(key) or {}
        if a.get("regime") == "ON" and (a.get("gap20") is not None) and a["gap20"] < 0:
            out.append(name)
    return out


def _rebalance_strategy_text(ctx: dict) -> str:
    cuts = defense_cuts(ctx)
    dips = dip_buy_candidates(ctx)
    cut_txt = ", ".join(f'{c["name"]}(12개월 {c["ret_1y"]:+.1f}%)' for c in cuts) or "없음"
    dip_txt = ", ".join(dips) or "없음"
    return (f"방어 컷(비중 절반): {cut_txt} / 눌림목 분할 매수 후보: {dip_txt} / "
            f"차익실현·저점매수는 '보유 비중이 목표의 1.2배 초과 → 초과분 매도, 0.8배 미만 → 부족분 매수' 밴드 규칙으로.")


def _strategy_html(ctx: dict) -> str:
    cuts = defense_cuts(ctx)
    dips = dip_buy_candidates(ctx)
    def box(title, txt, color):
        return (f'<span style="display:inline-block;background:{color}12;color:{color};border:1px solid {color}33;'
                f'border-radius:8px;padding:5px 9px;margin:2px 6px 2px 0;font-size:12px;font-weight:700">'
                f'{title}: {_esc(txt)}</span>')
    cut_txt = ", ".join(c["name"] for c in cuts) or "없음"
    dip_txt = ", ".join(dips) or "없음"
    return (box("🛡 방어 컷(비중 절반)", cut_txt, "#b91c1c")
            + box("🔵 눌림목 분할 매수", dip_txt, "#2563eb"))


def _band_rule_html() -> str:
    return ('<div style="background:#f0f9ff;border:1px solid #bae6fd;border-radius:8px;padding:8px 12px;'
            'font-size:12px;color:#0c4a6e;line-height:1.6;margin:8px 0">'
            '<b>차익실현 · 저점매수 규칙(리밸런싱 밴드)</b><br>'
            '본인 계좌에서 각 자산의 실제 비중을 확인해, 목표의 <b>1.2배</b>를 넘으면 초과분만 팔아 목표로 복귀(차익실현), '
            '<b>0.8배</b> 아래면 부족분을 매수(저점매수)하세요. 예: 목표 30% 자산이 36%를 넘으면 매도. '
            '하락 추세(방어 컷 대상) 자산의 저점매수는 한 번에 하지 말고 3회 분할로. '
            '이 방식은 특정 주의 등락(소음)이 아니라 비중 이탈에만 반응합니다.</div>')


def _market_overview(ctx: dict) -> str:
    moved = []
    for key, name, tic, _ in ASSETS:
        if tic is None:
            continue
        a = ctx.get("assets", {}).get(key) or {}
        r = a.get("ret_1w")
        if r is not None:
            moved.append((abs(r), r, name, a.get("trend", "")))
    if not moved:
        return "이번 주 자산별 가격 데이터를 기준으로 배분과 추세 신호를 점검합니다."
    moved.sort(reverse=True)
    top = moved[:3]
    parts = [f"{name} {r:+.1f}%" for _, r, name, _ in top]
    return ("이번 주 변동이 컸던 자산은 " + ", ".join(parts)
            + "입니다. 표준 배분은 그대로 두고, 추세 신호(방어 컷)와 비중 밴드 이탈에만 대응합니다.")


def deterministic_ai(ctx: dict) -> dict:
    """AI 실패 시 지표만으로 해설 구성."""
    cut_keys = {c["key"] for c in defense_cuts(ctx)}
    def cmt(a):
        r = a.get("ret_1w")
        base = a.get("trend", "")
        if a.get("signal_kr"):
            base += f" (신호: {a['signal_kr']})"
        if r is None:
            return base + "."
        direction = "상승" if r > 0 else ("하락" if r < 0 else "보합")
        return f"{base}. 이번 주 {r:+.1f}% {direction}."
    def act(a):
        if a.get("key") == "CASH":
            return "리밸런싱 실탄 — 방어 컷 발생 시 이 비중이 늘어남."
        if a.get("key") in cut_keys:
            return "방어 컷: 목표 비중의 절반만 유지(레짐 OFF + 12개월 음수). 레짐 복귀 시 원상복구."
        if a.get("regime") == "ON" and (a.get("gap20") is not None) and a["gap20"] < 0:
            return "눌림목 분할 매수 후보: 상승 추세 속 조정 — 2~3회 나눠 매수."
        if a.get("regime") == "ON":
            return "정상 보유: 목표 비중 유지, 밴드(±20%) 이탈 시에만 매매."
        return "관망: 신규 매수 보류, 밴드 규칙만 적용."
    fx = ctx.get("fx") or {}
    fxn = (f"달러-원 {fx['usdkrw']:,.0f}원, 1주 {fx['chg_1w']:+.1f}%."
           if fx.get("usdkrw") is not None and fx.get("chg_1w") is not None else "")
    return {"overview": _market_overview(ctx),
            "strategy": _rebalance_strategy_text(ctx),
            "assets": [{"key": k, "comment": cmt(a), "action": act(a)} for k, a in ctx["assets"].items()],
            "global_notes": [{"key": k, "comment": f"{g['name']}: {g['trend']}, 3개월 {g['ret_3m']:+.1f}%."
                              + (" 추세·모멘텀상 매수 신호." if g.get("buy_signal") else "")}
                             for k, g in ctx["global_sub"].items() if g.get("ret_3m") is not None],
            "fx_note": fxn,
            "risks": "고정 배분 기반 참고용 자료입니다. 투자 권유가 아닙니다."}

# ------------------------- 차트 -------------------------
def _chart_png(closes, title):
    if not closes or len(closes) < 60:
        return None
    c = closes[-252:]
    x = range(len(c))
    fig, ax = plt.subplots(figsize=(4.8, 2.2), dpi=150)
    ax.plot(x, c, lw=1.5, color="#111827")
    if len(closes) >= 200:
        ma = [sum(closes[i - 199:i + 1]) / 200 for i in range(len(closes) - len(c), len(closes))]
        ax.plot(x, ma, lw=1.0, color="#ef4444", label="200일선" if _KFONT else "MA200")
        ax.legend(fontsize=7, loc="upper left", frameon=False)
    ax.set_title(title, fontsize=9, loc="left", fontweight="bold", color="#111827")
    ax.margins(x=0); ax.grid(True, alpha=0.15, lw=0.5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(labelsize=7, length=0); ax.set_xticks([])
    b = io.BytesIO()
    fig.savefig(b, format="png", bbox_inches="tight"); plt.close(fig)
    return b.getvalue()


# ------------------------- HTML -------------------------
def _esc(s): return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _chip(label, color="#6b7280", strong=False):
    w = "700" if strong else "600"
    return (f'<span style="display:inline-block;background:{color}1a;color:{color};border-radius:6px;'
            f'padding:1px 7px;margin:1px 4px 1px 0;font-size:11px;font-weight:{w}">{label}</span>')


_TREND_COLOR = {"상승 추세": "#15803d", "추세 양호·단기 조정": "#ca8a04",
                "바닥 다지기·반등 시도": "#2563eb", "하락 추세": "#b91c1c"}


def _pct_chip(label, v):
    if v is None:
        return ""
    return _chip(f"{label} {v:+.1f}%", "#15803d" if v >= 0 else "#b91c1c")


def _rebalance_hint(asset):
    r = asset.get("ret_1w")
    if r is None:
        return '<span style="color:#6b7280">목표 비중 점검</span>'
    if r > 0:
        return '<span style="color:#b91c1c;font-weight:700">초과분 매도 점검</span>'
    if r < 0:
        return '<span style="color:#15803d;font-weight:700">부족분 매수 점검</span>'
    return '<span style="color:#6b7280">목표 비중 점검</span>'


def _weights_table(ctx):
    rows = ""
    stable = ctx["portfolios"]["STABLE"]["weights"]
    aggressive = ctx["portfolios"]["AGGRESSIVE"]["weights"]
    for key, name, _, _ in ASSETS:
        rows += (f'<tr><td style="padding:5px 8px;border-bottom:1px solid #f1f5f9">{_esc(name)}</td>'
                 f'<td align="center" style="padding:5px;border-bottom:1px solid #f1f5f9;font-weight:700">{stable[key]}%</td>'
                 f'<td align="center" style="padding:5px;border-bottom:1px solid #f1f5f9;font-weight:700">{aggressive[key]}%</td></tr>')
    rows += (f'<tr style="background:#f8fafc;font-weight:700"><td style="padding:5px 8px">합계</td>'
             f'<td align="center" style="padding:5px">{sum(stable.values())}%</td>'
             f'<td align="center" style="padding:5px">{sum(aggressive.values())}%</td></tr>')
    return (
        '<table role="presentation" width="100%" style="border-collapse:collapse;border:1px solid #e5e7eb;'
        'border-radius:10px;background:#fff;font-size:12px;overflow:hidden;max-width:460px">'
        '<tr style="background:#f8fafc;color:#6b7280;font-size:11px">'
        '<td style="padding:6px 8px">자산군</td><td align="center">안정형</td>'
        '<td align="center">공격형</td></tr>' + rows + '</table>')


def _profile_note_boxes(ctx, notes):
    nmap = {n.get("key"): n.get("comment", "") for n in (notes or []) if isinstance(n, dict)}
    boxes = ""
    for key in PROFILE_ORDER:
        p = ctx["portfolios"][key]
        comment = nmap.get(key, "")
        line = _weights_line(p["weights"])
        boxes += (f'<div style="border:1px solid #e5e7eb;border-radius:8px;padding:9px 12px;margin:7px 0;background:#fff">'
                  f'<div style="font-size:13px;font-weight:700">{_esc(p["name"])} '
                  f'<span style="color:#6b7280;font-size:11px;font-weight:400">목표 배분</span></div>'
                  f'<div style="font-size:12px;color:#374151;margin-top:3px;line-height:1.5">{_esc(line)}</div>'
                  + (f'<div style="font-size:12px;color:#111;margin-top:4px;line-height:1.5">{_esc(comment)}</div>' if comment else "")
                  + '</div>')
    return boxes


def _asset_card(a, d):
    tr = a.get("trend", "")
    chips = (_chip(tr, _TREND_COLOR.get(tr, "#6b7280"), True)
             + _pct_chip("1주", a.get("ret_1w")) + _pct_chip("1개월", a.get("ret_1m"))
             + _pct_chip("3개월", a.get("ret_3m")) + _pct_chip("6개월", a.get("ret_6m")))
    if a.get("signal_kr"):
        import market_signals as _MS
        meta = _MS.STATE_META.get(a.get("signal"))
        chips += _chip(f'{meta[0] if meta else ""} {a["signal_kr"]}', meta[2] if meta else "#6b7280", True)
    gtr, tn = a.get("gap_tr"), a.get("trend_n", 200)
    if gtr is not None:
        chips += _chip(f'{tn}일선 {gtr:+.1f}%', "#15803d" if gtr >= 0 else "#b91c1c")
    cmt = f'<div style="font-size:13px;color:#111;margin-top:6px;line-height:1.55">{_esc(d.get("comment"))}</div>' \
        if d.get("comment") else ""
    act = (f'<div style="font-size:12px;color:#1d4ed8;background:#eff6ff;border-radius:6px;'
           f'padding:5px 8px;margin-top:6px">🎯 {_esc(d.get("action"))}</div>') if d.get("action") else ""
    howto = f'<div style="color:#9ca3af;font-size:11px;margin-top:5px">{_esc(a.get("howto"))}</div>'
    chart_td = (f'<td width="44%" valign="top" style="padding:12px 12px 12px 0">'
                f'<img src="cid:wchart_{a["key"]}" style="width:100%;border-radius:6px"></td>'
                if a.get("closes") else "")
    left_w = "56%" if a.get("closes") else "100%"
    return (
        f'<table role="presentation" width="100%" style="border-collapse:collapse;border:1px solid #e5e7eb;'
        f'border-radius:10px;margin:10px 0;background:#fff;overflow:hidden"><tr>'
        f'<td width="{left_w}" valign="top" style="padding:12px 14px">'
        f'<div style="font-size:15px;font-weight:700">{_esc(a["name"])} '
        f'<span style="color:#6b7280;font-size:12px;font-weight:400">{_esc(a["ticker"])}</span></div>'
        f'<div style="margin:5px 0 0">{chips}</div>{cmt}{act}{howto}</td>'
        f'{chart_td}</tr></table>')


def _global_rows(ctx, notes):
    nmap = {n.get("key"): n.get("comment", "") for n in (notes or []) if isinstance(n, dict)}
    rows = ""
    for key, g in ctx["global_sub"].items():
        sig = _chip("매수 신호", "#15803d", True) if g.get("buy_signal") else ""
        chips = _pct_chip("1주", g.get("ret_1w")) + _pct_chip("3개월", g.get("ret_3m"))
        if g.get("gap200") is not None:
            chips += _chip(f'200일선 {g["gap200"]:+.1f}%', "#15803d" if g["gap200"] >= 0 else "#b91c1c")
        cmt = _esc(nmap.get(key, ""))
        rows += (f'<div style="border:1px solid #e5e7eb;border-radius:8px;padding:9px 12px;margin:6px 0;background:#fff">'
                 f'<div style="font-size:13px;font-weight:700">{_esc(g["name"])} '
                 f'<span style="color:#6b7280;font-size:11px;font-weight:400">{_esc(g["ticker"])}</span> {sig}</div>'
                 f'<div style="margin:3px 0 0">{chips}</div>'
                 + (f'<div style="font-size:12px;color:#374151;margin-top:4px;line-height:1.5">{cmt}</div>' if cmt else "")
                 + '</div>')
    return rows


def _adjusted_line(ctx) -> str:
    """방어 컷이 있을 때 '이번 주 적용 배분'을 표 아래 한 줄로."""
    cuts = defense_cuts(ctx)
    if not cuts:
        return ('<div style="font-size:11px;color:#6b7280;margin:6px 0 0">이번 주 방어 컷 대상 없음 — '
                '표준 배분 그대로 적용.</div>')
    names = {key: name for key, name, _, _ in ASSETS}
    lines = ""
    for pk in PROFILE_ORDER:
        p = ctx["portfolios"][pk]
        adj = adjusted_weights(p["weights"], cuts)
        txt = " · ".join(f"{names.get(k, k)} {v:g}%" for k, v in adj.items())
        lines += f'<div>{_esc(p["name"])}(적용): {_esc(txt)}</div>'
    cut_names = ", ".join(c["name"] for c in cuts)
    return (f'<div style="font-size:11px;color:#b91c1c;margin:6px 0 0;line-height:1.6">'
            f'🛡 방어 컷 적용({_esc(cut_names)} — 레짐 OFF + 12개월 음수 → 절반, 컷분은 현금성으로):'
            f'{lines}</div>')


def _rule_desc(ctx) -> str:
    stable = _weights_line(ctx["portfolios"]["STABLE"]["weights"])
    aggressive = _weights_line(ctx["portfolios"]["AGGRESSIVE"]["weights"])
    return f'안정형({stable}) / 공격형({aggressive})'


def render_html(ctx, d, verdict_html=""):
    dmap = {a.get("key"): a for a in (d.get("assets") or []) if isinstance(a, dict)}
    cards = "".join(_asset_card(a, dmap.get(k, {})) for k, a in ctx["assets"].items())
    fx = ctx.get("fx") or {}
    fx_line = ""
    if fx.get("usdkrw") is not None:
        chg = f" (1주 {fx['chg_1w']:+.1f}%)" if fx.get("chg_1w") is not None else ""
        note = f" — {_esc(d.get('fx_note'))}" if d.get("fx_note") else ""
        fx_line = (f'<div style="background:#f8fafc;border-left:3px solid #6b7280;padding:8px 12px;'
                   f'font-size:13px;line-height:1.6;margin:10px 0"><b>💱 환율</b> '
                   f'달러-원 {fx["usdkrw"]:,.0f}원{chg}{note}</div>')
    return (
        f'<div style="font-family:-apple-system,BlinkMacSystemFont,\'Segoe UI\',\'Malgun Gothic\',sans-serif;'
        f'max-width:700px;margin:0 auto;color:#111">'
        f'<h2 style="margin:6px 0">📊 주간 자산배분 리포트 '
        f'<span style="color:#9ca3af;font-size:12px">({_esc(ctx["as_of"])} 기준)</span></h2>'
        f'<div style="background:#f8fafc;border-left:3px solid #6b7280;padding:8px 12px;font-size:13px;'
        f'line-height:1.6;margin:8px 0"><b>🧭 이번 주</b> {_esc(d.get("overview"))}<br>'
        f'<div style="margin-top:6px"><b>🎯 대응 후보</b> {_strategy_html(ctx)}</div></div>'
        f'<h3 style="margin:16px 0 6px">⚖️ 표준 자산 배분 <span style="color:#9ca3af;font-size:12px">'
        f'(안정형·공격형 — 근거: 전통 60/40 틀 + 금 5~15% · 코인 1~5% 연구 권장 범위)</span></h3>'
        f'{_weights_table(ctx)}'
        f'{_adjusted_line(ctx)}'
        f'{_band_rule_html()}'
        f'<h3 style="margin:18px 0 2px">📈 자산군별 흐름</h3>{cards}'
        f'<h3 style="margin:18px 0 6px">🌏 해외 주요 지역 <span style="color:#9ca3af;font-size:12px">'
        f'(유럽·일본·중국 — 매수 신호 시 표시)</span></h3>'
        f'{_global_rows(ctx, d.get("global_notes"))}'
        f'{fx_line}'
        f'{verdict_html}'
        f'<div style="font-size:11px;color:#9ca3af;margin-top:14px;line-height:1.5">'
        f'⚠️ {_esc(d.get("risks"))}<br>규칙 기반 참고용 자료이며 투자 권유가 아닙니다. '
        f'판단·책임은 본인에게 있습니다.</div></div>')


# ------------------------- 실행 -------------------------
def run(no_email: bool = False):
    ctx = gather()
    if not ctx["assets"]:
        _log("자산 데이터 수집 실패 → 발송 중단.")
        return
    cuts = defense_cuts(ctx)
    _log(f"자산 {len(ctx['assets'])} · 참고지역 {len(ctx['global_sub'])} · "
         f"표준 배분 안정형 {STABLE_WEIGHTS} · 공격형 {AGGRESSIVE_WEIGHTS} · "
         f"방어 컷 {[c['name'] for c in cuts] or '없음'}")
    d = build_ai(ctx) or deterministic_ai(ctx)
    # 자산 comment 누락분은 지표 해설로 메꿈
    det = deterministic_ai(ctx)
    dmap = {a.get("key"): a for a in (d.get("assets") or []) if isinstance(a, dict)}
    detmap = {a["key"]: a for a in det["assets"]}
    d["assets"] = [dmap.get(k) or detmap[k] for k in ctx["assets"]]
    for f in ("overview", "strategy", "risks"):
        if not (d.get(f) or "").strip():
            d[f] = det[f]

    # 상단 대응 후보는 지표 기반 고정 포맷으로 표시한다.
    d["strategy"] = det["strategy"]
    if not (d.get("fx_note") or "").strip():
        d["fx_note"] = det.get("fx_note", "")
    if not d.get("global_notes"):
        d["global_notes"] = det["global_notes"]

    images = []
    for k, a in ctx["assets"].items():
        png = _chart_png(a["closes"], f'{a["name"]} ({a["ticker"]})')
        if png:
            images.append((f"wchart_{k}", png))
    html = render_html(ctx, d, verdict_html=_verdict_summary_html())

    os.makedirs("output", exist_ok=True)
    import base64
    prev = html
    for cid, png in images:
        prev = prev.replace(f"cid:{cid}", "data:image/png;base64," + base64.b64encode(png).decode())
    with open("output/weekly_report.html", "w", encoding="utf-8") as f:
        f.write(prev)
    _log("미리보기 output/weekly_report.html 저장")

    if not no_email:
        R.send_email(f"[주간 자산배분] {ctx['as_of']} 리포트", html, images)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="주간 자산배분 리포트")
    ap.add_argument("--no-email", action="store_true")
    args = ap.parse_args()
    run(no_email=args.no_email)
