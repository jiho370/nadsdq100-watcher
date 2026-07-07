#!/usr/bin/env python3
"""
market_signals.py — 지수·코인 6개 핵심 자산의 추세 신호 엔진 + 전일 세계시장 요약.

규칙(STRATEGY.md §1):
  · 주식 지수: 200일선 ±1% 히스테리시스(3일 확인) 레짐 + 12-1 모멘텀, 눌림선=20일선
  · 코인:     120일선 ±3% 히스테리시스(3일 확인) 레짐 + 3개월 모멘텀, 눌림선=50일선
  · 상태 5단계: 적극매수 / 눌림목분할매수 / 보유 / 축소검토 / 위험회피
  · 변동성 타깃 노출 W=min(1, 타깃/실현변동성60일) 은 참고 지표로만 표기.

신호는 전부 '종가 시계열만으로' 상태를 복원(stateless)하므로 상태파일이 필요 없다.
"""
from __future__ import annotations
import math

# ------------------------- 자산 정의 -------------------------
CORE_ASSETS = [
    # key, 이름, 야후 티커, 종류
    ("NDX",    "나스닥 100", "^NDX",    "equity"),
    ("SPX",    "S&P 500",    "^GSPC",   "equity"),
    ("KOSPI",  "코스피",     "^KS11",   "equity"),
    ("KOSDAQ", "코스닥",     "^KQ11",   "equity"),
    ("BTC",    "비트코인",   "BTC-USD", "crypto"),
    ("ETH",    "이더리움",   "ETH-USD", "crypto"),
]
WORLD_ASSETS = [
    ("DJI",  "다우존스",   "^DJI"),
    ("N225", "닛케이 225", "^N225"),
    ("DAX",  "독일 DAX",   "^GDAXI"),
    ("FTSE", "영국 FTSE",  "^FTSE"),
    ("HSI",  "홍콩 항셍",  "^HSI"),
    ("SSE",  "상해종합",   "000001.SS"),
    ("FX",   "달러-원",    "KRW=X"),
]

PARAMS = {
    "equity": {"trend_ma": 200, "band": 0.01, "confirm": 3, "mom": "12_1", "dip_ma": 20, "vol_target": 0.15},
    "crypto": {"trend_ma": 120, "band": 0.03, "confirm": 3, "mom": "3m",   "dip_ma": 50, "vol_target": 0.40},
}

STATE_META = {
    "aggressive_buy": ("🟢", "적극 매수",        "#15803d", "정기 적립 계속 + 신규 매수 가능"),
    "dip_buy":        ("🔵", "눌림목 분할 매수", "#2563eb", "상승 추세 속 조정 — 2~3회 분할 매수 (기대값 높은 진입 구간)"),
    "hold":           ("🟡", "보유",             "#ca8a04", "기존 보유 유지, 신규 매수는 보류"),
    "reduce":         ("🟠", "축소 검토",        "#c2410c", "신규 중단, 반등 시 일부 축소"),
    "risk_off":       ("🔴", "위험 회피",        "#b91c1c", "신규 중단 + 비중 절반 이상 축소 권고"),
}


# ------------------------- 계산 유틸 -------------------------
def _sma(closes, w, idx=None):
    """closes[:idx+1] 기준 w일 단순이동평균. 데이터 부족 시 None."""
    i = len(closes) - 1 if idx is None else idx
    if i + 1 < w:
        return None
    seg = closes[i - w + 1: i + 1]
    return sum(seg) / w


def _ret(closes, days):
    if len(closes) <= days:
        return None
    p0, p1 = closes[-days - 1], closes[-1]
    return (p1 / p0 - 1) * 100 if p0 else None


def _mom_12_1(closes):
    """12-1 모멘텀: 최근 1개월 제외 12개월 수익률(%)."""
    if len(closes) < 252:
        return None
    p0, p1 = closes[-252], closes[-21]
    return (p1 / p0 - 1) * 100 if p0 else None


def _realized_vol(closes, w=60):
    if len(closes) < w + 1:
        return None
    rets = [(closes[i] / closes[i - 1] - 1) for i in range(len(closes) - w, len(closes))]
    m = sum(rets) / len(rets)
    var = sum((r - m) ** 2 for r in rets) / max(len(rets) - 1, 1)
    return math.sqrt(var) * math.sqrt(252)


def _rsi(closes, w=14):
    if len(closes) < w + 1:
        return None
    gains, losses = [], []
    for i in range(len(closes) - w, len(closes)):
        d = closes[i] - closes[i - 1]
        gains.append(max(d, 0.0)); losses.append(max(-d, 0.0))
    ag, al = sum(gains) / w, sum(losses) / w
    if al == 0:
        return 100.0
    rs = ag / al
    return 100 - 100 / (1 + rs)


def regime_state(closes, trend_ma, band, confirm):
    """히스테리시스+확인일수 레짐을 종가 시계열만으로 복원.
    반환: (state: 'ON'|'OFF'|None, days_in_state: int)"""
    n = len(closes)
    if n < trend_ma + confirm:
        return None, 0
    state, streak_dir, streak = None, None, 0
    since = 0
    for i in range(trend_ma - 1, n):
        ma = _sma(closes, trend_ma, i)
        if ma is None:
            continue
        c = closes[i]
        raw = "ON" if c > ma * (1 + band) else ("OFF" if c < ma * (1 - band) else None)
        if raw and raw != state:
            if raw == streak_dir:
                streak += 1
            else:
                streak_dir, streak = raw, 1
            if streak >= confirm:
                state, since = raw, i
                streak_dir, streak = None, 0
        else:
            streak_dir, streak = None, 0
    return state, (n - 1 - since) if state else 0


def analyze(closes: list, kind: str) -> dict:
    """한 자산의 신호 일체를 계산."""
    p = PARAMS[kind]
    state, days = regime_state(closes, p["trend_ma"], p["band"], p["confirm"])
    mom = _mom_12_1(closes) if p["mom"] == "12_1" else _ret(closes, 63)
    ma_tr = _sma(closes, p["trend_ma"])
    ma_dip = _sma(closes, p["dip_ma"])
    price = closes[-1]
    gap_tr = (price / ma_tr - 1) * 100 if ma_tr else None
    gap_dip = (price / ma_dip - 1) * 100 if ma_dip else None
    rv = _realized_vol(closes)
    exposure = min(1.0, p["vol_target"] / rv) if rv and rv > 0 else None
    hi52 = max(closes[-252:]) if len(closes) >= 30 else None
    prox52 = (price / hi52 - 1) * 100 if hi52 else None

    mom_pos = (mom is not None and mom > 0)
    if state == "ON" and mom_pos:
        sig = "dip_buy" if (gap_dip is not None and gap_dip < 0) else "aggressive_buy"
    elif state == "ON":
        sig = "hold"
    elif state == "OFF" and mom_pos:
        sig = "reduce"
    elif state == "OFF":
        sig = "risk_off"
    else:
        sig = "hold"   # 데이터 부족 등 → 중립
    return {
        "price": price, "signal": sig, "regime": state, "regime_days": days,
        "trend_ma_n": p["trend_ma"], "gap_trend": gap_tr,
        "dip_ma_n": p["dip_ma"], "gap_dip": gap_dip,
        "mom_label": "12-1 모멘텀" if p["mom"] == "12_1" else "3개월 모멘텀",
        "mom": mom, "rsi": _rsi(closes), "prox52": prox52,
        "vol_ann": rv, "exposure": exposure,
        "ret_1d": _ret(closes, 1), "ret_1w": _ret(closes, 5), "ret_1m": _ret(closes, 21),
        "ret_3m": _ret(closes, 63), "ret_6m": _ret(closes, 126), "ret_1y": _ret(closes, 252),
    }


# ------------------------- 데이터 수집 -------------------------
def fetch_closes(yf, tickers: list[str]) -> dict:
    """야후에서 2년 일봉 종가. {ticker: {'closes': [...], 'dates': [...]}}"""
    df = yf.download(tickers, period="2y", interval="1d",
                     auto_adjust=True, progress=False, threads=True)
    close = df["Close"] if "Close" in getattr(df, "columns", []) else df
    out = {}
    for t in tickers:
        try:
            s = close[t].dropna() if hasattr(close, "columns") else close.dropna()
            if len(s) >= 30:
                out[t] = {"closes": [float(v) for v in s.tolist()],
                          "dates": [d.date().isoformat() for d in s.index]}
        except Exception:
            pass
    return out


def gather(yf) -> dict:
    """핵심 6자산 신호 + 세계시장 요약 데이터."""
    tickers = [t for _, _, t, _ in CORE_ASSETS] + [t for _, _, t in WORLD_ASSETS]
    raw = fetch_closes(yf, tickers)
    core, world = [], []
    for key, name, tic, kind in CORE_ASSETS:
        d = raw.get(tic)
        if not d:
            continue
        core.append({"key": key, "name": name, "ticker": tic, "kind": kind,
                     "as_of": d["dates"][-1], "closes": d["closes"],
                     **analyze(d["closes"], kind)})
    for key, name, tic in WORLD_ASSETS:
        d = raw.get(tic)
        if not d:
            continue
        c = d["closes"]
        world.append({"key": key, "name": name, "ticker": tic, "as_of": d["dates"][-1],
                      "price": c[-1], "ret_1d": _ret(c, 1), "ret_1w": _ret(c, 5),
                      "ret_1m": _ret(c, 21)})
    return {"core": core, "world": world}


def lean_for_ai(sig: dict) -> list:
    """AI 프롬프트 주입용(시세 배열 제외, 반올림)."""
    out = []
    for a in sig.get("core", []):
        d = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in a.items() if k != "closes"}
        meta = STATE_META.get(a["signal"])
        d["signal_kr"] = meta[1] if meta else a["signal"]
        d["action"] = meta[3] if meta else ""
        out.append(d)
    return out


# ------------------------- HTML -------------------------
def _esc(s): return (str(s or "")).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _chip(label, color="#6b7280", strong=False):
    w = "700" if strong else "600"
    return (f'<span style="display:inline-block;background:{color}1a;color:{color};border-radius:6px;'
            f'padding:1px 7px;margin:1px 4px 1px 0;font-size:11px;font-weight:{w}">{label}</span>')


def _pct(label, v, nd=1):
    if v is None:
        return ""
    return _chip(f"{label} {v:+.{nd}f}%", "#15803d" if v >= 0 else "#b91c1c")


def _fmt_price(a):
    p = a.get("price")
    if p is None:
        return ""
    if a.get("key") in ("BTC", "ETH"):
        return f"${p:,.0f}"
    if a.get("key") == "FX":
        return f"{p:,.1f}원"
    return f"{p:,.1f}"


def world_table_html(sig: dict) -> str:
    """전일 세계시장 요약 표(핵심 6자산 + 세계 지수 + 환율)."""
    rows = ""
    items = list(sig.get("core", [])) + list(sig.get("world", []))
    for a in items:
        r1 = a.get("ret_1d"); r5 = a.get("ret_1w"); r21 = a.get("ret_1m")
        def cell(v):
            if v is None:
                return '<td align="right" style="padding:4px 8px;color:#9ca3af">—</td>'
            col = "#15803d" if v >= 0 else "#b91c1c"
            return f'<td align="right" style="padding:4px 8px;color:{col};font-weight:600">{v:+.2f}%</td>'
        rows += (f'<tr style="border-bottom:1px solid #f1f5f9">'
                 f'<td style="padding:4px 8px;font-weight:600">{_esc(a["name"])}'
                 f' <span style="color:#9ca3af;font-size:10px">{_esc(a.get("as_of", ""))}</span></td>'
                 f'<td align="right" style="padding:4px 8px">{_fmt_price(a)}</td>'
                 f'{cell(r1)}{cell(r5)}{cell(r21)}</tr>')
    return (
        '<table role="presentation" width="100%" style="border-collapse:collapse;border:1px solid #e5e7eb;'
        'border-radius:10px;background:#fff;font-size:12px;overflow:hidden">'
        '<tr style="background:#f8fafc;color:#6b7280;font-size:11px">'
        '<td style="padding:5px 8px">시장 (기준일)</td><td align="right" style="padding:5px 8px">종가</td>'
        '<td align="right" style="padding:5px 8px">전일</td><td align="right" style="padding:5px 8px">1주</td>'
        '<td align="right" style="padding:5px 8px">1개월</td></tr>' + rows + '</table>')


def signal_cards_html(sig: dict, chart_cids: dict | None = None) -> str:
    """핵심 6자산 신호 카드."""
    cards = ""
    for a in sig.get("core", []):
        emoji, label, color, action = STATE_META.get(a["signal"], ("", a["signal"], "#6b7280", ""))
        chips = _pct("전일", a.get("ret_1d"), 2) + _pct("1개월", a.get("ret_1m")) + _pct("6개월", a.get("ret_6m"))
        if a.get("gap_trend") is not None:
            chips += _chip(f'{a["trend_ma_n"]}일선 {a["gap_trend"]:+.1f}%',
                           "#15803d" if a["gap_trend"] >= 0 else "#b91c1c")
        if a.get("mom") is not None:
            chips += _chip(f'{a["mom_label"]} {a["mom"]:+.1f}%',
                           "#15803d" if a["mom"] >= 0 else "#b91c1c")
        if a.get("exposure") is not None and a["exposure"] < 1:
            chips += _chip(f'변동성 참고 노출 {a["exposure"]*100:.0f}%', "#7c3aed")
        chart = ""
        if chart_cids and a["key"] in chart_cids:
            chart = (f'<td width="40%" valign="top" style="padding:10px 10px 10px 0">'
                     f'<img src="cid:{chart_cids[a["key"]]}" style="width:100%;border-radius:6px"></td>')
        cards += (
            f'<table role="presentation" width="100%" style="border-collapse:collapse;border:1px solid #e5e7eb;'
            f'border-radius:10px;margin:8px 0;background:#fff;overflow:hidden"><tr>'
            f'<td valign="top" style="padding:10px 12px">'
            f'<div style="font-size:14px;font-weight:700">{_esc(a["name"])} '
            f'<span style="color:#6b7280;font-size:11px;font-weight:400">{_fmt_price(a)}</span> '
            f'{_chip(f"{emoji} {label}", color, True)}</div>'
            f'<div style="margin:4px 0 0">{chips}</div>'
            f'<div style="font-size:12px;color:#1d4ed8;background:#eff6ff;border-radius:6px;'
            f'padding:4px 8px;margin-top:6px">🎯 {_esc(action)}</div></td>{chart}</tr></table>')
    legend = ('<div style="font-size:10px;color:#9ca3af;margin-top:4px;line-height:1.5">'
              '신호 규칙: 주식 지수 = 200일선 ±1% 히스테리시스(3일 확인) + 12-1 모멘텀 · '
              '코인 = 120일선 ±3% + 3개월 모멘텀. 눌림목 = 상승 레짐 속 20일선(코인 50일선) 아래. '
              '자세한 근거는 STRATEGY.md.</div>')
    return cards + legend
