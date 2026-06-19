#!/usr/bin/env python3
"""
nasdaq100_low_per_report.py  (A단계)

매일 나스닥-100 구성종목 중 PER(주가수익비율)이 가장 낮은 N개 종목을 추려
추세 지표(이동평균 20/50/200, RSI 14, MACD 12/26/9)를 계산하고,
이메일로 보낼 본문(HTML + 텍스트)과 제목을 생성한다.

- 데이터 소스 : Financial Modeling Prep (FMP)   → 환경변수 FMP_API_KEY 필요
- 실행 환경   : Claude Code 클라우드 routine (매일 1회, 미국장 마감 후)
- 발송(B단계) : generate_report()가 돌려주는 (subject, html, text)를 받아 처리

규칙
  * PER <= 0 또는 결측치는 "저평가"가 아니라 적자/무수익 신호이므로 제외한다.
  * 전일 명단과 비교해 신규 편입/이탈 종목을 함께 표시한다(STATE_FILE 사용).
"""

from __future__ import annotations

import os
import sys
import json
from datetime import datetime, timezone, timedelta

import requests
import pandas as pd
import numpy as np

import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# ----------------------------- 설정 -----------------------------
FMP_API_KEY   = os.environ.get("FMP_API_KEY", "").strip()
FMP_BASE      = os.environ.get("FMP_BASE", "https://financialmodelingprep.com/stable")
TOP_N         = int(os.environ.get("TOP_N", "10"))          # PER 하위 N개
HISTORY_DAYS  = int(os.environ.get("HISTORY_DAYS", "400"))  # MA200(영업일 200) + 여유
STATE_FILE    = os.environ.get("STATE_FILE", "state_prev_list.json")  # 전일 명단 저장
TIMEOUT       = 30
KST           = timezone(timedelta(hours=9))

RSI_PERIOD    = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
MA_WINDOWS    = (20, 50, 200)


# ------------------------- HTTP 유틸 ----------------------------
def _get_json(path: str, params: dict | None = None, retries: int = 3):
    """FMP GET 호출. 간단한 재시도/백오프 포함."""
    if not FMP_API_KEY:
        raise RuntimeError("환경변수 FMP_API_KEY가 비어 있습니다.")
    params = dict(params or {})
    params["apikey"] = FMP_API_KEY
    url = f"{FMP_BASE}/{path.lstrip('/')}"
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=TIMEOUT)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < retries - 1:
                import time
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"FMP 호출 실패: {url} :: {last_err}")


# ------------------------- 데이터 수집 --------------------------
# 나스닥-100 구성종목 (지수는 3·6·9·12월 분기 리밸런싱 — 그때만 갱신하면 됩니다)
# 출처: slickcharts.com/nasdaq100. 클래스 중복주(GOOGL/GOOG 등) 포함.
NASDAQ100_SYMBOLS = [
    "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "AVGO", "TSLA", "META", "MU",
    "WMT", "AMD", "ASML", "INTC", "AMAT", "LRCX", "CSCO", "ARM", "COST", "KLAC",
    "SNDK", "NFLX", "PLTR", "TXN", "MRVL", "WDC", "STX", "QCOM", "LIN", "PANW",
    "ADI", "TMUS", "PEP", "AMGN", "CRWD", "APP", "GILD", "HON", "ISRG", "SHOP",
    "BKNG", "VRTX", "SBUX", "PDD", "CDNS", "FTNT", "MAR", "CEG", "MNST", "SNPS",
    "ADP", "CSX", "ABNB", "MELI", "CMCSA", "NXPI", "DDOG", "MDLZ", "ADBE", "MPWR",
    "DASH", "ROST", "INTU", "ORLY", "AEP", "CTAS", "LITE", "WBD", "REGN", "PCAR",
    "BKR", "MCHP", "FAST", "FANG", "EA", "FER", "XEL", "EXC", "ODFL", "TTWO",
    "IDXX", "CCEP", "KDP", "ADSK", "MSTR", "PYPL", "ALNY", "PAYX", "TRI", "AXON",
    "ROP", "WDAY", "DXCM", "CPRT", "GEHC", "KHC", "VRSK", "INSM", "CTSH", "ZS", "CHTR",
]


def get_nasdaq100_symbols() -> list[str]:
    """나스닥-100 구성종목 티커 목록.

    FMP 무료 플랜은 지수 구성종목 엔드포인트(nasdaq_constituent)가 막혀 있어(HTTP 403)
    위 정적 목록을 사용한다. 분기 리밸런싱(3/6/9/12월) 때만 목록을 갱신하면 된다.
    유료 FMP 플랜이라면 환경변수 USE_FMP_CONSTITUENT=1 을 주면 API로 자동 조회한다.
    """
    if os.environ.get("USE_FMP_CONSTITUENT") == "1":
        try:
            data = _get_json("nasdaq_constituent")
            syms = [row["symbol"] for row in data if row.get("symbol")]
            if syms:
                return sorted(set(syms))
        except Exception as e:  # noqa: BLE001
            print(f"[경고] FMP 구성종목 조회 실패 → 정적 목록 사용: {e}", file=sys.stderr)
    return list(NASDAQ100_SYMBOLS)


def get_quotes(symbols: list[str]) -> dict[str, dict]:
    """각 종목의 시세/PER. stable quote는 종목당 1콜(symbol= 쿼리). 실패 종목은 건너뜀.

    PER(pe)이 없으면 price/eps로 보정한다. 100종목이면 ~100콜(무료 250/일 한도 내).
    """
    out: dict[str, dict] = {}
    for sym in symbols:
        try:
            data = _get_json("quote", {"symbol": sym})
        except Exception as e:  # noqa: BLE001
            print(f"[경고] {sym} 시세 조회 실패: {e}", file=sys.stderr)
            continue
        row = (data[0] if isinstance(data, list) and data
               else data if isinstance(data, dict) else None)
        if not row:
            continue
        pe = row.get("pe")
        if pe is None:
            eps, price = row.get("eps"), row.get("price")
            try:
                if eps and price:
                    pe = float(price) / float(eps)
            except (TypeError, ValueError, ZeroDivisionError):
                pe = None
        row["pe"] = pe
        out[sym] = row
    return out


def get_history(symbol: str, days: int = HISTORY_DAYS) -> pd.DataFrame:
    """일봉 종가 시계열(과거→현재 오름차순). 컬럼: date(index), close.

    stable: historical-price-eod/full?symbol=&from=&to=  → 보통 배열을 직접 반환.
    (v3 호환 위해 {"historical":[...]} 형태도 처리)
    """
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=int(days * 1.6) + 10)  # 주말/휴장 감안 여유
    data = _get_json(
        "historical-price-eod/full",
        {"symbol": symbol, "from": start.isoformat(), "to": end.isoformat()},
    )
    hist = data.get("historical", []) if isinstance(data, dict) else (data or [])
    if not hist:
        return pd.DataFrame(columns=["close"])
    df = pd.DataFrame(hist)
    if "date" not in df.columns or "close" not in df.columns:
        return pd.DataFrame(columns=["close"])
    df = df[["date", "close"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df.dropna(subset=["close"])


# ------------------------- 지표 계산 ----------------------------
def _rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Wilder 평활 = ewm(alpha=1/period, adjust=False)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi


def _macd(close: pd.Series):
    ema_fast = close.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=MACD_SLOW, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=MACD_SIGNAL, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist


def _pct_change(close: pd.Series, periods: int) -> float:
    """periods 거래일 전 종가 대비 변동률(%). 데이터 부족 시 NaN."""
    if len(close) > periods:
        prev = close.iloc[-1 - periods]
        if prev and not pd.isna(prev):
            return (float(close.iloc[-1]) / float(prev) - 1.0) * 100.0
    return float("nan")


def compute_indicators(df: pd.DataFrame) -> dict | None:
    """가장 최근 시점의 추세 지표 묶음을 반환. 데이터 부족 시 None 가능 필드는 NaN."""
    if df.empty:
        return None
    close = df["close"]
    last = float(close.iloc[-1])

    ma = {w: (close.rolling(w).mean().iloc[-1] if len(close) >= w else np.nan) for w in MA_WINDOWS}
    rsi_series = _rsi(close)
    rsi_val = float(rsi_series.iloc[-1]) if not np.isnan(rsi_series.iloc[-1]) else np.nan
    macd, signal, hist = _macd(close)
    macd_val, sig_val, hist_val = float(macd.iloc[-1]), float(signal.iloc[-1]), float(hist.iloc[-1])

    # 골든/데드크로스: 최근 5거래일 내 MA50 ↔ MA200 교차 여부
    cross = None
    if len(close) >= 205:
        ma50 = close.rolling(50).mean()
        ma200 = close.rolling(200).mean()
        diff = (ma50 - ma200).dropna()
        if len(diff) >= 6:
            recent = np.sign(diff.iloc[-6:])
            if (recent.iloc[0] < 0) and (recent.iloc[-1] > 0):
                cross = "golden"
            elif (recent.iloc[0] > 0) and (recent.iloc[-1] < 0):
                cross = "death"

    return {
        "price": last,
        "ma20": ma[20], "ma50": ma[50], "ma200": ma[200],
        "above_ma200": (not np.isnan(ma[200])) and last > ma[200],
        "rsi": rsi_val,
        "macd": macd_val, "macd_signal": sig_val, "macd_hist": hist_val,
        "macd_up": hist_val > 0,            # MACD가 시그널선 위(상승 모멘텀)
        "cross": cross,                      # 'golden' / 'death' / None
        "chg_1d": _pct_change(close, 1),     # 일간(전 거래일 대비)
        "chg_1w": _pct_change(close, 5),     # 주간(약 5거래일)
        "chg_1m": _pct_change(close, 21),    # 월간(약 21거래일)
    }


# --------------------- 명단 편입/이탈 비교 -----------------------
def load_prev_list() -> list[str]:
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f).get("symbols", [])
    except Exception:  # noqa: BLE001
        return []


def save_curr_list(symbols: list[str]) -> None:
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump({"symbols": symbols, "saved_at": datetime.now(KST).isoformat()}, f, ensure_ascii=False)
    except Exception as e:  # noqa: BLE001
        print(f"[경고] 상태 파일 저장 실패(편입/이탈 비교는 다음 실행부터): {e}", file=sys.stderr)


# ------------------------- 포맷 헬퍼 ----------------------------
def _pct(x) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.2f}%"


def _money(x) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"${x:,.2f}"


def _parse_change_pct(q: dict) -> float:
    v = q.get("changesPercentage")
    if v is None:
        return np.nan
    if isinstance(v, str):
        v = v.replace("%", "").replace("+", "").strip()
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _rsi_zone(r) -> str:
    if r is None or np.isnan(r):
        return ""
    if r >= 70:
        return "과매수"
    if r <= 30:
        return "과매도"
    return ""


# ------------------------- 본문 생성 ----------------------------
def build_rows(symbols: list[str], quotes: dict) -> list[dict]:
    rows = []
    for rank, sym in enumerate(symbols, start=1):
        q = quotes.get(sym, {})
        ind = compute_indicators(get_history(sym)) or {}
        rows.append({
            "rank": rank,
            "symbol": sym,
            "name": q.get("name", sym),
            "per": q.get("pe"),
            "price": ind.get("price", q.get("price")),
            "chg_1d": ind.get("chg_1d"),
            "chg_1w": ind.get("chg_1w"),
            "chg_1m": ind.get("chg_1m"),
            "above_ma200": ind.get("above_ma200"),
            "ma200": ind.get("ma200"),
            "rsi": ind.get("rsi"),
            "macd_up": ind.get("macd_up"),
            "cross": ind.get("cross"),
        })
    return rows


def build_email(rows: list[dict], new_in: list[str], dropped: list[str], asof: str):
    chgs = [r["chg_1d"] for r in rows if r["chg_1d"] is not None and not np.isnan(r["chg_1d"])]
    avg_chg = float(np.mean(chgs)) if chgs else float("nan")
    above_cnt = sum(1 for r in rows if r["above_ma200"])

    subject = f"[나스닥100 PER 최저 {len(rows)}] {asof} · 일간평균 {_pct(avg_chg)} · MA200↑ {above_cnt}/{len(rows)}"

    # ---------- HTML ----------
    def ma200_badge(r):
        if r["above_ma200"] is None:
            return '<span style="color:#999">—</span>'
        return ('<span style="color:#15803d">▲ 위</span>' if r["above_ma200"]
                else '<span style="color:#b91c1c">▼ 아래</span>')

    def chg_html(c):
        if c is None or np.isnan(c):
            return '<span style="color:#999">—</span>'
        color = "#15803d" if c >= 0 else "#b91c1c"
        return f'<span style="color:{color}">{c:+.2f}%</span>'

    def rsi_html(r):
        v, z = r["rsi"], _rsi_zone(r["rsi"])
        if v is None or np.isnan(v):
            return '<span style="color:#999">—</span>'
        color = "#b45309" if z else "#374151"
        ztxt = f' <span style="font-size:11px;color:#b45309">{z}</span>' if z else ""
        return f'<span style="color:{color}">{v:.0f}</span>{ztxt}'

    def macd_html(r):
        if r["macd_up"] is None:
            return "—"
        return ('<span style="color:#15803d">＋ 상승</span>' if r["macd_up"]
                else '<span style="color:#b91c1c">－ 하락</span>')

    trs = []
    for r in rows:
        per = "—" if r["per"] in (None, "") else f"{float(r['per']):.1f}"
        trs.append(
            "<tr>"
            f'<td style="padding:8px 6px;text-align:center;color:#6b7280">{r["rank"]}</td>'
            f'<td style="padding:8px 6px"><b>{r["symbol"]}</b>'
            f'<div style="font-size:11px;color:#9ca3af">{r["name"][:24]}</div></td>'
            f'<td style="padding:8px 6px;text-align:right;font-weight:600">{per}</td>'
            f'<td style="padding:8px 6px;text-align:right">{_money(r["price"])}</td>'
            f'<td style="padding:8px 6px;text-align:right">{chg_html(r["chg_1d"])}</td>'
            f'<td style="padding:8px 6px;text-align:right">{chg_html(r["chg_1w"])}</td>'
            f'<td style="padding:8px 6px;text-align:right">{chg_html(r["chg_1m"])}</td>'
            f'<td style="padding:8px 6px;text-align:center">{ma200_badge(r)}</td>'
            f'<td style="padding:8px 6px;text-align:center">{rsi_html(r)}</td>'
            f'<td style="padding:8px 6px;text-align:center">{macd_html(r)}</td>'
            "</tr>"
        )

    def chips(items, color):
        if not items:
            return '<span style="color:#9ca3af">없음</span>'
        return " ".join(
            f'<span style="background:{color};color:#fff;border-radius:4px;padding:1px 6px;font-size:12px;margin-right:4px">{s}</span>'
            for s in items
        )

    crosses = [r for r in rows if r["cross"]]
    cross_note = ""
    if crosses:
        items = "".join(
            f'<li>{r["symbol"]}: '
            + ("골든크로스(50일선이 200일선 상향 돌파)" if r["cross"] == "golden"
               else "데드크로스(50일선이 200일선 하향 돌파)")
            + " — 최근 5거래일 내</li>"
            for r in crosses
        )
        cross_note = (
            '<div style="margin-top:14px;font-size:13px;color:#374151">'
            '<b>📌 추세 전환 신호</b><ul style="margin:6px 0 0;padding-left:18px">'
            f'{items}</ul></div>'
        )

    html = f"""\
<div style="font-family:-apple-system,'Segoe UI',Roboto,'Apple SD Gothic Neo',sans-serif;max-width:680px;margin:0 auto;color:#111827">
  <h2 style="margin:0 0 4px">📉 나스닥100 · PER 최저 {len(rows)} 종목</h2>
  <div style="color:#6b7280;font-size:13px;margin-bottom:14px">{asof} 마감 기준 · 적자/무수익(PER≤0)은 제외</div>

  <table style="width:100%;border-collapse:collapse;font-size:13px;background:#fff">
    <tr style="background:#f9fafb;color:#6b7280;font-size:12px;text-align:center">
      <td style="padding:8px 6px">#</td>
      <td style="padding:8px 6px;text-align:left">종목</td>
      <td style="padding:8px 6px;text-align:right">PER</td>
      <td style="padding:8px 6px;text-align:right">종가</td>
      <td style="padding:8px 6px;text-align:right">1일</td>
      <td style="padding:8px 6px;text-align:right">1주</td>
      <td style="padding:8px 6px;text-align:right">1개월</td>
      <td style="padding:8px 6px">vs 200일선</td>
      <td style="padding:8px 6px">RSI(14)</td>
      <td style="padding:8px 6px">MACD</td>
    </tr>
    {''.join(trs)}
  </table>

  <div style="margin-top:14px;font-size:13px;line-height:1.7">
    <div>평균 일간 등락률 <b>{_pct(avg_chg)}</b> · 200일선 위 <b>{above_cnt}/{len(rows)}</b> 종목</div>
    <div>신규 편입: {chips(new_in, "#2563eb")}</div>
    <div>명단 이탈: {chips(dropped, "#6b7280")}</div>
  </div>
  {cross_note}

  <div style="margin-top:18px;font-size:11px;color:#9ca3af;border-top:1px solid #eee;padding-top:8px">
    데이터: Financial Modeling Prep · 지표는 일봉 종가 기준 자동 계산값이며 투자 권유가 아닙니다.
  </div>
</div>"""

    # ---------- 텍스트(폴백) ----------
    lines = [f"나스닥100 PER 최저 {len(rows)} — {asof} 마감 (PER<=0 제외)",
             "변동률은 [1일 / 1주 / 1개월] 순", ""]
    for r in rows:
        per = "—" if r["per"] in (None, "") else f"{float(r['per']):.1f}"
        ma = "MA200▲" if r["above_ma200"] else ("MA200▼" if r["above_ma200"] is not None else "MA200—")
        rsi = "—" if (r["rsi"] is None or np.isnan(r["rsi"])) else f"RSI{r['rsi']:.0f}"
        macd = "MACD+" if r["macd_up"] else ("MACD-" if r["macd_up"] is not None else "MACD—")
        chg3 = f"{_pct(r['chg_1d'])} / {_pct(r['chg_1w'])} / {_pct(r['chg_1m'])}"
        lines.append(f"{r['rank']:>2}. {r['symbol']:<6} PER {per:<5} {_money(r['price']):>10}  [{chg3}]  {ma} {rsi} {macd}")
    lines += ["", f"평균 등락 {_pct(avg_chg)} · 200일선 위 {above_cnt}/{len(rows)}"]
    lines.append(f"신규 편입: {', '.join(new_in) or '없음'} / 이탈: {', '.join(dropped) or '없음'}")
    if crosses:
        lines.append("추세 전환: " + ", ".join(f"{r['symbol']}({'골든' if r['cross']=='golden' else '데드'})" for r in crosses))
    text = "\n".join(lines)

    return subject, html, text


# ------------------------- 오케스트레이션 ------------------------
def generate_report():
    """전체 파이프라인. (subject, html, text, rows) 반환 → B단계에서 발송에 사용."""
    asof = datetime.now(KST).strftime("%Y-%m-%d")

    universe = get_nasdaq100_symbols()
    quotes = get_quotes(universe)

    # PER<=0 / 결측 제외 후 오름차순 정렬, 하위 N개
    candidates = []
    for sym in universe:
        pe = quotes.get(sym, {}).get("pe")
        try:
            pe = float(pe)
        except (TypeError, ValueError):
            continue
        if pe > 0:
            candidates.append((sym, pe))
    candidates.sort(key=lambda x: x[1])
    picked = [s for s, _ in candidates[:TOP_N]]

    prev = load_prev_list()
    new_in = [s for s in picked if s not in prev]
    dropped = [s for s in prev if s not in picked]

    rows = build_rows(picked, quotes)
    subject, html, text = build_email(rows, new_in, dropped, asof)

    save_curr_list(picked)
    return subject, html, text, rows


def send_email(subject: str, html: str, text: str) -> None:
    """SMTP로 메일 발송. 자격증명은 환경변수(또는 GitHub Secrets)에서만 읽는다."""
    host   = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    port   = int(os.environ.get("SMTP_PORT", "465"))
    user   = os.environ.get("SMTP_USER", "").strip()
    pw     = os.environ.get("SMTP_PASS", "").strip()
    to     = os.environ.get("EMAIL_TO", user).strip()
    sender = os.environ.get("EMAIL_FROM", user).strip()
    if not (user and pw and to):
        raise RuntimeError("SMTP_USER / SMTP_PASS / EMAIL_TO 환경변수가 필요합니다.")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to
    msg.attach(MIMEText(text, "plain", "utf-8"))
    msg.attach(MIMEText(html, "html", "utf-8"))

    recipients = [a.strip() for a in to.split(",") if a.strip()]
    ctx = ssl.create_default_context()
    with smtplib.SMTP_SSL(host, port, context=ctx) as s:
        s.login(user, pw)
        s.sendmail(sender, recipients, msg.as_string())


def main():
    subject, html, text, rows = generate_report()
    os.makedirs("output", exist_ok=True)
    with open("output/email.html", "w", encoding="utf-8") as f:
        f.write(html)
    with open("output/email.txt", "w", encoding="utf-8") as f:
        f.write(text)
    with open("output/subject.txt", "w", encoding="utf-8") as f:
        f.write(subject)
    print("제목:", subject)
    print("-" * 60)
    print(text)
    print("-" * 60)

    # SMTP 설정이 있으면 발송, 없으면 파일만 생성(로컬 미리보기용)
    if os.environ.get("SMTP_USER") and os.environ.get("EMAIL_TO"):
        send_email(subject, html, text)
        print("✅ 이메일 발송 완료 →", os.environ.get("EMAIL_TO"))
    else:
        print("(SMTP 미설정: 파일만 생성했습니다. 메일까지 받으려면 "
              "SMTP_USER/SMTP_PASS/EMAIL_TO 를 설정하세요)")


if __name__ == "__main__":
    main()
