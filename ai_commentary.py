#!/usr/bin/env python3
"""
ai_commentary.py  —  S&P500 데일리 리포트용 'AI 해석 레이어'

설계 원칙 (가장 중요)
  · 숫자는 코드가, 서술은 AI가.
    - 수치(지표·수익률·PER 등)는 sp500_daily_report.py 가 계산한 값만 AI에 '입력'으로 넘긴다.
    - AI 는 새 수치를 만들지 않는다(환각 차단). 해석·요약·교차검증(서술)만 담당.
  · 결정론: temperature=0, 모델 고정. 같은 입력이면 거의 같은 출력.
  · 안전한 폴백: anthropic 미설치 / 키 없음 / 호출 실패 시 예외를 던지지 않고
    빈 dict({}) 를 반환한다. 호출부는 값이 없으면 기존 정적 텍스트를 그대로 쓰면 된다.

제공 기능 (원하신 3가지: 총평 + 뉴스 + 검증)
  1) market_overview : 그날 시세/레짐/탐욕지수 기반 시황 총평 (정적 sector_briefings 대체)
  2) rationales      : 종목별 '왜 지금' 근거 + 최근 뉴스 요약 + 호재/악재 라벨
  3) warnings        : 규칙 점수가 뽑은 종목에 대한 AI 교차검증(모순·리스크 플래그)

공개 API
  build_ai_layer(picks, ind_map, regime, news_by_sym=None, sector_briefs=None) -> dict
      반환: {} (비활성/실패)  또는
            {"market_overview": str,
             "rationales": {sym: {"why": str, "news": str, "flag": "호재"|"악재"|"중립"|None}},
             "warnings":   [{"sym": str, "note": str}]}

환경변수
  ANTHROPIC_API_KEY   필수(없으면 폴백)
  AI_ENABLED          "1"(기본) / "0" 이면 강제 비활성
  AI_MODEL            기본 "claude-sonnet-4-6"
  AI_MAX_PICKS        AI에 넘길 최대 종목 수(기본 12) — 비용/지연 통제
  AI_TIMEOUT          호출 타임아웃 초(기본 40)
"""
from __future__ import annotations

import os
import re
import sys
import json
import math

import shutil
import subprocess

# anthropic SDK 는 '있으면 사용, 없으면 폴백'. import 실패가 전체를 막지 않게 한다.
try:
    import anthropic
except Exception:
    anthropic = None


# ----------------------------- 설정 -----------------------------
# AI_BACKEND:
#   "api" (기본) — Anthropic API(ANTHROPIC_API_KEY 필요, 종량 과금)
#   "cli"        — 로컬 Claude Code CLI(`claude -p`) 호출. Pro/Max '구독'으로 동작, API 키 불필요.
#                  단, 이 코드를 실행하는 머신에 Claude Code 가 설치·로그인돼 있어야 함(=내 PC).
AI_BACKEND    = os.environ.get("AI_BACKEND", "api").strip().lower()
CLAUDE_BIN    = os.environ.get("CLAUDE_BIN", "claude")   # PATH 에 없으면 절대경로 지정
AI_MODEL      = os.environ.get("AI_MODEL", "claude-haiku-4-5")   # 해석 레이어는 저가 모델로 충분
AI_MAX_PICKS  = int(os.environ.get("AI_MAX_PICKS", "12"))
AI_TIMEOUT    = float(os.environ.get("AI_TIMEOUT", "120"))


def _enabled() -> bool:
    if os.environ.get("AI_ENABLED", "1") != "1":
        return False
    if AI_BACKEND == "cli":
        # 구독 경로: claude 실행파일이 있어야 함(키 불필요)
        return shutil.which(CLAUDE_BIN) is not None or os.path.exists(CLAUDE_BIN)
    # api 경로: SDK + 키 필요
    if anthropic is None:
        return False
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    return True


def _log(msg: str):
    print(f"[AI] {msg}", file=sys.stderr)


def _isnan(x) -> bool:
    try:
        return x is None or (isinstance(x, float) and math.isnan(x))
    except Exception:
        return True


def _r(x, nd=1):
    """수치를 안전하게 반올림(없으면 None). AI 입력 JSON 을 깔끔히 유지."""
    if _isnan(x):
        return None
    try:
        return round(float(x), nd)
    except Exception:
        return None


# ------------------- 입력(컨텍스트) 빌더 -------------------
def _compact_indicator(sym: str, ind: dict) -> dict:
    """ind_map[sym] 에서 AI 가 해석에 쓸 '계산된 수치'만 추려 넘긴다.
    여기 들어간 값은 모두 코드가 계산한 사실이며, AI 는 이 값만 인용해야 한다."""
    return {
        "symbol": sym,
        "price": _r(ind.get("price"), 2),
        "rsi": _r(ind.get("rsi"), 0),
        "above_ma200": bool(ind.get("above_ma200")),
        "macd_up": bool(ind.get("macd_up")),
        "cross": ind.get("cross"),
        "vol_ann_pct": _r((ind.get("vol_ann") or float("nan")) * 100, 0),
        "ret_1w_pct": _r(ind.get("chg_1w"), 1),
        "ret_1m_pct": _r(ind.get("chg_1m"), 1),
        "ret_3m_pct": _r(ind.get("chg_3m"), 1),
        "ret_6m_pct": _r(ind.get("chg_6m"), 1),
        "ret_1y_pct": _r(ind.get("chg_1y"), 1),
    }


def _build_context(picks, ind_map, regime, news_by_sym, sector_briefs) -> dict:
    """AI 에 넘길 단일 JSON 컨텍스트. 숫자는 전부 여기서 확정한다."""
    syms = []
    for item in (picks or []):
        sym = item[0] if isinstance(item, (list, tuple)) else item
        if sym in (ind_map or {}):
            syms.append(sym)
    syms = syms[:AI_MAX_PICKS]

    stocks = []
    for sym in syms:
        row = _compact_indicator(sym, ind_map[sym])
        heads = (news_by_sym or {}).get(sym) or []
        # 헤드라인은 최근 5건까지, 제목만(원문 본문은 넘기지 않음 → 비용/저작권 통제)
        row["recent_headlines"] = [str(h)[:160] for h in heads[:5]]
        stocks.append(row)

    return {
        "market": {
            "spy_vs_ma200_pct": _r(regime.get("gap_pct"), 1),
            "risk_on": bool(regime.get("risk_on")),
            "fear_greed_score": _r(regime.get("fng_score"), 0),
            "fear_greed_rating": regime.get("fng_rating"),
        },
        "sector_briefings": sector_briefs or {},
        "stocks": stocks,
    }


# ------------------------- 프롬프트 -------------------------
_SYSTEM = (
    "당신은 한국 개인투자자에게 보내는 S&P500 데일리 메일의 '해석'을 담당하는 애널리스트다. "
    "엄격한 규칙을 따른다:\n"
    "1) 제공된 JSON 안의 수치만 사용한다. 새로운 숫자(가격·수익률·목표가 등)를 절대 만들지 않는다.\n"
    "2) 단정적 매수/매도 권유를 하지 않는다. '관찰/검토' 어조를 쓴다.\n"
    "3) 모든 출력은 한국어. 간결하게.\n"
    "4) recent_headlines 가 비어 있으면 뉴스 요약은 빈 문자열로 둔다(뉴스를 지어내지 않는다).\n"
    "5) 반드시 지정된 JSON 스키마 하나만 출력한다(코드블록·설명 없이)."
)

_INSTRUCTION = (
    "아래 CONTEXT(JSON)를 바탕으로 다음을 생성하라.\n"
    "- market_overview: 시장 국면 2~3문장. spy_vs_ma200_pct, fear_greed 로 추세/심리만 서술.\n"
    "- rationales: stocks 의 각 symbol 에 대해 why(왜 지금 관찰 대상인지, 지표 근거 1~2문장), "
    "news(recent_headlines 요약 1문장, 없으면 \"\"), flag(\"호재\"/\"악재\"/\"중립\" 중 뉴스 기준, 뉴스 없으면 null).\n"
    "- warnings: 지표끼리 또는 지표와 뉴스가 모순되는 종목만 골라 {symbol, note} 로 경고. 없으면 빈 배열.\n\n"
    "출력 스키마:\n"
    '{"market_overview": "...", '
    '"rationales": [{"symbol":"AAA","why":"...","news":"...","flag":"호재|악재|중립|null"}], '
    '"warnings": [{"symbol":"AAA","note":"..."}]}\n\n'
    "CONTEXT:\n"
)


def _extract_json(text: str) -> dict | None:
    """모델 출력에서 첫 번째 JSON 객체를 견고하게 추출."""
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    # 중괄호 균형으로 첫 객체 잘라내기
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except Exception:
                    return None
    return None


def _normalize(parsed: dict) -> dict:
    """모델 출력(JSON)을 호출부가 쓰기 쉬운 형태로 정규화."""
    out = {"market_overview": "", "rationales": {}, "warnings": []}
    if not isinstance(parsed, dict):
        return out

    mo = parsed.get("market_overview")
    if isinstance(mo, str):
        out["market_overview"] = mo.strip()

    for r in parsed.get("rationales") or []:
        if not isinstance(r, dict):
            continue
        sym = r.get("symbol")
        if not sym:
            continue
        flag = r.get("flag")
        if isinstance(flag, str) and flag.lower() in ("null", "none", ""):
            flag = None
        out["rationales"][sym] = {
            "why": (r.get("why") or "").strip(),
            "news": (r.get("news") or "").strip(),
            "flag": flag,
        }

    for w in parsed.get("warnings") or []:
        if isinstance(w, dict) and w.get("symbol") and w.get("note"):
            out["warnings"].append({"sym": w["symbol"], "note": (w["note"] or "").strip()})

    return out


# --------------------------- 공개 API ---------------------------
def build_ai_layer(picks, ind_map, regime,
                   news_by_sym: dict | None = None,
                   sector_briefs: dict | None = None) -> dict:
    """AI 해석 레이어 1회 생성. 실패/비활성 시 {} 반환(예외 없음)."""
    if not _enabled():
        _log("비활성(키 없음/미설치/AI_ENABLED=0) → 기존 정적 텍스트 사용.")
        return {}
    if not picks:
        return {}

    ctx = _build_context(picks, ind_map, regime, news_by_sym, sector_briefs)
    if not ctx["stocks"]:
        return {}

    try:
        if AI_BACKEND == "cli":
            text = _call_cli(ctx)
        else:
            text = _call_api(ctx)
        if not text:
            _log("빈 응답 → 폴백.")
            return {}
        parsed = _extract_json(text)
        if parsed is None:
            _log("응답 JSON 파싱 실패 → 폴백.")
            return {}
        result = _normalize(parsed)
        _log(f"[{AI_BACKEND}] 생성 완료: 총평 {len(result['market_overview'])}자 · "
             f"근거 {len(result['rationales'])}종목 · 경고 {len(result['warnings'])}건.")
        return result
    except Exception as e:
        _log(f"호출 실패({type(e).__name__}: {e}) → 폴백.")
        return {}


def _call_api(ctx: dict) -> str:
    """Anthropic API 백엔드(종량 과금)."""
    client = anthropic.Anthropic(timeout=AI_TIMEOUT)
    msg = client.messages.create(
        model=AI_MODEL,
        max_tokens=2000,
        temperature=0,
        system=_SYSTEM,
        messages=[{"role": "user",
                   "content": _INSTRUCTION + json.dumps(ctx, ensure_ascii=False)}],
    )
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


def _call_cli(ctx: dict) -> str:
    """로컬 Claude Code CLI(`claude -p`) 백엔드 — Pro/Max '구독'으로 동작(API 키 불필요).
    시스템 지시는 단일 프롬프트에 합쳐 버전 차이에 견고하게. JSON 엔벨로프/평문 모두 처리."""
    prompt = (_SYSTEM + "\n\n" + _INSTRUCTION + json.dumps(ctx, ensure_ascii=False))
    # 프롬프트를 argv 로 넘기면 Windows 명령줄 길이 제한(WinError 206)에 걸린다 → stdin 으로 전달.
    cmd = [CLAUDE_BIN, "-p", "--output-format", "json"]
    # 한국어 Windows 기본 인코딩(cp949)이 claude 의 UTF-8 출력을 못 읽어 깨진다 → UTF-8 강제.
    proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=AI_TIMEOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"claude CLI rc={proc.returncode}: {(proc.stderr or '')[:200]}")
    out = (proc.stdout or "").strip()
    # --output-format json 은 {"result":"<응답텍스트>", "is_error":bool, ...} 엔벨로프를 준다.
    try:
        env = json.loads(out)
        if isinstance(env, dict):
            if env.get("is_error"):   # 미로그인/한도초과 등 → 폴백(에러문구를 본문으로 쓰지 않음)
                raise RuntimeError(f"claude CLI is_error: {str(env.get('result'))[:160]}")
            if isinstance(env.get("result"), str):
                return env["result"]
    except json.JSONDecodeError:
        pass
    return out   # 평문으로 떨어진 경우 그대로(이후 _extract_json 이 처리)


# --------- 2단계(phase) 핸드오프: Cowork 스케줄 태스크에서 'Claude가 해석가' ---------
def dump_context(path, picks, ind_map, regime,
                 news_by_sym: dict | None = None,
                 sector_briefs: dict | None = None) -> bool:
    """phase1: AI 입력 컨텍스트(숫자 확정본)를 JSON 파일로 저장.
    스케줄 태스크의 Claude 가 이 파일만 읽고 해석을 작성한다(새 숫자 생성 금지). 대상 없으면 False."""
    ctx = _build_context(picks, ind_map, regime, news_by_sym, sector_briefs)
    if not ctx["stocks"]:
        return False
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ctx, f, ensure_ascii=False, indent=2)
    return True


def load_ai_result(path) -> dict:
    """phase2: Claude 가 작성한 해석 JSON 파일을 읽어 _AI 형태로 정규화. 실패 시 {} (폴백)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read()
        parsed = _extract_json(txt)
        return _normalize(parsed) if parsed else {}
    except Exception as e:
        _log(f"ai-json 로드 실패({e}) → 폴백.")
        return {}


# --------------------- 뉴스 헤드라인 수집(헬퍼) ---------------------
# yfinance .news가 실제 속보(로이터·PR Newswire 등)와 분석·의견 기사(Zacks·Trefis 등 "5 Stocks
# to Buy", "Is X a Buy?" 류)를 구분 없이 섞어 준다 — 후자는 "주가를 움직인 뉴스"가 아니라
# 사후 논평이라 걸러낸다(2026-07-15, 지호 님 리포트 — "이건 새 소식이 아니라 분석 기사").
_ANALYSIS_PUBLISHERS = {
    "zacks", "motley fool", "the motley fool", "trefis", "24/7 wall st.", "24/7 wall st",
    "simply wall st", "simply wall st.", "barchart", "investorplace", "tipranks",
    "insider monkey", "validea", "gurufocus", "seeking alpha", "benzinga insights",
    "smarter analyst", "insidermonkey.com",
}
_ANALYSIS_TITLE_RE = re.compile(
    r"\b(is\s+\S+\s+(a\s+)?(buy|sell|good stock)|buy,?\s*sell\s*or\s*hold|"
    r"\d+\s+(stocks?|reasons?|things?)\b|should you (buy|sell)|"
    r"what if\b|is it time to|here'?s why|trending stock)", re.IGNORECASE)


def _is_analysis_headline(title: str, publisher: str) -> bool:
    if (publisher or "").strip().lower() in _ANALYSIS_PUBLISHERS:
        return True
    return bool(_ANALYSIS_TITLE_RE.search(title or ""))


def fetch_news_headlines(symbols: list[str], yf_module, max_items: int = 5) -> dict[str, list[str]]:
    """AI 입력용: 종목별 '최근 뉴스 제목' 리스트를 반환.
    기존 fetch_news_flags(True/False)와 달리 실제 제목을 모아 AI 가 호재/악재를 판단하게 한다.
    yfinance 신·구 스키마(title / content.title) 모두 처리. 실패는 조용히 빈 리스트.
    분석·의견성 기사(발행처 블록리스트 + 제목 패턴)는 제외 — "속보"만 남긴다.
    (호출부에서 from ai_commentary import fetch_news_headlines; fetch_news_headlines(syms, yf))"""
    out: dict[str, list[str]] = {}
    if yf_module is None:
        return {s: [] for s in symbols}
    for sym in symbols:
        titles: list[str] = []
        try:
            for it in (yf_module.Ticker(sym).news or []):
                t, pub = None, None
                if isinstance(it, dict):
                    t = it.get("title")
                    pub = it.get("publisher")
                    content = it.get("content") if isinstance(it.get("content"), dict) else None
                    if content:
                        t = t or content.get("title")
                        pub = pub or ((content.get("provider") or {}).get("displayName")
                                      if isinstance(content.get("provider"), dict) else None)
                if t and not _is_analysis_headline(str(t), pub):
                    titles.append(str(t))
                if len(titles) >= max_items:
                    break
        except Exception:
            titles = []
        out[sym] = titles
    return out


# ------------------------- 단독 실행(스모크 테스트) -------------------------
if __name__ == "__main__":
    demo_ind = {
        "AAPL": {"price": 210.5, "rsi": 58, "above_ma200": True, "macd_up": True,
                 "cross": None, "vol_ann": 0.24, "chg_1w": 1.2, "chg_1m": 3.4,
                 "chg_3m": 8.1, "chg_6m": 12.5, "chg_1y": 22.0},
    }
    demo_regime = {"gap_pct": 4.2, "risk_on": True, "fng_score": 62, "fng_rating": "탐욕"}
    demo_news = {"AAPL": ["Apple unveils new AI features at WWDC", "Analysts raise price target"]}
    res = build_ai_layer([("AAPL", 5.0, "데모")], demo_ind, demo_regime, demo_news)
    print(json.dumps(res, ensure_ascii=False, indent=2) if res
          else "[폴백] AI 비활성 — 키를 설정하면 실제 총평/근거/검증이 생성됩니다.")
