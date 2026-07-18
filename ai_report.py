#!/usr/bin/env python3
"""
ai_report.py — 2단계(검증→서술) AI 리포트 생성기. 비용 최소화 설계.

역할 분담 (STRATEGY.md §5 + 2026-07 비용 개편):
  · 규칙(코드)      = 후보 발굴 + '실행 계획' 확정.
                      매수 분할 가격·비율, 손절선, 관찰→매수 전환 조건, 매도 처분 계획은
                      전부 entry_plan.py 가 지표로 계산한다. AI는 이 숫자를 바꿀 수 없다.
  · 프로필 캐시     = 종목 사업 설명(②펀더멘털 축)은 sp500_profiles.json /
                      kospi200_profiles.json 에서 재사용(분기 1회 gen_profiles.py 로 생성).
                      매일 AI에게 "무슨 회사인지" 다시 묻지 않는다.
  · AI 1단계(검증)  = sonnet + web_search(≤REPORT_WEB_USES회): 후보별 최신 악재·촉매 점검,
                      verdict(유지/강등/제외) + 뉴스 한 줄. 출력은 초압축 JSON(토큰 절약).
                      ※ pregen.json(전날 밤 PC에서 구독 CLI로 생성)이 있으면 이 단계를
                        통째로 건너뛴다 → 검색 비용 0.
  · AI 2단계(서술)  = haiku(검색 없음): '최종 확정된' 종목만 대상으로 상세 서술
                      (summary + 4축 points + 실행 코멘트). 토큰 대부분이 저가 모델로 간다.
  · 코드(최종)      = verdict 반영해 최종 목록 확정. AI는 종목 '추가' 불가(할루시네이션 차단).

호출 경로:
  AI_BACKEND=api  → Anthropic API (GitHub Actions). 모델은 아래 env로 지정.
  AI_BACKEND=cli  → 로컬 claude -p (Pro 구독, PC에서 pregen.py 가 사용).
실패 사다리: 검증 실패 → 전원 '유지' 취급으로 서술만 / 서술 실패 → deterministic_report
  (프로필+계획 덕에 무AI여도 꽤 상세) → 발송은 절대 안 거른다.

모델 env (opus는 코드에서 차단):
  REPORT_MODEL_VERIFY  기본 claude-sonnet-5      — 검증(웹검색 필요 → 판단력 있는 모델)
  REPORT_MODEL_WRITE   기본 claude-haiku-4-5     — 서술(입력에 사실이 다 있음 → 저가 모델)
"""
from __future__ import annotations
import os, sys, json, shutil, subprocess

try:
    import anthropic
except Exception:
    anthropic = None
try:
    from ai_commentary import _extract_json  # noqa
except Exception:
    def _extract_json(t):
        try: return json.loads(t)
        except Exception: return None
import entry_plan as EP
import expectancy_report as EXR

AI_BACKEND      = os.environ.get("AI_BACKEND", "cli").strip().lower()
CLAUDE_BIN      = os.environ.get("CLAUDE_BIN", "claude")
REPORT_WEB_USES = int(os.environ.get("REPORT_WEB_USES", "4"))   # 8→4: 검색이 최대 비용원
REPORT_WEB      = os.environ.get("REPORT_WEB", "1") == "1"
AI_TIMEOUT      = float(os.environ.get("AI_TIMEOUT", "1200"))


def _no_opus(name: str, fallback: str) -> str:
    """opus 계열 모델 차단(사용자 정책: 비용). 지정돼도 sonnet/haiku 폴백."""
    return fallback if "opus" in (name or "").lower() else name

MODEL_VERIFY = _no_opus(os.environ.get("REPORT_MODEL_VERIFY",
                        os.environ.get("REPORT_MODEL", "claude-sonnet-5")), "claude-sonnet-5")
MODEL_WRITE  = _no_opus(os.environ.get("REPORT_MODEL_WRITE", "claude-haiku-4-5"), "claude-haiku-4-5")

# 최종 채택 수(코드가 확정) — 관찰 폐지(2026-07-13): 관찰 슬롯을 매수 후보로 전환.
# 미국 4+4 → 매수 8, 한국 3+2 → 매수 5. AI 강등분은 관찰 대신 '제외된 후보' 박스에 사유 표기.
# 2026-07-17(지호 님 요청): 미국 추천폭을 8→10으로 확대(판단의 폭을 넓히기 위해) — 보유
# 상한(daily_ai_report.py US_MAX_HOLD)은 8로 그대로 둠, 추천(추천풀)과 실제 편입은 별개.
FINAL_BUY      = int(os.environ.get("REPORT_FINAL_BUY", "10"))
FINAL_WATCH    = int(os.environ.get("REPORT_FINAL_WATCH", "0"))
KR_FINAL_BUY   = int(os.environ.get("KR_FINAL_BUY", "5"))
KR_FINAL_WATCH = int(os.environ.get("KR_FINAL_WATCH", "0"))
MIN_BUY        = 3


def _log(m): print(f"[REPORT] {m}", file=sys.stderr)


def _enabled():
    if os.environ.get("AI_ENABLED", "1") != "1":
        return False
    if AI_BACKEND == "cli":
        return shutil.which(CLAUDE_BIN) is not None or os.path.exists(CLAUDE_BIN)
    return anthropic is not None and bool(os.environ.get("ANTHROPIC_API_KEY"))


# ------------------------- 프로필 캐시 -------------------------
_PROFILES = None

def _profiles() -> dict:
    """{sym: "한줄 + 상세"} — sp500_profiles.json(tickers) + kospi200_profiles.json.
    gen_profiles.py 가 분기 1회 채운다. 없어도 동작(빈 문자열)."""
    global _PROFILES
    if _PROFILES is not None:
        return _PROFILES
    out = {}
    try:
        d = json.load(open("sp500_profiles.json", encoding="utf-8"))
        for sym, t in (d.get("tickers") or {}).items():
            txt = " ".join(x for x in (t.get("one_liner"), t.get("detail")) if x).strip()
            if txt:
                out[sym] = txt
    except Exception:
        pass
    try:
        d = json.load(open("kospi200_profiles.json", encoding="utf-8"))
        for sym, t in (d.get("tickers") or {}).items():
            txt = " ".join(x for x in (t.get("one_liner"), t.get("detail")) if x).strip()
            if txt:
                out[sym] = txt
    except Exception:
        pass
    _PROFILES = out
    return out


_PROFILE_PARTS = None


def _profile_parts() -> dict:
    """{sym: (one_liner, detail)} — _profiles()와 별도 유지: one_liner(브랜드 인지용,
    ①종목 설명)와 detail(사업구조·전략, ②사업)을 서로 겹치지 않게 화면에 나눠 쓰기 위함
    (2026-07-15 — 지호 님 피드백: 종목설명·②사업이 같은 문장을 반복하고 있었음)."""
    global _PROFILE_PARTS
    if _PROFILE_PARTS is not None:
        return _PROFILE_PARTS
    out = {}
    for path in ("sp500_profiles.json", "kospi200_profiles.json"):
        try:
            d = json.load(open(path, encoding="utf-8"))
            for sym, t in (d.get("tickers") or {}).items():
                ol, det = (t.get("one_liner") or "").strip(), (t.get("detail") or "").strip()
                if ol or det:
                    out[sym] = (ol, det)
        except Exception:
            pass
    _PROFILE_PARTS = out
    return out


# ------------------------- 계획 주입(코드 확정) -------------------------
def attach_plans(groups: dict):
    """모든 후보에 entry_plan 결과를 붙인다(렌더·AI 컨텍스트 공용). 제자리 수정."""
    for key, krw in (("buy_now", False), ("kr_buy", True)):
        for c in groups.get(key) or []:
            c["plan"] = EP.buy_plan(c, krw=krw)
            c["plan_text"] = EP.plan_text(c["plan"])
    for key, krw in (("watch", False), ("kr_watch", True)):
        for c in groups.get(key) or []:
            c["trigger"] = EP.watch_trigger(c, krw=krw)
            c["plan"] = EP.buy_plan(c, krw=krw)          # 전환 시 쓸 분할 계획(표시용)
            c["plan_text"] = EP.plan_text(c["plan"])
    for s in groups.get("sells") or []:
        s["plan"] = EP.sell_plan(s)
    for s in groups.get("kr_sells") or []:
        s["plan"] = EP.sell_plan(s, krw=True)


# ------------------------- 1단계: 검증(sonnet + 웹검색) -------------------------
# 2026-07-16 재설계(Fable 5 자문): 이 필터는 "정량 팩터 랭킹을 신뢰하고 검증 가능한
# 개별종목 악재만 걸러내는 최소개입 거부권(veto)"이어야 한다 — 순위를 재구성하는
# 큐레이터가 아니다. 근거: 이 프로젝트의 2단계 재랭킹 백테스트에서 검증 안 된 재정렬은
# 전부 원래 팩터 신호보다 나빴다(STRATEGY.md §3 "2단계 재랭킹 검증"). 종목 간 비교·상대
# 우열 판단을 금지해야 그 결론과 충돌하지 않는다.
_V_SYSTEM = (
    "당신은 규칙 기반으로 선정된 주식 후보를 '최신 정보로 검증'하는 애널리스트다. 한국어로 답한다.\n"
    "임무: 각 후보의 심각한 악재(실적 쇼크·가이던스 하향·소송/규제·회계 이슈·공매도 리포트)와 "
    "다가올 촉매(실적 발표·신제품·이벤트)를 확인해 verdict를 부여한다.\n"
    "규칙:\n"
    "1) 웹검색 횟수가 제한된다. 종목별로 따로 검색하지 말고 섹터·공통 이슈로 묶어 검색하고, "
    "급등락 종목(1주 ±7% 이상)과 kind=sell 종목을 우선 확인한다.\n"
    "2) verdict은 종목마다 독립적으로 판단한다 — 다른 후보와 비교하거나 상대적으로 더/덜 "
    "매력적인지 순위를 매기지 않는다(정량 팩터 랭킹이 이미 그 역할을 함, 당신은 순위를 "
    "재조정하는 게 아니라 개별종목 결격사유만 확인).\n"
    "3) verdict은 둘 중 하나다(관찰·보류 같은 중간 등급 없음 — 애매하게 남겨두지 말고 결정할 것):\n"
    "   - '매수유지': 확인된 악재가 없거나, 하락이 그 종목만의 문제가 아니라 시장·업종 전반의 "
    "조정 때문임이 확인된 경우.\n"
    "   - '제외': 그 종목 고유의 검증된 부정적 정보 — 실적 쇼크·가이던스 하향·거버넌스 훼손"
    "(예: 자사주 매입 후 소각 대신 교환사채 발행 등으로 주주환원 기대를 저버림)·소송/규제·회계 "
    "이슈·공매도 리포트 등 매수 논거를 약화시키는 사안. 회계부정·상장폐지처럼 극단적인 사안만 "
    "해당하는 게 아니다 — '이걸 알고도 이 종목을 사겠는가'라는 질문에 아니라고 답하게 되는 "
    "사안이면 제외한다. 애매하면 매수유지 쪽으로 밀어붙이지 말고 제외한다(제외되면 다음 순위 "
    "후보가 자동으로 채워지므로 신중한 제외에 대한 부담이 없음). 사유 필수.\n"
    "3-1) verdict='제외'면 severity를 반드시 붙인다(매수유지면 빈칸). 이건 '얼마나 심각해 "
    "보이는가'라는 인상이 아니라 사건의 유형을 규칙대로 분류하는 것이다. 판별축 두 가지: "
    "(A)확정성 — 문서화된 사건(절차 개시·판결·공시·규제조치)인가, 아니면 주장·전망·의견·수급"
    "(애널리스트 의견·공매도 리포트·내부자 매도 등)인가. (B)비중 — 그 사건이 매출·기업가치의 "
    "대략 30% 이상을 차지하는 핵심 사업에 대한 것인가.\n"
    "   'severity=구조적'은 아래 목록에 해당하고 (A)(B)를 모두 충족할 때만 부여한다:\n"
    "   ① 회계부정·감사의견 거절/한정·재무제표 신뢰성 붕괴(대규모 정정, 규제기관 회계조사 착수 포함)\n"
    "   ② 파산·법정관리·워크아웃·상장폐지 절차 개시\n"
    "   ③ 규제기관의 핵심사업 금지·인허가 취소·강제 판매중단(조사 착수·경고장은 해당 안 됨)\n"
    "   ④ 핵심 제품·라이선스·특허·최대고객의 확정적 상실(비중 (B) 충족 시)\n"
    "   ⑤ 자기자본 대비 중대한(대략 20% 이상) 확정 판결·합의금(소송 '제기'는 해당 안 됨)\n"
    "   ⑥ 신용등급 투기등급 강등에 유동성 위험이 적시된 경우\n"
    "   'severity=일시적' = 그 외 전부. 특히 다음은 절대 구조적이 아니다: 한두 분기 실적 미스·"
    "가이던스 하향, 애널리스트 의견·목표가 하향, 공매도 리포트 등 제3자 주장(규제기관·감사인이 "
    "확인하기 전), 내부자 매도·외국인 매도세 등 수급, 섹터·거시·관세 역풍, 비핵심 파이프라인·"
    "단일 수주의 실패, 주가 급락 그 자체.\n"
    "   방향에 주의: verdict는 애매하면 '제외'지만 severity는 애매하면 '일시적'이다 — 구조적 "
    "판정은 보유 종목 매도로까지 이어질 수 있는 별도의 무거운 결정이므로, 문서화된 사실을 "
    "특정할 수 없으면 절대 구조적으로 올리지 않는다. 구조적이면 verdict_reason에 근거 사건과 "
    "날짜를 반드시 명시한다.\n"
    "4) 확인 안 된 내용은 쓰지 않는다. 수치를 지어내지 않는다.\n"
    "5) 출력은 지정된 JSON 하나만. 문장은 짧게(뉴스·촉매 각 한 줄).\n"
    "6) 종목 데이터에 prev_verdict(전날 판정)가 있으면 오늘 판정이 그것과 달라질 때만 "
    "change_reason에 '전날 판정 이후 새로 확인된 사실'을 날짜와 함께 명시한다(예: "
    "'7/16 실적발표에서 가이던스 상향'). prev_verdict와 같은 사실을 다르게 재해석했을 "
    "뿐이거나 오늘 검색에서 그 사실을 못 찾았을 뿐이면 판정을 바꾸지 말고 prev_verdict를 "
    "그대로 유지한다 — '새 정보 없이 뒤집기 금지'가 원칙. prev_verdict가 '제외'였고 "
    "prev_severity가 '구조적'이었다면, 복귀의 change_reason은 그 구조적 사건이 해소·정정·"
    "기각됐음을 보여주는 문서화된 사실이어야 한다(주가 반등·의견 상향은 근거가 아니다)."
)

_V_SCHEMA = (
    '{"market_overview":"전일 미국+세계 시장 1-2문장(제공된 market 수치+검색 근거)",\n'
    ' "macro":"환율·한국·코인 흐름 한 줄",\n'
    ' "signal_note":"제공된 신호 중 오늘 가장 중요한 변화 1-2문장(등급은 바꾸지 말 것)",\n'
    ' "risks":"이번 주 공통 리스크 1-2문장(FOMC/실적시즌/지정학 등 구체적으로)",\n'
    ' "stocks":[{"symbol":"AAA","verdict":"매수유지|제외",\n'
    '   "severity":"제외일 때만: 구조적|일시적(매수유지면 빈칸)",\n'
    '   "verdict_reason":"제외 사유(유지면 빈칸, 구조적이면 사건+날짜 필수)",\n'
    '   "change_reason":"전날과 판정이 다를 때만: 새로 확인된 사실+날짜(없으면 빈칸)",\n'
    '   "news":"최근 이슈 한 줄(없으면 빈칸)","catalyst":"다가올 이벤트/촉매 한 줄(없으면 빈칸)","flag":"호재|악재|중립"}]}'
)


def _normalize_severity(v: dict):
    """severity 페일세이프(2026-07-17, Fable 5 자문) — 모델이 태그를 빠뜨리거나 오타를
    내거나(구버전 pregen 캐시 포함) 값이 {구조적,일시적} 밖이면 무조건 '일시적'로 강제한다.
    태그 누락이 절대 매도 방아쇠(구조적) 쪽으로 새지 않게 하는 안전망 — 매수 게이트
    (_apply_verdicts)엔 영향 없음(제외면 severity 무관하게 후보에서 빠짐)."""
    if (v.get("verdict") or "").strip() != "제외":
        v["severity"] = ""
        return
    if (v.get("severity") or "").strip() != "구조적":
        v["severity"] = "일시적"


def _weekly_note(market: dict) -> str:
    """market['weekly_recap']=True(월요일)면 '어제' 대신 지난주 전체를 요약하도록 지시.
    코스피/코스닥은 금요일 이후 휴장이라 '전일'로 쓰면 주말 흐름이 누락된다."""
    return ("- 오늘은 월요일이다. market_overview·macro·risks는 '어제'가 아니라 지난 금요일 "
            "종가부터 오늘 아침까지 주말 포함 한 주간의 시장 흐름을 요약할 것(하루치가 아님).\n"
            if market.get("weekly_recap") else "")


def _v_stock(c, kind, prior=None):
    """검증 입력은 최소 필드만(토큰 절약). 판단에 필요한 것: 정체+최근 급등락+헤드라인.
    prior={"verdict","reason","date"}가 있으면 전날 판정을 같이 줘서(2026-07-17, 지호 님
    지적 — REGN/KCC/003030이 새 근거 없이 하루 만에 뒤집힘) 모델이 '새 정보 없이 뒤집기'를
    피하게 한다."""
    r = c.get("ret") or {}
    d = {"symbol": c.get("symbol"), "name": c.get("name"), "kind": kind,
         "sector": c.get("sector"), "ret_1w": r.get("1w"), "ret_1m": r.get("1m"),
         "hot": bool(c.get("hot")), "headlines": (c.get("headlines") or [])[:4]}
    if prior:
        d["prev_verdict"] = prior.get("verdict")
        d["prev_reason"] = prior.get("reason") or ""
        d["prev_date"] = prior.get("date")
        d["prev_severity"] = prior.get("severity") or ""
    return d


def verify_stage(groups, market) -> dict:
    """반환: {"by_sym":{sym:{verdict,verdict_reason,change_reason,news,catalyst,flag}},
             "market_overview","macro","signal_note","risks"}  실패 시 {}."""
    as_of = str(market.get("as_of") or "").strip()
    prior_us, prior_kr = {}, {}
    if as_of:
        try:
            import ai_verdict_log as AVL
            prior_us = {s: h[0] for s, h in AVL.history_by_symbol("us", as_of).items() if h}
            prior_kr = {s: h[0] for s, h in AVL.history_by_symbol("kr", as_of).items() if h}
        except Exception:
            pass
    stocks = ([_v_stock(c, "buy", prior_us.get(str(c.get("symbol")))) for c in groups.get("buy_now") or []]
              + [_v_stock(c, "watch", prior_us.get(str(c.get("symbol")))) for c in groups.get("watch") or []]
              + [_v_stock(c, "buy", prior_kr.get(str(c.get("symbol")))) for c in groups.get("kr_buy") or []]
              + [_v_stock(c, "watch", prior_kr.get(str(c.get("symbol")))) for c in groups.get("kr_watch") or []]
              + [{"symbol": s.get("symbol"), "name": s.get("name"), "kind": "sell",
                  "reason": s.get("reason")} for s in (groups.get("sells") or []) + (groups.get("kr_sells") or [])])
    instr = (
        "후보 목록을 검증하라. 각 symbol마다 stocks 항목 하나씩 반드시 출력.\n"
        + _weekly_note(market) +
        f"출력 스키마(JSON 하나만):\n{_V_SCHEMA}\n\n"
        f"CONTEXT.market = {json.dumps(market, ensure_ascii=False)}\n"
        f"CONTEXT.stocks = {json.dumps(stocks, ensure_ascii=False)}\n")
    try:
        try:
            text = (_call_cli(instr, REPORT_WEB, system=_V_SYSTEM, model=MODEL_VERIFY) if AI_BACKEND == "cli"
                    else _call_api(instr, REPORT_WEB, system=_V_SYSTEM, model=MODEL_VERIFY,
                                   max_tokens=3000, temperature=0))
        except Exception as e1:
            _log(f"검증 1차 실패({type(e1).__name__}) → 웹검색 없이 재시도")
            text = (_call_cli(instr, False, system=_V_SYSTEM, model=MODEL_VERIFY) if AI_BACKEND == "cli"
                    else _call_api(instr, False, system=_V_SYSTEM, model=MODEL_VERIFY,
                                   max_tokens=3000, temperature=0))
        p = _extract_json(text or "")
        if not isinstance(p, dict):
            return {}
        by = {}
        for r in p.get("stocks") or []:
            if isinstance(r, dict) and r.get("symbol"):
                _normalize_severity(r)
                by[str(r["symbol"])] = r
        return {"by_sym": by,
                "market_overview": (p.get("market_overview") or "").strip(),
                "macro": (p.get("macro") or "").strip(),
                "signal_note": (p.get("signal_note") or "").strip(),
                "risks": (p.get("risks") or "").strip()}
    except Exception as e:
        _log(f"검증 실패({type(e).__name__}: {e})"); return {}


# ------------------------- 2단계: 서술(haiku, 검색 없음) -------------------------
_W_SYSTEM = (
    "당신은 한국 개인투자자(수신자: 투자에 관심 있는 아버지)용 아침 주식 보고서의 서술을 쓰는 애널리스트다.\n"
    "규칙:\n"
    "1) 제공된 JSON의 수치·사실만 사용한다. 새 숫자·뉴스를 만들지 않는다.\n"
    "2) points는 정확히 3개, 순서대로 ①추세(지표 수치 인용) ②펀더멘털·사업(profile 요약+재무 수치) "
    "③촉매 또는 리스크(verified.catalyst, 없으면 주의점). 뉴스 항목은 넣지 않는다"
    "(2026-07-17부로 카드에서 뉴스 섹션 제외 — 관련 없는 기사가 붙는 문제로 보유종목 전용"
    " 별도 메일을 새로 설계할 때까지 보류).\n"
    "각 point는 구체 수치를 포함한 완결된 한 문장.\n"
    "2-1) summary는 '이 회사가 무슨 사업을 하는 회사인지'만 쓰는 회사 소개 문장이다. "
    "RSI·PE·ROE·수익률·%, '조정/급락/저평가/과열/바닥/반등/진입' 같은 시세·투자판단 표현은 "
    "summary에 절대 넣지 않는다(그 내용은 전부 points가 다룬다 — summary와 역할이 겹치면 안 됨). "
    "profile을 근거로 무엇을 만들어 파는 회사인지 구체적으로 1~2문장. 나쁜 예(투자근거 섞임): "
    "'도료·화학사로 PE 2.0(최저)·ROE 19.66%(최고)의 초우량 저평가에 1개월 -25% 조정 중'. "
    "좋은 예(사업 설명만): '건축 내외장용 도료와 자동차 도료, 방수·방음재 등 건축자재를 만들어 "
    "파는 회사다. 건설·자동차 업황에 실적이 연동된다.' 글자 수로 문장을 끊지 말고 자연스러운 "
    "문장 1~2개로 끝맺는다.\n"
    "3) plan(매수 계획)과 trigger(전환 조건)는 코드가 확정한 값이다 — 바꾸거나 새로 만들지 말 것. "
    "comment에는 뉴스·촉매를 반영한 실행 조언 한 줄만(예: '실적 발표 22일 전이라 2차분은 발표 후에').\n"
    "4) 쉬운 한국어. 한자 금지. 이동평균은 '20일선/50일선/200일선'. 단정적 수익 보장 금지.\n"
    "5) 출력은 지정된 JSON 하나만."
)

_W_SCHEMA = (
    '{"market_overview":"(요청된 경우만) 전일 시장 1-2문장","macro":"(요청된 경우만) 한 줄",\n'
    ' "signal_note":"(요청된 경우만) 1-2문장","risks":"(요청된 경우만) 1-2문장",\n'
    ' "stocks":[{"symbol":"AAA","name":"회사명","category":"세부분류(반도체는 팹리스·파운드리·메모리·장비 등)",\n'
    '   "summary":"무슨 사업을 하는 회사인지 1-2문장(투자 근거는 쓰지 말 것)","points":["①","②","③"],\n'
    '   "comment":"계획에 덧붙일 실행 조언 한 줄"}],\n'
    ' "sells":[{"symbol":"CCC","comment":"왜 지금 정리인지 한 줄(reason+뉴스 결합)"}]}'
)


def _w_stock(c, kind, vmap):
    """서술 입력: 지표+프로필+검증결과. closes 등 무거운 필드는 제외."""
    v = vmap.get(str(c.get("symbol"))) or {}
    r = c.get("ret") or {}
    gap200 = None
    if c.get("price") and c.get("ma200"):
        gap200 = round((c["price"] / c["ma200"] - 1) * 100, 1)
    d = {"symbol": c.get("symbol"), "name": c.get("name"), "kind": kind,
         "sector": c.get("sector"), "industry": c.get("industry"),
         "price": c.get("price"), "rsi": c.get("rsi"), "pe": c.get("pe"),
         "gap200_pct": gap200, "ret": r, "hot": bool(c.get("hot")),
         "roe": c.get("roe"), "rev_growth": c.get("rev_growth"),
         "profit_margin": c.get("profit_margin"),
         "profile": _profiles().get(str(c.get("symbol")), ""),
         # 2026-07-17: headlines·verified.news 제외(뉴스 섹션 전체 보류) — catalyst/flag만
         # 남김(호재·악재 색상칩·촉매 문구는 뉴스와 별개 용도).
         "verified": {k: v.get(k) for k in ("catalyst", "flag") if v.get(k)},
         "plan": c.get("plan_text") or ""}
    if kind == "watch":
        d["trigger"] = c.get("trigger") or ""
    return d


def write_stage(final_pairs, sells, market, vmap, need_market: bool) -> dict:
    """final_pairs=[(cand,kind)] 최종 확정 종목만 서술(풀 전체가 아님 → 토큰 절약).
    need_market=True면 시황 문장도 haiku가 작성(검증 단계를 건너뛴 pregen 경로)."""
    stocks = [_w_stock(c, kind, vmap) for c, kind in final_pairs]
    sell_in = [{"symbol": s.get("symbol"), "name": s.get("name"), "reason": s.get("reason"),
                "ret_pct": s.get("ret_pct"),
                "verified": (vmap.get(str(s.get("symbol"))) or {}).get("news", "")} for s in sells]
    mk = ("- market_overview/macro/signal_note/risks 도 작성하라(market 수치 근거).\n"
          if need_market else "- 시황 필드는 생략(이미 있음). stocks/sells만.\n")
    instr = (
        "각 종목의 상세 서술을 작성하라. symbol마다 stocks 항목 하나씩 반드시 출력.\n" + mk
        + _weekly_note(market) +
        f"출력 스키마(JSON 하나만):\n{_W_SCHEMA}\n\n"
        f"CONTEXT.market = {json.dumps(market, ensure_ascii=False)}\n"
        f"CONTEXT.stocks = {json.dumps(stocks, ensure_ascii=False)}\n"
        f"CONTEXT.sells = {json.dumps(sell_in, ensure_ascii=False)}\n")

    def _once():
        return (_call_cli(instr, False, system=_W_SYSTEM, model=MODEL_WRITE) if AI_BACKEND == "cli"
                else _call_api(instr, False, system=_W_SYSTEM, model=MODEL_WRITE, max_tokens=6000))

    # verify_stage와 동일하게 실패는 조용히 삼키지 않는다 — 이전엔 JSON 파싱 실패 시 아무 로그
    # 없이 빈 dict만 반환해 원인 파악이 안 됐다(예: 응답이 max_tokens에 잘려 중괄호가 안 닫힘).
    # 1회 재시도 후에도 실패하면 원문 일부를 로그에 남긴다.
    text = None
    for attempt in (1, 2):
        try:
            text = _once()
        except Exception as e:
            _log(f"서술 호출 실패(시도 {attempt}/2, {type(e).__name__}: {e})")
            continue
        p = _extract_json(text or "")
        if isinstance(p, dict) and p.get("stocks"):
            return p
        _log(f"서술 응답 파싱 실패(시도 {attempt}/2) — 원문 앞부분: {(text or '')[:200]!r}")
    return {}


# ------------------------- 2-b: 시황 4문장만(경량, 종목 JSON 없음) -------------------------
_M_SYSTEM = (
    "당신은 한국 개인투자자용 아침 주식 보고서의 '시황 총평만' 쓰는 애널리스트다.\n"
    "규칙: 1) 제공된 수치만 사용, 새 숫자·뉴스를 만들지 않는다. 2) 쉬운 한국어, 한자 금지. "
    "3) 각 필드 1~2문장. 4) 출력은 지정된 JSON 하나만."
)
_M_SCHEMA = (
    '{"market_overview":"전일 미국+세계 시장 1-2문장(제공된 수치 근거)",\n'
    ' "macro":"환율·한국·코인 흐름 한 줄",\n'
    ' "signal_note":"제공된 신호 중 오늘 가장 중요한 변화 1-2문장(등급은 바꾸지 말 것)",\n'
    ' "risks":"이번 주 공통 리스크 1-2문장"}')


def write_market_stage(market) -> dict:
    """시황 4문장만 작성(종목 JSON 없음 → 토큰 최소). pregen에 market_written이 없을 때
    (한국 메일 — 저녁 pregen 시점엔 미국장이 아직 개장 전이라 밤에 못 씀) 발송 시점에 쓰는
    경량 호출. 실패해도 빈 dict(필드가 비게 될 뿐, 발송은 계속됨)."""
    if not _enabled():
        return {}
    instr = (f"아래 수치로 시황 총평을 작성하라.\n" + _weekly_note(market) +
             f"출력 스키마(JSON 하나만):\n{_M_SCHEMA}\n\n"
             f"CONTEXT.market = {json.dumps(market, ensure_ascii=False)}\n")
    try:
        text = (_call_cli(instr, False, system=_M_SYSTEM, model=MODEL_WRITE) if AI_BACKEND == "cli"
                else _call_api(instr, False, system=_M_SYSTEM, model=MODEL_WRITE, max_tokens=600))
        p = _extract_json(text or "")
        return p if isinstance(p, dict) else {}
    except Exception as e:
        _log(f"시황 서술 실패({type(e).__name__}: {e})"); return {}


# ------------------------- verdict 사후추적 로그(2026-07-14) -------------------------
def _log_verdicts(buy_pool, kr_buy_pool, vmap, market):
    """검증 후보 전원(최종 채택 여부 무관)의 verdict+당일가를 ai_verdict_log.py에 기록.
    AI 검증이 실제로 이후 수익률과 상관 있는지 사후검증하기 위한 데이터 축적 —
    실패해도 리포트 파이프라인엔 영향 없게 무조건 흡수."""
    try:
        import ai_verdict_log as AVL
        date = str(market.get("as_of") or "").strip()
        if not date:
            return
        entries = []
        for pool, mkt in ((buy_pool, "us"), (kr_buy_pool, "kr")):
            for c in pool:
                sym = str(c.get("symbol"))
                v = vmap.get(sym) or {}
                entries.append({"date": date, "market": mkt, "symbol": sym,
                                "name": c.get("name", ""), "verdict": v.get("verdict") or "매수유지",
                                "severity": v.get("severity", ""),
                                "reason": v.get("verdict_reason", ""), "price": c.get("price")})
        AVL.log(entries)
    except Exception as e:
        _log(f"verdict 로그 기록 생략({type(e).__name__}: {e})")


STICKY_EXCLUDE_DAYS = int(os.environ.get("AI_STICKY_EXCLUDE_DAYS", "5"))
STICKY_EXCLUDE_DAYS_STRUCTURAL = int(os.environ.get("AI_STICKY_EXCLUDE_DAYS_STRUCTURAL", "20"))


def _apply_sticky_exclusion(vmap: dict, pool: list, market: str, as_of: str,
                            sticky_days=STICKY_EXCLUDE_DAYS,
                            sticky_days_structural=STICKY_EXCLUDE_DAYS_STRUCTURAL):
    """제외는 즉시, 복귀는 최소 sticky_days(거래일 근사) 저지 — 비대칭 히스테리시스
    (2026-07-17, Fable 5 자문). REGN·KCC(002380)·003030처럼 새 근거 없이 하루~며칠 만에
    제외→매수유지로 뒤집히는 걸 프롬프트 순응에 기대지 않고 코드 레벨에서 막는다.
    severity(구조적/일시적, 같은 날 도입)에 따라 점착 기간을 차등: 구조적(회계부정·상장폐지
    절차 등, verdict='제외')은 며칠 만에 해소되는 성질이 아니므로 20일, 일시적은 5일.
    오늘 change_reason이 채워져 있으면(모델이 새 정보를 인용) 점착을 걸지 않고 통과시킨다
    — '제외 자체를 못 풀게'가 아니라 '근거 없는 복귀만' 막는 게 목적. vmap은 제자리 수정."""
    if not as_of:
        return
    import datetime as _dt
    try:
        today = _dt.date.fromisoformat(as_of)
    except Exception:
        return
    import ai_verdict_log as AVL
    hist = AVL.history_by_symbol(market, as_of)
    for c in pool:
        sym = str(c.get("symbol"))
        v = vmap.get(sym)
        if not v or (v.get("verdict") or "").strip() == "제외":
            continue
        if (v.get("change_reason") or "").strip():
            continue
        h = hist.get(sym) or []
        if not h or (h[0].get("verdict") or "").strip() != "제외":
            continue
        limit = sticky_days_structural if (h[0].get("severity") or "").strip() == "구조적" else sticky_days
        since = None
        for e in h:
            if (e.get("verdict") or "").strip() != "제외":
                break
            since = e.get("date")
        if not since:
            continue
        try:
            since_d = _dt.date.fromisoformat(since)
        except Exception:
            continue
        days_elapsed = (today - since_d).days
        if days_elapsed < limit:
            v["verdict"] = "제외"
            v["severity"] = h[0].get("severity") or "일시적"
            v["verdict_reason"] = (h[0].get("reason") or v.get("verdict_reason") or "").strip() \
                or "점착 유지(새 근거 없이 복귀 저지)"
            v["change_reason"] = f"점착 유지 — {since} 제외 이후 {days_elapsed}일 경과(<{limit}일), 복귀 근거 없음"


# ------------------------- verdict 적용(기존 로직 유지) -------------------------
def _apply_verdicts(buy_pool, watch_pool, vmap, n_buy, n_watch):
    """1단계 verdict 반영해 최종 목록 확정. AI가 없으면 전원 '유지'로 동작.
    2026-07-16 2차 수정(지호 님 피드백 — KCC 사례: 실적 부진+자사주 소각 기대를 저버린 거버넌스
    이슈를 '관찰강등'으로 분류해 매수 목록에 그대로 남긴 게 부적절했음, "관찰 없어졌으니 강등이면
    빠지는 게 맞다"). 관찰(watch) 슬롯이 폐지된 지금은 '강등'이 갈 곳이 없다 — verdict가
    '매수유지'가 아니면(제외든 구버전 캐시의 관찰강등이든) 전부 빼고, 넓힌 풀(kr_stocks.N_BUY=8
    등)에서 팩터 랭킹 순으로 결정론적으로 채운다(AI가 고르는 게 아니라 팩터 랭킹이 그대로 채움).
    반환 4번째 값은 shortfall — 풀 전체가 제외로 소진돼 n_buy에 못 미치면 양수(그 경우 억지로
    더 아래 순위까지 끌어오지 않는다 — 그런 날은 이례적인 상황이니 경보로 다루는 게 숫자를
    맞추는 것보다 유용)."""
    survivors, excluded = [], []
    for c in buy_pool:
        v = ((vmap.get(str(c["symbol"])) or {}).get("verdict") or "매수유지").strip()
        if v == "매수유지":
            survivors.append(c)
        else:
            excluded.append(c)   # '제외'든 구버전 '관찰강등' 캐시든 전부 뺀다
    watch_keep = []
    for c in watch_pool:
        v = ((vmap.get(str(c["symbol"])) or {}).get("verdict") or "").strip()
        if v and v != "매수유지":
            excluded.append(c)
        else:
            watch_keep.append(c)
    final_buy = survivors[:n_buy]
    shortfall = max(0, n_buy - len(final_buy))
    used = {c["symbol"] for c in final_buy}
    final_watch = [c for c in watch_keep if c["symbol"] not in used][:n_watch]
    return final_buy, final_watch, excluded, shortfall


# ------------------------- 조립 -------------------------
def _mk_item(c, kind, vmap, wmap):
    """후보 c + 검증(vmap) + 서술(wmap) → 렌더용 아이템. AI가 없어도 최소 구성이 된다."""
    sym = str(c["symbol"])
    v = vmap.get(sym) or {}
    w = wmap.get(sym) or {}
    flag = v.get("flag") if v.get("flag") in ("호재", "악재", "중립") else None
    pts = [str(x).strip() for x in (w.get("points") or []) if str(x).strip()][:4]
    item = {"symbol": sym,
            "name": (w.get("name") or c.get("name") or "").strip(),
            "category": (w.get("category") or c.get("industry") or c.get("sector") or "").strip(),
            "summary": (w.get("summary") or c.get("score_reason") or "").strip(),
            "points": pts, "flag": flag,
            # 2026-07-17: 카드의 별도 뉴스 줄(📰)도 points의 ③뉴스와 같은 이유로 보류 —
            # v.get("news")는 verify_stage가 여전히 채우지만(검증 판정용) 카드엔 안 보여준다.
            "news": "",
            "catalyst": (v.get("catalyst") or "").strip(),
            "comment": (w.get("comment") or "").strip(),
            "verdict_reason": (v.get("verdict_reason") or "").strip(),
            "plan": c.get("plan") or {}, "hot": bool(c.get("hot")),
            "rank": c.get("rank"), "pool_size": c.get("pool_size"),
            "already_held": bool(c.get("already_held"))}
    if kind == "watch":
        item["trigger"] = c.get("trigger") or ""
    return item


def build_report(groups: dict, market: dict, pregen: dict | None = None) -> dict:
    """groups={"buy_now","watch","kr_buy","kr_watch","sells","kr_sells"(선택)}
    pregen = pregen.py 가 미리 만든 dict. by_sym 이 있으면 검증 단계 생략(검색 비용 0).
    written(종목별 서술)까지 있으면 서술 단계도 생략(캐시에 없는 심볼만 _auto_fields 무료 대체)
    → pregen 이 완전할 때(PC가 켜져 있던 날)는 발송 시점 AI 호출이 0회가 된다.
    실패 시 {} 반환(호출부가 deterministic_report 폴백).

    주의: _enabled()(CLI 바이너리·API 키 유무)만으로 조기 종료하면 안 된다 — pregen에 이미
    written(사전서술)까지 있으면 이 함수는 AI를 한 번도 안 부르고 캐시만으로 완성된다. API 키를
    뺀 상태에서도(로컬 CLI가 없어도) pregen 캐시가 있으면 정상 동작해야 하므로, pregen이 없을 때만
    _enabled()를 확인한다."""
    attach_plans(groups)          # 계획은 AI 유무와 무관하게 항상 확정
    have_pregen = bool(pregen and pregen.get("by_sym"))
    if not _enabled() and not have_pregen:
        _log("AI 비활성 · pregen 도 없음 → 폴백."); return {}
    buy_pool, watch_pool = groups.get("buy_now") or [], groups.get("watch") or []
    kr_buy_pool, kr_watch_pool = groups.get("kr_buy") or [], groups.get("kr_watch") or []
    sells = (groups.get("sells") or []) + (groups.get("kr_sells") or [])
    # 메일 분리 후 KR 전용/US 전용 groups 로도 호출된다 — 어느 쪽이든 내용이 있으면 진행
    if not (buy_pool or watch_pool or kr_buy_pool or kr_watch_pool or sells):
        return {}
    try:
        # ── 1단계: 검증. pregen 있으면 생략(밤/아침에 구독 CLI로 이미 검증됨 → $0)
        if have_pregen:
            # pregen 은 by_sym(종목 검증) + night_notes(밤 시점 시황 배경) + (있으면)
            # written/market_written(사전서술)을 갖는다.
            ver = {"by_sym": pregen["by_sym"]}
            if pregen.get("night_notes"):
                market = dict(market, night_notes=str(pregen["night_notes"])[:900])
            _log(f"pregen 사용({pregen.get('generated','?')}) → 검증 단계 생략(검색 0회)")
        else:
            ver = verify_stage(groups, market)
            if not ver:
                _log("검증 실패 → 전원 유지로 서술만 진행"); ver = {"by_sym": {}}
        vmap = ver.get("by_sym") or {}
        for v in vmap.values():   # pregen 캐시 경로(구버전 severity 없는 캐시 포함) 페일세이프
            _normalize_severity(v)
        as_of = str(market.get("as_of") or "").strip()
        try:
            _apply_sticky_exclusion(vmap, buy_pool, "us", as_of)
            _apply_sticky_exclusion(vmap, kr_buy_pool, "kr", as_of)
        except Exception as e:
            _log(f"제외 점착성 적용 생략({type(e).__name__}: {e})")
        _log_verdicts(buy_pool, kr_buy_pool, vmap, market)

        # ── 코드가 최종 목록 확정
        fb, fw, fx, fshort = _apply_verdicts(buy_pool, watch_pool, vmap, FINAL_BUY, FINAL_WATCH)
        kfb, kfw, kfx, kshort = _apply_verdicts(kr_buy_pool, kr_watch_pool, vmap,
                                                 KR_FINAL_BUY, KR_FINAL_WATCH)
        # shortfall(제외가 너무 많아 풀이 목표에 못 미침)을 억지로 더 아래 순위로 채우지 않는다
        # — 대신 경보로 표시(MIN_BUY=서킷브레이커 임계값). 지호 님 질문 대응(2026-07-16, Fable 5 자문).
        if fshort or kshort:
            _log(f"[경보] 목표 미달 — 미국 부족 {fshort} · 한국 부족 {kshort} "
                 f"(제외가 이례적으로 많음 → 시장 상황 점검 요망)")
        final_pairs = ([(c, "buy") for c in fb] + [(c, "watch") for c in fw]
                       + [(c, "buy") for c in kfb] + [(c, "watch") for c in kfw])

        # ── 2단계: 서술. pregen에 사전서술(written)이 있으면 캐시만 쓰고 write_stage 자체를
        #    건너뛴다(API 0회). 캐시에 없는 심볼(후보풀 변동 등 드문 경우)만 _auto_fields 무료 대체.
        cached_written = (pregen or {}).get("written") or {}
        cached_sells = (pregen or {}).get("sells_written") or {}
        cached_market = (pregen or {}).get("market_written") or {}
        if cached_written:
            wmap, gaps = {}, 0
            for c, _kind in final_pairs:
                sym = str(c["symbol"])
                if sym in cached_written:
                    wmap[sym] = cached_written[sym]
                else:
                    wmap[sym] = _auto_fields(c); gaps += 1
            smap = {str(s["symbol"]): {"comment": cached_sells.get(str(s["symbol"]), "")} for s in sells}
            market_fields = cached_market or (write_market_stage(market) if not ver.get("market_overview") else {})
            _log(f"사전서술 캐시 사용({len(cached_written)}종목 · 부족분 {gaps}건 무료 대체)")
        else:
            need_market = not bool(ver.get("market_overview"))
            parsed = write_stage(final_pairs, sells, market, vmap, need_market)
            wmap = {str(r["symbol"]): r for r in (parsed.get("stocks") or []) if isinstance(r, dict) and r.get("symbol")}
            smap = {str(r["symbol"]): r for r in (parsed.get("sells") or []) if isinstance(r, dict) and r.get("symbol")}
            market_fields = parsed

        def pick(field):
            return (ver.get(field) or (market_fields.get(field) or "")).strip()

        risks_text = pick("risks")
        # MIN_BUY(서킷브레이커): 최종 매수 종목 수가 이 문턱 밑으로 떨어지면 억지로 채우지
        # 않고 대신 눈에 띄게 경고한다(기존 ⚠️ 고지 줄에 자연히 노출됨, 별도 템플릿 수정 불필요).
        # buy_pool이 비어있으면 그 시장은 애초에 이번 실행 대상이 아니었던 것(예: --kr 단독
        # 실행 시 미국 그룹은 원래 빈 채로 들어옴) — AI가 다 걸러낸 것과 구분해야 오탐이 안 남.
        warn_bits = []
        if kr_buy_pool and len(kfb) < MIN_BUY:
            warn_bits.append(f"한국 매수 후보가 {len(kfb)}종목으로 최소기준({MIN_BUY}) 미달")
        if buy_pool and len(fb) < MIN_BUY:
            warn_bits.append(f"미국 매수 후보가 {len(fb)}종목으로 최소기준({MIN_BUY}) 미달")
        if warn_bits:
            risks_text = ("[AI 제외 급증] " + " · ".join(warn_bits) + " — 시장 상황 점검 요망. "
                           + risks_text)

        out = {
            "market_overview": pick("market_overview"), "macro": pick("macro"),
            "signal_note": pick("signal_note"), "risks": risks_text,
            "buy_now": [_mk_item(c, "buy", vmap, wmap) for c in fb],
            "watch": [_mk_item(c, "watch", vmap, wmap) for c in fw],
            "kr_buy": [_mk_item(c, "buy", vmap, wmap) for c in kfb],
            "kr_watch": [_mk_item(c, "watch", vmap, wmap) for c in kfw],
            "ai_excluded": [{"symbol": c["symbol"], "name": c.get("name", ""),
                             "reason": ((vmap.get(str(c["symbol"])) or {}).get("verdict_reason") or "").strip(),
                             "severity": ((vmap.get(str(c["symbol"])) or {}).get("severity") or "").strip()}
                            for c in fx + kfx],
            "sells": [{"symbol": s["symbol"], "name": s.get("name", ""), "reason": s.get("reason", ""),
                       "since": s.get("since"), "ret_pct": s.get("ret_pct"), "plan": s.get("plan", ""),
                       "comment": ((smap.get(str(s["symbol"])) or {}).get("comment")
                                   or (vmap.get(str(s["symbol"])) or {}).get("news") or "").strip()}
                      for s in sells],
        }
        _log(f"[{AI_BACKEND}] 완료: 미국 {len(out['buy_now'])}/{len(out['watch'])} · "
             f"한국 {len(out['kr_buy'])}/{len(out['kr_watch'])} · 제외 {len(out['ai_excluded'])} · "
             f"매도 {len(out['sells'])} · pregen={'O' if pregen else 'X'}")
        return out
    except Exception as e:
        _log(f"호출 실패({type(e).__name__}: {e}) → 폴백."); return {}


# ------------------------- API/CLI 호출 -------------------------
def _call_api(instruction, web=True, system=None, model=None, max_tokens=6000, temperature=None):
    """Anthropic API. 시스템 프롬프트에 cache_control — 같은 실행 내 재시도/재호출 때
    캐시 적중(입력 90% 할인). 일 1회 실행이라 날짜 간 캐시는 TTL(5분)상 해당 없음.
    temperature=0(검증 단계 전용, 2026-07-17 Fable 5 자문): 분류 작업이라 창의성이
    불필요 — 모델 자체의 출력 분산을 한 축 제거해 같은 입력에 더 재현 가능한 판정이
    나오게 한다(검색 결과 분산은 못 줄이지만 공짜로 줄일 수 있는 분산은 줄임)."""
    client = anthropic.Anthropic(timeout=AI_TIMEOUT)
    kw = {}
    if web:
        kw["tools"] = [{"type": "web_search_20250305", "name": "web_search",
                        "max_uses": REPORT_WEB_USES}]
    if temperature is not None:
        kw["temperature"] = temperature
    msg = client.messages.create(
        model=_no_opus(model or MODEL_VERIFY, "claude-sonnet-5"),
        max_tokens=max_tokens,
        system=[{"type": "text", "text": system or _V_SYSTEM,
                 "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": instruction}], **kw)
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


def _call_cli(instruction, web=True, system=None, model=None):
    """로컬 claude -p (Pro 구독 — API 키·과금 없음). pregen.py 와 weekly_report.py 가 사용.
    model 지정 시 --model 로 전달한다. 이전엔 이 인자가 아예 없어서 CLI 기본 모델이
    검증·서술 단계 모두에 쓰였다(sonnet/haiku 분리가 API 경로에만 적용되던 버그) — 구독
    한도(토큰) 소모가 컸던 원인. 이제 verify_stage=MODEL_VERIFY, write_stage/write_market_stage
    =MODEL_WRITE 를 CLI 에도 그대로 전달한다.

    --tools(허용 도구 '전체 목록')를 항상 명시한다 — 이전엔 --allowedTools 만 썼는데, 이건
    '프롬프트 없이 실행되는' 도구를 사전승인할 뿐 다른 도구(Write 등)를 목록에서 빼지 않는다.
    그 결과 서술 단계(web=False라 이 플래그 자체가 안 붙던 경로)에서 모델이 결과를 파일로
    저장하려 시도 → 권한 대기 상태로 빠져 JSON 대신 '저장 권한이 필요합니다' 같은 텍스트만
    반환하는 문제가 실사용에서 확인됐다. --tools 는 '가용한' 도구 자체를 제한하므로(''=전체 비활성)
    검증 단계도 WebSearch/WebFetch 외엔 아예 못 쓰게 되어 이 문제가 구조적으로 막힌다."""
    prompt = (system or _V_SYSTEM) + "\n\n" + instruction
    cmd = [CLAUDE_BIN, "-p", "--output-format", "json"]
    if model:
        cmd += ["--model", _no_opus(model, "claude-sonnet-5")]
    cmd += ["--tools", "WebSearch,WebFetch" if web else ""]
    to = AI_TIMEOUT if web else min(AI_TIMEOUT, 300)
    proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=to)
    out = (proc.stdout or "").strip()
    if proc.returncode != 0:
        msg = (proc.stderr or "").strip() or out
        try:
            env = json.loads(out)
            if isinstance(env, dict):
                msg = str(env.get("result") or env.get("error") or msg)
        except Exception:
            pass
        raise RuntimeError(f"claude CLI rc={proc.returncode}: {msg[:400] or '(출력 없음)'}")
    try:
        env = json.loads(out)
        if isinstance(env, dict):
            if env.get("is_error"):
                raise RuntimeError(f"claude is_error: {str(env.get('result'))[:160]}")
            if isinstance(env.get("result"), str):
                return env["result"]
    except json.JSONDecodeError:
        pass
    return out


# ------------------------- 무AI 서술(공용) -------------------------
def _smart_truncate(text: str, max_len: int) -> str:
    """문장 중간이 아니라 마지막 '.'(또는 '다.')에서 자른다 — 안 되면 단어 경계 + '…'."""
    text = (text or "").strip()
    if len(text) <= max_len:
        return text
    cut = text[:max_len]
    dot = cut.rfind(". ")
    if dot == -1:
        dot = cut.rfind(".") if cut.endswith(".") else -1
    if dot >= max_len * 0.4:                    # 너무 짧게 잘리는 건 피함(원문의 40% 이상은 남길 때만)
        return cut[:dot + 1]
    sp = cut.rfind(" ")
    return (cut[:sp] if sp >= max_len * 0.4 else cut).rstrip("· ,") + "…"


def _first_sentence(text: str, max_len: int) -> str:
    """첫 문장만(있으면) — 요약 라인용. 문장부호 없으면 스마트 자르기로 폴백."""
    text = (text or "").strip()
    if not text:
        return ""
    dot = text.find(". ")
    if dot == -1 and text.endswith("."):
        dot = len(text) - 1
    if dot != -1 and dot + 1 <= max_len:
        return text[:dot + 1]
    return _smart_truncate(text, max_len)


def _auto_fields(c) -> dict:
    """AI 없이 지표+프로필만으로 만드는 최소 서술 필드(비용 $0). deterministic_report와
    build_report의 '사전서술 캐시에 없는 심볼(드묾)' 대체용이 공용으로 쓴다."""
    sym = str(c.get("symbol"))
    r = c.get("ret") or {}
    pts = []
    if c.get("rsi") is not None or r.get("3m") is not None:
        pts.append("①추세: " + " · ".join(x for x in (
            f"RSI {c['rsi']:.0f}" if c.get("rsi") is not None else "",
            f"3개월 {r['3m']:+.1f}%" if r.get("3m") is not None else "",
            f"6개월 {r['6m']:+.1f}%" if r.get("6m") is not None else "") if x))
    one_liner, detail = _profile_parts().get(sym, ("", ""))
    if detail:
        pts.append("②사업: " + _smart_truncate(detail, 140))
    # 2026-07-17 지호 님 지시로 뉴스 포인트 일단 제외(HPQ에 무관한 "Domino's..." 기사가
    # 붙던 사례 — 관련성 필터로 근본 원인은 고쳤지만, 보유종목만 모아 보내는 별도 뉴스
    # 메일을 새로 설계할 때까지는 카드에서 뉴스 자체를 빼두기로 함). headlines 수집·
    # fetch_news_headlines 관련성 필터는 그대로 유지 — 그 별도 메일에서 재사용할 것.
    # 요약 라인(=종목 설명): one_liner(브랜드 인지용) 우선 — ②사업(detail)과 겹치지 않게
    # 서로 다른 소스로 분리(2026-07-15, 지호 님 피드백: 둘이 같은 문장을 반복하고 있었음).
    # one_liner도 없으면(구프로필 과도기) detail 첫 문장으로, 그마저 없으면 내부 라벨 폴백.
    summary = one_liner or _first_sentence(detail, 90) or (c.get("score_reason") or "")[:90]
    return {"name": c.get("name", ""), "category": c.get("industry") or c.get("sector", ""),
            "summary": summary, "points": pts, "comment": ""}


# ------------------------- 무AI 폴백 -------------------------
def deterministic_report(groups: dict, market: dict) -> dict:
    """AI 실패 시 지표+프로필+계획만으로 구성. 계획·프로필 덕에 무AI여도 꽤 상세하다."""
    attach_plans(groups)
    spy = market.get("spy", {}) or {}; fg = market.get("fear_greed", {}) or {}
    gap, sc, rt = spy.get("gap_pct"), fg.get("score"), fg.get("rating")
    mo = "지표 기반 자동 선정본"
    if gap is not None:
        mo = f"SPY 200일선 대비 {gap:+.1f}%" + (f", 탐욕지수 {sc:.0f}({rt})" if sc is not None else "") + " — 지표 기반 자동 선정."

    def item(c, kind):
        sym = str(c.get("symbol"))
        af = _auto_fields(c)
        d = {"symbol": sym, "name": af["name"], "category": af["category"], "summary": af["summary"],
             "points": af["points"], "news": "", "catalyst": "", "comment": "", "flag": None,
             "verdict_reason": "", "plan": c.get("plan") or {}, "hot": bool(c.get("hot")),
             "already_held": bool(c.get("already_held"))}
        if kind == "watch":
            d["trigger"] = c.get("trigger") or "20일선 회복 확인 후 매수 검토"
        return d
    sells = (groups.get("sells") or []) + (groups.get("kr_sells") or [])
    return {"market_overview": mo, "macro": "", "signal_note": "",
            "risks": "지표 기반 자동본(AI 검증 생략). 투자 권유 아님.",
            "buy_now": [item(c, "buy") for c in (groups.get("buy_now") or [])[:FINAL_BUY]],
            "watch": [item(c, "watch") for c in (groups.get("watch") or [])[:FINAL_WATCH]],
            "kr_buy": [item(c, "buy") for c in (groups.get("kr_buy") or [])[:KR_FINAL_BUY]],
            "kr_watch": [item(c, "watch") for c in (groups.get("kr_watch") or [])[:KR_FINAL_WATCH]],
            "ai_excluded": [],
            "sells": [{"symbol": s.get("symbol"), "name": s.get("name", ""), "reason": s.get("reason", ""),
                       "since": s.get("since"), "ret_pct": s.get("ret_pct"),
                       "plan": s.get("plan", ""), "comment": ""} for s in sells]}


# ------------------------- HTML 렌더 -------------------------
def _esc(s): return (str(s) if s is not None else "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _chip(label, color="#6b7280", strong=False):
    w = "700" if strong else "600"
    return (f'<span style="display:inline-block;background:{color}1a;color:{color};border-radius:6px;'
            f'padding:1px 7px;margin:1px 4px 1px 0;font-size:11px;font-weight:{w}">{label}</span>')


def _metric_chips(m):
    if not m:
        return ""
    out = []
    if m.get("price") is not None:
        label = f'{m["price"]:,.0f}원' if m.get("krw") else f'${m["price"]:,.2f}'
        out.append(_chip(label, "#111827", True))
    if m.get("pe") is not None: out.append(_chip(f'PER {m["pe"]:.1f}'))
    if m.get("rsi") is not None:
        rc = "#b91c1c" if m["rsi"] >= 75 else ("#15803d" if m["rsi"] <= 40 else "#6b7280")
        out.append(_chip(f'RSI {m["rsi"]:.0f}', rc))
    if m.get("gap200") is not None:
        out.append(_chip(f'200일선 {m["gap200"]:+.0f}%', "#15803d" if m["gap200"] >= 0 else "#b91c1c"))
    if m.get("ret6m") is not None:
        out.append(_chip(f'6개월 {m["ret6m"]:+.0f}%', "#15803d" if m["ret6m"] >= 0 else "#b91c1c"))
    return "".join(out)


def _plan_table(plan: dict, comment: str = ""):
    """entry_plan.buy_plan 결과를 분할매수 표로. 코드 확정값 — 리포트의 '실행' 핵심."""
    if not plan or not plan.get("tranches"):
        return ""
    krw = plan.get("krw", False)
    rows = "".join(
        f'<tr><td style="padding:2px 8px;color:#374151">{t["label"]}</td>'
        f'<td style="padding:2px 8px;font-weight:700">{EP._fmt(t["price"], krw)}</td>'
        f'<td style="padding:2px 8px">{t["pct"]}%</td>'
        f'<td style="padding:2px 8px;color:#6b7280">{_esc(t["basis"])}</td></tr>'
        for t in plan["tranches"])
    stop = plan.get("stop") or {}
    stop_row = (f'<tr><td style="padding:2px 8px;color:#b91c1c">손절</td>'
                f'<td style="padding:2px 8px;font-weight:700;color:#b91c1c">{EP._fmt(stop.get("price"), krw)}</td>'
                f'<td style="padding:2px 8px;color:#b91c1c">전량</td>'
                f'<td style="padding:2px 8px;color:#b91c1c">{_esc(stop.get("basis"))}</td></tr>') if stop else ""
    note = (f'<div style="font-size:11px;color:#6b7280;margin-top:2px">{_esc(plan.get("note"))}</div>'
            if plan.get("note") else "")
    cmt = (f'<div style="font-size:12px;color:#1d4ed8;margin-top:3px">💬 {_esc(comment)}</div>'
           if comment else "")
    return (
        '<div style="background:#eff6ff;border-radius:6px;padding:6px 8px;margin-top:6px">'
        '<div style="font-size:12px;font-weight:700;color:#1d4ed8">🎯 매수 계획 (규칙 확정)</div>'
        f'<table style="border-collapse:collapse;font-size:12px;margin-top:2px">{rows}{stop_row}</table>'
        f'{note}{cmt}</div>')


def _card(i, r, metrics_by_sym, kind, is_kr=False):
    """is_kr=True면 헤더에 종목코드 대신 종목명을 쓴다(6자리 코드는 사람이 읽기 어려움 — 미국
    티커(AAPL 등)는 그 자체로 의미가 있어 그대로 둠)."""
    flag_color = {"호재": "#15803d", "악재": "#b91c1c", "중립": "#6b7280"}
    sym = r.get("symbol"); m = metrics_by_sym.get(sym, {})
    fl = r.get("flag")
    flag_chip = _chip(fl, flag_color.get(fl, "#6b7280"), True) if fl else ""
    cat_chip = _chip(_esc(r.get("category")), "#7c3aed") if r.get("category") else ""
    hot_chip = _chip("과열·분할", "#c2410c", True) if (kind == "buy" and r.get("hot")) else ""
    # 2026-07-17(지호 님 요청): 보유중/신규를 둘 다 명시적으로 표기해 구별되게 — 국장은
    # already_held를 이미 쓰고 있었는데 미장엔 안 켜져 있었음(daily_ai_report.run_us에서 신규 배선).
    held_chip = (_chip("보유중", "#0369a1", True) if r.get("already_held")
                else _chip("신규", "#15803d", True)) if kind == "buy" else ""
    pts = "".join(f'<li style="margin:1px 0">{_esc(p)}</li>' for p in r.get("points", []))
    pts_html = (f'<ul style="margin:6px 0 0;padding-left:16px;font-size:12px;color:#374151;'
                f'line-height:1.55">{pts}</ul>') if pts else ""
    # 분기 C-3(NEXT_STEPS_SONNET.md): 순위 사실은 항상 표시, 0~10 점수는
    # score_calibration.load_calibration()이 None이면(G2 미통과) expectancy_report가 자동 숨김.
    fact_html = EXR.rank_fact_html(r.get("rank"), r.get("pool_size")) + \
        EXR.score_line_html(r.get("rank"), r.get("pool_size"))
    if kind == "buy":
        act = _plan_table(r.get("plan"), r.get("comment"))
    else:
        act = ""
        if r.get("trigger") or r.get("plan"):
            act = (f'<div style="font-size:12px;color:#c2410c;background:#fff7ed;border-radius:6px;'
                   f'padding:5px 8px;margin-top:6px">&#9203; 매수 전환 조건: {_esc(r.get("trigger"))}'
                   + (f'<div style="color:#9a3412;margin-top:2px">전환 시 계획: {_esc(EP.plan_text(r.get("plan") or {}))}</div>'
                      if r.get("plan") else "")
                   + (f'<div style="color:#1d4ed8;margin-top:2px">&#128172; {_esc(r.get("comment"))}</div>'
                      if r.get("comment") else "")
                   + '</div>')
    _nw = r.get("news") or ""
    news = (f'<div style="color:#6b7280;font-size:11px;margin-top:5px;line-height:1.5">&#128480; {_esc(_nw)}</div>'
            if (_nw and fl != "중립" and "확인" not in _nw) else "")
    cata = (f'<div style="color:#7c3aed;font-size:11px;margin-top:3px;line-height:1.5">&#128197; {_esc(r.get("catalyst"))}</div>'
            if r.get("catalyst") else "")
    chart = f'<img src="cid:chart_{sym}" style="width:100%;border-radius:6px">'
    header = (f'{i}. {_esc(r.get("name"))}' if is_kr else
              f'{i}. {_esc(sym)} '
              f'<span style="color:#6b7280;font-size:12px;font-weight:400">{_esc(r.get("name"))}</span>')
    return (
        f'<table role="presentation" width="100%" style="border-collapse:collapse;border:1px solid #e5e7eb;'
        f'border-radius:10px;margin:10px 0;background:#fff;overflow:hidden"><tr>'
        f'<td width="56%" valign="top" style="padding:12px 14px">'
        f'<div style="font-size:15px;font-weight:700">{header}</div>'
        f'<div style="margin:4px 0 2px">{cat_chip}{held_chip}{hot_chip}{flag_chip}</div>'
        f'<div style="font-size:13px;color:#111;margin-top:4px;line-height:1.5">{_esc(r.get("summary"))}</div>'
        f'{fact_html}{pts_html}{act}{news}{cata}</td>'
        f'<td width="44%" valign="top" style="padding:12px 12px 12px 0">{chart}'
        f'<div style="margin-top:6px">{_metric_chips(m)}</div></td></tr></table>')


def _sell_card(i, s, is_kr=False):
    ret = s.get("ret_pct")
    ret_chip = (_chip(f'편입 후 {ret:+.0f}%', "#15803d" if (ret or 0) >= 0 else "#b91c1c", True)
                if ret is not None else "")
    since = f' · {_esc(s.get("since"))} 편입' if s.get("since") else ""
    plan = (f'<div style="font-size:12px;color:#7f1d1d;background:#fff1f2;border-radius:6px;'
            f'padding:5px 8px;margin-top:5px"><b>처분 계획:</b> {_esc(s.get("plan"))}</div>') if s.get("plan") else ""
    cmt = (f'<div style="font-size:12px;color:#374151;margin-top:4px;line-height:1.5">{_esc(s.get("comment"))}</div>'
           if s.get("comment") else "")
    header = (f'{i}. {_esc(s.get("name"))}{since}' if is_kr else
              f'{i}. {_esc(s.get("symbol"))} '
              f'<span style="color:#6b7280;font-size:12px;font-weight:400">{_esc(s.get("name"))}{since}</span>')
    return (
        f'<div style="border:1px solid #fecaca;border-radius:10px;padding:11px 13px;margin:8px 0;background:#fef2f2">'
        f'<div style="font-size:14px;font-weight:700">{header} {ret_chip}</div>'
        f'<div style="font-size:12px;color:#b91c1c;margin-top:3px">&#9888; {_esc(s.get("reason"))}</div>{plan}{cmt}</div>')


def _excluded_html(items, is_kr=False):
    if not items:
        return ""
    def _badge(x):
        # 2026-07-17(지호 님 제안 — 매도는 장기 펀더멘탈 훼손일 때만): 구조적/일시적을 표기해
        # 독자가 "이건 그냥 단기 뉴스라 매수만 안 하는 것"과 "회사 자체가 흔들리는 것"을 구분.
        sev = (x.get("severity") or "").strip()
        if sev == "구조적":
            return ' <span style="font-size:10px;color:#991b1b;font-weight:700">[구조적]</span>'
        if sev == "일시적":
            return ' <span style="font-size:10px;color:#9ca3af">[일시적]</span>'
        return ""
    rows = "".join(
        (f'<div style="font-size:12px;color:#374151;margin:3px 0">'
         f'<b>{_esc(x.get("name"))}</b>{_badge(x)} — {_esc(x.get("reason") or "사유 미기재")}</div>' if is_kr else
         f'<div style="font-size:12px;color:#374151;margin:3px 0">'
         f'<b>{_esc(x.get("symbol"))}</b> {_esc(x.get("name"))}{_badge(x)} — {_esc(x.get("reason") or "사유 미기재")}</div>')
        for x in items)
    return (
        '<div style="border:1px solid #e5e7eb;border-radius:10px;padding:10px 13px;margin:10px 0;background:#f8fafc">'
        '<div style="font-size:13px;font-weight:700;color:#0e7490">AI 검증에서 제외된 후보</div>'
        '<div style="font-size:11px;color:#9ca3af;margin:2px 0 4px">규칙이 뽑았지만 최신 뉴스·리스크 점검에서 탈락</div>'
        + rows + '</div>')


def holdings_table_html(summary: list, krw: bool = False, chart_cid: str | None = None,
                        totals: dict | None = None, name_map: dict | None = None) -> str:
    """holdings.live_summary() 결과를 보유현황 표로. summary 없으면 빈 문자열(섹션 자체 생략).
    totals={"strategy","bench","index_name"} — 전체 투입자산 기준 누적수익률 vs 지수(있으면 표기).
    name_map={symbol:name} — 한국은 종목코드(숫자)만으론 못 알아보므로 이름으로 치환
    (2026-07-15, 지호 님 피드백). 없으면 symbol 그대로(미국은 티커 자체가 읽을 만해 그대로 둠)."""
    if not summary:
        return ""
    name_map = name_map or {}
    rows = "".join(
        f'<tr><td style="padding:3px 8px">{_esc(name_map.get(r["symbol"], r["symbol"]))}</td>'
        f'<td style="padding:3px 8px;color:#6b7280">{_esc(r.get("since"))}</td>'
        f'<td style="padding:3px 8px">{EP._fmt(r.get("entry"), krw)}</td>'
        f'<td style="padding:3px 8px">{EP._fmt(r.get("price"), krw)}</td>'
        f'<td style="padding:3px 8px;font-weight:700;color:{"#15803d" if r["ret_pct"] >= 0 else "#b91c1c"}">'
        f'{r["ret_pct"]:+.1f}%</td>'
        f'<td style="padding:3px 8px;color:#6b7280">{r.get("held_days") if r.get("held_days") is not None else "-"}일</td></tr>'
        for r in summary)
    chart = (f'<img src="cid:{chart_cid}" style="width:100%;max-width:640px;border-radius:8px;margin:8px 0">'
             if chart_cid else "")
    totals_html = ""
    if totals:
        sc = "#15803d" if totals["strategy"] >= 0 else "#b91c1c"
        bc = "#15803d" if totals["bench"] >= 0 else "#b91c1c"
        totals_html = (
            f'<div style="font-size:13px;margin:2px 0 6px">전체 투입자산 기준 '
            f'<b style="color:{sc}">{totals["strategy"]:+.1f}%</b>'
            f' <span style="color:#9ca3af">vs</span> {_esc(totals["index_name"])}(동일시점·동일금액) '
            f'<b style="color:{bc}">{totals["bench"]:+.1f}%</b></div>')
    return (
        '<h3 style="margin:18px 0 4px">&#128202; 보유현황</h3>' + totals_html +
        '<table role="presentation" style="border-collapse:collapse;font-size:12px;width:100%;max-width:640px">'
        '<tr style="color:#6b7280;text-align:left"><th style="padding:3px 8px">종목</th>'
        '<th style="padding:3px 8px">매수일</th><th style="padding:3px 8px">진입가</th>'
        '<th style="padding:3px 8px">현재가</th><th style="padding:3px 8px">수익률</th>'
        '<th style="padding:3px 8px">보유일</th></tr>' + rows + '</table>' + chart)


def render_report_html(report, as_of="", metrics_by_sym=None, market_html="", signals_html="",
                       kr_sells=None, banner="", title=None, show_spy=True, is_kr=False,
                       market_label="전일", holdings_html=""):
    """일일 리포트 HTML — 메일 2통 분리 지원.
    title    = 헤더 제목(없으면 기본). KR 장전/US 마감 메일이 각자 지정.
    show_spy = SPY 큰 차트 표시 여부(KR 전용 메일은 SPY 데이터가 없어 False).
    is_kr    = 국장 메일 여부. True면 sells/ai_excluded 카드에서 종목코드 대신 이름을 쓴다
               (kr_buy/kr_watch/kr_sells 카드는 국장 소속이 확정이라 항상 이름만 표시).
    market_label = 세계시장 요약 표의 기준(기본 "전일", 월요일엔 "전주" — 주말분 누락 방지).
    미국/한국 섹션은 해당 카드가 있을 때만 그린다."""
    if not report:
        return ""
    metrics_by_sym = metrics_by_sym or {}
    buy_cards = "".join(_card(i, r, metrics_by_sym, "buy", is_kr=False) for i, r in enumerate(report.get("buy_now", []), 1))
    watch_cards = "".join(_card(i, r, metrics_by_sym, "watch", is_kr=False) for i, r in enumerate(report.get("watch", []), 1))
    kr_buy_cards = "".join(_card(i, r, metrics_by_sym, "buy", is_kr=True) for i, r in enumerate(report.get("kr_buy", []), 1))
    kr_watch_cards = "".join(_card(i, r, metrics_by_sym, "watch", is_kr=True) for i, r in enumerate(report.get("kr_watch", []), 1))
    sells = report.get("sells", [])
    sell_html = ""
    if sells:
        sell_cards = "".join(_sell_card(i, s, is_kr=is_kr) for i, s in enumerate(sells, 1))
        sell_html = ('<h3 style="margin:18px 0 2px">&#128308; 매도 후보 · 차익실현 <span style="color:#9ca3af;font-size:12px">'
                     '(보유 종목 중 추세 이탈 — 처분 계획 포함)</span></h3>' + sell_cards)
    kr_sell_html = ""
    if kr_sells:
        kr_sell_cards = "".join(_sell_card(i, s, is_kr=True) for i, s in enumerate(kr_sells, 1))
        kr_sell_html = ('<h3 style="margin:18px 0 2px">&#128308; 한국 매도 후보 · 차익실현 <span style="color:#9ca3af;font-size:12px">'
                        '(보유 종목 중 추세 이탈)</span></h3>' + kr_sell_cards)
    sub = f' <span style="color:#9ca3af;font-size:12px">({_esc(as_of)} 종가 기준)</span>' if as_of else ""
    spy = ('<img src="cid:spy_chart" style="width:100%;max-width:640px;border-radius:8px;margin:8px 0">'
           if show_spy else "")
    banner_html = (f'<div style="background:#fffbeb;border:1px solid #fcd34d;border-radius:8px;padding:7px 11px;'
                   f'font-size:12px;color:#92400e;margin:8px 0">&#8505;&#65039; {_esc(banner)}</div>') if banner else ""
    # 분기 C-3: 전략 레벨 기대값 박스(검증된 사실만, G1 실측치 기준) — 미국장 메일 · 주 1회만.
    expectancy_html = EXR.expectancy_box_html() if not is_kr else ""
    market_sec = ""
    if market_html:
        market_sec = (f'<h3 style="margin:14px 0 6px">&#127760; {market_label} 시장 요약 <span style="color:#9ca3af;font-size:12px">'
                      f'(나스닥·다우존스·닛케이·유럽·글로벌·비트코인·환율 — {market_label} 등락)</span></h3>' + market_html)
    signals_sec = ""
    if signals_html:
        note = _esc(report.get("signal_note") or "")
        note_html = (f'<div style="font-size:13px;color:#111;margin:4px 0 8px;line-height:1.55">{note}</div>'
                     if note else "")
        signals_sec = ('<h3 style="margin:18px 0 4px">&#129517; 지수·코인 추세 신호 <span style="color:#9ca3af;font-size:12px">'
                       '(규칙 기반 — STRATEGY.md)</span></h3>' + note_html + signals_html)
    kr_note = (
        '<div style="font-size:12px;color:#374151;background:#f8fafc;border-radius:6px;'
        'padding:6px 9px;margin:2px 0 8px;line-height:1.5">저PER(저평가)·저PBR(저평가)·고배당'
        ' 상위 종목 — 지수를 그대로 담는 핵심자산(코어)을 보완하는 위성자산(새틀라이트) '
        '전략입니다. 요즘 코스피는 삼전·하이닉스 쏠림으로 지수 자체가 오르는 장이라 개별종목'
        '만으로 지수를 이기긴 어려워, 위험 대비 수익(샤프지수) 기준 코어65:새틀35 비중이 '
        '가장 안정적이었습니다(8년 백테스트 — 근거는 주간 배분 리포트 참고).</div>')
    kr_sec = ""
    if kr_buy_cards or kr_watch_cards:
        kr_sec = (
            '<h3 style="margin:18px 0 2px">&#127472;&#127479; 코스피200 매수 후보</h3>'
            + kr_note
            + (kr_buy_cards or '<div style="font-size:12px;color:#6b7280">해당 없음</div>')
            + ('<h3 style="margin:18px 0 2px">&#127472;&#127479; 코스피200 관찰 · 내려오면 매수</h3>' + kr_watch_cards if kr_watch_cards else ""))
    # 미국 섹션도 (한국처럼) 카드가 있을 때만 — KR 전용 메일에서 빈 헤더 방지
    us_note = (
        '<div style="font-size:12px;color:#374151;background:#f8fafc;border-radius:6px;'
        'padding:6px 9px;margin:2px 0 8px;line-height:1.5">퀄리티·주주환원 팩터(자산 대비 '
        '수익성·연구개발 집약도·자사주매입 등 계량 지표) 점수 상위 종목입니다. 과최적화 '
        '위험(PBO)과 통계적 유의성(DSR) 검증을 통과한 가중치로 순위를 매기고, 최신 뉴스·'
        '리스크는 AI가 추가로 확인합니다.</div>')
    us_sec = ""
    if buy_cards or watch_cards:
        us_sec = (
            f'<h3 style="margin:16px 0 2px">&#11088; 미국(S&amp;P500) 추천 종목</h3>{us_note}{buy_cards}'
            + (f'<h3 style="margin:18px 0 2px">&#128064; 미국 관찰 · 내려오면 매수 <span style="color:#9ca3af;font-size:12px">'
               f'(좋은 종목이나 지금은 조정 중 · AI 강등 포함)</span></h3>{watch_cards}' if watch_cards else ""))
    head = _esc(title) if title else "&#128200; 데일리 시장 점검 · 매수·매도 후보"
    # 2026-07-15: 차트+추천종목을 지수·코인 추세 신호보다 위로(지호 님 요청) — KR 전용 호출은
    # spy/us_sec가 원래 빈 문자열이라 이 순서 변경으로 KR 레이아웃엔 영향 없음.
    return (
        f'<div style="font-family:-apple-system,BlinkMacSystemFont,\'Segoe UI\',\'Malgun Gothic\',sans-serif;'
        f'max-width:700px;margin:0 auto;color:#111">'
        f'<h2 style="margin:6px 0">{head}{sub}</h2>'
        f'{banner_html}'
        f'{expectancy_html}'
        f'<div style="background:#f8fafc;border-left:3px solid #6b7280;padding:8px 12px;font-size:13px;'
        f'line-height:1.6;margin:8px 0"><b>&#129517; 시장</b> {_esc(report.get("market_overview"))}<br>'
        f'<b>&#127760; 환율·한국·코인</b> {_esc(report.get("macro"))}</div>'
        f'{spy}'
        f'{us_sec}'
        f'{market_sec}'
        f'{signals_sec}'
        f'{_excluded_html(report.get("ai_excluded"), is_kr=is_kr)}'
        f'{sell_html}'
        f'{kr_sec}'
        f'{kr_sell_html}'
        f'{holdings_html}'
        f'<div style="font-size:11px;color:#9ca3af;margin-top:14px;line-height:1.5">'
        f'&#9888;&#65039; {_esc(report.get("risks"))}<br>정보 제공용이며 투자 권유가 아닙니다. 판단·책임은 본인에게 있습니다.<br>'
        f'매도 규칙: 6개월 정기 재평가 또는 200일선 -3% 이탈 (미국·한국 공통). 전략 근거: STRATEGY.md</div>'
        f'</div>')
