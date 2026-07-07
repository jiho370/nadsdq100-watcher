#!/usr/bin/env python3
"""
ai_report.py — Claude(구독 CLI 또는 API)로 'S&P500 종목추천 보고서' 생성.

구조(중요): 종목 선정은 코드가 '점수(백테스트 최적 가중치) + 진입 적합도'로 확정한다.
  · 지금 매수(buy_now)  = 점수 상위 & 과열 아님(지금 진입하기 좋은)
  · 관찰(watch)         = 점수 상위지만 과열/비쌈(좋은 종목, 내려오면 매수)
AI 는 종목을 바꾸지 않고 각 종목의 '설명'만 쓴다(=매번 동일 종목, 일관성 보장).

무료(구독) 경로: AI_BACKEND=cli → 로컬 claude -p. 실패/키없음 시 {} 반환(호출부 폴백).
_call_cli/_call_api 는 system 파라미터로 주간 리포트(weekly_report.py)도 재사용.
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

AI_BACKEND     = os.environ.get("AI_BACKEND", "cli").strip().lower()
CLAUDE_BIN     = os.environ.get("CLAUDE_BIN", "claude")
REPORT_MODEL   = os.environ.get("REPORT_MODEL", "claude-sonnet-4-6")
REPORT_WEB_USES = int(os.environ.get("REPORT_WEB_USES", "6"))
REPORT_WEB     = os.environ.get("REPORT_WEB", "1") == "1"     # 기본 ON(웹검색으로 뉴스·환율). 5pm 배치라 시간 무방.
AI_TIMEOUT     = float(os.environ.get("AI_TIMEOUT", "1200"))  # 20분 — 웹검색 넉넉히


def _log(m): print(f"[REPORT] {m}", file=sys.stderr)


def _enabled():
    if os.environ.get("AI_ENABLED", "1") != "1":
        return False
    if AI_BACKEND == "cli":
        return shutil.which(CLAUDE_BIN) is not None or os.path.exists(CLAUDE_BIN)
    return anthropic is not None and bool(os.environ.get("ANTHROPIC_API_KEY"))


_SYSTEM = (
    "당신은 한국 개인투자자(수신자: 투자에 관심 있는 아버지)에게 매일 보내는 '시장 점검·종목추천 보고서'"
    "(미국 S&P500 + 코스피200 + 지수·코인 신호)를 쓰는 애널리스트다. 규칙:\n"
    "1) 종목 지표 수치는 제공된 JSON 값만 인용한다. 새로운 종목 수치를 지어내지 않는다.\n"
    "2) 종목 목록은 이미 확정돼 있다. 종목을 추가/삭제/이동하지 말고, 주어진 각 종목의 '설명'만 쓴다.\n"
    "3) 뉴스·시황·환율은 web_search 로 최신 정보를 찾고, 제공된 headlines 도 참고한다. 확인 안 되면 빈 값(지어내지 않음).\n"
    "4) 단정적 매수/매도 단언·수익 보장 금지. 핵심만 아주 간결하게(요약·불릿). 어려운 말 금지.\n"
    "5) 한자(漢字) 절대 금지(예: 前日比 → 전일 대비). 이동평균은 '20일선/50일선/200일선'.\n"
    "6) flag가 '중립'이거나 재료가 없으면 news는 빈 문자열(\"\"). '확인 어려움' 같은 문구 금지.\n"
    "7) 최종 출력은 지정된 JSON 하나만(코드블록·군더더기 없이)."
)

_SCHEMA = (
    '{\n'
    '  "market_overview": "시장 한 줄 요약(추세+심리)",\n'
    '  "macro": "환율(USD/KRW)·코스피 등 한 줄",\n'
    '  "buy_now": [ {"symbol":"AAA","name":"회사명","category":"세부분류(반도체·파운드리 등)",\n'
    '     "summary":"무슨 회사+왜 지금 진입 좋은지 한 줄","points":["핵심1","핵심2"],\n'
    '     "entry":"지금 진입 방법 한 줄(예: 현재가 분할 매수)","news":"최근 이슈 한 줄(없으면 \\"\\")",\n'
    '     "flag":"호재|악재|중립"} ],\n'
    '  "watch": [ {"symbol":"BBB","name":"회사명","category":"세부분류",\n'
    '     "summary":"왜 좋은 종목인지 한 줄","points":["핵심1","핵심2"],\n'
    '     "target":"어디까지 내려오면 매수 검토(예: 50일선 근처/과열 진정 시)","news":"...(없으면 \\"\\")",\n'
    '     "flag":"호재|악재|중립"} ],\n'
    '  "kr_buy": [ {"symbol":"005930","name":"회사명","category":"업종",\n'
    '     "summary":"무슨 회사+왜 지금 진입 좋은지 한 줄","points":["핵심1","핵심2"],\n'
    '     "entry":"지금 진입 방법 한 줄","news":"...(없으면 \\"\\")","flag":"호재|악재|중립"} ],\n'
    '  "kr_watch": [ {"symbol":"000660","name":"회사명","category":"업종",\n'
    '     "summary":"왜 좋은 종목인지 한 줄","points":["핵심1"],\n'
    '     "target":"어디까지 오면 매수 검토","news":"...","flag":"호재|악재|중립"} ],\n'
    '  "signal_note": "핵심 6자산(나스닥100·S&P500·코스피·코스닥·비트코인·이더리움) 신호 요약 1-2문장",\n'
    '  "sells": [ {"symbol":"CCC","comment":"왜 지금 정리를 고려할 만한지 한 줄(reason+뉴스)"} ],\n'
    '  "risks": "공통 유의사항 한 줄"\n'
    '}'
)


def _lean(cands):
    return [{k: v for k, v in c.items() if k != "closes"} for c in cands]


def _instruction(buy_now, watch, sells, market, kr_buy=None, kr_watch=None):
    kr_part = ""
    if kr_buy or kr_watch:
        kr_part = (
            "- kr_buy/kr_watch(코스피200 종목): 미국 종목과 같은 요령으로 각 종목 설명만 쓴다. "
            "symbol은 6자리 종목코드 그대로. 수치는 CONTEXT 값만.\n"
            f"CONTEXT.kr_buy = {json.dumps(_lean(kr_buy or []), ensure_ascii=False)}\n"
            f"CONTEXT.kr_watch = {json.dumps(_lean(kr_watch or []), ensure_ascii=False)}\n")
    sig_part = ""
    if market.get("signals"):
        sig_part = ("- signal_note: market.signals(핵심 6자산의 추세 신호·상태)를 근거로 오늘 가장 중요한 "
                    "신호 변화를 1-2문장으로. 신호 등급 자체는 코드가 확정했으니 바꾸지 말 것.\n")
    return (
        "아래 그룹의 시장·종목 보고서를 작성하라. 종목·신호는 이미 확정이니 바꾸지 말고 설명만 쓴다.\n"
        + sig_part + kr_part +
        "- buy_now(지금 매수 검토): 점수 상위 & 상승 추세가 살아있는 종목(과열이어도 강하면 포함). "
        "각 종목 왜 '지금' 좋은지 + entry(지금 진입 방법). 단 hot=true(과열) 종목은 entry에 "
        "'한 번에 말고 분할 매수'를 반드시 강조.\n"
        "- watch(관찰·눌림목): 점수 상위지만 지금 하락·조정 중인 종목. "
        "각 종목 왜 좋은지 + target(반등/조정 마무리 확인 후 매수, 예: 20일선 회복 시).\n"
        "- sells(매도 검토): 이미 매도 시그널이 뜬 보유 종목(reason 참고). 각 종목에 왜 지금 "
        "정리를 고려할 만한지 comment 한 줄(reason + 뉴스). 없으면 빈 배열.\n"
        "- category(세부분류)는 구체적으로(특히 반도체는 팹리스·파운드리·메모리·장비 등으로).\n"
        "- news: web_search + 제공된 headlines 로 각 종목 최근 이슈 1문장(없으면 빈 문자열, 지어내지 말 것).\n"
        "- market_overview: market 집계(SPY 200일선 이격·탐욕지수·섹터 등락·world)로 전일 미국+세계 시장 1-2문장.\n"
        "- macro: 환율(USD/KRW)·한국(코스피/코스닥)·코인(BTC/ETH) 전일 흐름 한 줄(market.signals·world 값 사용, "
        "web_search 보강. 모르면 빈 문자열, '확인 어려움' 쓰지 말 것).\n\n"
        f"출력 스키마(JSON):\n{_SCHEMA}\n\n"
        f"CONTEXT.market = {json.dumps(market, ensure_ascii=False)}\n\n"
        f"CONTEXT.buy_now = {json.dumps(_lean(buy_now), ensure_ascii=False)}\n\n"
        f"CONTEXT.watch = {json.dumps(_lean(watch), ensure_ascii=False)}\n\n"
        f"CONTEXT.sells = {json.dumps(sells, ensure_ascii=False)}\n"
    )


def _desc_map(items):
    out = {}
    for r in items or []:
        if isinstance(r, dict) and r.get("symbol"):
            out[r["symbol"]] = r
    return out


def _norm_item(sym, d, fallback, kind):
    """AI 설명(d)을 고정 종목(sym)에 매핑. 없으면 fallback(candidate)로 최소 구성."""
    d = d or {}
    flag = d.get("flag") if d.get("flag") in ("호재", "악재", "중립") else None
    pts = d.get("points")
    if isinstance(pts, str):
        pts = [pts]
    pts = [str(x).strip() for x in (pts or []) if str(x).strip()][:3]
    item = {
        "symbol": sym, "name": (d.get("name") or fallback.get("name") or "").strip(),
        "category": (d.get("category") or fallback.get("sector") or "").strip(),
        "summary": (d.get("summary") or "").strip(),
        "points": pts, "news": (d.get("news") or "").strip(), "flag": flag,
    }
    if kind == "buy":
        item["entry"] = (d.get("entry") or "").strip()
        item["hot"] = bool(fallback.get("hot"))
    else:
        item["target"] = (d.get("target") or d.get("entry") or "").strip()
    return item


def build_report(groups: dict, market: dict) -> dict:
    """groups={"buy_now":[cand...],"watch":[cand...]} → 보고서 dict. 실패/비활성 시 {}."""
    if not _enabled():
        _log("비활성 → 폴백."); return {}
    buy_now = groups.get("buy_now") or []
    watch = groups.get("watch") or []
    sells = groups.get("sells") or []
    kr_buy = groups.get("kr_buy") or []
    kr_watch = groups.get("kr_watch") or []
    if not (buy_now or watch or sells):
        return {}
    try:
        instr = _instruction(buy_now, watch, sells, market, kr_buy, kr_watch)
        try:                                  # 1차: 기본은 웹검색 OFF(빠름). REPORT_WEB=1이면 ON.
            text = _call_cli(instr, REPORT_WEB) if AI_BACKEND == "cli" else _call_api(instr, REPORT_WEB)
        except Exception as e1:               # 실패 시 웹검색 없이 재시도(새 형식 유지)
            _log(f"1차 실패({type(e1).__name__}) → 웹검색 없이 재시도")
            text = _call_cli(instr, False) if AI_BACKEND == "cli" else _call_api(instr, False)
        parsed = _extract_json(text or "")
        if parsed is None:
            _log("JSON 파싱 실패 → 폴백."); return {}
        bmap = _desc_map(parsed.get("buy_now"))
        wmap = _desc_map(parsed.get("watch"))
        smap = _desc_map(parsed.get("sells"))
        kbmap = _desc_map(parsed.get("kr_buy"))
        kwmap = _desc_map(parsed.get("kr_watch"))
        out = {
            "market_overview": (parsed.get("market_overview") or "").strip(),
            "macro": (parsed.get("macro") or "").strip(),
            "signal_note": (parsed.get("signal_note") or "").strip(),
            "risks": (parsed.get("risks") or "").strip(),
            # 멤버십은 코드가 강제(점수·진입·매도시그널 기준). AI는 설명만.
            "buy_now": [_norm_item(c["symbol"], bmap.get(c["symbol"]), c, "buy") for c in buy_now],
            "watch": [_norm_item(c["symbol"], wmap.get(c["symbol"]), c, "watch") for c in watch],
            "kr_buy": [_norm_item(c["symbol"], kbmap.get(c["symbol"]), c, "buy") for c in kr_buy],
            "kr_watch": [_norm_item(c["symbol"], kwmap.get(c["symbol"]), c, "watch") for c in kr_watch],
            "sells": [{"symbol": s["symbol"], "name": s.get("name", ""), "reason": s.get("reason", ""),
                       "since": s.get("since"), "ret_pct": s.get("ret_pct"),
                       "comment": ((smap.get(s["symbol"]) or {}).get("comment") or "").strip()}
                      for s in sells],
        }
        _log(f"[{AI_BACKEND}] 생성: 지금매수 {len(out['buy_now'])} · 관찰 {len(out['watch'])} · 매도 {len(out['sells'])}")
        return out
    except Exception as e:
        _log(f"호출 실패({type(e).__name__}: {e}) → 폴백."); return {}


def _call_api(instruction, web=True, system=None):
    client = anthropic.Anthropic(timeout=AI_TIMEOUT)
    kw = {}
    if web:
        kw["tools"] = [{"type": "web_search_20250305", "name": "web_search", "max_uses": REPORT_WEB_USES}]
    msg = client.messages.create(
        model=REPORT_MODEL, max_tokens=4000, system=system or _SYSTEM,
        messages=[{"role": "user", "content": instruction}], **kw)
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


def _call_cli(instruction, web=True, system=None):
    prompt = (system or _SYSTEM) + "\n\n" + instruction
    cmd = [CLAUDE_BIN, "-p", "--output-format", "json"]
    if web:
        cmd += ["--allowedTools", "WebSearch,WebFetch"]
    to = AI_TIMEOUT if web else min(AI_TIMEOUT, 150)   # 웹 없으면 빠름
    proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=to)
    out = (proc.stdout or "").strip()
    if proc.returncode != 0:
        # 에러 상세를 최대한 노출(빈 stderr 대비 stdout·JSON 엔벨로프까지 확인)
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


def deterministic_report(groups: dict, market: dict) -> dict:
    """AI 생성 실패 시 지표만으로 새 형식(지금매수/관찰/매도) 리포트를 구성 → 발송 누락 방지."""
    spy = market.get("spy", {}) or {}; fg = market.get("fear_greed", {}) or {}
    gap, sc, rt = spy.get("gap_pct"), fg.get("score"), fg.get("rating")
    mo = "지표 기반 자동 선정본"
    if gap is not None:
        mo = f"SPY 200일선 대비 {gap:+.1f}%" + (f", 탐욕지수 {sc:.0f}({rt})" if sc is not None else "") + " — 지표 기반 자동 선정."

    def item(c, kind):
        d = {"symbol": c.get("symbol"), "name": c.get("name", ""), "category": c.get("sector", ""),
             "summary": (c.get("score_reason") or "")[:70], "points": [], "news": "", "flag": None}
        if kind == "buy":
            d["entry"] = "현재가 분할 매수 검토" + (" (과열 — 소량/눌림 대기)" if c.get("hot") else "")
            d["hot"] = bool(c.get("hot"))
        else:
            d["target"] = "20일선/50일선 회복 확인 후 매수 검토"
        return d
    return {"market_overview": mo, "macro": "", "signal_note": "",
            "risks": "지표 기반 자동본(AI 해설 생략). 투자 권유 아님.",
            "buy_now": [item(c, "buy") for c in groups.get("buy_now", [])],
            "watch": [item(c, "watch") for c in groups.get("watch", [])],
            "kr_buy": [item(c, "buy") for c in groups.get("kr_buy", [])],
            "kr_watch": [item(c, "watch") for c in groups.get("kr_watch", [])],
            "sells": [{"symbol": s.get("symbol"), "name": s.get("name", ""), "reason": s.get("reason", ""),
                       "since": s.get("since"), "ret_pct": s.get("ret_pct"), "comment": ""}
                      for s in groups.get("sells", [])]}


# ------------------------- HTML 렌더 -------------------------
def _esc(s): return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


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


def _card(i, r, metrics_by_sym, kind):
    flag_color = {"호재": "#15803d", "악재": "#b91c1c", "중립": "#6b7280"}
    sym = r.get("symbol"); m = metrics_by_sym.get(sym, {})
    fl = r.get("flag")
    flag_chip = _chip(fl, flag_color.get(fl, "#6b7280"), True) if fl else ""
    cat_chip = _chip(_esc(r.get("category")), "#7c3aed") if r.get("category") else ""
    hot_chip = _chip("⚠️과열·분할", "#c2410c", True) if (kind == "buy" and r.get("hot")) else ""
    pts = "".join(f'<li style="margin:1px 0">{_esc(p)}</li>' for p in r.get("points", []))
    pts_html = (f'<ul style="margin:6px 0 0;padding-left:16px;font-size:12px;color:#374151;'
                f'line-height:1.55">{pts}</ul>') if pts else ""
    if kind == "buy":
        act = (f'<div style="font-size:12px;color:#1d4ed8;background:#eff6ff;border-radius:6px;'
               f'padding:5px 8px;margin-top:6px">🎯 지금 진입: {_esc(r.get("entry"))}</div>') if r.get("entry") else ""
    else:
        act = (f'<div style="font-size:12px;color:#c2410c;background:#fff7ed;border-radius:6px;'
               f'padding:5px 8px;margin-top:6px">⏳ 매수 조건: {_esc(r.get("target"))}</div>') if r.get("target") else ""
    _nw = r.get("news") or ""
    show = _nw and fl != "중립" and ("확인" not in _nw)
    news = (f'<div style="color:#6b7280;font-size:11px;margin-top:5px;line-height:1.5">🗞 {_esc(_nw)}</div>'
            if show else "")
    chart = f'<img src="cid:chart_{sym}" style="width:100%;border-radius:6px">'
    return (
        f'<table role="presentation" width="100%" style="border-collapse:collapse;border:1px solid #e5e7eb;'
        f'border-radius:10px;margin:10px 0;background:#fff;overflow:hidden"><tr>'
        f'<td width="56%" valign="top" style="padding:12px 14px">'
        f'<div style="font-size:15px;font-weight:700">{i}. {_esc(sym)} '
        f'<span style="color:#6b7280;font-size:12px;font-weight:400">{_esc(r.get("name"))}</span></div>'
        f'<div style="margin:4px 0 2px">{cat_chip}{hot_chip}{flag_chip}</div>'
        f'<div style="font-size:13px;color:#111;margin-top:4px;line-height:1.5">{_esc(r.get("summary"))}</div>'
        f'{pts_html}{act}{news}</td>'
        f'<td width="44%" valign="top" style="padding:12px 12px 12px 0">{chart}'
        f'<div style="margin-top:6px">{_metric_chips(m)}</div></td></tr></table>')


def _sell_card(i, s):
    ret = s.get("ret_pct")
    ret_chip = (_chip(f'추천이후 {ret:+.0f}%', "#15803d" if (ret or 0) >= 0 else "#b91c1c", True)
                if ret is not None else "")
    since = f' · {_esc(s.get("since"))} 추천' if s.get("since") else ""
    cmt = f'<div style="font-size:12px;color:#374151;margin-top:4px;line-height:1.5">{_esc(s.get("comment"))}</div>' if s.get("comment") else ""
    return (
        f'<div style="border:1px solid #fecaca;border-radius:10px;padding:11px 13px;margin:8px 0;background:#fef2f2">'
        f'<div style="font-size:14px;font-weight:700">{i}. {_esc(s.get("symbol"))} '
        f'<span style="color:#6b7280;font-size:12px;font-weight:400">{_esc(s.get("name"))}{since}</span> {ret_chip}</div>'
        f'<div style="font-size:12px;color:#b91c1c;margin-top:3px">⚠ {_esc(s.get("reason"))}</div>{cmt}</div>')


def render_report_html(report, as_of="", metrics_by_sym=None, market_html="", signals_html="",
                       kr_sells=None, banner=""):
    """일일 리포트 HTML.
    market_html  = 전일 세계시장 요약 표(market_signals.world_table_html)
    signals_html = 핵심 6자산 신호 카드(market_signals.signal_cards_html)
    banner       = 휴장 안내 등 상단 배너 텍스트"""
    if not report:
        return ""
    metrics_by_sym = metrics_by_sym or {}
    buy_cards = "".join(_card(i, r, metrics_by_sym, "buy") for i, r in enumerate(report.get("buy_now", []), 1))
    watch_cards = "".join(_card(i, r, metrics_by_sym, "watch") for i, r in enumerate(report.get("watch", []), 1))
    kr_buy_cards = "".join(_card(i, r, metrics_by_sym, "buy") for i, r in enumerate(report.get("kr_buy", []), 1))
    kr_watch_cards = "".join(_card(i, r, metrics_by_sym, "watch") for i, r in enumerate(report.get("kr_watch", []), 1))
    sells = report.get("sells", [])
    sell_html = ""
    if sells:
        sell_cards = "".join(_sell_card(i, s) for i, s in enumerate(sells, 1))
        sell_html = ('<h3 style="margin:18px 0 2px">🔴 미국 매도 · 차익실현 검토 <span style="color:#9ca3af;font-size:12px">'
                     '(이전 추천 종목 중 추세 이탈)</span></h3>' + sell_cards)
    kr_sell_html = ""
    if kr_sells:
        kr_sell_cards = "".join(_sell_card(i, s) for i, s in enumerate(kr_sells, 1))
        kr_sell_html = ('<h3 style="margin:18px 0 2px">🔴 한국 매도 · 차익실현 검토 <span style="color:#9ca3af;font-size:12px">'
                        '(이전 추천 종목 중 추세 이탈)</span></h3>' + kr_sell_cards)
    sub = f' <span style="color:#9ca3af;font-size:12px">({_esc(as_of)} 종가 기준)</span>' if as_of else ""
    spy = '<img src="cid:spy_chart" style="width:100%;max-width:640px;border-radius:8px;margin:8px 0">'
    banner_html = (f'<div style="background:#fffbeb;border:1px solid #fcd34d;border-radius:8px;padding:7px 11px;'
                   f'font-size:12px;color:#92400e;margin:8px 0">ℹ️ {_esc(banner)}</div>') if banner else ""
    market_sec = ""
    if market_html:
        market_sec = ('<h3 style="margin:14px 0 6px">🌐 전일 시장 요약 <span style="color:#9ca3af;font-size:12px">'
                      '(미국 · 한국 · 코인 · 세계)</span></h3>' + market_html)
    signals_sec = ""
    if signals_html:
        note = _esc(report.get("signal_note") or "")
        note_html = (f'<div style="font-size:13px;color:#111;margin:4px 0 8px;line-height:1.55">{note}</div>'
                     if note else "")
        signals_sec = ('<h3 style="margin:18px 0 4px">🧭 지수·코인 추세 신호 <span style="color:#9ca3af;font-size:12px">'
                       '(규칙 기반 — STRATEGY.md)</span></h3>' + note_html + signals_html)
    kr_sec = ""
    if kr_buy_cards or kr_watch_cards:
        kr_sec = (
            '<h3 style="margin:18px 0 2px">🇰🇷 코스피200 지금 매수 검토 <span style="color:#9ca3af;font-size:12px">'
            '(펀더멘탈 필터 + 추세 상위)</span></h3>' + (kr_buy_cards or '<div style="font-size:12px;color:#6b7280">해당 없음</div>')
            + ('<h3 style="margin:18px 0 2px">🇰🇷 코스피200 관찰 · 내려오면 매수</h3>' + kr_watch_cards if kr_watch_cards else ""))
    return (
        f'<div style="font-family:-apple-system,BlinkMacSystemFont,\'Segoe UI\',\'Malgun Gothic\',sans-serif;'
        f'max-width:700px;margin:0 auto;color:#111">'
        f'<h2 style="margin:6px 0">📈 데일리 시장 점검 · 종목추천{sub}</h2>'
        f'{banner_html}'
        f'<div style="background:#f8fafc;border-left:3px solid #6b7280;padding:8px 12px;font-size:13px;'
        f'line-height:1.6;margin:8px 0"><b>🧭 시장</b> {_esc(report.get("market_overview"))}<br>'
        f'<b>🌐 환율·한국·코인</b> {_esc(report.get("macro"))}</div>'
        f'{market_sec}'
        f'{signals_sec}'
        f'{spy}'
        f'<h3 style="margin:16px 0 2px">⭐ 미국(S&P500) 지금 매수 검토 <span style="color:#9ca3af;font-size:12px">'
        f'(점수 상위 · 200일선 위 · 52주 고점 -25% 이내)</span></h3>{buy_cards}'
        f'<h3 style="margin:18px 0 2px">👀 미국 관찰 · 내려오면 매수 <span style="color:#9ca3af;font-size:12px">'
        f'(좋은 종목이나 지금은 조정 중)</span></h3>{watch_cards}'
        f'{sell_html}'
        f'{kr_sec}'
        f'{kr_sell_html}'
        f'<div style="font-size:11px;color:#9ca3af;margin-top:14px;line-height:1.5">'
        f'⚠️ {_esc(report.get("risks"))}<br>정보 제공용이며 투자 권유가 아닙니다. 판단·책임은 본인에게 있습니다.<br>'
        f'매도 규칙: 트레일링 -20% 또는 200일선 -3% 이탈 (미국·한국 공통).</div>'
        f'</div>')
