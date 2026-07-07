#!/usr/bin/env python3
"""
daily_ai_report.py — 완전 자동(무인) 'AI 종목추천 보고서' 러너.

GitHub Actions(장 마감 직후)에서 이 파일 하나만 실행하면:
  1) 데이터 수집(sp500_daily_report 계산 재사용)
  2) 후보/시황 컴팩트화(export_data 재사용)
  3) Claude API 로 보고서 생성(web_search 로 뉴스·환율·글로벌 반영)  [ai_report]
  4) 차트 렌더(후보 시세 배열로 matplotlib)
  5) 이메일 발송 → 아빠(EMAIL_TO)
보고서 생성 실패/키없음 시 → 기존 규칙기반 메일(daily_main)로 자동 폴백.

실행:  python daily_ai_report.py            # 발송
       python daily_ai_report.py --no-email # 미리보기만(output/ai_report.html)
"""
from __future__ import annotations
import os, sys, io, json, argparse

import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm

# 차트에 한글(‘20일선’ 등) 표기 위해 시스템 한글 폰트 등록(있으면). 없으면 영문 폴백.
_KFONT = None
for _p in (r"C:\Windows\Fonts\malgun.ttf", r"C:\Windows\Fonts\malgunsl.ttf",
           "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"):
    if os.path.exists(_p):
        try:
            _fm.fontManager.addfont(_p)
            _KFONT = _fm.FontProperties(fname=_p).get_name()
            plt.rcParams["font.family"] = _KFONT
            break
        except Exception:
            pass
plt.rcParams["axes.unicode_minus"] = False

# yfinance 소음(상장폐지 경고·HTTP404) 억제 — 화면을 깔끔하게.
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

import sp500_daily_report as R
import export_data as E
import ai_report as AR

MAX_CANDIDATES = int(os.environ.get("REPORT_MAX_CANDIDATES", "60"))
_MA_LABEL = {20: "20일선", 50: "50일선", 200: "200일선"} if _KFONT else {20: "MA20", 50: "MA50", 200: "MA200"}
_CLOSE_LABEL = "종가" if _KFONT else "Close"


def _ma(arr, w):
    """단순 이동평균(끝쪽만 유효, 앞은 NaN)."""
    a = np.asarray(arr, dtype=float)
    if len(a) < w:
        return np.full(len(a), np.nan)
    c = np.cumsum(np.insert(a, 0, 0.0))
    out = np.full(len(a), np.nan)
    out[w - 1:] = (c[w:] - c[:-w]) / w
    return out


def _stock_chart_png(closes, ticker, big=False):
    """종가 + 이동평균선(20/50/200) 차트 PNG. big=True면 고해상도(SPY 등 넓게 보이는 차트)."""
    if not closes or len(closes) < 30:
        return None
    x = range(len(closes))
    figsize = (7.6, 2.7) if big else (4.8, 2.4)
    dpi = 200 if big else 150            # 흐릿함 개선: 표시 크기 대비 픽셀 충분히
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot(x, closes, lw=1.6, color="#111827", label=_CLOSE_LABEL)
    for w, col in [(20, "#f59e0b"), (50, "#3b82f6"), (200, "#ef4444")]:
        if len(closes) >= w:
            ax.plot(x, _ma(closes, w), lw=1.1, color=col, label=_MA_LABEL[w])
    ax.set_title(ticker, fontsize=10, loc="left", color="#111827", fontweight="bold")
    ax.legend(fontsize=8 if big else 7, loc="upper left", frameon=False, ncol=4,
              handlelength=1.1, columnspacing=0.9, borderpad=0.1)
    ax.margins(x=0)
    ax.grid(True, alpha=0.15, lw=0.5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(labelsize=7, length=0)
    ax.set_xticks([])
    b = io.BytesIO()
    fig.savefig(b, format="png", bbox_inches="tight"); plt.close(fig)
    return b.getvalue()


def _charts_and_metrics(candidates: dict, market: dict, report: dict):
    """추천 종목별 (이동평균 포함) 차트 + SPY 차트 이미지, 그리고 지표칩용 metrics_by_sym."""
    images = []
    by_sym = {c["symbol"]: c for c in candidates.get("candidates", [])}
    # SPY(라인만)
    spy = market.get("spy_closes") or []
    if spy:
        png = _stock_chart_png(spy, "S&P 500 (SPY)", big=True)   # 고해상도
        if png:
            images.append(("spy_chart", png))
    metrics = {}
    for r in (report.get("buy_now", []) + report.get("watch", [])):
        sym = r.get("symbol"); c = by_sym.get(sym)
        if not c:
            continue
        png = _stock_chart_png(c.get("closes") or [], sym)
        if png:
            images.append((f"chart_{sym}", png))
        price, ma200, ret = c.get("price"), c.get("ma200"), (c.get("ret") or {})
        gap200 = ((price / ma200 - 1) * 100) if (price and ma200) else None
        metrics[sym] = {"price": price, "pe": c.get("pe"), "rsi": c.get("rsi"),
                        "gap200": gap200, "ret6m": ret.get("6m")}
    return images, metrics


_LAST_SENT = os.path.join("output", "last_sent.json")


def _load_last_sent() -> dict:
    try:
        with open(_LAST_SENT, encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _save_last_sent(d: dict):
    os.makedirs("output", exist_ok=True)
    with open(_LAST_SENT, "w", encoding="utf-8") as f:
        json.dump(d, f)


def run(no_email: bool = False, force: bool = False):
    import datetime as _dt
    R._require_yf()
    today_kst = _dt.datetime.now(R.KST).date().isoformat()
    last = _load_last_sent()

    # 중복 발송 가드: 같은 KST 날짜에 이미 발송했으면 생략(재실행·수동 트리거 대비)
    if not force and not no_email and last.get("sent_kst") == today_kst:
        print(f"[중복] 오늘({today_kst}) 이미 발송 → 생략", file=sys.stderr)
        return

    # ── 1) 지수·코인 신호 + 세계시장 요약 (미국 종목과 독립 — 실패해도 리포트 계속)
    import market_signals as MS
    signals = {}
    try:
        signals = MS.gather(R.yf)
        print(f"[신호] 핵심자산 {len(signals.get('core', []))} · 세계 {len(signals.get('world', []))}",
              file=sys.stderr)
    except Exception as e:
        print(f"[경고] 지수·코인 신호 수집 실패({type(e).__name__}: {e}) → 해당 섹션 생략", file=sys.stderr)

    # ── 2) 코스피200 선별 (실패해도 리포트 계속)
    import kr_stocks as KR
    kr = {}
    kr_sells = []
    try:
        kr = KR.select(R.yf) or {}
        if kr.get("ind_map"):
            kr_sells = KR.update_holdings([c["symbol"] for c in kr.get("buy", [])],
                                          kr["ind_map"], today_kst)
    except Exception as e:
        print(f"[경고] 코스피200 선별 실패({type(e).__name__}: {e}) → 해당 섹션 생략", file=sys.stderr)

    # ── 3) 미국(S&P500) 파이프라인 (기존)
    data = R.gather_universe_data(with_volume=True)
    as_of = R._last_data_date(data["hist"])

    # 미국 휴장 안내(발송은 계속 — 한국·코인·세계 시황은 매일 새로움)
    banner = ""
    if as_of and last.get("us_as_of") == as_of:
        banner = f"미국 휴장 — 미국 종목 추천은 직전 거래일({as_of}) 종가 기준입니다."
    if kr.get("as_of") and last.get("kr_as_of") == kr.get("as_of"):
        banner += (" " if banner else "") + "한국 휴장 — 한국 종목은 직전 거래일 기준입니다."
    # 후보 선정: 백테스트가 찾은 '최적 가중치'(best_weights.json) 우선 → 없으면 최우수모델 → 하이브리드.
    scored, info, model_used = E.select_pool(data, MAX_CANDIDATES)
    print(f"[선정] 방식='{model_used}' 로 후보 {len(scored)}종목", file=sys.stderr)
    candidates = {"as_of": as_of, "model": model_used, "count": min(len(scored), MAX_CANDIDATES),
                  "candidates": E.build_candidates(data, info, scored, MAX_CANDIDATES)}
    market = {"as_of": as_of, **E.build_market(data)}

    # 지금 매수 5 + 관찰(내려오면) 5 로 '점수순 고정' 분할 (AI는 종목을 못 바꿈)
    buy_now, watch = E.split_by_entry(candidates["candidates"], k=5)

    # 추천 이력 자동 추적 → 보유 갱신 + 매도 시그널(느슨/장기보유)
    import holdings as H
    hstate = H.load()
    sells = H.update(hstate, [c["symbol"] for c in buy_now], data["ind_map"], as_of)
    if sells:                                     # 매도 종목 회사명 보강
        sinfo = R.get_info_for([s["symbol"] for s in sells])
        for s in sells:
            s["name"] = R._company_name(s["symbol"], sinfo.get(s["symbol"], {}))
    H.save(hstate)
    print(f"[분할] 지금매수 {len(buy_now)} · 관찰 {len(watch)} · 매도검토 {len(sells)} · 보유 {len(hstate.get('holdings', {}))}",
          file=sys.stderr)
    # 뉴스는 야후 헤드라인을 미리 받아 AI에 주입(웹검색 불필요 → 타임아웃 방지)
    try:
        from ai_commentary import fetch_news_headlines
        heads = fetch_news_headlines([c["symbol"] for c in buy_now + watch], getattr(R, "yf", None))
        for c in buy_now + watch:
            c["headlines"] = (heads.get(c["symbol"]) or [])[:4]
    except Exception as _e:
        print(f"[정보] 뉴스 헤드라인 수집 생략({_e})", file=sys.stderr)

    # 신호·세계시장을 AI 컨텍스트에 주입
    if signals:
        market["signals"] = MS.lean_for_ai(signals)
        market["world"] = [{k: (round(v, 2) if isinstance(v, float) else v)
                            for k, v in w.items()} for w in signals.get("world", [])]

    groups = {"buy_now": buy_now, "watch": watch, "sells": sells,
              "kr_buy": kr.get("buy") or [], "kr_watch": kr.get("watch") or []}
    report = AR.build_report(groups, market)
    if not report:
        # AI 실패해도 지표 기반 새 형식으로 무조건 발송(발송 누락 방지). 옛 4섹션 폴백 폐기.
        print("[정보] AI 생성 실패 → 지표 기반(무AI) 리포트로 발송", file=sys.stderr)
        report = AR.deterministic_report(groups, market)

    images, metrics = _charts_and_metrics(candidates, market, report)

    # 한국 종목 차트·지표칩 + 지수·코인 신호 차트
    for c in (kr.get("buy") or []) + (kr.get("watch") or []):
        png = _stock_chart_png(c.get("closes") or [], f'{c["name"]} ({c["symbol"]})')
        if png:
            images.append((f"chart_{c['symbol']}", png))
        gap200 = ((c["price"] / c["ma200"] - 1) * 100) if (c.get("price") and c.get("ma200")) else None
        metrics[c["symbol"]] = {"price": c.get("price"), "pe": c.get("pe"), "rsi": c.get("rsi"),
                                "gap200": gap200, "ret6m": (c.get("ret") or {}).get("6m"), "krw": True}
    sig_cids = {}
    for a in signals.get("core", []):
        png = _stock_chart_png(a.get("closes") or [], a["name"])
        if png:
            cid = f"sig_{a['key']}"
            images.append((cid, png)); sig_cids[a["key"]] = cid

    market_html = MS.world_table_html(signals) if signals else ""
    signals_html = MS.signal_cards_html(signals, sig_cids) if signals else ""
    html = AR.render_report_html(report, as_of, metrics, market_html=market_html,
                                 signals_html=signals_html, kr_sells=kr_sells, banner=banner)

    os.makedirs("output", exist_ok=True)
    # 미리보기는 이미지를 data-URI 로 바꿔 파일 단독으로 열리게
    prev = html
    import base64
    for cid, png in images:
        prev = prev.replace(f"cid:{cid}", "data:image/png;base64," + base64.b64encode(png).decode())
    with open("output/ai_report.html", "w", encoding="utf-8") as f:
        f.write(prev)
    print(f"[정보] 미리보기 output/ai_report.html "
          f"(지금매수 {len(report.get('buy_now', []))} · 관찰 {len(report.get('watch', []))} · "
          f"매도 {len(report.get('sells', []))})", file=sys.stderr)

    if not no_email:
        subject = f"[데일리] {today_kst} 시장 점검 · 종목추천"
        if R.send_email(subject, html, images):
            _save_last_sent({"sent_kst": today_kst, "us_as_of": as_of,
                             "kr_as_of": kr.get("as_of")})   # 발송 성공 시에만 기록(실패하면 다음날 재시도)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="AI 종목추천 보고서(완전 자동)")
    ap.add_argument("--no-email", action="store_true")
    ap.add_argument("--weekly", action="store_true", help="주간 자산배분 리포트 강제 실행")
    ap.add_argument("--daily", action="store_true", help="요일과 무관하게 일일 리포트 강제 실행")
    ap.add_argument("--force", action="store_true", help="중복 발송 체크 무시")
    args = ap.parse_args()
    import datetime as _dt
    # 요일은 반드시 KST 기준(깃허브 러너는 UTC — 22:30 UTC 실행 시 KST는 다음날 07:30)
    _dow = _dt.datetime.now(R.KST).weekday()   # 월=0 … 일=6
    if args.weekly or (_dow == 6 and not args.daily):
        # 일요일(KST): 개별 종목 대신 자산군별 주간 리포트
        import weekly_report
        weekly_report.run(no_email=args.no_email)
    elif _dow == 0 and not (args.daily or args.no_email):
        # 월요일(KST): 발송 없음(주말 — 새 거래일 없음). --daily 로 강제 가능.
        print("[휴무] 월요일은 발송하지 않습니다(화~토 일일, 일 주간). --daily 로 강제 실행 가능.",
              file=sys.stderr)
    else:
        run(no_email=args.no_email, force=args.force)
