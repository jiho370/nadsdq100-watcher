#!/usr/bin/env python3
"""
daily_ai_report.py — 메일 2통 분리 러너 (2026-07-09 개편).

  · 한국장 메일 (--kr) : 월~금 KST 08:00 발송(장전). 전날 한국장 마감 데이터 기준.
        내용 = 전일 세계시장 요약(밤사이 미국 마감 포함, 코드 생성) + 지수·코인 신호
               + 코스피200 매수/관찰/매도. AI 검증은 전날 저녁 pregen_kr.json(구독 CLI)
               이 있으면 재사용(검색 0회), 없으면 API 폴백.
  · 미국장 메일 (--us) : 화~토 KST 17:00 발송(미국장 개장 전). 그날 새벽 마감 데이터 기준.
        내용 = 미국 시황 + S&P500 매수/관찰/매도. 당일 아침~오후 pregen_us.json 재사용.
  · 주간   (--weekly) : 일요일 자산배분 리포트(기존 weekly_report).

실행:  python daily_ai_report.py --kr [--no-email]
       python daily_ai_report.py --us [--no-email]
       (플래그 없으면 KST 시간으로 자동: 일요일=주간, 오전=--kr, 오후=--us)
AI 실패 시 지표+계획 기반(deterministic)으로 무조건 발송 — 발송 누락 없음.
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

logging.getLogger("yfinance").setLevel(logging.CRITICAL)

import sp500_daily_report as R
import export_data as E
import ai_report as AR

MAX_CANDIDATES = int(os.environ.get("REPORT_MAX_CANDIDATES", "60"))
_MA_LABEL = {20: "20일선", 50: "50일선", 200: "200일선"} if _KFONT else {20: "MA20", 50: "MA50", 200: "MA200"}
_CLOSE_LABEL = "종가" if _KFONT else "Close"
_PORT_LABEL = "보유 포트폴리오" if _KFONT else "My holdings"
_BENCH_SUFFIX = "(동일시점·동일금액)" if _KFONT else "(same timing/amount)"
_CUM_RET_LABEL = "누적수익률(%)" if _KFONT else "Cumulative return (%)"


def _ma(arr, w):
    a = np.asarray(arr, dtype=float)
    if len(a) < w:
        return np.full(len(a), np.nan)
    c = np.cumsum(np.insert(a, 0, 0.0))
    out = np.full(len(a), np.nan)
    out[w - 1:] = (c[w:] - c[:-w]) / w
    return out


def _stock_chart_png(closes, ticker, big=False):
    """종가 + 이동평균선(20/50/200) 차트 PNG."""
    if not closes or len(closes) < 30:
        return None
    x = range(len(closes))
    figsize = (7.6, 2.7) if big else (4.8, 2.4)
    dpi = 200 if big else 150
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


def _holdings_compare_chart_png(series: dict, index_name: str):
    """포트폴리오(각 종목 진입일에 동일 금액 투입 가정) 누적수익률 vs 같은 날짜에 같은 금액을
    지수에 넣었을 때의 누적수익률 — 시계열 라인 비교."""
    dates = series.get("dates") or []
    if not dates:
        return None
    port = [v if v is not None else np.nan for v in series["portfolio"]]
    bench = [v if v is not None else np.nan for v in series["bench"]]
    x = np.arange(len(dates))
    fig, ax = plt.subplots(figsize=(7.6, 3.0), dpi=150)
    ax.plot(x, port, lw=1.8, color="#2563eb", label=_PORT_LABEL)
    ax.plot(x, bench, lw=1.4, color="#9ca3af", label=f"{index_name} {_BENCH_SUFFIX}")
    ax.axhline(0, color="#111827", lw=0.8)
    ticks = np.linspace(0, len(dates) - 1, min(6, len(dates))).astype(int)
    ax.set_xticks(ticks); ax.set_xticklabels([dates[i][5:] for i in ticks], fontsize=8)
    ax.set_ylabel(_CUM_RET_LABEL, fontsize=9)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.grid(True, axis="y", alpha=0.15, lw=0.5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(labelsize=8, length=0)
    b = io.BytesIO()
    fig.savefig(b, format="png", bbox_inches="tight"); plt.close(fig)
    return b.getvalue()


def _bench_series(signals, key):
    """지수·코인 신호(market_signals.gather 결과)에서 key(예: 'KOSPI')의 날짜·종가 시계열을 뽑는다."""
    for a in (signals or {}).get("core", []):
        if a.get("key") == key:
            return a.get("dates") or [], a.get("closes") or []
    return [], []


def _holdings_section(hstate, ind_map, price_map, bench_dates, bench_closes, index_name, krw=False,
                      name_map=None):
    """holdings.py 보유현황 표+포트폴리오 비교차트를 조립. (html, images) — 데이터 없으면 ("", [])."""
    import holdings as H
    summary = H.live_summary(hstate, ind_map)
    if not summary:
        return "", []
    series = H.portfolio_series(summary, price_map, bench_dates, bench_closes)
    images, chart_cid, totals = [], None, None
    if series:
        png = _holdings_compare_chart_png(series, index_name)
        if png:
            chart_cid = "holdings_cmp"
            images.append((chart_cid, png))
        lp = [v for v in series["portfolio"] if v is not None]
        lb = [v for v in series["bench"] if v is not None]
        if lp and lb:
            totals = {"strategy": lp[-1], "bench": lb[-1], "index_name": index_name}
    return AR.holdings_table_html(summary, krw=krw, chart_cid=chart_cid, totals=totals,
                                  name_map=name_map), images


# ------------------------- 공용 헬퍼 -------------------------
_LAST_SENT = os.path.join("output", "last_sent.json")


def _load_last_sent() -> dict:
    try:
        with open(_LAST_SENT, encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _save_last_sent(update: dict):
    """부분 갱신 — KR/US 메일이 서로의 기록을 덮어쓰지 않게 merge."""
    d = _load_last_sent(); d.update(update)
    os.makedirs("output", exist_ok=True)
    with open(_LAST_SENT, "w", encoding="utf-8") as f:
        json.dump(d, f)


def _load_pregen(name: str, today_kst: str):
    """output/pregen_{kr|us}.json — 대상일(for_kst)이 오늘이면 반환, 아니면 None."""
    try:
        with open(f"output/pregen_{name}.json", encoding="utf-8") as f:
            pg = json.load(f)
        if pg.get("for_kst") == today_kst and pg.get("by_sym"):
            print(f"[pregen:{name}] {pg.get('generated')} 생성본 사용({len(pg['by_sym'])}종목)",
                  file=sys.stderr)
            return pg
        print(f"[pregen:{name}] 대상일 불일치({pg.get('for_kst')} != {today_kst}) → API 검증",
              file=sys.stderr)
    except Exception:
        pass
    return None


def _gather_signals():
    """지수·코인 신호 + 세계시장 요약(코드 생성, AI 비용 0). 실패해도 리포트 계속."""
    import market_signals as MS
    try:
        signals = MS.gather(R.yf)
        print(f"[신호] 핵심자산 {len(signals.get('core', []))} · 세계 {len(signals.get('world', []))}",
              file=sys.stderr)
        return MS, signals
    except Exception as e:
        print(f"[경고] 지수·코인 신호 수집 실패({type(e).__name__}: {e}) → 해당 섹션 생략", file=sys.stderr)
        return MS, {}


def _signal_images(signals, when=None):
    """when 지정 시 그 메일에 표시되는 추세신호 자산(코스피/코스닥/금 또는 나스닥100/S&P500/
    비트코인)만 차트를 만든다 — 안 쓸 이미지를 만들지 않는다."""
    sig_cids, images = {}, []
    items = [a for a in signals.get("core", []) if not when or a.get("when") == when]
    for a in items:
        png = _stock_chart_png(a.get("closes") or [], a["name"])
        if png:
            cid = f"sig_{a['key']}"
            images.append((cid, png)); sig_cids[a["key"]] = cid
    return images, sig_cids


def _attach_headlines(cands, suffix=""):
    """야후 헤드라인(무료)을 후보에 주입 — AI 웹검색 의존 축소. 한국은 '.KS' 접미사."""
    try:
        from ai_commentary import fetch_news_headlines
        ysyms = {c["symbol"]: c["symbol"] + suffix for c in cands}
        heads = fetch_news_headlines(list(ysyms.values()), getattr(R, "yf", None))
        for c in cands:
            c["headlines"] = (heads.get(ysyms[c["symbol"]]) or [])[:4]
    except Exception as e:
        print(f"[정보] 뉴스 헤드라인 수집 생략({e})", file=sys.stderr)


def _preview_and_send(html, images, subject, out_name, no_email, sent_update):
    os.makedirs("output", exist_ok=True)
    import base64
    prev = html
    for cid, png in images:
        prev = prev.replace(f"cid:{cid}", "data:image/png;base64," + base64.b64encode(png).decode())
    with open(f"output/{out_name}", "w", encoding="utf-8") as f:
        f.write(prev)
    print(f"[정보] 미리보기 output/{out_name}", file=sys.stderr)
    if not no_email:
        if R.send_email(subject, html, images):
            _save_last_sent(sent_update)   # 발송 성공 시에만 기록(실패하면 다음 실행 때 재시도)


# ------------------------- 한국장 메일 (장전 08:00) -------------------------
def run_kr(no_email: bool = False, force: bool = False):
    import datetime as _dt
    R._require_yf()
    today_kst = _dt.datetime.now(R.KST).date().isoformat()
    last = _load_last_sent()
    if not force and not no_email and last.get("sent_kr_kst") == today_kst:
        print(f"[중복] 오늘({today_kst}) 한국장 메일 이미 발송 → 생략", file=sys.stderr)
        return

    MS, signals = _gather_signals()

    import kr_stocks as KR
    kr, kr_sells = {}, []
    try:
        kr = KR.select(R.yf) or {}
        if kr.get("ind_map"):
            kr_sells = KR.update_holdings([], kr["ind_map"], today_kst,
                                          pool_syms=kr.get("pool"))
            for s in kr_sells:
                cl = (kr["ind_map"].get(s["symbol"]) or {}).get("closes")
                if cl:
                    s["closes"] = cl
    except Exception as e:
        print(f"[경고] 코스피200 선별 실패({type(e).__name__}: {e})", file=sys.stderr)

    banner = ""
    if kr.get("as_of") and last.get("kr_as_of") == kr.get("as_of"):
        banner = f"한국 휴장 — 직전 거래일({kr['as_of']}) 종가 기준입니다."

    kr_cands = (kr.get("buy") or []) + (kr.get("watch") or [])
    _attach_headlines(kr_cands, suffix=".KS")

    # 시황 컨텍스트: 신호(국장 표시분=코스피/코스닥/금만)+세계(밤사이 미국 마감 포함 — 코드 계산이라 비용 0)
    is_monday = _dt.datetime.now(R.KST).weekday() == 0
    market = {"as_of": kr.get("as_of"), "weekly_recap": is_monday}
    if signals:
        market["signals"] = MS.lean_for_ai(signals, when="kr")
        market["world"] = [{k: (round(v, 2) if isinstance(v, float) else v)
                            for k, v in w.items()} for w in signals.get("world", [])]

    groups = {"kr_buy": kr.get("buy") or [], "kr_watch": kr.get("watch") or [],
              "kr_sells": kr_sells}
    report = AR.build_report(groups, market, pregen=_load_pregen("kr", today_kst))
    if not report:
        print("[정보] AI 실패 → 지표+계획 기반 리포트로 발송", file=sys.stderr)
        report = AR.deterministic_report(groups, market)

    # 최종 매수만 보유목록 편입
    holdings_html = ""
    holdings_images = []
    if kr.get("ind_map"):
        try:
            KR.add_holdings([r["symbol"] for r in report.get("kr_buy", [])], kr["ind_map"], today_kst)
            import holdings as H
            kr_state = H.load(KR.KR_HOLDINGS)
            bench_dates, bench_closes = _bench_series(signals, "KOSPI")
            price_map = {sym: {"dates": (kr["ind_map"].get(sym) or {}).get("dates") or [],
                               "closes": (kr["ind_map"].get(sym) or {}).get("closes") or []}
                         for sym in kr_state.get("holdings", {})}
            # 보유현황 표 종목명: 종목코드(숫자)만으론 못 알아보므로 이름으로 치환(2026-07-15).
            # 오늘 후보풀(kr_cands)엔 있지만, 보유 중인데 오늘 후보풀 밖으로 밀린 종목은 코스피200
            # 캐시(kr_stocks._cached_universe)로 보강 — 그래도 없으면 표시 단계에서 코드 그대로.
            name_map = {c["symbol"]: c.get("name") for c in kr_cands if c.get("name")}
            missing = [s for s in kr_state.get("holdings", {}) if s not in name_map]
            if missing:
                try:
                    uni = KR._cached_universe() or {}
                    for s in missing:
                        n = (uni.get(s) or {}).get("name")
                        if n:
                            name_map[s] = n
                except Exception:
                    pass
            holdings_html, holdings_images = _holdings_section(
                kr_state, kr["ind_map"], price_map, bench_dates, bench_closes, "코스피", krw=True,
                name_map=name_map)
        except Exception as e:
            print(f"[경고] 한국 보유목록 갱신 실패({e})", file=sys.stderr)

    # 차트·지표칩(최종 종목) + 신호 차트
    images, metrics = list(holdings_images), {}
    kr_by_sym = {c["symbol"]: c for c in kr_cands}
    for r in report.get("kr_buy", []) + report.get("kr_watch", []):
        c = kr_by_sym.get(r.get("symbol"))
        if not c:
            continue
        png = _stock_chart_png(c.get("closes") or [], c["name"])   # 국장은 차트 제목도 코드 없이 이름만
        if png:
            images.append((f"chart_{c['symbol']}", png))
        gap200 = ((c["price"] / c["ma200"] - 1) * 100) if (c.get("price") and c.get("ma200")) else None
        metrics[c["symbol"]] = {"price": c.get("price"), "pe": c.get("pe"), "rsi": c.get("rsi"),
                                "gap200": gap200, "ret6m": (c.get("ret") or {}).get("6m"), "krw": True}
    sig_images, sig_cids = _signal_images(signals, when="kr")
    images += sig_images

    # 전일(월요일엔 전주) 시장 요약(나스닥·다우존스·닛케이·유럽·글로벌·비트코인·환율)은 국장 메일에만
    # 붙는다 — 밤사이 미국장 등 요약이 필요한 건 이쪽뿐이고, 추세신호(코스피/코스닥/금)와도 안 겹친다.
    market_html = MS.world_table_html(signals, weekly=is_monday) if signals else ""
    signals_html = MS.signal_cards_html(signals, sig_cids, when="kr") if signals else ""
    html = AR.render_report_html(report, kr.get("as_of") or "", metrics,
                                 market_html=market_html, signals_html=signals_html,
                                 banner=banner, show_spy=False, is_kr=True,
                                 market_label=("전주" if is_monday else "전일"),
                                 title="🇰🇷 장전 시장 점검 · 코스피200 매수·매도 후보",
                                 holdings_html=holdings_html)
    _preview_and_send(html, images, f"[장전] {today_kst} 한국 시장 점검 · 매수·매도 후보",
                      "kr_report.html", no_email,
                      {"sent_kr_kst": today_kst, "kr_as_of": kr.get("as_of")})


# ------------------------- 미국장 메일 (마감 후 17:00) -------------------------
def run_us(no_email: bool = False, force: bool = False):
    import datetime as _dt
    R._require_yf()
    today_kst = _dt.datetime.now(R.KST).date().isoformat()
    last = _load_last_sent()
    if not force and not no_email and last.get("sent_us_kst") == today_kst:
        print(f"[중복] 오늘({today_kst}) 미국장 메일 이미 발송 → 생략", file=sys.stderr)
        return

    MS, signals = _gather_signals()

    data = R.gather_universe_data(with_volume=True)
    as_of = R._last_data_date(data["hist"])
    banner = ""
    if as_of and last.get("us_as_of") == as_of:
        banner = f"미국 휴장 — 직전 거래일({as_of}) 종가 기준입니다."

    scored, info, model_used = E.select_pool(data, MAX_CANDIDATES)
    print(f"[선정] 방식='{model_used}' 로 후보 {len(scored)}종목", file=sys.stderr)
    candidates = {"as_of": as_of, "candidates": E.build_candidates(data, info, scored, MAX_CANDIDATES)}
    market = {"as_of": as_of, **E.build_market(data)}

    # 관찰 폐지(2026-07-13): 풀 전체를 매수 후보로 — split_by_entry는 hot 태그 부여용으로만 호출
    pool_k = int(os.environ.get("REPORT_POOL", "10"))
    buy_now, _ = E.split_by_entry(candidates["candidates"], k=pool_k)
    watch = []

    import holdings as H
    hstate = H.load()
    # pool_syms = 오늘 팩터 후보풀(60) — 6개월 경과 보유종목이 풀 밖이면 정기 재평가 매도(2026-07 재검증)
    sells = H.update(hstate, [], data["ind_map"], as_of,
                     pool_syms={s for s, _, _ in scored})
    if sells:
        sinfo = R.get_info_for([s["symbol"] for s in sells])
        for s in sells:
            s["name"] = R._company_name(s["symbol"], sinfo.get(s["symbol"], {}))
            hist_s = data["hist"].get(s["symbol"])
            if hist_s is not None:
                s["closes"] = [round(float(v), 2) for v in hist_s.dropna().tail(252).tolist()]
    print(f"[후보풀] 매수 {len(buy_now)} · 관찰 {len(watch)} · 매도검토 {len(sells)}", file=sys.stderr)
    _attach_headlines(buy_now + watch)

    if signals:
        market["signals"] = MS.lean_for_ai(signals, when="us")
        # 전일 시장 요약(world)은 국장 메일 전용 — 미장 메일 AI 컨텍스트엔 안 넣는다(중복·불필요).

    groups = {"buy_now": buy_now, "watch": watch, "sells": sells}
    report = AR.build_report(groups, market, pregen=_load_pregen("us", today_kst))
    if not report:
        print("[정보] AI 실패 → 지표+계획 기반 리포트로 발송", file=sys.stderr)
        report = AR.deterministic_report(groups, market)

    # 편입: 보유 상한 10 + 동일 회사 복수 클래스(GOOG/GOOGL 등) 중복 배제 — 팔아야 산다
    H.add(hstate, [r["symbol"] for r in report.get("buy_now", [])], data["ind_map"], as_of,
          max_n=int(os.environ.get("US_MAX_HOLD", "8")))
    H.save(hstate)

    # 보유현황(라이브 트래킹): 전체 투입자산 기준 누적수익률 vs SPY(동일시점·동일금액) 시계열
    spy_series = data.get("spy")
    bench_dates, bench_closes = [], []
    if spy_series is not None and not spy_series.empty:
        s = spy_series.dropna()
        bench_dates = [d.date().isoformat() for d in s.index]
        bench_closes = [float(v) for v in s.tolist()]
    price_map = {}
    for sym in hstate.get("holdings", {}):
        hs = data["hist"].get(sym)
        if hs is not None and len(hs):
            hs = hs.dropna()
            price_map[sym] = {"dates": [d.date().isoformat() for d in hs.index],
                              "closes": [float(v) for v in hs.tolist()]}
    holdings_html, holdings_images = _holdings_section(
        hstate, data["ind_map"], price_map, bench_dates, bench_closes, "S&P500")

    # 차트: SPY(큰 차트) + 종목 + 신호
    images, metrics = list(holdings_images), {}
    spy_closes = market.get("spy_closes") or []
    if spy_closes:
        png = _stock_chart_png(spy_closes, "S&P 500 (SPY)", big=True)
        if png:
            images.append(("spy_chart", png))
    by_sym = {c["symbol"]: c for c in candidates["candidates"]}
    for r in report.get("buy_now", []) + report.get("watch", []):
        c = by_sym.get(r.get("symbol"))
        if not c:
            continue
        png = _stock_chart_png(c.get("closes") or [], r["symbol"])
        if png:
            images.append((f"chart_{r['symbol']}", png))
        gap200 = ((c["price"] / c["ma200"] - 1) * 100) if (c.get("price") and c.get("ma200")) else None
        metrics[r["symbol"]] = {"price": c.get("price"), "pe": c.get("pe"), "rsi": c.get("rsi"),
                                "gap200": gap200, "ret6m": (c.get("ret") or {}).get("6m")}
    sig_images, sig_cids = _signal_images(signals, when="us")
    images += sig_images

    # 전일 시장 요약 표는 국장 메일 전용 — 미장 메일엔 안 붙인다(market_html="").
    signals_html = MS.signal_cards_html(signals, sig_cids, when="us") if signals else ""
    html = AR.render_report_html(report, as_of, metrics,
                                 market_html="", signals_html=signals_html,
                                 banner=banner, show_spy=bool(spy_closes),
                                 title="🇺🇸 미국장 마감 점검 · S&P500 매수·매도 후보",
                                 holdings_html=holdings_html)
    _preview_and_send(html, images, f"[미국 마감] {today_kst} 시장 점검 · 매수·매도 후보",
                      "us_report.html", no_email,
                      {"sent_us_kst": today_kst, "us_as_of": as_of})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="AI 종목추천 보고서 — 메일 2통(한국 장전/미국 마감)")
    ap.add_argument("--kr", action="store_true", help="한국장 장전 메일")
    ap.add_argument("--us", action="store_true", help="미국장 마감 메일")
    ap.add_argument("--weekly", action="store_true", help="주간 자산배분 리포트")
    ap.add_argument("--daily", action="store_true", help="(수동) 한국+미국 둘 다 실행")
    ap.add_argument("--no-email", action="store_true")
    ap.add_argument("--force", action="store_true", help="중복 발송 체크 무시")
    args = ap.parse_args()
    import datetime as _dt
    now = _dt.datetime.now(R.KST)   # 요일·시각은 반드시 KST 기준(러너는 UTC)
    if args.weekly:
        import weekly_report
        weekly_report.run(no_email=args.no_email)
    elif args.kr:
        run_kr(no_email=args.no_email, force=args.force)
    elif args.us:
        run_us(no_email=args.no_email, force=args.force)
    elif args.daily:
        run_kr(no_email=args.no_email, force=args.force)
        run_us(no_email=args.no_email, force=args.force)
    else:
        # 플래그 없음(수동/구 스케줄 호환): 일요일=주간, 오전=한국장, 오후=미국장
        if now.weekday() == 6:
            import weekly_report
            weekly_report.run(no_email=args.no_email)
        elif now.hour < 12:
            run_kr(no_email=args.no_email, force=args.force)
        else:
            run_us(no_email=args.no_email, force=args.force)
