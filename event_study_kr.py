#!/usr/bin/env python3
"""
event_study_kr.py — KR_STRATEGY_OPTIONS.md Phase 1: DART 공시 이벤트 스터디 공용 프레임.

§2-A(자사주 소각)·§2-B(PEAD 잠정실적)·§3-C(무상증자·유상증자·CB·최대주주변동)의 이벤트별
CAR(누적초과수익) + t-stat + 서브기간 안정성을 한 프레임에서 측정한다. 검정력 이점(§5-6):
이벤트 수가 곧 표본이라 기존 팩터 백테스트의 T_eff=8 문제에서 자유롭다.

PIT 정합(§5-4): DART 접수일(rcept_dt) 기준, 진입은 그 다음 거래일(T+1) — 공시 당일 종가에
이미 반영됐을 수 있는 반응을 룩어헤드로 취하지 않는다. [기재정정] 공시는 원 이벤트의 정정
재공시라 이벤트 중복이 되므로 제외한다.

한계(정직 고지): 가격 패널이 코스피200 PIT union(276종목 — 대형주·생존 편향)이다. 이벤트
'반응이 존재하는가'를 측정하는 용도(§5 취지 그대로)이며, 스몰캡까지 포함한 실전 전략화는
§5-3 상장폐지 처분가 인프라(backtest_kr_core) 완성 후. 소각 이벤트는 §7.3 구조변화
(2026-02 상법개정 의무화)로 자발성 시대(~2026-02) 표본만 유효 — 서브기간으로 분리 확인.

실행(PC, DART_API_KEY 환경변수 필요 — opendart.fss.or.kr 무료 발급):
  python event_study_kr.py --collect          # 공시목록 수집(캐시·재개 가능)
  python event_study_kr.py                     # 이벤트 스터디 실행
  python event_study_kr.py --self-test
결과: output/event_study_kr.json (이벤트유형별 CAR·t-stat·서브기간)
"""
from __future__ import annotations
import os, sys, re, json, time, argparse, datetime as dt
import numpy as np
import pandas as pd

DISC_CACHE = "output/kr_disclosures.json"       # 원 공시목록 캐시(rcept_dt·report_nm·corp)
OUT_PATH = "output/event_study_kr.json"
LIST_URL = "https://opendart.fss.or.kr/api/list.json"
PBLNTF_TYPES = ("B", "I")                        # B 주요사항보고 · I 거래소공시(잠정실적·최대주주)
WINDOWS = (5, 20, 60)                            # CAR 측정 창(거래일)
SUBS = [("full", None, None), ("2018-2021", None, "2021-12-31"),
        ("2022-2023", "2022-01-01", "2023-12-31"), ("2024+", "2024-01-01", None)]

# report_nm(정정표기 제거 후) → 이벤트 유형. 순서대로 첫 매치 채택.
EVENT_PATTERNS = [
    ("소각",        re.compile(r"자기주식소각결정")),
    ("자사주취득",  re.compile(r"자기주식취득결정")),          # 신탁계약 체결은 별도(취득 아님)
    ("무상증자",    re.compile(r"무상증자결정")),
    ("유상증자",    re.compile(r"유상증자결정")),
    ("CB발행",      re.compile(r"전환사채권발행결정")),
    ("잠정실적",    re.compile(r"영업\(잠정\)실적|잠정실적")),
    ("최대주주변동", re.compile(r"최대주주등소유주식변동")),
]


def _log(m): print(f"[이벤트KR] {m}", file=sys.stderr)


def _classify(report_nm: str) -> str | None:
    nm = re.sub(r"^\[[^\]]*\]", "", report_nm).strip()   # [기재정정] 등 접두 제거
    if report_nm.startswith("[기재정정]") or report_nm.startswith("[첨부정정]"):
        return None                                       # 정정 재공시 = 이벤트 중복 → 제외
    for name, pat in EVENT_PATTERNS:
        if pat.search(nm):
            return name
    return None


# ------------------------- DART 공시목록 수집 -------------------------
def collect(stock_codes: list[str], bgn="20180101", end=None, sleep_s=0.12):
    key = os.environ.get("DART_API_KEY")
    if not key:
        _log("환경변수 DART_API_KEY 필요 — opendart.fss.or.kr 무료 발급"); sys.exit(1)
    import requests
    from kr_factor_ic import _corp_map
    end = end or dt.date.today().strftime("%Y%m%d")
    cache = {}
    if os.path.exists(DISC_CACHE):
        with open(DISC_CACHE, encoding="utf-8") as f:
            cache = json.load(f)
    cmap = _corp_map(key)
    n_req = 0
    for i, sc in enumerate(stock_codes):
        corp = cmap.get(sc)
        if not corp:
            continue
        rec = cache.setdefault(sc, {})
        done = set(rec.get("_done", []))
        evs = rec.setdefault("events", [])
        for ty in PBLNTF_TYPES:
            tag = f"{ty}:{bgn}:{end}"
            if tag in done:
                continue
            page, total_page = 1, 1
            ok = True
            while page <= total_page:
                try:
                    js = requests.get(LIST_URL, timeout=30, params={
                        "crtfc_key": key, "corp_code": corp, "bgn_de": bgn, "end_de": end,
                        "pblntf_ty": ty, "page_count": "100", "page_no": str(page)}).json()
                    n_req += 1
                except Exception as e:
                    _log(f"{sc} {ty} p{page} 요청 실패({e}) — 재개 대상"); ok = False; break
                st = js.get("status")
                if st == "020":
                    _log("DART 사용한도 초과(020) — 저장 후 중단, 내일 재실행 시 재개")
                    _save(cache); return cache
                if st == "013":                       # 해당 없음(공시 0건) — 정상 완료
                    break
                if st != "000":
                    _log(f"{sc} {ty} status {st}({js.get('message')}) — 재개 대상"); ok = False; break
                total_page = int(js.get("total_page") or 1)
                for it in (js.get("list") or []):
                    et = _classify(it.get("report_nm", ""))
                    if et:
                        evs.append({"date": it["rcept_dt"], "type": et,
                                    "nm": it["report_nm"]})
                page += 1
                time.sleep(sleep_s)
            if ok:
                done.add(tag)
        # 중복 제거(같은 날·같은 유형 1건)
        seen = set(); uniq = []
        for e in sorted(evs, key=lambda x: (x["date"], x["type"])):
            k = (e["date"], e["type"])
            if k not in seen:
                seen.add(k); uniq.append(e)
        rec["events"] = uniq
        rec["_done"] = sorted(done)
        if (i + 1) % 20 == 0 or i == len(stock_codes) - 1:
            _save(cache)
            _log(f"저장 {i+1}/{len(stock_codes)}종목 (요청 누적 {n_req})")
    return cache


def _save(cache):
    os.makedirs("output", exist_ok=True)
    with open(DISC_CACHE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)


# ------------------------- CAR 이벤트 스터디 -------------------------
def _abn(panel: pd.DataFrame, bench: pd.Series, sym: str, e_idx: int, k: int) -> float | None:
    """CAR[e+1, e+k] = Σ(종목수익 − 벤치수익). 공시일 다음 거래일부터(룩어헤드 방지)."""
    if sym not in panel.columns or e_idx + k >= len(panel):
        return None
    px = panel[sym].to_numpy(dtype=float)
    b = bench.to_numpy(dtype=float)
    s = 0.0
    for t in range(e_idx + 1, e_idx + k + 1):
        if not (np.isfinite(px[t]) and np.isfinite(px[t - 1]) and px[t - 1] > 0
                and np.isfinite(b[t]) and np.isfinite(b[t - 1]) and b[t - 1] > 0):
            return None
        s += (px[t] / px[t - 1] - 1) - (b[t] / b[t - 1] - 1)
    return s * 100                                        # %p


def _event_index(panel: pd.DataFrame, date_str: str) -> int | None:
    ts = pd.Timestamp(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}")
    pos = panel.index.searchsorted(ts)                    # 공시일 이상 첫 거래일
    return int(pos) if 0 <= pos < len(panel) else None


def _tstat(xs: list[float]) -> tuple[float, float, int]:
    a = np.array([x for x in xs if x is not None and np.isfinite(x)], dtype=float)
    if len(a) < 3:
        return (float("nan"), float("nan"), len(a))
    m, sd = float(a.mean()), float(a.std(ddof=1))
    t = m / (sd / np.sqrt(len(a))) if sd else 0.0
    return (round(m, 3), round(t, 2), len(a))


def car_study(events: list[dict], panel: pd.DataFrame, bench: pd.Series) -> dict:
    """이벤트유형별 CAR[+1,+K] 평균·t·서브기간. events: [{date,type,sym}]."""
    bench = bench.reindex(panel.index).ffill()
    by_type: dict[str, list] = {}
    for e in events:
        by_type.setdefault(e["type"], []).append(e)
    out = {}
    for et, evs in sorted(by_type.items()):
        rows = {}
        for k in WINDOWS:
            per_sub = {}
            for tag, a, b in SUBS:
                cars = []
                for e in evs:
                    d = e["date"]
                    if a and d < a.replace("-", ""):
                        continue
                    if b and d > b.replace("-", ""):
                        continue
                    ei = _event_index(panel, d)
                    if ei is None:
                        continue
                    cars.append(_abn(panel, bench, e["sym"], ei, k))
                m, t, n = _tstat(cars)
                per_sub[tag] = {"mean_car_pct": m, "t": t, "n": n}
            rows[f"car_{k}d"] = per_sub
        out[et] = rows
    return out


def pead_study(events: list[dict], panel: pd.DataFrame, bench: pd.Series) -> dict:
    """§2-B: 잠정실적 공시의 발표반응(CAR[+1,+2]) 부호로 나눠 이후 드리프트(CAR[+3,+60]) 측정.
    SUE 재구성(컨센서스 필요) 없이 '반응 방향 드리프트'로 PEAD를 검정한다(문서 §2-B 변형2)."""
    bench = bench.reindex(panel.index).ffill()
    pos_drift, neg_drift = [], []
    for e in events:
        if e["type"] != "잠정실적":
            continue
        ei = _event_index(panel, e["date"])
        if ei is None:
            continue
        react = _abn(panel, bench, e["sym"], ei, 2)       # 발표 직후 2거래일 반응
        if react is None:
            continue
        # 드리프트 = CAR[+3,+60] (반응창 이후) — _abn을 e+2 기준으로 재사용
        drift = _abn(panel, bench, e["sym"], ei + 2, 58)
        if drift is None:
            continue
        (pos_drift if react > 0 else neg_drift).append(drift)
    mp, tp, npn = _tstat(pos_drift)
    mn, tn, nnn = _tstat(neg_drift)
    spread = (mp - mn) if (np.isfinite(mp) and np.isfinite(mn)) else float("nan")
    return {"positive_reaction": {"drift_car_pct": mp, "t": tp, "n": npn},
            "negative_reaction": {"drift_car_pct": mn, "t": tn, "n": nnn},
            "spread_pct": round(spread, 3) if np.isfinite(spread) else None,
            "note": "발표반응(+2일) 양수 그룹 vs 음수 그룹의 이후 드리프트(+3~+60일). "
                    "spread>0 이면 PEAD 존재(반응 방향으로 계속 밀림)."}


def run(save=True):
    import benchmarks_kr as B
    panel, membership, fundamentals, flows, mktcaps, bench = B.load_research_data()
    if not os.path.exists(DISC_CACHE):
        _log(f"{DISC_CACHE} 없음 — 먼저 --collect 실행"); sys.exit(1)
    with open(DISC_CACHE, encoding="utf-8") as f:
        cache = json.load(f)
    events = []
    for sym, rec in cache.items():
        if sym not in panel.columns:                      # 가격 패널에 없는 종목 제외
            continue
        for e in rec.get("events", []):
            events.append({"date": e["date"], "type": e["type"], "sym": sym})
    _log(f"이벤트 {len(events)}건 (패널 내 종목 한정) · 유형 "
         f"{dict((k, sum(1 for e in events if e['type']==k)) for k in set(e['type'] for e in events))}")
    car = car_study(events, panel, bench)
    pead = pead_study(events, panel, bench)
    payload = {"as_of": panel.index[-1].date().isoformat(),
               "universe": "코스피200 PIT union 276종목(대형주·생존 편향 — 반응 존재 측정용)",
               "n_events_total": len(events),
               "judgment": "채택 게이트: |t|≥3(HLZ) 또는 (|t|≥2 & 서브기간 2/3 동일부호). "
                           "다중검정: 이벤트 7유형 × 창 3 — 예산 등록.",
               "car_by_type": car, "pead": pead}
    if save:
        os.makedirs("output", exist_ok=True)
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _log(f"저장: {OUT_PATH}")
    return payload


# ------------------------- self-test -------------------------
def self_test():
    _log("[self-test] 합성: 소각 이벤트 후 +드리프트 심어 CAR·t 검출, 분류 정규식 확인")
    assert _classify("주요사항보고서(자기주식소각결정)") == "소각"
    assert _classify("[기재정정]주요사항보고서(자기주식소각결정)") is None
    assert _classify("연결재무제표기준영업(잠정)실적(공정공시)") == "잠정실적"
    assert _classify("분기보고서") is None

    idx = pd.bdate_range("2020-01-01", periods=700)
    rng = np.random.default_rng(3)
    bench = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, 700))), index=idx)
    # AAA: 이벤트 5건 다음날부터 20일 초과수익 +0.3%/일 심음(표본≥3 확보)
    e_idx = [100, 200, 300, 400, 500]
    ar = rng.normal(0, 0.008, 700)
    for e in e_idx:
        ar[e + 1:e + 21] += 0.003
    aaa = bench.to_numpy() * np.exp(np.cumsum(ar)) / bench.iloc[0] * 100
    panel = pd.DataFrame({"AAA": aaa}, index=idx)
    events = [{"date": idx[e].strftime("%Y%m%d"), "type": "소각", "sym": "AAA"} for e in e_idx]
    res = car_study(events, panel, bench)
    c20 = res["소각"]["car_20d"]["full"]
    assert c20["mean_car_pct"] > 3 and c20["n"] == 5, c20
    _log(f"[self-test] 통과: 심은 소각 드리프트 CAR20 {c20['mean_car_pct']}%p (n={c20['n']})")


def main():
    ap = argparse.ArgumentParser(description="DART 이벤트 스터디(Phase 1)")
    ap.add_argument("--collect", action="store_true", help="공시목록 수집만")
    ap.add_argument("--bgn", default="20180101")
    ap.add_argument("--limit", type=int, default=0, help="수집 종목 수 제한(테스트용)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test(); return
    if args.collect:
        import benchmarks_kr as B
        panel, *_ = B.load_research_data()
        syms = list(panel.columns)
        if args.limit:
            syms = syms[:args.limit]
        collect(syms, bgn=args.bgn)
        return
    run()


if __name__ == "__main__":
    main()
