#!/usr/bin/env python3
"""
regime_kr.py — KR_STRATEGY_OPTIONS.md §2-E: 레짐 라벨 생성 + 팩터 IC 레짐별 분해.

"한국 팩터 IC 평균이 0인 것은 '항상 0'이 아니라 레짐별 +/−가 상쇄된 결과일 수 있다"는
가설의 진단 도구. **전략이 아니라 측정 인프라다** — 여기서 나온 레짐×팩터 조합을
곧바로 전략으로 채택하지 않는다(다중검정 폭발 방지, §2-E 함정 항목).

사전 등록 레짐(3종 고정 — 결과 보고 후 수정 금지):
  R1 시장폭(breadth): 코스피200 멤버 중 200일선 상회 비율 — broad(>50%) / narrow(<30%) / mid
  R2 외국인 자금: 코스피 전체 외국인 순매수 20일 누적 부호 — in / out
  R3 원달러 추세: USDKRW 20일 변화 부호 — won_weak(환율 상승=외인 이탈) / won_strong

분해 대상: backtest_kr의 11팩터 × 지평(1m/3m/6m) × 위 레짐. 스냅샷(63일 간격) 단위
Spearman IC를 스냅샷 날짜의 레짐 라벨로 그룹화. 셀당 표본이 작으므로(n<8이면 표기만)
부호의 일관성만 읽는다.

실행: python regime_kr.py            # output/regime_kr.json
      python regime_kr.py --self-test
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import pandas as pd

REGIME_CACHE = "output/kr_regime_cache.json"
OUT_PATH = "output/regime_kr.json"
FACTORS = ["mom12_1", "mom6", "hi52_prox", "low_vol", "value", "pbr_inv",
           "div_yield", "roe", "frgn_flow", "inst_flow", "size"]
HORIZONS = ["1m", "3m", "6m"]


def _log(m): print(f"[레짐KR] {m}", file=sys.stderr)


# ------------------------- 외부 데이터(캐시) -------------------------
def _load_regime_cache():
    try:
        with open(REGIME_CACHE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_regime_cache(c):
    os.makedirs("output", exist_ok=True)
    with open(REGIME_CACHE, "w", encoding="utf-8") as f:
        json.dump(c, f, ensure_ascii=False)


def fetch_market_frgn_flow(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """코스피 전체 외국인 일별 순매수거래대금(원). 연 단위 청크로 pykrx 조회, 캐시."""
    cache = _load_regime_cache()
    rec = cache.setdefault("mkt_frgn", {})
    from pykrx import stock as K
    for y in range(start.year, end.year + 1):
        tag = str(y)
        if tag in cache.get("mkt_frgn_done", []) and y < end.year:
            continue
        a = max(start, pd.Timestamp(f"{y}-01-01")).strftime("%Y%m%d")
        b = min(end, pd.Timestamp(f"{y}-12-31")).strftime("%Y%m%d")
        try:
            df = K.get_market_trading_value_by_date(a, b, "KOSPI")
            col = next(c for c in df.columns if "외국인" in str(c))
            for d, v in df[col].items():
                rec[d.strftime("%Y-%m-%d")] = float(v)
            cache.setdefault("mkt_frgn_done", [])
            if tag not in cache["mkt_frgn_done"]:
                cache["mkt_frgn_done"].append(tag)
        except Exception as e:
            _log(f"시장 수급 조회 실패 {y}: {e}")
        _save_regime_cache(cache)
    s = pd.Series(rec)
    s.index = pd.to_datetime(s.index)
    return s.sort_index().loc[start:end]


def fetch_usdkrw(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """원달러 환율 종가(yfinance KRW=X), 캐시."""
    cache = _load_regime_cache()
    rec = cache.get("usdkrw", {})
    have_end = max(rec) if rec else None
    if not rec or (have_end and pd.Timestamp(have_end) < end - pd.Timedelta(days=7)):
        import yfinance as yf
        df = yf.download("KRW=X", start=start.strftime("%Y-%m-%d"), progress=False)
        cl = df["Close"]
        if isinstance(cl, pd.DataFrame):
            cl = cl.iloc[:, 0]
        rec = {d.strftime("%Y-%m-%d"): float(v) for d, v in cl.dropna().items()}
        cache["usdkrw"] = rec
        _save_regime_cache(cache)
    s = pd.Series(rec)
    s.index = pd.to_datetime(s.index)
    return s.sort_index().loc[start:end]


# ------------------------- 레짐 라벨 -------------------------
def build_labels(panel: pd.DataFrame, membership: dict, mkt_frgn: pd.Series,
                 usdkrw: pd.Series) -> pd.DataFrame:
    """일별 레짐 라벨 DataFrame [breadth, R1, frgn20, R2, fx20, R3]."""
    from benchmarks_kr import _membership_segments
    ma200 = panel.rolling(200, min_periods=200).mean()
    breadth = pd.Series(np.nan, index=panel.index)
    for i0, i1, members in _membership_segments(panel, membership):
        cols = [c for c in panel.columns if c in set(members)]
        if len(cols) < 20:
            continue
        above = panel.iloc[i0:i1][cols] > ma200.iloc[i0:i1][cols]
        valid = ma200.iloc[i0:i1][cols].notna()
        breadth.iloc[i0:i1] = above.sum(axis=1) / valid.sum(axis=1).replace(0, np.nan)
    r1 = pd.cut(breadth, [-np.inf, 0.30, 0.50, np.inf], labels=["narrow", "mid", "broad"])
    f20 = mkt_frgn.reindex(panel.index).fillna(0).rolling(20).sum()
    r2 = pd.Series(np.where(f20 > 0, "in", "out"), index=panel.index)
    fx = usdkrw.reindex(panel.index).ffill()
    fx20 = fx.pct_change(20)
    r3 = pd.Series(np.where(fx20 > 0, "won_weak", "won_strong"), index=panel.index)
    return pd.DataFrame({"breadth": breadth, "R1": r1, "frgn20": f20, "R2": r2,
                         "fx20": fx20, "R3": r3})


# ------------------------- IC 분해 -------------------------
def snapshot_ics(snaps) -> pd.DataFrame:
    """행=스냅샷, 열=(factor, horizon) Spearman IC + date."""
    rows = []
    for s in snaps:
        row = {"date": s["date"]}
        raw = s["raw"]
        for h in HORIZONS:
            fwd = s["fwd"][h]
            fr = fwd.rank()
            for f in FACTORS:
                if f in raw.columns and raw[f].notna().sum() >= 15:
                    v = raw[f].rank().corr(fr.reindex(raw.index))
                    if pd.notna(v):
                        row[f"{f}|{h}"] = float(v)
        rows.append(row)
    return pd.DataFrame(rows).set_index("date")


def decompose(ic_df: pd.DataFrame, labels: pd.DataFrame) -> dict:
    """레짐(R1/R2/R3)별 팩터 IC 평균·표본수·t(단순)."""
    lbl = labels.copy()
    lbl.index = [d.date().isoformat() for d in lbl.index]
    out = {}
    for rname in ("R1", "R2", "R3"):
        groups = {}
        reg = lbl[rname].reindex(ic_df.index)
        for g in reg.dropna().unique():
            sub = ic_df[reg == g]
            cell = {}
            for col in ic_df.columns:
                a = sub[col].dropna()
                if len(a) < 3:
                    continue
                t = float(a.mean() / (a.std(ddof=1) / np.sqrt(len(a)))) if a.std(ddof=1) > 0 else 0.0
                cell[col] = {"ic": round(float(a.mean()), 4), "n": int(len(a)),
                             "t": round(t, 2)}
            groups[str(g)] = cell
        out[rname] = groups
    return out


def run(save=True):
    from benchmarks_kr import load_research_data
    import backtest_kr as BK
    panel, membership, fundamentals, flows, mktcaps, bench = load_research_data()
    _log("시장 외국인 수급·환율 수집(캐시)…")
    mkt_frgn = fetch_market_frgn_flow(panel.index[0], panel.index[-1])
    usdkrw = fetch_usdkrw(panel.index[0], panel.index[-1])
    labels = build_labels(panel, membership, mkt_frgn, usdkrw)
    snaps, _, _ = BK.build_kr_snaps(panel, bench, membership, fundamentals,
                                    rebal_days=63, flows=flows, mktcaps=mktcaps)
    _log(f"스냅샷 {len(snaps)}개 · 레짐 분포: "
         f"R1 {labels['R1'].value_counts().to_dict()} · R2 {labels['R2'].value_counts().to_dict()} · "
         f"R3 {labels['R3'].value_counts().to_dict()}")
    ic_df = snapshot_ics(snaps)
    dec = decompose(ic_df, labels)
    overall = {c: {"ic": round(float(ic_df[c].mean()), 4), "n": int(ic_df[c].notna().sum())}
               for c in ic_df.columns}
    payload = {"as_of": panel.index[-1].date().isoformat(),
               "n_snapshots": len(snaps),
               "regime_days": {r: labels[r].value_counts().to_dict() for r in ("R1", "R2", "R3")},
               "overall_ic": overall,
               "by_regime": dec,
               "labels_daily": {d.date().isoformat(): {"R1": str(labels["R1"].loc[d]),
                                                       "R2": str(labels["R2"].loc[d]),
                                                       "R3": str(labels["R3"].loc[d])}
                                for d in labels.index if pd.notna(labels["breadth"].loc[d])},
               "note": ("진단 도구 — 셀당 n이 작아(63일 스냅) 부호 일관성만 읽을 것. "
                        "레짐×팩터 조합의 전략 채택은 별도 사전등록 검증 필요(§2-E 함정)."),
               "budget": "사전 등록: 레짐 3 × 팩터 11 × 지평 3 — 진단 전용, 채택 시행 0회"}
    if save:
        os.makedirs("output", exist_ok=True)
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        _log(f"저장: {OUT_PATH}")
    return payload, labels, ic_df


# ------------------------- self-test -------------------------
def self_test():
    _log("[self-test] 합성: narrow 레짐에서만 IC가 죽는 팩터를 심고 분해가 탐지하는지")
    rng = np.random.default_rng(9)
    n = 40   # 스냅샷 수
    dates = [f"2020-{1 + i % 12:02d}-{1 + i // 12:02d}" for i in range(n)]
    dates = sorted(dates)
    regime = ["narrow" if i < n // 2 else "broad" for i in range(n)]
    ic_rows = []
    for i in range(n):
        # good_f: broad에서 IC=0.3, narrow에서 -0.1 / dead_f: 항상 0
        good = (0.3 if regime[i] == "broad" else -0.1) + rng.normal(0, 0.05)
        ic_rows.append({"date": dates[i], "good_f|6m": good,
                        "dead_f|6m": rng.normal(0, 0.05)})
    ic_df = pd.DataFrame(ic_rows).set_index("date")
    lbl = pd.DataFrame({"R1": regime, "R2": "in", "R3": "won_weak", "breadth": 0.5},
                       index=pd.to_datetime(dates))
    dec = decompose(ic_df, lbl)
    g = dec["R1"]
    assert g["broad"]["good_f|6m"]["ic"] > 0.2 and g["narrow"]["good_f|6m"]["ic"] < 0, dec["R1"]
    assert abs(g["broad"]["dead_f|6m"]["ic"]) < 0.1
    # 라벨 빌더 스모크: 합성 패널로 breadth 라벨 산출
    idx = pd.bdate_range("2019-01-01", periods=500)
    up = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0.001, 0.005, (500, 30)), axis=0)),
                      index=idx, columns=[f"{i:06d}" for i in range(30)])
    membership = {idx[0].strftime("%Y%m%d"): list(up.columns)}
    flat = pd.Series(1000.0, index=idx)
    lab = build_labels(up, membership, flat * 0, flat)
    tail = lab["R1"].dropna().iloc[-50:]
    assert (tail == "broad").mean() > 0.9, tail.value_counts()   # 전부 상승 종목 → broad
    _log("[self-test] 통과: 레짐별 IC 분해 탐지 · breadth 라벨 OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="한국 레짐 라벨 + 팩터 IC 레짐별 분해(진단)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    self_test() if args.self_test else run()
