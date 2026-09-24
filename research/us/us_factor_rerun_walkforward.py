#!/usr/bin/env python3
"""
us_factor_rerun_walkforward.py — 지호 님 요청(2026-09-24): "1:2:2와 그 3개 팩터도 결함 있는 데이터로
정했다면 다 다시 돌려보자."

왜 '같은 탐색 다시 돌리기'가 아닌가: 2026-07-18의 665조합 탐색은 (1) 공시 후 주식분할 미보정 —
나중에 분할할(=이후 크게 오른) 종목의 과거 시총이 작게 잡혀 가격 분모 팩터 점수가 부풀려지는
미래참조, (2) PIT 백테스트의 과거 종료 종목 삭제 위에서 돌았다. 같은 10년에서 다시 1등을 뽑으면
이미 여러 번 본 표본에서 고르는 셈이라, 이번엔 기간을 나눈다.

A. 팩터 진단: 월초마다 전 팩터(펀더멘탈 24 + 모멘텀 2 + 기술 4)의 6개월 순방향 순위IC.
   학습(2021년까지 결과 확정)·검증(2022~) 구간별 평균과 Newey-West t(6개월 중첩 보정, lag 5).
B. 학습 구간 탐색: 학습 IC 상위 6개(양수만) × 가중치 {0,1,2} 격자(기존 665조합과 같은 방식),
   top10 동일비중 6개월 순초과수익(vs SPY)의 샤프로 1등 선정. 결과 확정일이 검증 시작 전인
   스냅샷만 사용. 학습 행렬 PBO/DSR.
C. 검증 구간(2022-01~) 1회 평가 — 계좌 NAV, 분기 리밸런싱을 3개 슬리브로 교대(시점 운 제거),
   아래 후보만(사전 고정):
     train_best        B의 1등, 동일비중 top10
     train_best_cap    같은 종목, 시총비중(종목당 20% 상한)
     w122_live         1:2:2 + 라이브 필터(live_z·floor 3.25·100일선), 동일비중, 빈 슬롯 현금
     w122_nofilter     1:2:2 필터 없음, 동일비중
     w122_live_cap     w122_live 종목, 시총비중(20% 상한)
     gp_only           int_gp_assets 단독(분할 보정 후에도 IC가 유지된 유일한 팩터), 동일비중
   비교: SPY, QQQ, RSP(동일비중 S&P500 — 동일비중 전략의 공정한 비교군)
   하한·클립·100일선·종목 수는 다시 튜닝하지 않는다(w122_live vs w122_nofilter 비교만).

한계: 검증 구간 4.7년(2022 약세장 포함 1개 국면 전환)으로 짧다. 라이브의 180일+후보풀 끈적 보유가
아니라 분기 전량 재구성. 시총은 수정주가×(순이익/EPS, 분할 보정) 추정.

실행: python -m research.us.us_factor_rerun_walkforward [--years 10] [--test-start 2022-01-01]
결과: output/us_factor_rerun_walkforward.json
"""
from __future__ import annotations
import argparse
import json
import sys

import numpy as np
import pandas as pd

import backtest_costs as BC
import backtest_weights as BW
import overfit_stats as OS
import research.us.us_capweight_momentum_validation as V

TOPN = 10
HOLD = 126                  # 6개월 순방향(거래일)
N_PICK = 6                  # 학습 IC 상위 팩터 수(기존 탐색과 동일)
CAP_W = 0.20                # 시총비중 변형의 종목당 상한(10종목이라 9%는 불가능)
SLEEVES = 3                 # 분기 리밸런싱 × 월 교대
EXCLUDE = {"ma100_gap"}     # 가중 팩터가 아니라 필터용


def _log(m): print(f"[팩터재검증]  {m}", file=sys.stderr)


def nw_t(x: np.ndarray, lag: int = 5) -> float:
    """Newey-West t — 월 간격 스냅샷의 6개월 순방향 수익은 5개월씩 겹친다."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    e = x - x.mean()
    s = e @ e / n
    for k in range(1, min(lag, n - 1) + 1):
        s += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    return float(x.mean() / np.sqrt(s / n)) if s > 0 else float("nan")


def zscore(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(BW._z).fillna(0.0)


def cap_weights(syms, mcap: pd.Series) -> dict:
    m = mcap.reindex(syms).dropna()
    m = m[m > 0]
    if m.empty:
        return {}
    w = m / m.sum()
    lim = max(CAP_W, 1.0 / len(w))
    for _ in range(100):
        over = w > lim + 1e-12
        if not over.any():
            break
        ex = float((w[over] - lim).sum())
        w[over] = lim
        room = ~over
        w[room] += ex * w[room] / w[room].sum()
    return dict(w / w.sum())


def run(years: float = 10, test_start: str = "2022-01-01") -> dict:
    import export_data as E
    import sp500_daily_report as R
    import tech_factors as T
    pit = BC.load_pit()
    panel, _, _ = BC.build_panel_pit(years, pit)
    funds = BW.load_funds() or {}
    dupes = V._share_class_dupes(funds)
    etf = pd.DataFrame(R.download_histories(["SPY", "QQQ", "RSP", "BIL"], period=f"{int(years)}y",
                                            drop_stale=False)).reindex(panel.index).ffill()
    rf = etf["BIL"].pct_change().fillna(0.0)
    cross = T.build_panels(panel)
    idx = panel.index
    t_test = int(idx.searchsorted(pd.Timestamp(test_start)))
    months = V.month_starts(idx, BW.LOOKBACK)
    cost = V.COST

    raws, mcaps = {}, {}
    for p in months:
        raw = BW._raw_frame(panel, p, funds, True, cross)
        if raw is None or raw.empty:
            continue
        raw = raw.loc[raw.index.intersection(BC.membership_asof(pit, idx[p].date().isoformat()))]
        raw = raw.loc[[s for s in raw.index if s not in dupes]]
        raws[p] = raw
        d = idx[p].date().isoformat()
        mcaps[p] = pd.Series({s: (V.shares_asof(funds.get(s) or {}, d) or np.nan) * float(panel.iloc[p][s])
                              for s in raw.index})
    ps = sorted(raws)
    _log(f"스냅샷 {len(ps)}개(월초), 검증 시작 {idx[t_test].date()}")
    factors = sorted(set().union(*[set(r.columns) for r in raws.values()]) - EXCLUDE)

    def fwd(p):
        e = p + 1
        if e + HOLD >= len(idx):
            return None, None
        r = panel.iloc[e + HOLD] / panel.iloc[e] - 1
        return r, float(etf["SPY"].iloc[e + HOLD] / etf["SPY"].iloc[e] - 1)

    # ---------- A. 팩터 IC ----------
    ic = {f: {} for f in factors}
    for p in ps:
        r, _ = fwd(p)
        if r is None:
            continue
        for f in factors:
            if f in raws[p]:
                v = raws[p][f].rank().corr(r.reindex(raws[p].index).rank())
                if pd.notna(v):
                    ic[f][p] = v
    train_ps = [p for p in ps if p + 1 + HOLD < t_test]          # 결과 확정일 < 검증 시작
    test_ps = [p for p in ps if p >= t_test]
    ic_tbl = {}
    for f in factors:
        s = pd.Series(ic[f])
        tr, te = s[s.index.isin(train_ps)], s[s.index.isin(test_ps)]
        ic_tbl[f] = {"ic_all": round(float(s.mean()), 4), "t_all": round(nw_t(s.values), 2),
                     "ic_train": round(float(tr.mean()), 4) if len(tr) else None,
                     "t_train": round(nw_t(tr.values), 2) if len(tr) > 2 else None,
                     "ic_test": round(float(te.mean()), 4) if len(te) else None,
                     "t_test": round(nw_t(te.values), 2) if len(te) > 2 else None,
                     "n": int(len(s))}
    for f, v in sorted(ic_tbl.items(), key=lambda kv: -kv[1]["ic_all"]):
        _log(f"IC {f:18s} 전체 {v['ic_all']:+.4f}(t{v['t_all']:+.2f}) 학습 {v['ic_train']} 검증 {v['ic_test']}")

    # ---------- B. 학습 구간 탐색 ----------
    pos = [(f, ic_tbl[f]["ic_train"]) for f in factors if (ic_tbl[f]["ic_train"] or 0) > 0]
    picked = [f for f, _ in sorted(pos, key=lambda kv: -kv[1])[:N_PICK]]
    grid = BW._weight_grid(picked)
    _log(f"학습 선정 팩터 {picked} → 격자 {len(grid)}조합, 학습 스냅샷 {len(train_ps)}")
    zs = {p: zscore(raws[p].reindex(columns=picked)) for p in train_ps}
    fw = {p: fwd(p) for p in train_ps}
    M = np.zeros((len(grid), len(train_ps)))
    for i, w in enumerate(grid):
        wv = pd.Series(w)
        for j, p in enumerate(train_ps):
            top = (zs[p][list(w)] * wv).sum(axis=1).sort_values(ascending=False).index[:TOPN]
            r, b = fw[p]
            M[i, j] = cost.net(float(r.reindex(top).mean())) - b
    sh = M.mean(axis=1) / M.std(axis=1, ddof=1)
    best_i = int(np.argmax(sh))
    best_w = grid[best_i]
    trial = {"horizon": "factor_rerun_train", "universe": "pit", "cost": cost.describe(),
             "rebal_days": 21, "hold_days": HOLD, "trials": [json.dumps(g) for g in grid],
             "excess_returns": M.tolist()}
    rep = OS.analyze(trial, save=False)
    _log(f"학습 1등 {best_w} 학습 초과 {100*M[best_i].mean():+.2f}%/6개월 · PBO {rep['pbo']['pbo']} DSR {rep['dsr'].get('dsr')}")

    # ---------- C. 검증 구간 계좌 NAV ----------
    with open("output/best_weights.json", encoding="utf-8") as f:
        live_w = json.load(f)["weights"]

    def pick(p, mode):
        raw = raws[p]
        if mode.startswith("train_best"):
            z = zscore(raw.reindex(columns=list(best_w)))
            return list((z * pd.Series(best_w)).sum(axis=1).sort_values(ascending=False).index[:TOPN])
        if mode == "gp_only":
            return list(raw["int_gp_assets"].dropna().sort_values(ascending=False).index[:TOPN])
        w = {k: v for k, v in live_w.items() if k in raw.columns}
        score = sum(float(v) * E.live_z(raw[k], k) for k, v in w.items())
        if mode == "w122_nofilter":
            return list(score.sort_values(ascending=False).index[:TOPN])
        score = score[score >= E.SCORE_FLOOR]
        gap = raw["ma100_gap"].reindex(score.index)
        return list(score[gap > 0].sort_values(ascending=False).index[:TOPN])

    modes = ["train_best", "train_best_cap", "w122_live", "w122_nofilter", "w122_live_cap", "gp_only"]
    test_months = [p for p in ps if p >= t_test]
    navs = {}
    for mode in modes:
        sl = []
        for k in range(SLEEVES):
            sched = []
            for p in test_months[k::SLEEVES]:
                syms = pick(p, mode.replace("_cap", "") if mode.endswith("_cap") else mode)
                if mode.endswith("_cap"):
                    w = cap_weights(syms, mcaps[p])
                else:
                    w = {s: 1.0 / TOPN for s in syms}           # 모자라면 나머지 현금(라이브와 동일)
                sched.append((p + 1, w))
            sl.append(V.sleeve_nav(panel, rf, sched))
        t0 = max(s.index[0] for s in sl)
        navs[mode] = sum(s.loc[t0:] / s.loc[t0] for s in sl) / SLEEVES
    t0 = max(v.index[0] for v in navs.values())
    for k in ("SPY", "QQQ", "RSP"):
        navs[k] = etf[k]
    navs = {k: v.loc[t0:] / v.loc[t0] for k, v in navs.items()}
    res = {k: V.metrics(v, rf) for k, v in navs.items()}
    for k, v in res.items():
        _log(f"검증 {k:15s} CAGR {v['cagr_pct']:6.2f}% 샤프 {v['sharpe']:.2f} MDD {v['mdd_pct']:.1f}%")
    m = {k: V.monthly(v) for k, v in navs.items()}
    rf_m = (1 + rf).resample("ME").prod().sub(1).reindex(m["SPY"].index)
    boot = {f"{c}_vs_{b}": V.paired_bootstrap(m[c], m[b], rf_m)
            for c in modes for b in ("SPY", "RSP")}
    years_tbl = {k: {str(d.year): round(100 * float(x), 1)
                     for d, x in v.resample("YE").last().pct_change().dropna().items()}
                 for k, v in navs.items()}

    out = {"as_of": idx[-1].date().isoformat(), "test_start": idx[t_test].date().isoformat(),
           "n_snapshots": len(ps), "n_train": len(train_ps), "n_test_months": len(test_months),
           "ic": ic_tbl,
           "train_search": {"picked_factors": picked, "n_combos": len(grid), "best_weights": best_w,
                            "best_train_excess_pct_6m": round(100 * float(M[best_i].mean()), 2),
                            "best_train_sharpe_6m": round(float(sh[best_i]), 3),
                            "pbo": rep["pbo"]["pbo"], "dsr": rep["dsr"].get("dsr"), "passed": rep["passed"],
                            "rank_122_in_grid": None},
           "test_metrics": res, "test_calendar_year_pct": years_tbl, "test_bootstrap_monthly": boot,
           "notes": "검증 구간 1회 평가. 분기 전량 재구성(라이브 끈적 보유 아님). 현금=BIL."}
    tgt = {f: live_w.get(f, 0) for f in picked}
    for i, g in enumerate(grid):                     # 1:2:2의 세 팩터가 모두 학습 상위 6에 들었을 때만 존재
        if set(live_w) <= set(picked) and g == tgt:
            out["train_search"]["rank_122_in_grid"] = int((sh > sh[i]).sum()) + 1
    with open("output/us_factor_rerun_walkforward.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    _log("저장: output/us_factor_rerun_walkforward.json")
    return out


def self_test():
    x = np.r_[np.ones(20), -np.ones(20)] * 0.1 + 0.05
    assert abs(nw_t(np.full(10, 1.0) + np.linspace(-0.1, 0.1, 10))) > 5
    assert np.isfinite(nw_t(x))
    w = cap_weights(["A", "B", "C"], pd.Series({"A": 90.0, "B": 5.0, "C": 5.0}))
    assert abs(sum(w.values()) - 1) < 1e-9 and max(w.values()) <= 1 / 3 + 1e-9   # 3종목이면 상한 1/3로 완화
    _log("self-test 통과")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=10)
    ap.add_argument("--test-start", default="2022-01-01")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return
    run(a.years, a.test_start)


if __name__ == "__main__":
    main()
