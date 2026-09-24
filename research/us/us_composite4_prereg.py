#!/usr/bin/env python3
"""
us_composite4_prereg.py — 사전등록 1회 검증(2026-09-24, HISTORY.md §17).

방향과 구성은 우리 10년 데이터가 아니라 60년 대형주 데이터(§16, 롤링 10년 같은 부호 ≥85%)로 정했다.
이 파일을 실행하기 전에 아래 정의를 고정했고, 결과를 보고 바꾸지 않는다.

  합성점수 = z(mom12_1) + z(op_be) + z(accruals) − z(net_issuance)
    mom12_1      12-1개월 수익(모멘텀, +)
    op_be        영업이익/자기자본(FF 영업수익성의 근사, 자본>0만, +)
    accruals     −(순이익−영업현금흐름)/자산 — fundamentals_edgar 정의(이미 '낮은 발생액이 높은 값')
    net_issuance 주식수 전년 대비 증가율(분할 보정, 낮을수록 좋음 → 부호 반대로 더함)
    z는 매 시점 유니버스 내 표준화·±3 클립, 결측 팩터는 0(중립), 4개 중 2개 미만이면 제외.
  포트폴리오: S&P500 PIT, 합성 상위 30종목, 분기 리밸런싱 3개 슬리브 교대(시점 운 제거), 비용 반영.
    composite4_ew  동일비중
    composite4_cw  시총비중(종목당 10% 상한)
  비교(같은 창): SPY·QQQ·RSP·SPMO, capmom30(§13), 현행 1:2:2 라이브 계좌 NAV.
  함께 보고: 합성점수 6개월 순위IC(Newey-West t), 구간별(~2022·2023~) 성과, SPY 대비 짝지은 부트스트랩.

실행: python -m research.us.us_composite4_prereg
결과: output/us_composite4_prereg.json
"""
from __future__ import annotations
import json
import sys

import numpy as np
import pandas as pd

import backtest_costs as BC
import backtest_weights as BW
import fundamentals_edgar as F
import research.us.us_capweight_momentum_validation as V
from research.us.us_single_factor_study import _shares_now_prev, nw_t

TOPN = 30
SLEEVES = 3
CAP_W = 0.10
FACTORS = {"mom12_1": 1, "op_be": 1, "accruals": 1, "net_issuance": -1}


def _log(m): print(f"[4팩터사전등록]  {m}", file=sys.stderr)


def raw_factors(panel, funds, dupes, pit, p) -> tuple[pd.DataFrame, pd.Series]:
    idx = panel.index
    d = idx[p].date().isoformat()
    mem = BC.membership_asof(pit, d)
    cols = [c for c in panel.columns if c in mem and c not in dupes]
    px = panel.iloc[p][cols]
    alive = panel.iloc[p - 10:p + 1][cols].nunique() > 1
    univ = px[px.notna() & alive & panel.iloc[p - 252][cols].notna()].index
    rows, mcap = {}, {}
    for s in univ:
        rec = funds.get(s) or {}
        price = float(px[s])
        r = {"mom12_1": float(panel.iloc[p - 21][s] / panel.iloc[p - 252][s] - 1)}
        opinc, eq = F.asof(rec.get("opinc"), d), F.asof(rec.get("equity"), d)
        if opinc is not None and eq and eq > 0:
            r["op_be"] = opinc / eq
        acc = F.factor_values(rec, d, price).get("accruals")
        if acc is not None:
            r["accruals"] = acc
        sh, sh_prev = _shares_now_prev(rec, d)
        if sh and sh_prev:
            r["net_issuance"] = sh / sh_prev - 1
        if sh:
            mcap[s] = sh * price
        rows[s] = r
    return pd.DataFrame.from_dict(rows, orient="index").reindex(columns=list(FACTORS)), pd.Series(mcap, dtype=float)


def composite(raw: pd.DataFrame) -> pd.Series:
    z = raw.apply(BW._z)                       # ±3 클립 z
    enough = raw.notna().sum(axis=1) >= 2
    score = sum(sign * z[f].fillna(0.0) for f, sign in FACTORS.items())
    return score[enough]


def run(years: float = 10) -> dict:
    import sp500_daily_report as R
    pit = BC.load_pit()
    panel, _, _ = BC.build_panel_pit(years, pit)
    funds = BW.load_funds() or {}
    dupes = V._share_class_dupes(funds)
    etf = pd.DataFrame(R.download_histories(["SPY", "QQQ", "RSP", "SPMO", "BIL"], period=f"{int(years)}y",
                                            drop_stale=False)).reindex(panel.index).ffill()
    rf = etf["BIL"].pct_change().fillna(0.0)
    months = V.month_starts(panel.index, BW.LOOKBACK)

    scores, mcaps, ics = {}, {}, []
    for p in months:
        raw, mc = raw_factors(panel, funds, dupes, pit, p)
        sc = composite(raw)
        scores[p], mcaps[p] = sc, mc
        e = p + 1
        if e + 126 < len(panel):
            fwd = panel.iloc[e + 126][sc.index] / panel.iloc[e][sc.index] - 1
            ics.append(sc.rank().corr(fwd.rank()))
    _log(f"스냅샷 {len(scores)}개 · 합성 6개월 IC {np.nanmean(ics):+.4f} (NW t {nw_t(ics, 5):+.2f})")

    def weights(p, kind):
        top = scores[p].sort_values(ascending=False).index[:TOPN]
        if kind == "ew":
            return {s: 1.0 / len(top) for s in top}
        m = mcaps[p].reindex(top).dropna()
        m = m[m > 0]
        w = m / m.sum()
        for _ in range(100):
            over = w > CAP_W + 1e-12
            if not over.any():
                break
            ex = float((w[over] - CAP_W).sum())
            w[over] = CAP_W
            w[~over] += ex * w[~over] / w[~over].sum()
        return dict(w / w.sum())

    navs = {}
    for kind in ("ew", "cw"):
        sl = []
        for k in range(SLEEVES):
            sched = [(p + 1, weights(p, kind)) for p in months[k::SLEEVES]]
            sl.append(V.sleeve_nav(panel, rf, sched))
        t0 = max(s.index[0] for s in sl)
        navs[f"composite4_{kind}"] = sum(s.loc[t0:] / s.loc[t0] for s in sl) / SLEEVES
    logr = np.log(panel).diff()
    navs["capmom30"], _ = V.staggered(panel, rf, pit, logr, funds, dupes, 30, "capscore", {})
    live = V.live_nav(panel, funds, pit, rf)
    if live is not None:
        navs["live_122_top10"] = live
    t0 = max(v.index[0] for v in navs.values())
    for k in ("SPY", "QQQ", "RSP", "SPMO"):
        navs[k] = etf[k]
    navs = {k: v.loc[t0:] / v.loc[t0] for k, v in navs.items()}

    windows = {"full": (None, None), "to_2022": (None, "2022-12-31"), "2023_on": ("2023-01-01", None)}
    tbl = {w: {k: V.metrics(v.loc[a:b] / v.loc[a:b].iloc[0], rf) for k, v in navs.items()}
           for w, (a, b) in windows.items()}
    for k, v in tbl["full"].items():
        _log(f"{k:16s} CAGR {v['cagr_pct']:6.2f}% 샤프 {v['sharpe']:.2f} MDD {v['mdd_pct']:.1f}% | ~2022 "
             f"{tbl['to_2022'][k]['cagr_pct']:.2f}/{tbl['to_2022'][k]['sharpe']:.2f} | 2023~ "
             f"{tbl['2023_on'][k]['cagr_pct']:.2f}/{tbl['2023_on'][k]['sharpe']:.2f}")
    m = {k: V.monthly(v) for k, v in navs.items()}
    rf_m = (1 + rf).resample("ME").prod().sub(1).reindex(m["SPY"].index)
    boot = {f"{c}_vs_{b}": V.paired_bootstrap(m[c], m[b], rf_m)
            for c in ("composite4_ew", "composite4_cw") for b in ("SPY", "QQQ", "RSP")}
    for k, v in boot.items():
        _log(f"{k}: P(샤프 우위) {v['p_sharpe_better']} · P(CAGR 우위) {v['p_cagr_better']}")
    out = {"as_of": panel.index[-1].date().isoformat(), "window_start": t0.date().isoformat(),
           "definition": {"factors": FACTORS, "topn": TOPN, "sleeves": SLEEVES, "cap_weight_limit": CAP_W,
                          "direction_source": "HISTORY.md §16 (Ken French BIG, 1963-2026)"},
           "composite_ic_6m": round(float(np.nanmean(ics)), 4), "composite_ic_6m_nw_t": round(nw_t(ics, 5), 2),
           "windows": tbl, "bootstrap_monthly": boot}
    with open("output/us_composite4_prereg.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    _log("저장: output/us_composite4_prereg.json")
    return out


if __name__ == "__main__":
    run()
