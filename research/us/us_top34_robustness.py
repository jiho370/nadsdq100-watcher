#!/usr/bin/env python3
"""
us_top34_robustness.py — 지호 님 질문(2026-09-24): "3~4의 경우 그게 우연이었는지, 특정 종목에 의존한 건지,
꾸준한 건지 검증해줘" (HISTORY §18에서 top3 24.9%/0.97·top4 24.3%/0.95가 점추정 최고였음).

라이브 엔진(backtest_portfolio.simulate — 끈적 보유·180일+후보풀60 재평가, 10거래일 판정, 비용 반영)으로:
  1. 종목 의존: 보유했던 종목을 하나씩 영구 제외(그 슬롯은 다음 순위가 채움)하고 재실행 → CAGR 기여 순위.
     기여 상위 1·2·3·5종목을 동시에 제외해도 SPY를 이기는가.
  2. 순위의 가치(운 대 실력): 매 판정일 같은 필터 통과 후보풀 안에서 순서만 무작위로 섞어(매도 규칙의 후보풀
     집합은 그대로) 300회 → 실제 CAGR·샤프가 그 분포의 몇 퍼센타일인가.
  3. 판정 시점 운: 10거래일 판정 격자의 시작일을 0·2·4·6·8일 밀어 5가지.
  4. 꾸준함: 연도별 SPY 대비 초과, 롤링 12개월 SPY 초과 비율, 전·후반부.
비교로 top10(현행)도 같은 검정을 한다.

실행: python -m research.us.us_top34_robustness
결과: output/us_top34_robustness.json
"""
from __future__ import annotations
import json
import sys

import numpy as np
import pandas as pd

import backtest_costs as BC
import backtest_weights as BW
import research.us.us_capweight_momentum_validation as V
from research.us.us_topn_scorecap_sweep import live_scores

TOPNS = [3, 4, 10]
N_RANDOM = 300
STEP = 10
OFFSETS = [0, 2, 4, 6, 8]


def _log(m): print(f"[top3·4 검증]  {m}", file=sys.stderr)


def run(years: float = 10, seed: int = 20260924) -> dict:
    import sp500_daily_report as R
    import tech_factors as T
    import backtest_exec as BE
    from research.us import backtest_portfolio as BP

    pit = BC.load_pit()
    panel, _, _ = BC.build_panel_pit(years, pit)
    funds = BW.load_funds() or {}
    dupes = V._share_class_dupes(funds)
    etf = pd.DataFrame(R.download_histories(["SPY", "BIL"], period=f"{int(years)}y",
                                            drop_stale=False)).reindex(panel.index).ffill()
    rf = etf["BIL"].pct_change().fillna(0.0)
    cross = T.build_panels(panel)
    weights = BE._load_exec_weights()
    ma200 = panel.rolling(200, min_periods=200).mean()

    scores = {}
    for p in range(BW.LOOKBACK, len(panel) - 1, 2):       # 격자 오프셋 0·2·4·6·8일에 쓰일 날만 계산
        s = live_scores(panel, p, funds, cross, pit, weights)
        if s is not None:
            scores[p] = [x for x in s.index if x not in dupes]
    _log(f"점수 계산 {len(scores)}일")

    def decisions(offset=0, ban=frozenset(), shuffle_rng=None):
        out = []
        for p in range(BW.LOOKBACK + offset, len(panel) - 1, STEP):
            r = scores.get(p)
            if not r:
                continue
            r = [s for s in r if s not in ban][:60]
            if shuffle_rng is not None:
                r = list(shuffle_rng.permutation(r))
            if r:
                out.append((p, r))
        return out

    spy = etf["SPY"]

    def stats(nav):
        n = nav / nav.iloc[0]
        return V.metrics(n, rf)

    t_common = None
    res = {}
    for n in TOPNS:
        log = []
        base = BP.simulate(panel, ma200, decisions(), n, V.COST, ma200_backup=False, trade_log=log)
        t_common = base.index[0] if t_common is None else max(t_common, base.index[0])
        held = sorted({e["sym"] for e in log if e.get("action") == "buy"})
        b = stats(base)
        # 1) 종목 의존: 하나씩 제외
        loo = {}
        for s in held:
            nav = BP.simulate(panel, ma200, decisions(ban=frozenset([s])), n, V.COST, ma200_backup=False)
            loo[s] = round(b["cagr_pct"] - stats(nav)["cagr_pct"], 2)     # +면 그 종목이 CAGR에 기여
        ranked_contrib = sorted(loo.items(), key=lambda kv: -kv[1])
        drop_k = {}
        for k in (1, 2, 3, 5):
            ban = frozenset(s for s, _ in ranked_contrib[:k])
            nav = BP.simulate(panel, ma200, decisions(ban=ban), n, V.COST, ma200_backup=False)
            drop_k[f"without_top{k}"] = {"banned": sorted(ban), **stats(nav)}
        # 2) 무작위 순서 대조군
        rng = np.random.default_rng(seed + n)
        rnd = []
        for _ in range(N_RANDOM):
            nav = BP.simulate(panel, ma200, decisions(shuffle_rng=rng), n, V.COST, ma200_backup=False)
            st = stats(nav)
            rnd.append((st["cagr_pct"], st["sharpe"]))
        rnd = np.array(rnd)
        # 3) 판정 시점 오프셋
        offs = {}
        for off in OFFSETS:
            nav = BP.simulate(panel, ma200, decisions(offset=off), n, V.COST, ma200_backup=False)
            offs[off] = stats(nav)
        # 4) 꾸준함
        nb = base / base.iloc[0]
        s_ = spy.reindex(nb.index).ffill()
        s_ = s_ / s_.iloc[0]
        yr = pd.DataFrame({"algo": nb.resample("YE").last(), "spy": s_.resample("YE").last()})
        yr = pd.concat([pd.DataFrame({"algo": [1.0], "spy": [1.0]}, index=[nb.index[0]]), yr]).pct_change().dropna()
        roll = (nb.pct_change(252) - s_.pct_change(252)).dropna()
        half = nb.index[len(nb) // 2]
        res[f"top{n}"] = {
            "base": b, "n_symbols_held": len(held),
            "leave_one_out_cagr_contrib_top10": ranked_contrib[:10],
            "leave_one_out_cagr_contrib_bottom5": ranked_contrib[-5:],
            "drop_top_contributors": drop_k,
            "random_order": {"n": N_RANDOM, "cagr_mean": round(float(rnd[:, 0].mean()), 2),
                             "cagr_p5_p95": [round(float(np.percentile(rnd[:, 0], 5)), 2),
                                             round(float(np.percentile(rnd[:, 0], 95)), 2)],
                             "sharpe_mean": round(float(rnd[:, 1].mean()), 3),
                             "actual_cagr_pctile": round(100 * float((rnd[:, 0] < b["cagr_pct"]).mean()), 1),
                             "actual_sharpe_pctile": round(100 * float((rnd[:, 1] < b["sharpe"]).mean()), 1)},
            "offsets": {str(k): {"cagr_pct": v["cagr_pct"], "sharpe": v["sharpe"], "mdd_pct": v["mdd_pct"]}
                        for k, v in offs.items()},
            "calendar_excess_vs_spy_pct": {str(d.year): round(100 * float(r.algo - r.spy), 1) for d, r in yr.iterrows()},
            "rolling12m_beat_spy_pct": round(100 * float((roll > 0).mean()), 1),
            "halves": {"first": stats(base.loc[:half]), "second": stats(base.loc[half:])},
        }
        r = res[f"top{n}"]
        _log(f"top{n}: {b['cagr_pct']}%/{b['sharpe']} | 종목 {len(held)}개 · 기여1위 {ranked_contrib[0]} | "
             f"상위1·3·5 제외 CAGR {drop_k['without_top1']['cagr_pct']}/{drop_k['without_top3']['cagr_pct']}/"
             f"{drop_k['without_top5']['cagr_pct']} | 무작위 순서 대비 CAGR {r['random_order']['actual_cagr_pctile']}%ile "
             f"(무작위 평균 {r['random_order']['cagr_mean']}) | 오프셋 CAGR {min(v['cagr_pct'] for v in offs.values())}~"
             f"{max(v['cagr_pct'] for v in offs.values())} | 롤링12M SPY 초과 {r['rolling12m_beat_spy_pct']}%")
    spy_m = stats(spy.loc[t_common:])
    out = {"as_of": panel.index[-1].date().isoformat(), "spy": spy_m, "results": res,
           "note": "무작위 순서 대조군은 같은 필터(floor·100일선) 통과 풀 안에서 순위만 섞은 것 — 순위(1:2:2 점수)의 "
                   "가치를 잰다. 필터 자체의 가치는 포함하지 않는다."}
    with open("output/us_top34_robustness.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    _log(f"SPY {spy_m['cagr_pct']}%/{spy_m['sharpe']} · 저장: output/us_top34_robustness.json")
    return out


if __name__ == "__main__":
    run()
