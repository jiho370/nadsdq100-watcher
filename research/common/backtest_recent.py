#!/usr/bin/env python3
"""
backtest_recent.py — 최근 구간 워크포워드 재현성 확인 (지호 님 요청, 2026-07).

메인 전략의 가중치는 output/best_weights.json 에 이미 발행돼 있다(재탐색 없음 — 여기서는
그 가중치를 그대로 고정해두고 "최근에도 잘 작동하는가"만 별도로 확인한다). 종목/가중치
탐색이 없으므로 PBO(다중검정 과최적화 검사)는 해당 없음 — 이미 채택된 전략의 표본외
재현성만 본다.

기본값(지호 님 지정): 리밸 1주(5거래일) 간격 · 3개월(63거래일) 고정보유 · 상위 4종목 ·
최근 12개월 전~3개월 전 구간(3개월 전 시작분까지 오늘 기준으로 만기 확인 가능).

주의(리밸=1주, 보유=3개월): 인접 이벤트가 대부분 겹친다 — n_events_raw는 원표본 수일 뿐,
실제 독립정보량은 n_eff(= n_events_raw × rebal_days/hold_days)로 훨씬 작다. 표본 수만
보고 "충분하다"고 판단하지 말 것(SCORE_MODEL_DESIGN.md D3와 동일한 함정).

실행: python backtest_recent.py
      python backtest_recent.py --self-test
결과: output/backtest_recent.json
"""
from __future__ import annotations
import os, sys, json, math, argparse
from statistics import NormalDist
import numpy as np
import pandas as pd

import backtest_weights as BW
import backtest_costs as BC

OUT_PATH = "output/backtest_recent.json"


def _log(m): print(f"[최근검증] {m}", file=sys.stderr)


def _load_weights():
    try:
        with open("output/best_weights.json", encoding="utf-8") as f:
            w = (json.load(f).get("weights") or {})
        if any(w.values()):
            return w
    except Exception:
        pass
    raise RuntimeError("output/best_weights.json 없음 — 먼저 backtest_costs.py --publish-weights 실행 필요")


def _select_topn(panel, p, funds, cross, pit, weights, topn):
    raw = BW._raw_frame(panel, p, funds, bool(funds), cross)
    if raw is None or raw.empty:
        return []
    date = panel.index[p].date().isoformat()
    idx = raw.index.intersection(BC.membership_asof(pit, date))
    if len(idx) < topn:
        return []
    raw = raw.loc[idx]
    w = {k: v for k, v in weights.items() if k in raw.columns}
    if not w:
        return []
    z = raw[list(w)].apply(BW._z).fillna(0.0)
    score = (z * pd.Series(w)).sum(axis=1)
    return list(score.sort_values(ascending=False).index[:topn])


def run_recent(panel, spy, funds, pit, weights, start_months_ago=12, end_months_ago=3,
              rebal_days=5, hold_days=63, topn=4, cost=None):
    import tech_factors as T
    cost = cost or BC.CostModel("us", commission_bps=0.0, slippage_bps=5.0)
    cross = T.build_panels(panel)
    spy = spy.reindex(panel.index).ffill()
    n = len(panel)
    today = panel.index[-1]
    start = today - pd.DateOffset(months=start_months_ago)
    end = today - pd.DateOffset(months=end_months_ago)

    ps = [p for p in range(BW.LOOKBACK, n - hold_days - 1, rebal_days)
          if start <= panel.index[p] <= end]
    if not ps:
        raise RuntimeError("지정 구간에 리밸런싱 시점 없음 — 기간·데이터 길이를 확인하세요.")

    events, prev_basket, turns = [], None, []
    for p in ps:
        basket = _select_topn(panel, p, funds, cross, pit, weights, topn)
        if not basket:
            continue
        e = p + 1
        if e + hold_days >= n:
            continue
        if prev_basket is not None:
            turns.append(1 - len(set(basket) & prev_basket) / max(len(basket), 1))
        prev_basket = set(basket)
        fwd = (panel.iloc[e + hold_days][basket] / panel.iloc[e][basket] - 1).dropna()
        if fwd.empty:
            continue
        gross = float(fwd.mean())
        net = float(np.mean([cost.net(x) for x in fwd]))
        bench = float(spy.iloc[e + hold_days] / spy.iloc[e] - 1)
        events.append({"date": panel.index[p].date().isoformat(), "basket": basket,
                       "gross_pct": round(100 * gross, 2), "net_pct": round(100 * net, 2),
                       "excess_pct": round(100 * (net - bench), 2), "bench_pct": round(100 * bench, 2)})

    if len(events) < 4:
        raise RuntimeError(f"이벤트 수 부족({len(events)}) — 구간·리밸간격을 조정하세요.")

    ex = np.array([e["excess_pct"] / 100 for e in events])
    net = np.array([e["net_pct"] / 100 for e in events])
    T_raw = len(ex)
    t_eff = max(int(round(T_raw * min(rebal_days / hold_days, 1.0))), 3)
    mean_ex = float(ex.mean())
    sd_ex = float(ex.std(ddof=1)) if T_raw > 1 else 0.0
    se = sd_ex / math.sqrt(t_eff) if sd_ex > 0 else 0.0
    t_stat = mean_ex / se if se > 0 else 0.0
    p_value = 2 * (1 - NormalDist().cdf(abs(t_stat))) if se > 0 else 1.0

    payload = {
        "as_of": today.date().isoformat(), "weights_used": weights,
        "window": {"start": start.date().isoformat(), "end": end.date().isoformat()},
        "rebal_days": rebal_days, "hold_days": hold_days, "topn": topn,
        "n_events_raw": T_raw, "n_eff": t_eff,
        "turnover_pct": round(100 * float(np.mean(turns)), 1) if turns else None,
        "mean_net_pct": round(100 * float(net.mean()), 2),
        "mean_excess_pct": round(100 * mean_ex, 2),
        "std_excess_pct": round(100 * sd_ex, 2),
        "win_rate_pct": round(100 * float((net > 0).mean()), 1),
        "worst_net_pct": round(100 * float(net.min()), 1),
        "t_stat_eff": round(t_stat, 2),
        "p_value_approx": round(p_value, 4),
        "significant_at_5pct": bool(se > 0 and p_value < 0.05),
        "events": events,
        "note": ("이미 채택된 가중치(재탐색 없음)를 최근 구간에 재적용한 표본외 재현성 확인 — "
                "다중검정 과최적화 검사(PBO)는 여기선 해당 없음(탐색 자체가 없으므로). "
                "리밸·보유 중첩으로 n_eff(유효표본)가 n_events_raw보다 훨씬 작음에 유의"
                "(SCORE_MODEL_DESIGN.md D3와 동일한 함정).")}
    os.makedirs("output", exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    _log(f"저장: {OUT_PATH} (이벤트 {T_raw}개 · 유효표본 {t_eff} · 평균초과 {payload['mean_excess_pct']}%p "
         f"· t={t_stat:.2f} · p≈{p_value:.3f})")
    return payload


# ------------------------- self-test -------------------------
def self_test():
    _log("[self-test] 합성 데이터로 최근구간 검증 로직 점검")
    panel, spy, funds, opens = BW._synthetic()
    pit = BC._synthetic_pit(panel)
    weights = {"gross_margin": 1, "shareholder_yield": 1}
    cost = BC.CostModel("us", commission_bps=0.0, slippage_bps=5.0)
    payload = run_recent(panel, spy, funds, pit, weights,
                         start_months_ago=12, end_months_ago=3,
                         rebal_days=5, hold_days=63, topn=4, cost=cost)
    assert payload["n_events_raw"] > 4
    assert payload["n_eff"] < payload["n_events_raw"], "겹침 보정이 적용 안 됨"
    assert payload["n_eff"] == max(int(round(payload["n_events_raw"] * 5 / 63)), 3)
    _log(f"[self-test] 통과: 이벤트 {payload['n_events_raw']} · 유효표본 {payload['n_eff']}")


def main():
    ap = argparse.ArgumentParser(description="최근 구간 워크포워드 재현성 확인(재탐색 없음, 고정 가중치)")
    ap.add_argument("--years", type=float, default=2.5, help="데이터 다운로드 기간(룩백+구간 커버용)")
    ap.add_argument("--start-months-ago", type=int, default=12)
    ap.add_argument("--end-months-ago", type=int, default=3)
    ap.add_argument("--rebal-days", type=int, default=5)
    ap.add_argument("--hold-days", type=int, default=63)
    ap.add_argument("--topn", type=int, default=4)
    ap.add_argument("--market", default="us", choices=["us", "kospi", "kosdaq"])
    ap.add_argument("--commission-bps", type=float, default=0.0)
    ap.add_argument("--slippage-bps", type=float, default=5.0)
    ap.add_argument("--pit-file", default=None)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test(); return
    weights = _load_weights()
    pit = BC.load_pit(args.pit_file)
    panel, spy, _ = BC.build_panel_pit(args.years, pit)
    funds = BW.load_funds()
    cost = BC.CostModel(args.market, args.commission_bps, args.slippage_bps)
    run_recent(panel, spy, funds, pit, weights,
              start_months_ago=args.start_months_ago, end_months_ago=args.end_months_ago,
              rebal_days=args.rebal_days, hold_days=args.hold_days, topn=args.topn, cost=cost)


if __name__ == "__main__":
    main()
