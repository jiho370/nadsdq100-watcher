#!/usr/bin/env python3
"""
us_factor_formula_sweep.py — 지호 님 요청(2026-07-17): "gp_assets(수익성)·rd_mktcap(R&D
비중)·shareholder_yield(주주환원) 비율을 세밀하게 조정하는 것과 품질게이트·순위변환을
한 번에 백테스트". Fable 5 자문 반영: 섹터중립화(2026-07-10 backtest_sector_neutral.py)는
이미 시행·기각(OOS 붕괴, rd_mktcap=0 선택)됐으므로 재등록하지 않음. 대신:
  (1) 현행 3팩터(int_gp_assets·rd_mktcap·shareholder_yield) 가중치 비율을 0~3 격자로
      세밀 스윕(backtest_weights._weight_grid 재사용, 현행 1:2:2보다 촘촘)
  (2) rd_mktcap의 formulation 3종 비교: raw(현행, z-score ±3클립) / rank(순위→[-3,3]
      선형매핑, 극단치 지배 완화) / qgate(품질게이트 — int_gp_assets나 shareholder_yield가
      양수인 종목에만 R&D 가점)
  (1)×(2) 전체를 하나의 등록된 시행 세트로 묶어 overfit_stats(PBO/DSR)에 넣는다(다중검정
  정직하게 반영). 판정은 Fable 5 권고대로 "수익 개선"이 아니라 "섹터쏠림(HHI) 개선 +
  성과 비열등성" 프레임 — 이번 동기가 리스크 관리지 수익 극대화가 아니므로.

실행: python us_factor_formula_sweep.py
결과: output/us_factor_formula_sweep.json · output/pbo_report_us_factor_formula.json
"""
from __future__ import annotations
import os, sys, json, itertools, math
import numpy as np
import pandas as pd

import backtest_weights as BW
import overfit_stats as OS

TD_H = "6m"   # 판정 지표(backtest_weights.py와 동일하게 6개월 forward)
TD_DAYS = 126
LOOKBACK = 252
COST = BW.COST
TOPN = 8
LEVELS = (0, 1, 2, 3)   # 현행(0,1,2)보다 촘촘 — 지호 님 "세밀하게" 요청 반영
FACTORS = ["int_gp_assets", "rd_mktcap", "shareholder_yield"]


def _log(m): print(f"[팩터공식] {m}", file=sys.stderr)


def _weight_grid_fine():
    """backtest_weights._weight_grid 재사용 — 3팩터·레벨 0~3, gcd로 비율 중복 제거."""
    return BW._weight_grid(FACTORS, LEVELS)


def _rd_variants(raw_rd: pd.Series, z_gp: pd.Series, z_sy: pd.Series) -> dict:
    """rd_mktcap의 3가지 formulation. 입력은 그 시점 유니버스 원값(raw_rd)과 이미 계산된
    int_gp_assets·shareholder_yield의 z-score(qgate 조건용)."""
    sd = raw_rd.std()
    z_raw = ((raw_rd - raw_rd.mean()) / sd).clip(-3, 3) if sd and not np.isnan(sd) else raw_rd * 0.0
    n = len(raw_rd)
    rank = raw_rd.rank(method="average")
    z_rank = ((rank - 0.5) / n - 0.5) * 6 if n > 1 else raw_rd * 0.0   # [-3,3] 선형매핑
    qgate_mask = (z_gp > 0) | (z_sy > 0)
    z_qgate = z_raw.where(qgate_mask, 0.0)
    return {"raw": z_raw.fillna(0.0), "rank": z_rank.fillna(0.0), "qgate": z_qgate.fillna(0.0)}


def build_snaps(panel, funds, spy):
    """backtest_weights._raw_frame과 동일 시점 그리드(rebal_days=63)에서 3팩터 원값만 추출."""
    import fundamentals_edgar as F
    n = len(panel); max_h = max(BW.TD.values())
    ps = list(range(LOOKBACK, n - max_h - 1, 63))
    snaps = []
    for p in ps:
        price = panel.iloc[p]
        valid = [s for s in price.dropna().index if not np.isnan(panel.iloc[p - LOOKBACK][s])]
        if not valid:
            continue
        v = pd.Index(valid)
        date_iso = panel.index[p].date().isoformat()
        rows = {}
        for s in v:
            fv = F.factor_values(funds.get(s) or {}, date_iso, float(price[s]))
            rows[s] = {f: fv.get(f) for f in FACTORS}
        raw = pd.DataFrame(rows).T.astype(float)
        mom = (panel.iloc[p] / panel.iloc[p - 126] - 1).reindex(v)   # 유효종목 필터용(모멘텀 결측 제외)
        raw = raw[mom.notna()]
        if raw.empty or raw[FACTORS].isna().all(axis=None):
            continue
        e = p + 1
        fwd = panel.iloc[e + TD_DAYS][raw.index] / panel.iloc[e][raw.index] - 1
        bench = float(spy.iloc[e + TD_DAYS] / spy.iloc[e] - 1)
        snaps.append({"date": date_iso, "raw": raw, "fwd": fwd, "bench": bench})
    return snaps


def sector_of_map():
    import sp500_daily_report as R
    return R.fetch_wikipedia_sectors()


def eval_trial(snaps, weights: dict, rd_mode: str, sector_map: dict):
    """가중치+rd_mktcap formulation 조합 하나를 전체 스냅에서 평가.
    반환: {excess_ret(월별 리스트, 개별시행 판정용), max_sector_share(HHI 근사)}."""
    excess = []
    max_sec_counts = []
    for snap in snaps:
        raw = snap["raw"]
        z_gp = ((raw["int_gp_assets"] - raw["int_gp_assets"].mean()) / raw["int_gp_assets"].std()).clip(-3, 3).fillna(0.0)
        z_sy = ((raw["shareholder_yield"] - raw["shareholder_yield"].mean()) / raw["shareholder_yield"].std()).clip(-3, 3).fillna(0.0)
        rd_variants = _rd_variants(raw["rd_mktcap"], z_gp, z_sy)
        z_rd = rd_variants[rd_mode]
        comp = weights.get("int_gp_assets", 0) * z_gp + weights.get("rd_mktcap", 0) * z_rd \
             + weights.get("shareholder_yield", 0) * z_sy
        top = comp.sort_values(ascending=False).index[:TOPN]
        r = snap["fwd"].reindex(top).dropna()
        if len(r):
            net = float(r.mean()) - COST
            excess.append(net - snap["bench"])
        from collections import Counter
        secs = [sector_map.get(s, "(미상)") for s in top]
        c = Counter(secs)
        max_sec_counts.append(c.most_common(1)[0][1] if c else 0)
    return excess, max_sec_counts


def run(save=True):
    panel, spy, opens = BW.build_panel(10)
    funds = BW.load_funds()
    _log(f"패널 {panel.shape[1]}종목 x {panel.shape[0]}일 · 펀더멘탈 {'있음' if funds else '없음'}")
    snaps = build_snaps(panel, funds, spy)
    _log(f"스냅 {len(snaps)}개 확보")
    sector_map = sector_of_map()

    grid = _weight_grid_fine()
    _log(f"가중치 조합 {len(grid)}개 x formulation 3종 = {len(grid)*3}개 시행")

    trials = []   # {"label", "weights", "rd_mode", "excess": [...], "max_sec": [...]}
    for w in grid:
        for rd_mode in ("raw", "rank", "qgate"):
            excess, max_sec = eval_trial(snaps, w, rd_mode, sector_map)
            if len(excess) < 10:
                continue
            label = "·".join(f"{k}{v}" for k, v in w.items() if v) + f"[{rd_mode}]"
            trials.append({"label": label, "weights": w, "rd_mode": rd_mode,
                          "excess": excess, "max_sec": max_sec})

    if len(trials) < 2:
        raise RuntimeError("시행 부족")

    n_ev = min(len(t["excess"]) for t in trials)
    trial_data = {"horizon": "us_factor_formula", "universe": "sp500_full", "cost": {"round_trip_bps": COST*2*10000},
                 "rebal_days": 63, "hold_days": TD_DAYS,
                 "dates": [s["date"] for s in snaps[:n_ev]],
                 "trials": [t["label"] for t in trials],
                 "excess_returns": [t["excess"][:n_ev] for t in trials]}
    rpt = OS.analyze(trial_data, save=False)

    rows = []
    for t in trials:
        ex = np.array(t["excess"])
        ms = np.array(t["max_sec"])
        rows.append({"label": t["label"], "weights": t["weights"], "rd_mode": t["rd_mode"],
                    "n": len(ex), "excess_6m_mean_pct": round(100*float(ex.mean()), 2),
                    "excess_6m_sharpe": round(float(ex.mean()/ex.std()*math.sqrt(252/TD_DAYS)), 2) if ex.std() > 0 else 0.0,
                    "avg_max_sector_in_top8": round(float(ms.mean()), 2)})
    rows.sort(key=lambda r: r["excess_6m_sharpe"], reverse=True)

    # 현행 라이브(1,2,2 raw) 위치 확인
    live_label = "int_gp_assets1·rd_mktcap2·shareholder_yield2[raw]"
    live_rank = next((i+1 for i, r in enumerate(rows) if r["label"] == live_label), None)

    _log("상위 15개 시행(6M 초과수익 샤프 기준):")
    for r in rows[:15]:
        _log(f"  {r['label']:45s} 초과6M {r['excess_6m_mean_pct']:+6.2f}%p · 샤프 {r['excess_6m_sharpe']:5.2f} · "
             f"평균최대섹터쏠림 {r['avg_max_sector_in_top8']:.2f}")
    _log(f"현행 라이브({live_label}) 순위: {live_rank}/{len(rows)}")

    payload = {"as_of": panel.index[-1].date().isoformat(), "n_trials": len(trials),
              "grid_levels": LEVELS, "rows": rows, "live_config_label": live_label,
              "live_rank": live_rank,
              "pbo": rpt.get("pbo", {}).get("pbo"), "pbo_verdict": rpt.get("pbo_verdict"),
              "dsr": rpt.get("dsr", {}).get("dsr"), "dsr_verdict": rpt.get("dsr_verdict"),
              "passed": rpt.get("passed", False),
              "note": "판정 프레임(Fable 5 자문): 수익 개선이 아니라 섹터쏠림 개선+성과 "
                      "비열등성 — 섹터중립화는 2026-07-10 backtest_sector_neutral.py에서 "
                      "이미 시행·기각(OOS 붕괴)돼 재등록하지 않음."}
    if save:
        os.makedirs("output", exist_ok=True)
        with open("output/us_factor_formula_sweep.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        with open("output/pbo_report_us_factor_formula.json", "w", encoding="utf-8") as f:
            json.dump(rpt, f, ensure_ascii=False, indent=2)
        _log(f"저장: output/us_factor_formula_sweep.json · PBO {payload['pbo']} · DSR {payload['dsr']}")
    return payload


if __name__ == "__main__":
    run()
