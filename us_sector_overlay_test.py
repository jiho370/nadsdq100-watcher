#!/usr/bin/env python3
"""
us_sector_overlay_test.py — "섹터 오버레이가 실제로 돈을 더 버는가"를 라이브와 같은
조건(top8·섹터캡2·비용반영)에서 직접 검증(2026-09-01, 지호 님 요청 — "이중카운팅이면
어때. 돈만 잘 벌면 되지" → 논쟁 대신 실제로 테스트).

비교 3개 (전부 동일 PIT 패널·라이브 가중치 gp1·rd2·sy2 기준):
  BASE   : 현재 라이브 그대로 — top8, 섹터캡 2(전 섹터 동일)
  IT_IND : IT·산업재만 섹터캡 4로 완화(그 외 섹터는 캡 2 유지), top8
  EX_FIN : 금융 섹터 종목을 후보에서 아예 제외한 뒤 top8·섹터캡2

섹터는 us_sector_concentration_analysis.py에서 이미 확인한 재분류(오늘 위키피디아
목록에 없는 16개 티커를 실제 섹터로 수동 매핑)를 그대로 적용 — "금융 제외"가 실제로
BFH 등을 제외하도록 하기 위함(재분류 안 하면 이 티커들은 전부 "(미상)"이라 금융
제외 규칙이 사실상 아무것도 못 거름).

판정: 각 변형 vs BASE 페어드 차이(34분기, 포트폴리오 단위 초과수익) + t검정 +
overfit_stats(PBO/DSR)까지 이 저장소의 기존 검증 관례 그대로 적용.

실행: python us_sector_overlay_test.py
결과: output/us_sector_overlay_test.json
"""
from __future__ import annotations
import json, math
import numpy as np
import pandas as pd

import backtest_costs as BC
import us_factor_formula_pit_sweep as PS
import overfit_stats as OS

BW_COST = 0.001  # us_factor_formula_pit_sweep.py와 동일 관례(왕복 10bp 근사)

RECLASS = {
    "AIV": "Real Estate", "MAC": "Real Estate", "SLG": "Real Estate",
    "MTCH": "Communication Services", "LUMN": "Communication Services",
    "ETSY": "Consumer Discretionary", "GAP": "Consumer Discretionary", "BWA": "Consumer Discretionary",
    "WHR": "Consumer Discretionary", "SIG": "Consumer Discretionary", "BBWI": "Consumer Discretionary",
    "BFH": "Financials", "NOV": "Energy", "OGN": "Health Care", "ILMN": "Health Care",
    "QRVO": "Information Technology",
}


def _composite(raw: pd.DataFrame) -> pd.Series:
    z_gp = ((raw["int_gp_assets"] - raw["int_gp_assets"].mean()) / raw["int_gp_assets"].std()).clip(-3, 3).fillna(0.0)
    z_sy = ((raw["shareholder_yield"] - raw["shareholder_yield"].mean()) / raw["shareholder_yield"].std()).clip(-3, 3).fillna(0.0)
    z_rd = PS._rd_variant(raw["rd_mktcap"], z_gp, z_sy, "raw")
    return 1 * z_gp + 2 * z_rd + 2 * z_sy


def _sector_of(s, sector_map):
    return RECLASS.get(s, sector_map.get(s, "(미상)"))


def _pick_uniform_cap(ranked, sector_map, n, cap):
    out, per_sec = [], {}
    for s, sc in ranked:
        sec = _sector_of(s, sector_map)
        if per_sec.get(sec, 0) >= cap:
            continue
        out.append(s); per_sec[sec] = per_sec.get(sec, 0) + 1
        if len(out) >= n:
            break
    return out


def _pick_boost_cap(ranked, sector_map, n, boost_secs, boost_cap, base_cap):
    out, per_sec = [], {}
    for s, sc in ranked:
        sec = _sector_of(s, sector_map)
        cap = boost_cap if sec in boost_secs else base_cap
        if per_sec.get(sec, 0) >= cap:
            continue
        out.append(s); per_sec[sec] = per_sec.get(sec, 0) + 1
        if len(out) >= n:
            break
    return out


def _pick_exclude(ranked, sector_map, n, cap, exclude_secs):
    filtered = [(s, sc) for s, sc in ranked if _sector_of(s, sector_map) not in exclude_secs]
    return _pick_uniform_cap(filtered, sector_map, n, cap)


def _port_excess(picks, snap):
    r = snap["fwd"].reindex(picks).dropna()
    if not len(r):
        return np.nan
    net = float(r.mean()) - BW_COST
    return net - snap["bench"]


def run(years=10, topn=8, cap=2):
    pit = BC.load_pit()
    panel, spy, _ = BC.build_panel_pit(years, pit)
    with open("output/fundamentals_cache.json", encoding="utf-8") as f:
        funds = json.load(f)
    snaps = PS.build_snaps(panel, spy, funds, pit)
    sector_map = PS.sector_of_map()

    base_ex, itind_ex, exfin_ex = [], [], []
    dates = []
    for snap in snaps:
        raw = snap["raw"]
        comp = _composite(raw)
        ranked = [(s, comp[s]) for s in comp.sort_values(ascending=False).index]

        base_picks = _pick_uniform_cap(ranked, sector_map, topn, cap)
        itind_picks = _pick_boost_cap(ranked, sector_map, topn,
                                      {"Information Technology", "Industrials"}, 4, cap)
        exfin_picks = _pick_exclude(ranked, sector_map, topn, cap, {"Financials"})

        base_ex.append(_port_excess(base_picks, snap))
        itind_ex.append(_port_excess(itind_picks, snap))
        exfin_ex.append(_port_excess(exfin_picks, snap))
        dates.append(snap["date"])

    def _stats(ex):
        a = np.array(ex, dtype=float)
        a = a[~np.isnan(a)]
        return {"n": len(a), "mean_excess_6m_pct": round(100 * float(a.mean()), 2),
                "std_pct": round(100 * float(a.std(ddof=1)), 2),
                "hit_rate_pct": round(100 * float((a > 0).mean()), 1)}

    def _paired(challenger, base, label):
        pair = np.array([c - b if (c == c and b == b) else np.nan for c, b in zip(challenger, base)])
        pair = pair[~np.isnan(pair)]
        mean_diff = float(pair.mean()) if len(pair) else 0.0
        se = float(pair.std(ddof=1) / math.sqrt(len(pair))) if len(pair) > 1 else 0.0
        tstat = mean_diff / se if se > 0 else 0.0
        return {"label": label, "paired_diff_6m_pct": round(100 * mean_diff, 2),
                "paired_tstat": round(tstat, 2), "significant_p05": abs(tstat) > 2.03}  # df~33, 양측 5%

    trial_data = {"horizon": "us_sector_overlay_top8cap2", "universe": "sp500_pit",
                 "cost": {"approx_bps": BW_COST * 10000}, "rebal_days": PS.REBAL_DAYS,
                 "hold_days": PS.TD_DAYS, "dates": dates,
                 "trials": ["BASE(라이브 top8cap2)", "IT_IND(IT·산업재캡4)", "EX_FIN(금융제외)"],
                 "excess_returns": [[v if v == v else 0.0 for v in base_ex],
                                    [v if v == v else 0.0 for v in itind_ex],
                                    [v if v == v else 0.0 for v in exfin_ex]]}
    rpt = OS.analyze(trial_data, save=False)

    payload = {
        "n_quarters": len(snaps),
        "BASE": _stats(base_ex), "IT_IND": _stats(itind_ex), "EX_FIN": _stats(exfin_ex),
        "paired_vs_base": [_paired(itind_ex, base_ex, "IT_IND"), _paired(exfin_ex, base_ex, "EX_FIN")],
        "pbo": rpt.get("pbo", {}).get("pbo"), "dsr": rpt.get("dsr", {}).get("dsr"),
        "passed": rpt.get("passed", False),
        "note": "top8·섹터캡2 라이브와 동일 조건, 왕복비용 10bp 근사 반영. "
                "(미상) 16종목은 실제 섹터로 재분류 적용.",
    }
    with open("output/us_sector_overlay_test.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


if __name__ == "__main__":
    run()
