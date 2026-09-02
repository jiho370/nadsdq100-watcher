#!/usr/bin/env python3
"""
us_sector_concentration_analysis.py — 팩터 상위 20종목이 특정 시점에 특정 섹터로
쏠리는 경향이 있는지, 그리고 그 쏠림이 실제로 그 섹터의 이후 초과수익을 예측하는지
탐색(2026-09-01, 지호 님 요청 — "알고리즘이 종목 추천뿐 아니라 저평가 섹터도
집어낼 수 있는지").

⚠ 방법론적 한계(중요): sector_map은 sp500_daily_report.fetch_wikipedia_sectors()로
현재 시점 GICS 섹터를 그대로 과거 모든 리밸런싱일에 소급 적용한다(이 저장소의 다른
모든 섹터 관련 백테스트 — us_factor_formula_pit_sweep.py 등 — 와 동일한 기존 한계.
PIT(시점별) 섹터 데이터 자체가 없음). 섹터 재분류(예: GOOGL/META 2018년 커뮤니케이션
서비스 신설 편입)나 상장폐지·합병 종목의 섹터 소실은 반영 안 됨. 표본도 10년 34분기뿐
이라 통계적 검정력이 낮다 — 아래 결과는 PBO/DSR 같은 정식 게이트를 통과한 결론이
아니라 탐색적 참고 분석이다.

실행: python us_sector_concentration_analysis.py
결과: output/us_sector_concentration.json
"""
from __future__ import annotations
import json
from collections import Counter
import numpy as np
import pandas as pd

import backtest_costs as BC
import us_factor_formula_pit_sweep as PS


def _composite(raw: pd.DataFrame) -> pd.Series:
    """라이브 가중치(gp1·rd2·sy2, rd raw)로 종합점수 — us_factor_formula_pit_sweep.TRIALS[0]와 동일."""
    z_gp = ((raw["int_gp_assets"] - raw["int_gp_assets"].mean()) / raw["int_gp_assets"].std()).clip(-3, 3).fillna(0.0)
    z_sy = ((raw["shareholder_yield"] - raw["shareholder_yield"].mean()) / raw["shareholder_yield"].std()).clip(-3, 3).fillna(0.0)
    z_rd = PS._rd_variant(raw["rd_mktcap"], z_gp, z_sy, "raw")
    return 1 * z_gp + 2 * z_rd + 2 * z_sy


def run(years=10, topn=20, min_in_top=3):
    pit = BC.load_pit()
    panel, spy, _ = BC.build_panel_pit(years, pit)
    with open("output/fundamentals_cache.json", encoding="utf-8") as f:
        funds = json.load(f)
    snaps = PS.build_snaps(panel, spy, funds, pit)
    sector_map = PS.sector_of_map()

    # 섹터별 누적 통계(2026-09-01, 지호 님 요청 — "많이 나온 섹터 랭킹, 섹터별 수익률 랭킹"):
    # 매 분기 top20 등장횟수 + 그 섹터 전체(유니버스 기준)의 그 시점 이후 6개월 수익률을
    # 섹터마다 누적해뒀다가 루프가 끝난 뒤 두 개의 독립적 랭킹으로 집계한다.
    sector_stats = {}  # sec -> {"top20_counts": [...], "fwd_pcts": [...], "excess_pcts": [...]}
    # (미상) 정체 확인용(2026-09-01, 지호 님 요청 — "이거 (미상) 종목들이 뭔지 목록으로"):
    # sector_map(오늘 위키피디아 S&P500 목록)에 없는 티커 = 상장폐지·합병·인수 등으로
    # 지금은 지수에서 빠진 종목일 가능성이 큼. ticker -> {"dates":[...], "fwd_pcts":[...]}
    unclassified_detail = {}
    # 분기별 종목/섹터 상세(2026-09-01, 지호 님 요청 — "역대 분기별 뽑혔던 종목들/섹터
    # 리스트를 엑셀로"): top20 종목 하나하나의 순위·섹터·개별 이후 6개월 수익률.
    ticker_rows = []

    rows = []
    for snap in snaps:
        raw = snap["raw"]
        comp = _composite(raw)
        ranked = comp.sort_values(ascending=False).index.tolist()
        top = ranked[:topn]
        universe = list(raw.index)

        for rank, s in enumerate(top, 1):
            fr = snap["fwd"].get(s)
            fr_pct = round(float(fr) * 100, 2) if (fr is not None and fr == fr) else None
            ticker_rows.append({
                "date": snap["date"], "rank": rank, "ticker": s,
                "sector": sector_map.get(s, "(미상)"),
                "composite_score": round(float(comp[s]), 3),
                "fwd_6m_pct": fr_pct,
                "spy_fwd_6m_pct": round(snap["bench"] * 100, 2),
                "excess_vs_spy_pct": round(fr_pct - snap["bench"] * 100, 2) if fr_pct is not None else None,
            })

        for s in top:
            if s not in sector_map:
                fr = snap["fwd"].get(s)
                d = unclassified_detail.setdefault(s, {"dates": [], "fwd_pcts": []})
                d["dates"].append(snap["date"])
                if fr is not None and fr == fr:
                    d["fwd_pcts"].append(round(float(fr) * 100, 2))

        uni_cnt = Counter(sector_map.get(s, "(미상)") for s in universe)
        top_cnt = Counter(sector_map.get(s, "(미상)") for s in top)
        uni_n, top_n = len(universe), len(top)

        overrep = {}
        for sec, c in top_cnt.items():
            uni_share = uni_cnt.get(sec, 0) / uni_n
            if uni_share > 0:
                overrep[sec] = (c / top_n) / uni_share

        def _sector_fwd(sec):
            if not sec:
                return None
            members = [s for s in universe if sector_map.get(s, "(미상)") == sec]
            fr = snap["fwd"].reindex(members).dropna()
            return float(fr.mean()) if len(fr) else None

        # 이번 분기에 top20에 하나라도 있었던 섹터 전부(=1개 이상)를 대상으로 누적
        # — "그 섹터를 그때 하나라도 담았다면 실제로 그 섹터 자체는 어떻게 됐는지"
        for sec, c in top_cnt.items():
            fwd = _sector_fwd(sec)
            if fwd is None:
                continue
            st = sector_stats.setdefault(sec, {"top20_counts": [], "fwd_pcts": [], "excess_pcts": []})
            st["top20_counts"].append(c)
            st["fwd_pcts"].append(fwd * 100)
            st["excess_pcts"].append((fwd - snap["bench"]) * 100)

        candidates = [(sec, r) for sec, r in overrep.items() if top_cnt[sec] >= min_in_top]
        dominant, ratio = max(candidates, key=lambda x: x[1]) if candidates else (None, None)
        sec_fwd = _sector_fwd(dominant)

        # IT 제외 2군 쏠림(2026-09-01, 지호 님 질문 — "IT 제외하면?"): IT는 거의 항상 1위라
        # 진짜 로테이션이 있는지 보려면 IT를 뺀 나머지 중 1위를 따로 봐야 한다.
        candidates_ex = [(sec, r) for sec, r in candidates if sec != "Information Technology"]
        dominant_ex, ratio_ex = max(candidates_ex, key=lambda x: x[1]) if candidates_ex else (None, None)
        sec_fwd_ex = _sector_fwd(dominant_ex)

        rows.append({
            "date": snap["date"],
            "dominant_sector": dominant,
            "overrep_ratio": round(ratio, 2) if ratio else None,
            "n_in_top20": top_cnt.get(dominant, 0) if dominant else None,
            "sector_fwd_6m_pct": round(sec_fwd * 100, 2) if sec_fwd is not None else None,
            "spy_fwd_6m_pct": round(snap["bench"] * 100, 2),
            "sector_excess_vs_spy_pct": round((sec_fwd - snap["bench"]) * 100, 2) if sec_fwd is not None else None,
            "dominant_sector_ex_it": dominant_ex,
            "overrep_ratio_ex_it": round(ratio_ex, 2) if ratio_ex else None,
            "n_in_top20_ex_it": top_cnt.get(dominant_ex, 0) if dominant_ex else None,
            "sector_fwd_6m_pct_ex_it": round(sec_fwd_ex * 100, 2) if sec_fwd_ex is not None else None,
            "sector_excess_vs_spy_pct_ex_it": round((sec_fwd_ex - snap["bench"]) * 100, 2) if sec_fwd_ex is not None else None,
            "top20_sector_breakdown": dict(sorted(top_cnt.items(), key=lambda x: -x[1])),
        })

    valid = [r for r in rows if r["sector_excess_vs_spy_pct"] is not None]
    hit_rate = round(100 * sum(1 for r in valid if r["sector_excess_vs_spy_pct"] > 0) / len(valid), 1) if valid else None
    mean_excess = round(float(np.mean([r["sector_excess_vs_spy_pct"] for r in valid])), 2) if valid else None
    ratios = np.array([r["overrep_ratio"] for r in valid])
    excs = np.array([r["sector_excess_vs_spy_pct"] for r in valid])
    corr = round(float(np.corrcoef(ratios, excs)[0, 1]), 3) if len(valid) > 2 else None
    dominant_seq = Counter(r["dominant_sector"] for r in rows if r["dominant_sector"])

    valid_ex = [r for r in rows if r["sector_excess_vs_spy_pct_ex_it"] is not None]
    hit_rate_ex = round(100 * sum(1 for r in valid_ex if r["sector_excess_vs_spy_pct_ex_it"] > 0) / len(valid_ex), 1) if valid_ex else None
    mean_excess_ex = round(float(np.mean([r["sector_excess_vs_spy_pct_ex_it"] for r in valid_ex])), 2) if valid_ex else None
    dominant_seq_ex = Counter(r["dominant_sector_ex_it"] for r in rows if r["dominant_sector_ex_it"])
    # 최근 8개 분기(약 2년)만 따로 — "요즘" 패턴 확인용
    recent_ex = Counter(r["dominant_sector_ex_it"] for r in rows[-8:] if r["dominant_sector_ex_it"])

    # 랭킹 1: 많이 나온 섹터(전체 34분기 top20 누적 등장횟수 기준)
    freq_rank = sorted(
        ({"sector": sec, "total_top20_appearances": sum(st["top20_counts"]),
          "n_quarters_present": len(st["top20_counts"]),
          "avg_per_quarter_present": round(float(np.mean(st["top20_counts"])), 2)}
         for sec, st in sector_stats.items()),
        key=lambda x: -x["total_top20_appearances"])

    # 랭킹 2: 섹터별 수익률(그 섹터를 top20에 하나라도 담았던 분기들의 평균 이후 6개월
    # 수익률·초과수익 — 최소 3개 분기 이상 등장한 섹터만, 표본 너무 적은 섹터 배제)
    ret_rank = sorted(
        ({"sector": sec, "n_quarters": len(st["fwd_pcts"]),
          "avg_fwd_6m_pct": round(float(np.mean(st["fwd_pcts"])), 2),
          "avg_excess_vs_spy_pct": round(float(np.mean(st["excess_pcts"])), 2),
          "hit_rate_pct": round(100 * sum(1 for e in st["excess_pcts"] if e > 0) / len(st["excess_pcts"]), 1)}
         for sec, st in sector_stats.items() if len(st["fwd_pcts"]) >= 3),
        key=lambda x: -x["avg_excess_vs_spy_pct"])

    summary = {
        "n_snaps": len(snaps), "n_with_dominant_sector": len(valid),
        "hit_rate_pct": hit_rate, "mean_sector_excess_vs_spy_pct": mean_excess,
        "corr_overrep_ratio_vs_fwd_excess": corr,
        "dominant_sector_frequency": dict(dominant_seq.most_common()),
        "ex_it__hit_rate_pct": hit_rate_ex, "ex_it__mean_sector_excess_vs_spy_pct": mean_excess_ex,
        "ex_it__dominant_sector_frequency_전체": dict(dominant_seq_ex.most_common()),
        "ex_it__dominant_sector_frequency_최근8분기": dict(recent_ex.most_common()),
        "sector_frequency_ranking": freq_rank,
        "sector_return_ranking": ret_rank,
        "caveat": "sector_map은 현재 시점 GICS를 과거에 소급 적용(PIT 아님). 34분기 표본, 탐색적 분석.",
    }
    unclassified_list = sorted(
        ({"ticker": t, "n_quarters": len(d["dates"]), "dates": d["dates"],
          "avg_fwd_6m_pct": round(float(np.mean(d["fwd_pcts"])), 2) if d["fwd_pcts"] else None}
         for t, d in unclassified_detail.items()),
        key=lambda x: -x["n_quarters"])

    payload = {"summary": summary, "rows": rows, "unclassified_tickers": unclassified_list,
              "ticker_rows": ticker_rows}
    with open("output/us_sector_concentration.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return payload


if __name__ == "__main__":
    run()
