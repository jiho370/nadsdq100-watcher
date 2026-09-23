#!/usr/bin/env python3
"""
us_band_trading_extreme.py — 지호 님 실제 의도 반영 재검증 (2026-09-22, 채팅 피드백).

배경: us_algo8_band_trading_grid.py(2026-09-08)는 상승·하락 3~25%를 촘촘히(7×7) 훑었지만,
지호 님이 실제로 생각하던 구간은 "최소 15% 이상, 연 5~6회 정도만 매매 — 극단 이상치
위험관리 겸 익절"이었다. 기존 그리드에서 15%+ 영역은 3칸(15/20/25)뿐이라 성기고, 트랜치도
10~50%뿐이라 "포지션의 일부만 살짝 정리"하는 옵션(5~10%대 트랜치)이 아예 없었다 — 즉
지호 님이 실제로 관심있던 영역을 정면으로 겨냥한 그리드가 아니었다.

방법: 새로 구현하지 않고 us_algo8_band_trading_grid의 검증된 엔진(simulate_band·
build_context·DIRECTIONS)을 그대로 재사용 — 그리드 상수(RISE_GRID·FALL_GRID·TRANCHE_GRID)만
모듈 임포트 후 교체해서 run()을 그대로 호출한다(MAINTENANCE.md §1 "재구현 대신 재사용"
원칙). ProcessPoolExecutor 워커는 build_grid()가 만든 '이미 값이 확정된' 작업 튜플만
받으므로, 워커가 모듈을 재임포트(Windows spawn)해도 그리드 패치와 무관하게 동작한다.

새 그리드: 상승·하락 [15,20,25,30,40,50]%(6×6, 기존의 3칸→6칸으로 세분화) × 트랜치
[5,10,15,20,25,33,50]%(7단계, 5·10·15%를 추가해 "살짝만 정리"하는 경우까지 포함) + 전량 =
사이징 8종 × 방향 4종 = 1,152조합. 연 5~6회 근방 조합을 보고 시 강조 표시.

실행: python -m research.us.us_band_trading_extreme [--years 10] [--workers N]
결과: output/us_band_trading_extreme.json
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import research.us.us_algo8_band_trading_grid as G

G.RISE_GRID = [15, 20, 25, 30, 40, 50]
G.FALL_GRID = [15, 20, 25, 30, 40, 50]
G.TRANCHE_GRID = [5, 10, 15, 20, 25, 33, 50]

OUT_PATH = "output/us_band_trading_extreme.json"


def _log(m):
    print(f"[극단밴드그리드] {m}", file=sys.stderr)


def summarize_near_target(payload: dict, target=5.5, tol=1.5) -> list[dict]:
    """연 5~6회(target±tol) 근방 조합만 추려 CAGR 내림차순 — 지호 님이 실제로 궁금해한 영역."""
    rows = [r for r in payload["all_results"] if abs(r["trades_per_year"] - target) <= tol]
    rows.sort(key=lambda r: r["cagr_pct"], reverse=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description="극단 이상치 밴드매매(15%+, 연5~6회 목표) 재검증")
    ap.add_argument("--years", type=float, default=10)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    payload = G.run(years=args.years, workers=args.workers, save=False)
    payload["grid_design_note"] = (
        "지호 님 의도(15%+, 연5~6회, 극단 이상치 위험관리/익절) 전용 그리드 — "
        "상승/하락 15~50%(6단계) × 트랜치 5~50%(7단계)+전량. "
        "us_algo8_band_trading_grid.json(3~25% 그리드)과는 별개 파일."
    )
    near_target = summarize_near_target(payload)
    payload["near_5_6_trades_per_year"] = near_target[:30]

    os.makedirs("output", exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    _log(f"저장: {OUT_PATH}")

    a = payload["baseline_no_band_trading"]
    _log(f"대조군(밴드매매 없음): CAGR {a['cagr_pct']}% 샤프 {a['sharpe']} MDD {a['mdd_pct']}% "
        f"연매매 {a['trades_per_year']}건")
    _log(f"연 5~6회(±1.5) 근방 조합 {len(near_target)}개 중 CAGR 상위 10개:")
    for r in near_target[:10]:
        tranche_str = "" if r["tranche_pct"] is None else f"({r['tranche_pct']}%)"
        _log(f"  {r['direction']:>10s} 상승{r['rise_pct']:>2}%/하락{r['fall_pct']:>2}% "
            f"{r['sizing']:>7s}{tranche_str} -> CAGR {r['cagr_pct']:>6.2f}% "
            f"샤프 {r['sharpe']:>5.2f} MDD {r['mdd_pct']:>6.1f}% 연매매 {r['trades_per_year']:>4.1f}건")


if __name__ == "__main__":
    main()
