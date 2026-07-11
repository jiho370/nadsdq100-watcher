#!/usr/bin/env python3
"""
expectancy_report.py — 분기 C-3: 점수 대신 "기대값(캘리브레이션)" 사실만 리포트에 추가.
설계 근거: NEXT_STEPS_SONNET.md 판정 확정(G1 통과·G2 실패) + SCORE_MODEL_DESIGN.md §2.5.

G2(스냅샷 Spearman)가 미통과라 0~10 점수는 아직 비노출. 대신:
  1. 전략 레벨 기대값 박스 — backtest_costs_compare.json(pit_best) + pbo_report.json(G1) 실측치.
     숫자 하드코딩 금지, 주 1회(월요일)만 표기(회전 최소화).
  2. 종목별로는 순위 사실만 — "오늘 팩터 랭킹 N위 / 후보 M종목" (점수·기대수익 문구 금지).
  3. 점수 표시 코드는 만들어 두되 score_calibration.load_calibration()이 None이면 숨긴다.
     향후 재검증에서 G2가 통과하면 수동 개입 없이 자동으로 켜진다.

발송 로직(ai_report.py) 삽입 지점은 1곳(_card/render_report_html) — 이 모듈 자체는 순수 계산+렌더.

self-test: python expectancy_report.py --self-test
"""
from __future__ import annotations
import sys, json, datetime as dt

import score_calibration as SC

COMPARE_PATH = "output/backtest_costs_compare.json"
PBO_PATH = "output/pbo_report.json"
KST = dt.timezone(dt.timedelta(hours=9))


def _log(m): print(f"[기대값] {m}", file=sys.stderr)


def _esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_strategy_expectancy(horizon="6m"):
    """검증된 사실만 반환: backtest_costs_compare.json(pit_best) + pbo_report.json(G1).
    G1 미통과(passed!=True)거나 파일이 없으면 None — 미검증 수치는 리포트에 올리지 않는다."""
    compare = _load_json(COMPARE_PATH)
    pbo = _load_json(PBO_PATH)
    if not compare or not pbo or not pbo.get("passed"):
        return None
    pit_best = compare.get("pit_best") or {}
    ret, excess, win, worst = (pit_best.get(f"ret_{horizon}"), pit_best.get(f"excess_{horizon}"),
                               pit_best.get(f"win_{horizon}"), pit_best.get(f"worst_{horizon}"))
    if None in (ret, excess, win, worst):
        return None
    return {"horizon": horizon, "ret_pct": ret, "excess_pct": excess, "win_pct": win,
            "worst_pct": worst, "pbo_pct": round(pbo["pbo"]["pbo"] * 100, 1),
            "dsr": pbo["dsr"]["dsr"], "as_of": compare.get("as_of")}


def _is_weekly_cadence(now=None) -> bool:
    """주 1회만 표기(월요일) — 회전 최소화. 검증용으로 now 주입 가능."""
    now = now or dt.datetime.now(KST)
    return now.weekday() == 0


def expectancy_box_html(now=None) -> str:
    """전략 레벨 기대값 박스. 주간 케이던스 아니거나 G1 실측치가 없으면 빈 문자열."""
    if not _is_weekly_cadence(now):
        return ""
    exp = load_strategy_expectancy()
    if not exp:
        return ""
    h = {"1m": "1개월", "3m": "3개월", "6m": "6개월", "12m": "12개월"}.get(exp["horizon"], exp["horizon"])
    return (
        '<div style="background:#f0fdf4;border:1px solid #86efac;border-radius:8px;padding:8px 12px;'
        'margin:8px 0;font-size:12px;color:#14532d;line-height:1.6">'
        f'<b>이 추천 방식의 과거 실측</b>(비용·생존편향 반영, {_esc(exp["as_of"] or "")} 기준): '
        f'{h} 보유 시 평균 {exp["ret_pct"]:+.1f}% (S&amp;P500 대비 {exp["excess_pct"]:+.1f}%p), '
        f'승률 {exp["win_pct"]:.1f}%, 최악 {exp["worst_pct"]:.1f}% '
        f'&middot; 통계 검증: PBO {exp["pbo_pct"]:.1f}% &middot; DSR {exp["dsr"]:.2f}</div>')


def rank_fact_html(rank, pool_size) -> str:
    """검증된 사실만: 오늘 팩터 랭킹 N위 / 후보 M종목. 점수·기대수익 문구 없음."""
    if not rank or not pool_size:
        return ""
    return (f'<div style="font-size:11px;color:#6b7280;margin-top:3px">'
            f'오늘 팩터 랭킹 {int(rank)}위 / 후보 {int(pool_size)}종목</div>')


def score_line_html(rank, pool_size) -> str:
    """0~10 점수 표시 — score_calibration.load_calibration()이 None이면(G2 미통과) 항상 빈 문자열.
    향후 재검증에서 G2가 통과하면 수동 개입 없이 자동으로 켜진다."""
    cal = SC.load_calibration()
    if not cal or not rank or not pool_size:
        return ""
    pct = 1.0 - (int(rank) - 1) / int(pool_size)
    score = SC.score_from_percentile(pct)
    idx = min(max(score - 1, 0), 9)
    decs = cal.get("deciles") or []
    if idx >= len(decs):
        return ""
    dec = decs[idx]
    return (f'<div style="font-size:11px;color:#15803d;margin-top:2px;font-weight:600">'
            f'점수 {score} — 과거 동일 분위({cal.get("horizon", "6m")}): '
            f'평균 {dec["mean_ret_pct"]:+.1f}% &middot; 승률 {dec["win_rate_pct"]:.0f}% '
            f'&middot; 손익비 {dec["payoff"]:.2f}</div>')


# ------------------------- self-test -------------------------
def self_test():
    _log("[self-test] ① 주간 케이던스 ② 전략 박스(G1 실측) ③ 랭킹 사실 ④ 점수 게이트")
    mon = dt.datetime(2026, 7, 13, 9, 0, tzinfo=KST)   # 월요일
    tue = dt.datetime(2026, 7, 14, 9, 0, tzinfo=KST)   # 화요일
    assert _is_weekly_cadence(mon) and not _is_weekly_cadence(tue)
    assert expectancy_box_html(tue) == "", "월요일이 아니면 박스 숨김이어야 함"

    assert "랭킹 3위" in rank_fact_html(3, 60) and "후보 60종목" in rank_fact_html(3, 60)
    assert rank_fact_html(None, 60) == "" and rank_fact_html(3, None) == ""

    global COMPARE_PATH, PBO_PATH
    orig_paths = (COMPARE_PATH, PBO_PATH)
    import tempfile, os
    with tempfile.TemporaryDirectory() as td:
        cp, pp = os.path.join(td, "compare.json"), os.path.join(td, "pbo.json")
        fake_compare = {"as_of": "2026-07-10", "pit_best": {
            "ret_6m": 12.97, "excess_6m": 5.42, "win_6m": 87.5, "worst_6m": -20.1}}
        fake_pbo_pass = {"passed": True, "pbo": {"pbo": 0.15}, "dsr": {"dsr": 0.9711}}
        fake_pbo_fail = {"passed": False, "pbo": {"pbo": 0.5}, "dsr": {"dsr": 0.5}}
        json.dump(fake_compare, open(cp, "w", encoding="utf-8"))
        try:
            COMPARE_PATH, PBO_PATH = cp, pp
            json.dump(fake_pbo_pass, open(pp, "w", encoding="utf-8"))
            exp = load_strategy_expectancy()
            assert exp and exp["ret_pct"] == 12.97, f"G1 통과인데 실측치 못 읽음: {exp}"
            html = expectancy_box_html(mon)
            assert "평균" in html and "PBO 15.0%" in html and "DSR 0.97" in html, html

            json.dump(fake_pbo_fail, open(pp, "w", encoding="utf-8"))
            assert load_strategy_expectancy() is None, "G1 미통과인데 박스가 뜨면 안 됨"
        finally:
            COMPARE_PATH, PBO_PATH = orig_paths

    from unittest import mock
    with mock.patch.object(SC, "load_calibration", return_value=None):
        assert score_line_html(3, 60) == "", "G2 미통과(None)인데 점수가 노출됨"
    fake_cal = {"horizon": "6m", "deciles": [
        {"mean_ret_pct": 4.4, "win_rate_pct": 61.0, "payoff": 1.35}] * 10}
    with mock.patch.object(SC, "load_calibration", return_value=fake_cal):
        line = score_line_html(1, 10)
        assert "점수 10" in line and "4.4" in line, line

    _log("[self-test] 통과: 케이던스 · G1 실측 박스 · 랭킹 사실 · 점수 게이트(G2) 전부 OK")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="분기 C-3 기대값 리포트 — 순수 계산/렌더 모듈")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
    else:
        print("사용법: python expectancy_report.py --self-test", file=sys.stderr)
