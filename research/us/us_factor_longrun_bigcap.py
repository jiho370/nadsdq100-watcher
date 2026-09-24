#!/usr/bin/env python3
"""
us_factor_longrun_bigcap.py — 지호 님 질문(2026-09-24): "그럼 60년 볼까? 30년?"

우리 종목단위 데이터는 EDGAR(2009~)·야후(상장폐지 시세 없음) 때문에 30년 이상 늘릴 수 없어서,
켄 프렌치(CRSP) 2×3·5×5 규모 교차 포트폴리오의 **대형주(BIG) 쪽만** 써서 S&P500에 가까운
유니버스로 60년(1963-07~) 단일 팩터 Hi−Lo(시총가중 월수익)를 본다.

질문 두 가지를 한 번에:
  1) 부호가 시대와 무관하게 유지되나 — 10년 단위 구간, 10년 롤링 창에서 같은 부호인 비율
  2) "30년으로 방향을 정하면 다음 30년에도 맞나" — 1963-07~1994-12(앞 30년)의 부호로 방향을
     정해 1995-01~(뒤 30년)에 적용한 방향보정 수익. 앞 30년 기준의 사전 결정이라 뒤 30년은 표본외.

실행: python -m research.us.us_factor_longrun_bigcap
결과: output/us_factor_longrun_bigcap.json
"""
from __future__ import annotations
import json
import math
import sys

import numpy as np
import pandas as pd

from research.us.us_single_factor_study import _ff_first_tables

START, SPLIT = "1963-07", "1995-01"
# 라벨: (파일, Hi−Lo 해석) — Hi/Lo는 해당 지표의 높고 낮음(예: 순발행 Hi = 주식을 많이 찍은 기업)
FILES = {
    "value_BM": "6_Portfolios_2x3_CSV",
    "value_EP": "6_Portfolios_ME_EP_2x3_CSV",
    "value_CFP": "6_Portfolios_ME_CFP_2x3_CSV",
    "dividend_yield": "6_Portfolios_ME_DP_2x3_CSV",
    "profitability_OP": "6_Portfolios_ME_OP_2x3_CSV",
    "investment": "6_Portfolios_ME_INV_2x3_CSV",
    "momentum_12_2": "6_Portfolios_ME_Prior_12_2_CSV",
    "reversal_1m": "6_Portfolios_ME_Prior_1_0_CSV",
    "long_reversal_60_13": "6_Portfolios_ME_Prior_60_13_CSV",
    "accruals": "25_Portfolios_ME_AC_5x5_CSV",
    "net_share_issues": "25_Portfolios_ME_NI_5x5_CSV",
    "beta": "25_Portfolios_ME_BETA_5x5_CSV",
    "variance": "25_Portfolios_ME_VAR_5x5_CSV",
    "residual_variance": "25_Portfolios_ME_RESVAR_5x5_CSV",
}


def _log(m): print(f"[장기대형주]  {m}", file=sys.stderr)


def _stat(s: pd.Series) -> dict:
    s = s.dropna()
    if len(s) < 24:
        return {}
    return {"ann_pct": round(1200 * float(s.mean()), 2),
            "t": round(float(s.mean() / s.std(ddof=1) * math.sqrt(len(s))), 2),
            "months": int(len(s))}


def big_spread(fname: str) -> tuple[pd.Series, str, str]:
    df = _ff_first_tables(fname)["vw"]
    lo = next(c for c in df.columns if c.startswith("BIG Lo"))
    hi = next(c for c in df.columns if c.startswith("BIG Hi"))
    return (df[hi] - df[lo]).loc[START:], lo, hi


def analyze(s: pd.Series) -> dict:
    out = {"full": _stat(s), "first30": _stat(s.loc[:"1994-12"]), "last30": _stat(s.loc[SPLIT:])}
    dec = {}
    for y0 in range(1960, 2030, 10):
        sub = s.loc[str(max(y0, 1963)):str(y0 + 9)]
        if len(sub) >= 24:
            dec[f"{y0}s"] = _stat(sub)
    out["decades"] = dec
    roll = s.rolling(120).mean().dropna()
    full_sign = np.sign(s.mean())
    out["rolling10y_same_sign_as_full_pct"] = round(100 * float((np.sign(roll) == full_sign).mean()), 1)
    tstat = (s.rolling(120).mean() / s.rolling(120).std() * math.sqrt(120)).dropna()
    out["rolling10y_sig_same_sign_pct"] = round(100 * float(((tstat * full_sign) > 2).mean()), 1)
    out["rolling10y_sig_opposite_pct"] = round(100 * float(((tstat * full_sign) < -2).mean()), 1)
    # 앞 30년 부호로 방향 결정 → 뒤 30년 방향보정 수익
    sgn = np.sign(s.loc[:"1994-12"].mean())
    adj = sgn * s.loc[SPLIT:]
    out["first30_direction"] = "Hi" if sgn > 0 else "Lo"
    out["last30_direction_adjusted"] = _stat(adj)
    return out


def run() -> dict:
    res, last = {}, None
    for label, fname in FILES.items():
        s, lo, hi = big_spread(fname)
        r = analyze(s)
        r["columns"] = f"{hi} − {lo}"
        last = s.index[-1]
        res[label] = r
        a, b = r["first30"], r["last30"]
        _log(f"{label:20s} 전체 {r['full']['ann_pct']:+6.2f}%(t{r['full']['t']:+.2f}) | 앞30 {a['ann_pct']:+6.2f}(t{a['t']:+.2f}) "
             f"뒤30 {b['ann_pct']:+6.2f}(t{b['t']:+.2f}) | 앞30방향→뒤30 {r['last30_direction_adjusted']['ann_pct']:+6.2f}"
             f"(t{r['last30_direction_adjusted']['t']:+.2f}) | 롤링10년 같은부호 {r['rolling10y_same_sign_as_full_pct']}% "
             f"유의같은 {r['rolling10y_sig_same_sign_pct']}% 유의반대 {r['rolling10y_sig_opposite_pct']}%")
    adj = [r["last30_direction_adjusted"]["ann_pct"] for r in res.values()]
    kept = sum(1 for r in res.values() if r["last30_direction_adjusted"]["ann_pct"] > 0)
    summary = {"n_factors": len(res), "last30_direction_kept": kept,
               "last30_direction_adjusted_mean_ann_pct": round(float(np.mean(adj)), 2),
               "first30_abs_mean_ann_pct": round(float(np.mean([abs(r["first30"]["ann_pct"]) for r in res.values()])), 2)}
    _log(f"요약: 앞30년 방향이 뒤30년에도 맞은 팩터 {kept}/{len(res)} · 방향보정 평균 "
         f"{summary['last30_direction_adjusted_mean_ann_pct']}%/년 (앞30년 |평균| {summary['first30_abs_mean_ann_pct']}%)")
    out = {"source": "Kenneth R. French Data Library (CRSP), value-weighted, BIG size group only",
           "last_month": last.strftime("%Y-%m"),
           "start": START, "split": SPLIT, "summary": summary, "factors": res}
    with open("output/us_factor_longrun_bigcap.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    _log("저장: output/us_factor_longrun_bigcap.json")
    return out


if __name__ == "__main__":
    run()
