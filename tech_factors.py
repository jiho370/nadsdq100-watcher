#!/usr/bin/env python3
"""
tech_factors.py — 이동평균(20/60/200) 크로스오버 기반 기술 신호 팩터.
백테스트(과거 시점)와 라이브(현재)가 '같은 정의'를 쓰도록 한 곳에 둔다.

팩터(모두 '높을수록 좋다' 방향, 가격만으로 계산 = 미래참조 없음):
  gc60_200   : 60일선이 200일선 위(정배열 확립)이면서 '최근' 상향돌파일수록 높음
               = 현재 60>200 이고, 지난 126일 중 대부분이 200선 아래였을수록(=갓 돌파) 큰 값.
  squeeze2060: 20일선과 60일선이 '붙었다가(수렴) 위로 벌어지기 시작'할수록 높음
               = 최근 20일 스프레드가 바짝 붙어 있었고, 지금 20>60 으로 확산 중일 때 큰 값.
  ma_align   : 20>60>200 정배열 정도(0~1).

ma100_gap(2026-09-23 추가, 가중 팩터 아님 — CROSS_FACTORS 목록에 없음): 현재가/100일선-1.
us_momentum_overlay.py·us_ma100_ret3m_weight_sweep.py 백테스트로 검증 — "양음"(100일선
위/아래)은 뚜렷이 유의(DSR 0.9999·PBO 23.5%)하지만 "정도"(이격도 클수록 계속 좋아짐)는
근거가 약해(워크포워드 n=8, 유의하지 않음) 연속가중치가 아니라 export_data.py에서
문턱 필터(위/아래)로만 쓴다.
"""
from __future__ import annotations
import pandas as pd

CROSS_FACTORS = ["gc60_200", "squeeze2060", "ma_align", "resid_mom"]


def build_panels(panel: pd.DataFrame) -> dict:
    """가격 패널(dates×syms) → 팩터별 팩터패널(dates×syms) dict."""
    ma20 = panel.rolling(20, min_periods=20).mean()
    ma60 = panel.rolling(60, min_periods=60).mean()
    ma100 = panel.rolling(100, min_periods=100).mean()
    ma200 = panel.rolling(200, min_periods=200).mean()
    ma100_gap = panel / ma100 - 1.0

    # 잔차 모멘텀(Blitz-Huij-Martens): 시장베타 제거한 12-1 모멘텀 → 모멘텀 붕괴 완화.
    mret = panel.pct_change()
    mkt = mret.mean(axis=1)                       # 동일가중 시장 일별수익률(프록시)
    var = mkt.rolling(252, min_periods=120).var()
    beta = mret.rolling(252, min_periods=120).cov(mkt).div(var, axis=0)
    mom121 = panel.shift(21) / panel.shift(252) - 1
    mkt_px = (1 + mkt.fillna(0)).cumprod()
    mkt_mom = mkt_px.shift(21) / mkt_px.shift(252) - 1
    resid_mom = mom121.sub(beta.mul(mkt_mom, axis=0))

    above60 = (ma60 > ma200)
    # 최근성: 현재 200 위 & 지난 126일 중 아래였던 비율이 클수록(=갓 돌파) 높게.
    gc60_200 = above60.astype(float) * (1.0 - above60.rolling(126, min_periods=20).mean())

    spread = (ma20 - ma60) / ma60                       # 20-60 이격(양수=20이 위)
    tight = spread.abs().rolling(20, min_periods=5).min()   # 최근 20일 최소 이격(작을수록 바짝 붙었었음)
    squeeze2060 = spread.clip(lower=0) / (1.0 + tight * 50.0)  # 붙었다가 위로 벌어질수록 큰 값

    ma_align = (((ma20 > ma60).astype(int) + (ma60 > ma200).astype(int)
                 + (ma20 > ma200).astype(int)) / 3.0)

    return {"gc60_200": gc60_200, "squeeze2060": squeeze2060, "ma_align": ma_align,
            "resid_mom": resid_mom, "ma100_gap": ma100_gap}


def latest_by_sym(hist) -> dict:
    """라이브용: hist(dict[sym]->Series 또는 DataFrame) → {sym: {factor: 현재값}}.
    2026-09-23 버그 수정: df.iloc[-1](패널 전체의 마지막 '날짜' 행)을 그대로 쓰면, 단
    1종목이라도 다른 종목보다 하루 늦은 시세를 갖고 있을 때(실측: 503종목 중 502개는
    9/21 마지막인데 1개만 9/22) 그 날짜가 유니버스 전체 인덱스 끝이 돼버려 나머지
    502종목이 전부 결측 처리됐다(require_above_ma100 필터가 처음으로 이 결측을 실제
    소비하면서 드러남 — gc60_200 등 기존 팩터는 가중치에 안 쓰여 안 드러났었음).
    ffill로 각 종목의 '자기 마지막 실측값'을 유니버스 끝 날짜까지 이어붙여 해결."""
    panel = pd.DataFrame(hist) if isinstance(hist, dict) else hist
    panels = build_panels(panel)
    out = {}
    for f, df in panels.items():
        row = df.ffill().iloc[-1]
        for sym, v in row.items():
            out.setdefault(sym, {})[f] = (None if pd.isna(v) else float(v))
    return out
