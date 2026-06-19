#!/usr/bin/env python3
"""
nasdaq100_low_per_report.py  (v2.0)

매일 나스닥-100 구성종목을 분석해 이메일 리포트를 생성/발송한다.

리포트 구성
  1) PER(주가수익비율) 최저 N개 종목 — 기업 한 줄 설명 + 업종 + 추세 지표 + 차트
     · 변동률: 1일 / 1주 / 1개월 / 1년 / 3년 / 5년
     · 지표: 이동평균(20/50/200), RSI(14), MACD(12/26/9)
     · 종목별 가격 차트(이미지)를 메일에 임베드
  2) 추세 전환(하락→상승) 종목 — 나스닥100 전체를 스캔
  3) 추세 굳힘(상승추세 고착) 종목 — 나스닥100 전체를 스캔

데이터 소스 : Yahoo Finance (yfinance) — API 키 불필요
실행 환경   : GitHub Actions cron (매일 1회, 미국장 마감 후)

규칙
  * PER <= 0 또는 결측치는 "저평가"가 아니라 적자/무수익 신호이므로 PER 랭킹에서 제외한다.
  * 전일 명단과 비교해 신규 편입/이탈 종목을 함께 표시한다(STATE_FILE 사용).
  * 차트/지표는 일봉 종가 기준 자동 계산값이며 투자 권유가 아니다.
"""

from __future__ import annotations

import os
import io
import sys
import json
import base64
import time
from datetime import datetime, timezone, timedelta

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")            # 서버(헤드리스) 렌더링
import matplotlib.pyplot as plt

try:
    import yfinance as yf
except ImportError:              # 런타임 환경에 설치 필요 (requirements.txt 참고)
    yf = None

import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

# ----------------------------- 설정 -----------------------------
TOP_N          = int(os.environ.get("TOP_N", "10"))             # PER 하위 N개
HISTORY_PERIOD = os.environ.get("HISTORY_PERIOD", "5y")          # 차트·장기수익률·MA200용
STATE_FILE     = os.environ.get("STATE_FILE", "state_prev_list.json")
TREND_MAX      = int(os.environ.get("TREND_MAX", "6"))           # 추세 섹션별 최대 표시 종목 수
KST            = timezone(timedelta(hours=9))

RSI_PERIOD     = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
MA_WINDOWS     = (20, 50, 200)

# 거래일 기준 환산(대략): 1주=5, 1개월=21, 1년=252, 3년=756
P_1W, P_1M, P_1Y, P_3Y = 5, 21, 252, 756


# ============================================================
#  나스닥-100 구성종목 + 한 줄 설명 + 업종
#  지수는 분기(3/6/9/12월) 리밸런싱 때만 갱신하면 된다.
# ============================================================
NASDAQ100_INFO: dict[str, dict[str, str]] = {
    "NVDA": {"desc": "AI·데이터센터용 GPU 1위", "sector": "반도체"},
    "AAPL": {"desc": "아이폰·맥·서비스 생태계", "sector": "빅테크·하드웨어"},
    "MSFT": {"desc": "윈도우·오피스·Azure 클라우드", "sector": "빅테크·SW"},
    "AMZN": {"desc": "전자상거래·AWS 클라우드", "sector": "빅테크·유통"},
    "GOOGL": {"desc": "구글 검색·유튜브·클라우드(클래스A)", "sector": "빅테크·플랫폼"},
    "GOOG": {"desc": "구글 검색·유튜브·클라우드(클래스C)", "sector": "빅테크·플랫폼"},
    "AVGO": {"desc": "AI 네트워킹 반도체+인프라SW(브로드컴)", "sector": "반도체"},
    "TSLA": {"desc": "전기차·에너지·자율주행(테슬라)", "sector": "전기차"},
    "META": {"desc": "페이스북·인스타·AI·메타버스", "sector": "빅테크·플랫폼"},
    "MU": {"desc": "D램·낸드 메모리 반도체(마이크론)", "sector": "반도체"},
    "WMT": {"desc": "미국 최대 오프라인·온라인 유통(월마트)", "sector": "유통·리테일"},
    "AMD": {"desc": "CPU·GPU·데이터센터 반도체", "sector": "반도체"},
    "ASML": {"desc": "EUV 노광장비 독점 공급(네덜란드)", "sector": "반도체장비"},
    "INTC": {"desc": "CPU·파운드리 전환 중 종합반도체(인텔)", "sector": "반도체"},
    "AMAT": {"desc": "반도체 전공정 장비 1위(어플라이드)", "sector": "반도체장비"},
    "LRCX": {"desc": "식각·증착 반도체 장비(램리서치)", "sector": "반도체장비"},
    "CSCO": {"desc": "네트워크 장비·보안(시스코)", "sector": "통신장비"},
    "ARM": {"desc": "모바일·AI 칩 설계 IP(영국)", "sector": "반도체IP"},
    "COST": {"desc": "회원제 창고형 할인점(코스트코)", "sector": "유통·리테일"},
    "KLAC": {"desc": "반도체 검사·계측 장비(KLA)", "sector": "반도체장비"},
    "SNDK": {"desc": "낸드 플래시·SSD(샌디스크)", "sector": "반도체·저장장치"},
    "NFLX": {"desc": "글로벌 스트리밍 1위(넷플릭스)", "sector": "미디어·엔터"},
    "PLTR": {"desc": "정부·기업용 빅데이터 분석SW(팔란티어)", "sector": "소프트웨어"},
    "TXN": {"desc": "아날로그·임베디드 반도체(TI)", "sector": "반도체"},
    "MRVL": {"desc": "데이터센터·AI 커스텀 반도체(마벨)", "sector": "반도체"},
    "WDC": {"desc": "대용량 하드디스크(웨스턴디지털)", "sector": "저장장치"},
    "STX": {"desc": "하드디스크 드라이브(씨게이트)", "sector": "저장장치"},
    "QCOM": {"desc": "모바일 AP·통신 모뎀(퀄컴)", "sector": "반도체"},
    "LIN": {"desc": "세계 최대 산업용 가스(린데)", "sector": "소재·화학"},
    "PANW": {"desc": "차세대 사이버보안 플랫폼(팔로알토)", "sector": "사이버보안"},
    "ADI": {"desc": "아날로그·신호처리 반도체(ADI)", "sector": "반도체"},
    "TMUS": {"desc": "미국 이동통신 3사(T모바일)", "sector": "통신"},
    "PEP": {"desc": "음료·스낵(펩시코)", "sector": "식음료"},
    "AMGN": {"desc": "바이오 신약(암젠)", "sector": "바이오·제약"},
    "CRWD": {"desc": "클라우드 엔드포인트 보안(크라우드스트라이크)", "sector": "사이버보안"},
    "APP": {"desc": "모바일 앱 광고·수익화 플랫폼(앱러빈)", "sector": "SW·광고"},
    "GILD": {"desc": "항바이러스·항암 신약(길리어드)", "sector": "바이오·제약"},
    "HON": {"desc": "항공·자동화 복합 산업재(하니웰)", "sector": "산업재"},
    "ISRG": {"desc": "수술용 로봇 다빈치(인튜이티브서지컬)", "sector": "의료기기"},
    "SHOP": {"desc": "이커머스 구축 플랫폼(쇼피파이)", "sector": "SW·이커머스"},
    "BKNG": {"desc": "글로벌 온라인 여행 예약(부킹닷컴)", "sector": "여행·예약"},
    "VRTX": {"desc": "희귀질환 신약(버텍스)", "sector": "바이오·제약"},
    "SBUX": {"desc": "글로벌 커피 체인(스타벅스)", "sector": "식음료"},
    "PDD": {"desc": "중국 저가 이커머스·테무(핀둬둬)", "sector": "이커머스"},
    "CDNS": {"desc": "반도체 설계 EDA SW(케이던스)", "sector": "SW·EDA"},
    "FTNT": {"desc": "네트워크 방화벽 보안(포티넷)", "sector": "사이버보안"},
    "MAR": {"desc": "글로벌 호텔 체인(메리어트)", "sector": "여행·숙박"},
    "CEG": {"desc": "미국 최대 원자력 발전(컨스텔레이션)", "sector": "유틸리티"},
    "MNST": {"desc": "에너지 음료(몬스터)", "sector": "식음료"},
    "SNPS": {"desc": "반도체 설계 EDA·IP(시놉시스)", "sector": "SW·EDA"},
    "ADP": {"desc": "급여·인사 아웃소싱(ADP)", "sector": "SW·서비스"},
    "CSX": {"desc": "미 동부 화물 철도", "sector": "물류·운송"},
    "ABNB": {"desc": "숙박 공유 플랫폼(에어비앤비)", "sector": "여행·숙박"},
    "MELI": {"desc": "중남미 이커머스·핀테크(메르카도리브레)", "sector": "이커머스·핀테크"},
    "CMCSA": {"desc": "케이블·미디어(컴캐스트)", "sector": "미디어·통신"},
    "NXPI": {"desc": "차량용·산업용 반도체(NXP)", "sector": "반도체"},
    "DDOG": {"desc": "클라우드 모니터링·관측(데이터독)", "sector": "소프트웨어"},
    "MDLZ": {"desc": "과자·초콜릿(몬델리즈)", "sector": "식음료"},
    "ADBE": {"desc": "크리에이티브·문서 SW(어도비)", "sector": "소프트웨어"},
    "MPWR": {"desc": "전력관리 반도체(모놀리식파워)", "sector": "반도체"},
    "DASH": {"desc": "음식 배달 플랫폼(도어대시)", "sector": "플랫폼·배달"},
    "ROST": {"desc": "오프프라이스 의류 할인(로스)", "sector": "유통·리테일"},
    "INTU": {"desc": "세무·회계 SW(인튜이트)", "sector": "소프트웨어"},
    "ORLY": {"desc": "자동차 부품 유통(오라일리)", "sector": "유통·리테일"},
    "AEP": {"desc": "미 중서부 전력 유틸리티(AEP)", "sector": "유틸리티"},
    "CTAS": {"desc": "유니폼 렌탈·기업서비스(신타스)", "sector": "산업재·서비스"},
    "LITE": {"desc": "광통신·레이저 부품(루멘텀)", "sector": "광통신부품"},
    "WBD": {"desc": "영화·방송 미디어(워너브러더스디스커버리)", "sector": "미디어·엔터"},
    "REGN": {"desc": "항체 신약(리제네론)", "sector": "바이오·제약"},
    "PCAR": {"desc": "대형 트럭 제조(파카)", "sector": "산업재·자동차"},
    "BKR": {"desc": "유전 서비스·장비(베이커휴즈)", "sector": "에너지"},
    "MCHP": {"desc": "마이크로컨트롤러 반도체(마이크로칩)", "sector": "반도체"},
    "FAST": {"desc": "산업용 부품·체결재 유통(패스널)", "sector": "산업재·유통"},
    "FANG": {"desc": "셰일 원유·가스 생산(다이아몬드백)", "sector": "에너지"},
    "EA": {"desc": "비디오게임(일렉트로닉아츠)", "sector": "미디어·게임"},
    "FER": {"desc": "글로벌 인프라·건설 운영(페로비알)", "sector": "산업재·인프라"},
    "XEL": {"desc": "중서부 전력·가스 유틸리티(엑셀에너지)", "sector": "유틸리티"},
    "EXC": {"desc": "미국 최대 전력 배전(엑셀론)", "sector": "유틸리티"},
    "ODFL": {"desc": "LTL 화물 운송(올드도미니언)", "sector": "물류·운송"},
    "TTWO": {"desc": "GTA 등 게임(테이크투)", "sector": "미디어·게임"},
    "IDXX": {"desc": "동물병원 진단(아이덱스)", "sector": "의료기기·진단"},
    "CCEP": {"desc": "유럽 코카콜라 보틀러", "sector": "식음료"},
    "KDP": {"desc": "음료·커피(큐리그닥터페퍼)", "sector": "식음료"},
    "ADSK": {"desc": "3D 설계 CAD SW(오토데스크)", "sector": "소프트웨어"},
    "MSTR": {"desc": "비트코인 보유 SW기업(스트래티지)", "sector": "SW·암호화폐"},
    "PYPL": {"desc": "온라인 결제(페이팔)", "sector": "핀테크·결제"},
    "ALNY": {"desc": "RNA간섭 신약(알닐람)", "sector": "바이오·제약"},
    "PAYX": {"desc": "중소기업 급여·HR(페이첵스)", "sector": "SW·서비스"},
    "TRI": {"desc": "법률·금융 정보 서비스(톰슨로이터)", "sector": "정보서비스"},
    "AXON": {"desc": "테이저·바디캠 공공안전(액손)", "sector": "산업재·공공안전"},
    "ROP": {"desc": "다각화 소프트웨어·산업재(로퍼)", "sector": "SW·산업재"},
    "WDAY": {"desc": "클라우드 인사·재무 SW(워크데이)", "sector": "소프트웨어"},
    "DXCM": {"desc": "연속혈당측정기(덱스콤)", "sector": "의료기기"},
    "CPRT": {"desc": "온라인 중고차 경매(코파트)", "sector": "서비스·유통"},
    "GEHC": {"desc": "의료영상 장비(GE헬스케어)", "sector": "의료기기"},
    "KHC": {"desc": "가공식품(크래프트하인즈)", "sector": "식음료"},
    "VRSK": {"desc": "보험 데이터 분석(베리스크)", "sector": "정보서비스"},
    "INSM": {"desc": "희귀 폐질환 신약(인스메드)", "sector": "바이오·제약"},
    "CTSH": {"desc": "IT 컨설팅·아웃소싱(코그니전트)", "sector": "IT서비스"},
    "ZS": {"desc": "클라우드 제로트러스트 보안(지스케일러)", "sector": "사이버보안"},
    "CHTR": {"desc": "케이블 인터넷·방송(차터)", "sector": "미디어·통신"},
}

NASDAQ100_SYMBOLS = list(NASDAQ100_INFO.keys())


def get_nasdaq100_symbols() -> list[str]:
    """나스닥-100 구성종목 티커 목록 (정적)."""
    return list(NASDAQ100_SYMBOLS)


def info_of(sym: str) -> dict[str, str]:
    return NASDAQ100_INFO.get(sym, {"desc": "", "sector": "—"})


# ------------------------- yfinance 유틸 ------------------------
def _require_yf():
    if yf is None:
        raise RuntimeError("yfinance가 설치되어 있지 않습니다. `pip install yfinance` 후 실행하세요.")


def get_quotes(symbols: list[str]) -> dict[str, dict]:
    """각 종목의 trailingPE/가격/이름. 종목당 1콜(.info), 실패는 건너뜀.
    PER이 없으면 price/EPS로 보정한다. 100종목 .info 조회는 1~3분 걸릴 수 있다.
    """
    _require_yf()
    out: dict[str, dict] = {}
    fail = 0
    for sym in symbols:
        info = None
        for attempt in range(2):
            try:
                info = yf.Ticker(sym).info or {}
                break
            except Exception:  # noqa: BLE001
                if attempt == 0:
                    time.sleep(1.0)
        if info is None:
            fail += 1
            print(f"[경고] {sym} info 조회 실패 → 제외", file=sys.stderr)
            continue

        pe = info.get("trailingPE")
        price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")
        if pe is None:
            eps = info.get("trailingEps")
            try:
                if eps and price:
                    pe = float(price) / float(eps)
            except (TypeError, ValueError, ZeroDivisionError):
                pe = None
        out[sym] = {
            "pe": pe,
            "price": price,
            "name": info.get("shortName") or info.get("longName") or sym,
        }
    if fail:
        print(f"[정보] info 조회 실패 {fail}건 (나머지로 진행)", file=sys.stderr)
    return out


def download_histories(symbols: list[str], period: str = HISTORY_PERIOD) -> dict[str, pd.Series]:
    """모든 종목의 일봉 종가를 배치로 한 번에 받아 {sym: close(Series)}로 반환.
    배치 실패 시 종목별 개별 조회로 폴백한다.
    """
    _require_yf()
    out: dict[str, pd.Series] = {}
    try:
        data = yf.download(
            symbols, period=period, interval="1d", auto_adjust=True,
            group_by="ticker", threads=True, progress=False,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[경고] 배치 다운로드 실패({e}) → 개별 조회로 폴백", file=sys.stderr)
        data = None

    if data is not None and not data.empty:
        for sym in symbols:
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    if sym not in data.columns.get_level_values(0):
                        continue
                    close = data[sym]["Close"]
                else:  # 단일 종목인 경우
                    close = data["Close"]
                close = _clean_close(close)
                if not close.empty:
                    out[sym] = close
            except Exception:  # noqa: BLE001
                continue

    # 누락분 개별 보충
    missing = [s for s in symbols if s not in out]
    for sym in missing:
        try:
            raw = yf.Ticker(sym).history(period=period, interval="1d", auto_adjust=True)
            if raw is not None and not raw.empty and "Close" in raw.columns:
                close = _clean_close(raw["Close"])
                if not close.empty:
                    out[sym] = close
        except Exception:  # noqa: BLE001
            continue
    return out


def _clean_close(close: pd.Series) -> pd.Series:
    s = pd.to_numeric(close, errors="coerce")
    idx = pd.to_datetime(s.index)
    try:
        idx = idx.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    s.index = idx
    return s.dropna().sort_index()


# ------------------------- 지표 계산 ----------------------------
def _rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _macd(close: pd.Series):
    ema_fast = close.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=MACD_SLOW, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=MACD_SIGNAL, adjust=False).mean()
    return macd, signal, macd - signal


def _ret(close: pd.Series, periods: int) -> float:
    """periods 거래일 전 종가 대비 변동률(%)."""
    if len(close) > periods:
        prev = close.iloc[-1 - periods]
        if prev and not pd.isna(prev):
            return (float(close.iloc[-1]) / float(prev) - 1.0) * 100.0
    return float("nan")


def _ret_full(close: pd.Series) -> float:
    """확보된 가장 오래된 종가 대비 변동률(%) — 5년 보유분이 ~5년치면 5Y 수익률."""
    c = close.dropna()
    if len(c) >= 2 and c.iloc[0]:
        return (float(c.iloc[-1]) / float(c.iloc[0]) - 1.0) * 100.0
    return float("nan")


def compute_indicators(close: pd.Series) -> dict | None:
    """최근 시점 추세 지표 + 다기간 수익률 + 추세 분류."""
    if close is None or close.empty:
        return None
    last = float(close.iloc[-1])

    ma = {w: (close.rolling(w).mean() if len(close) >= w else pd.Series(dtype=float)) for w in MA_WINDOWS}
    ma_last = {w: (float(ma[w].iloc[-1]) if len(ma[w]) and not np.isnan(ma[w].iloc[-1]) else np.nan) for w in MA_WINDOWS}

    rsi_series = _rsi(close)
    rsi_val = float(rsi_series.iloc[-1]) if not np.isnan(rsi_series.iloc[-1]) else np.nan
    rsi_prev = float(rsi_series.iloc[-6]) if len(rsi_series) > 6 and not np.isnan(rsi_series.iloc[-6]) else np.nan

    macd, signal, hist = _macd(close)
    macd_val, sig_val, hist_val = float(macd.iloc[-1]), float(signal.iloc[-1]), float(hist.iloc[-1])

    # 50/200 골든·데드크로스 (최근 5거래일 내)
    cross = None
    if len(close) >= 205:
        ma50, ma200 = close.rolling(50).mean(), close.rolling(200).mean()
        diff = (ma50 - ma200).dropna()
        if len(diff) >= 6:
            recent = np.sign(diff.iloc[-6:])
            if recent.iloc[0] < 0 and recent.iloc[-1] > 0:
                cross = "golden"
            elif recent.iloc[0] > 0 and recent.iloc[-1] < 0:
                cross = "death"

    ind = {
        "price": last,
        "ma20": ma_last[20], "ma50": ma_last[50], "ma200": ma_last[200],
        "above_ma200": (not np.isnan(ma_last[200])) and last > ma_last[200],
        "rsi": rsi_val, "rsi_prev": rsi_prev,
        "macd": macd_val, "macd_signal": sig_val, "macd_hist": hist_val,
        "macd_up": hist_val > 0,
        "cross": cross,
        "chg_1d": _ret(close, 1),
        "chg_1w": _ret(close, P_1W),
        "chg_1m": _ret(close, P_1M),
        "chg_1y": _ret(close, P_1Y),
        "chg_3y": _ret(close, P_3Y),
        "chg_5y": _ret_full(close),
    }
    ind.update(_classify_trend(close, ma, hist, rsi_series))
    return ind


def _classify_trend(close, ma, hist, rsi_series) -> dict:
    """추세 전환(하락→상승) / 추세 굳힘(상승 고착) 분류 + 점수 + 한글 사유."""
    res = {
        "reversal": False, "reversal_score": 0.0, "reversal_reason": "",
        "solidified": False, "solidified_score": 0.0, "solidified_reason": "",
    }
    if len(close) < 60:
        return res

    last = float(close.iloc[-1])
    ma20s, ma50s, ma200s = ma[20], ma[50], ma[200]
    ma20 = float(ma20s.iloc[-1]) if len(ma20s) and not np.isnan(ma20s.iloc[-1]) else np.nan
    ma50 = float(ma50s.iloc[-1]) if len(ma50s) and not np.isnan(ma50s.iloc[-1]) else np.nan
    ma200 = float(ma200s.iloc[-1]) if len(ma200s) and not np.isnan(ma200s.iloc[-1]) else np.nan

    h = hist.dropna()
    rsi = rsi_series.dropna()
    if len(h) < 6 or len(rsi) < 6:
        return res
    rsi_now = float(rsi.iloc[-1])

    # ---- (1) 추세 전환: 최근 며칠 사이 하락→상승 신호 ----
    rev_reasons, rev_score = [], 0.0
    # MACD 히스토그램이 최근 4일 내 음→양 전환
    macd_flip = (h.iloc[-1] > 0) and (h.iloc[-5:-1] <= 0).any()
    if macd_flip:
        rev_reasons.append("MACD 히스토그램이 음(-)에서 양(+)으로 전환(상승 모멘텀 발생)")
        rev_score += 2.0
    # 종가가 최근 5일 내 20일선을 아래→위로 돌파
    if not np.isnan(ma20):
        above_now = last > ma20
        below_before = (close.iloc[-6:-1].values < ma20s.iloc[-6:-1].values).any()
        if above_now and below_before:
            rev_reasons.append("종가가 20일 이동평균선을 아래에서 위로 돌파")
            rev_score += 2.0
    # RSI가 50선을 상향 돌파(약세→중립/강세)
    if len(rsi) > 6:
        if rsi_now >= 50 and float(rsi.iloc[-6]) < 50:
            rev_reasons.append("RSI가 50선을 상향 돌파(매수 우위로 전환)")
            rev_score += 1.0
    # 직전이 실제 하락 추세였는지(20일선 < 50일선 또는 최근 1개월 음수)였을 때만 '전환'으로 인정
    prior_down = False
    if not np.isnan(ma20) and not np.isnan(ma50):
        prior_down = float(ma20s.iloc[-6]) < float(ma50s.iloc[-6]) if len(ma20s) > 6 and len(ma50s) > 6 else False
    mom_1m = _ret(close, P_1M)
    if not prior_down and not np.isnan(mom_1m) and mom_1m < 0:
        prior_down = True
    if rev_score >= 3.0 and prior_down:
        res["reversal"] = True
        res["reversal_score"] = rev_score
        res["reversal_reason"] = " · ".join(rev_reasons)

    # ---- (2) 추세 굳힘: 상승 정배열이 며칠째 유지 ----
    sol_reasons, sol_score = [], 0.0
    aligned = (not np.isnan(ma20) and not np.isnan(ma50) and not np.isnan(ma200)
               and last > ma20 > ma50 > ma200)
    if aligned:
        # 정배열(종가>20>50>200)이 며칠 연속 유지됐는지
        days = 0
        n = min(len(close), len(ma20s), len(ma50s), len(ma200s))
        for i in range(1, n + 1):
            c = float(close.iloc[-i])
            m20, m50, m200 = ma20s.iloc[-i], ma50s.iloc[-i], ma200s.iloc[-i]
            if np.isnan(m20) or np.isnan(m50) or np.isnan(m200):
                break
            if c > m20 > m50 > m200:
                days += 1
            else:
                break
        if days >= 3:
            sol_reasons.append(f"종가>20>50>200일선 정배열 {days}일째 유지")
            sol_score += 2.0 + min(days, 20) / 10.0
        if h.iloc[-1] > 0 and (h.iloc[-5:] > 0).all():
            sol_reasons.append("MACD가 5일 연속 시그널선 위(상승 모멘텀 지속)")
            sol_score += 1.0
        if 50 <= rsi_now <= 78:
            sol_reasons.append(f"RSI {rsi_now:.0f}로 건강한 강세 구간")
            sol_score += 1.0
        # 200일선 위 이격(과도하지 않은 상승)
        if not np.isnan(ma200) and ma200 > 0:
            gap = (last / ma200 - 1) * 100
            sol_reasons.append(f"200일선 대비 +{gap:.0f}%")
        if sol_score >= 3.0 and sol_reasons:
            res["solidified"] = True
            res["solidified_score"] = sol_score
            res["solidified_reason"] = " · ".join(sol_reasons)
    return res


# --------------------- 차트 이미지 생성 -------------------------
def make_chart_png(close: pd.Series, days: int | None, ma_windows=(50, 200),
                   figsize=(6.0, 1.85), up: bool = True, title: str | None = None) -> bytes:
    """가격 + 이동평균 차트 PNG. (라벨은 폰트 호환 위해 영문 사용)"""
    s = close.dropna()
    if days:
        s = s.iloc[-days:]
    fig, ax = plt.subplots(figsize=figsize)
    line_color = "#15803d" if up else "#b91c1c"
    ax.plot(s.index, s.values, color=line_color, lw=1.4, label="Close")
    ma_colors = {20: "#f59e0b", 50: "#2563eb", 200: "#9333ea"}
    for w in ma_windows:
        if len(close) >= w:
            m = close.rolling(w).mean().reindex(s.index)
            ax.plot(s.index, m.values, color=ma_colors.get(w, "#888"), lw=1.0, label=f"MA{w}")
    ax.fill_between(s.index, s.values, s.min(), color=line_color, alpha=0.06)
    ax.margins(x=0)
    ax.grid(True, alpha=0.15)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(labelsize=7, length=0)
    ax.yaxis.tick_right()
    if title:
        ax.set_title(title, fontsize=9, loc="left", color="#374151")
    ax.legend(fontsize=6.5, loc="upper left", frameon=False, ncol=4)
    fig.tight_layout(pad=0.4)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return buf.getvalue()


# --------------------- 명단 편입/이탈 비교 -----------------------
def load_prev_list() -> list[str]:
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f).get("symbols", [])
    except Exception:  # noqa: BLE001
        return []


def save_curr_list(symbols: list[str]) -> None:
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump({"symbols": symbols, "saved_at": datetime.now(KST).isoformat()}, f, ensure_ascii=False)
    except Exception as e:  # noqa: BLE001
        print(f"[경고] 상태 파일 저장 실패: {e}", file=sys.stderr)


# ------------------------- 포맷 헬퍼 ----------------------------
def _isnan(x) -> bool:
    return x is None or (isinstance(x, float) and np.isnan(x))


def _pct(x) -> str:
    return "—" if _isnan(x) else f"{x:+.2f}%"


def _money(x) -> str:
    return "—" if _isnan(x) else f"${x:,.2f}"


def _rsi_zone(r) -> str:
    if _isnan(r):
        return ""
    if r >= 70:
        return "과매수"
    if r <= 30:
        return "과매도"
    return ""


def _color_chg(c: float) -> str:
    return "#9ca3af" if _isnan(c) else ("#15803d" if c >= 0 else "#b91c1c")


# ------------------------- 본문 생성 ----------------------------
def _ret_chip(label: str, val: float) -> str:
    color = _color_chg(val)
    txt = "—" if _isnan(val) else f"{val:+.1f}%"
    return (f'<span style="display:inline-block;margin:2px 6px 2px 0;font-size:12px">'
            f'<span style="color:#9ca3af">{label}</span> '
            f'<b style="color:{color}">{txt}</b></span>')


def _badges(ind: dict) -> str:
    out = []
    if ind.get("above_ma200") is not None:
        if ind["above_ma200"]:
            out.append('<span style="background:#dcfce7;color:#15803d;border-radius:4px;padding:1px 6px;font-size:11px">200일선 위</span>')
        else:
            out.append('<span style="background:#fee2e2;color:#b91c1c;border-radius:4px;padding:1px 6px;font-size:11px">200일선 아래</span>')
    rsi = ind.get("rsi")
    if not _isnan(rsi):
        zone = _rsi_zone(rsi)
        z = f" {zone}" if zone else ""
        out.append(f'<span style="background:#f3f4f6;color:#374151;border-radius:4px;padding:1px 6px;font-size:11px">RSI {rsi:.0f}{z}</span>')
    if ind.get("macd_up") is not None:
        if ind["macd_up"]:
            out.append('<span style="background:#dcfce7;color:#15803d;border-radius:4px;padding:1px 6px;font-size:11px">MACD ＋</span>')
        else:
            out.append('<span style="background:#fee2e2;color:#b91c1c;border-radius:4px;padding:1px 6px;font-size:11px">MACD －</span>')
    if ind.get("cross") == "golden":
        out.append('<span style="background:#fef9c3;color:#a16207;border-radius:4px;padding:1px 6px;font-size:11px">골든크로스</span>')
    elif ind.get("cross") == "death":
        out.append('<span style="background:#fee2e2;color:#b91c1c;border-radius:4px;padding:1px 6px;font-size:11px">데드크로스</span>')
    return " ".join(out)


def _stock_card(rank, sym, name, per, ind, cid, sector, desc) -> str:
    per_txt = "—" if _isnan(per) else f"{float(per):.1f}"
    rank_html = (f'<span style="background:#111827;color:#fff;border-radius:50%;width:20px;height:20px;'
                 f'display:inline-block;text-align:center;line-height:20px;font-size:12px">{rank}</span> '
                 if rank else "")
    rets = (_ret_chip("1일", ind.get("chg_1d")) + _ret_chip("1주", ind.get("chg_1w"))
            + _ret_chip("1달", ind.get("chg_1m")) + _ret_chip("1년", ind.get("chg_1y"))
            + _ret_chip("3년", ind.get("chg_3y")) + _ret_chip("5년", ind.get("chg_5y")))
    return f"""
    <table style="width:100%;border-collapse:collapse;margin:10px 0;border:1px solid #eef0f3;border-radius:8px">
      <tr>
        <td style="padding:12px 14px;vertical-align:top;width:52%">
          <div style="font-size:15px">{rank_html}<b>{sym}</b>
            <span style="background:#eef2ff;color:#4338ca;border-radius:4px;padding:1px 6px;font-size:11px;margin-left:4px">{sector}</span>
          </div>
          <div style="font-size:12px;color:#6b7280;margin:2px 0 1px">{name}</div>
          <div style="font-size:12px;color:#374151;margin-bottom:6px">{desc}</div>
          <div style="font-size:13px;margin-bottom:6px">PER <b>{per_txt}</b>
            <span style="color:#9ca3af">·</span> 종가 <b>{_money(ind.get('price'))}</b></div>
          <div style="margin-bottom:6px">{rets}</div>
          <div>{_badges(ind)}</div>
        </td>
        <td style="padding:8px;vertical-align:middle;width:48%">
          <img src="cid:{cid}" alt="{sym} chart" style="width:100%;max-width:340px;height:auto;display:block"/>
        </td>
      </tr>
    </table>"""


def _trend_card(sym, name, ind, cid, sector, desc, reason) -> str:
    rets = (_ret_chip("1주", ind.get("chg_1w")) + _ret_chip("1달", ind.get("chg_1m"))
            + _ret_chip("3달", _ret_safe(ind, "chg_3m")) + _ret_chip("1년", ind.get("chg_1y")))
    return f"""
    <table style="width:100%;border-collapse:collapse;margin:10px 0;border:1px solid #eef0f3;border-radius:8px">
      <tr>
        <td style="padding:10px;vertical-align:middle;width:46%">
          <img src="cid:{cid}" alt="{sym} chart" style="width:100%;max-width:320px;height:auto;display:block"/>
        </td>
        <td style="padding:12px 14px;vertical-align:top;width:54%">
          <div style="font-size:15px"><b>{sym}</b>
            <span style="background:#eef2ff;color:#4338ca;border-radius:4px;padding:1px 6px;font-size:11px;margin-left:4px">{sector}</span></div>
          <div style="font-size:12px;color:#6b7280;margin:2px 0">{name} · {desc}</div>
          <div style="margin:6px 0">{rets}</div>
          <div style="font-size:12px;color:#374151;background:#f9fafb;border-radius:6px;padding:8px;line-height:1.6">
            <b>분석</b> · {reason}</div>
        </td>
      </tr>
    </table>"""


def _ret_safe(ind, key):
    return ind.get(key, float("nan"))


def build_report(picks_rows, reversal_rows, solid_rows, new_in, dropped, asof, images):
    """picks_rows / reversal_rows / solid_rows: [(rank, sym, name, per, ind)] 형태.
    images: cid->png 누적 딕셔너리(여기서 채움). HTML/텍스트/제목 반환."""
    chgs = [r[4].get("chg_1d") for r in picks_rows if not _isnan(r[4].get("chg_1d"))]
    avg_chg = float(np.mean(chgs)) if chgs else float("nan")
    above_cnt = sum(1 for r in picks_rows if r[4].get("above_ma200"))

    subject = (f"[나스닥100] {asof} · PER최저{len(picks_rows)} 일평균 {_pct(avg_chg)} "
               f"· 추세전환 {len(reversal_rows)} · 굳힘 {len(solid_rows)}")

    # ----- PER 최저 카드 -----
    per_cards = []
    for rank, sym, name, per, ind in picks_rows:
        cid = f"per_{sym}"
        meta = info_of(sym)
        per_cards.append(_stock_card(rank, sym, name, per, ind, cid, meta["sector"], meta["desc"]))

    # ----- 추세 전환 카드 -----
    rev_cards = []
    for _, sym, name, per, ind in reversal_rows:
        cid = f"rev_{sym}"
        meta = info_of(sym)
        rev_cards.append(_trend_card(sym, name, ind, cid, meta["sector"], meta["desc"], ind.get("reversal_reason", "")))

    # ----- 추세 굳힘 카드 -----
    sol_cards = []
    for _, sym, name, per, ind in solid_rows:
        cid = f"sol_{sym}"
        meta = info_of(sym)
        sol_cards.append(_trend_card(sym, name, ind, cid, meta["sector"], meta["desc"], ind.get("solidified_reason", "")))

    def chips(items, color):
        if not items:
            return '<span style="color:#9ca3af">없음</span>'
        return " ".join(
            f'<span style="background:{color};color:#fff;border-radius:4px;padding:1px 6px;font-size:12px;margin-right:4px">{s}</span>'
            for s in items)

    rev_section = ""
    if rev_cards:
        rev_section = f"""
      <h3 style="margin:26px 0 4px">🔄 추세 전환 — 하락에서 상승으로 (나스닥100 전체)</h3>
      <div style="color:#6b7280;font-size:12px;margin-bottom:6px">최근 며칠 사이 하락세를 멈추고 상승 신호(MACD 전환·20일선 돌파·RSI 50 회복)가 나타난 종목입니다.</div>
      {''.join(rev_cards)}"""
    else:
        rev_section = ('<h3 style="margin:26px 0 4px">🔄 추세 전환 — 하락에서 상승으로</h3>'
                       '<div style="color:#9ca3af;font-size:13px">오늘은 조건을 충족한 종목이 없습니다.</div>')

    sol_section = ""
    if sol_cards:
        sol_section = f"""
      <h3 style="margin:26px 0 4px">📈 추세 굳힘 — 상승 흐름 고착 (나스닥100 전체)</h3>
      <div style="color:#6b7280;font-size:12px;margin-bottom:6px">종가가 20·50·200일선 위에 정배열로 며칠째 유지되며 상승 추세가 굳어진 종목입니다.</div>
      {''.join(sol_cards)}"""
    else:
        sol_section = ('<h3 style="margin:26px 0 4px">📈 추세 굳힘 — 상승 흐름 고착</h3>'
                       '<div style="color:#9ca3af;font-size:13px">오늘은 조건을 충족한 종목이 없습니다.</div>')

    html = f"""\
<div style="font-family:-apple-system,'Segoe UI',Roboto,'Apple SD Gothic Neo',sans-serif;max-width:720px;margin:0 auto;color:#111827">
  <h2 style="margin:0 0 4px">📊 나스닥100 데일리 리포트</h2>
  <div style="color:#6b7280;font-size:13px;margin-bottom:8px">{asof} 마감 기준 · 변동률은 1일/1주/1달/1년/3년/5년 · 차트는 가격+이동평균(영문 라벨)</div>

  <h3 style="margin:18px 0 2px">💰 PER 최저 {len(picks_rows)} 종목</h3>
  <div style="color:#6b7280;font-size:12px;margin-bottom:4px">적자·무수익(PER≤0)은 제외 · 저PER = 이익 대비 주가가 낮음(저평가 후보이나 업황·일회성 요인 확인 필요)</div>
  {''.join(per_cards)}

  <div style="margin:12px 0;font-size:13px;line-height:1.7">
    <div>PER최저군 평균 일간 등락 <b>{_pct(avg_chg)}</b> · 200일선 위 <b>{above_cnt}/{len(picks_rows)}</b> 종목</div>
    <div>신규 편입: {chips(new_in, "#2563eb")}</div>
    <div>명단 이탈: {chips(dropped, "#6b7280")}</div>
  </div>

  {rev_section}
  {sol_section}

  <div style="margin-top:22px;font-size:11px;color:#9ca3af;border-top:1px solid #eee;padding-top:8px;line-height:1.6">
    데이터: Yahoo Finance (yfinance) · 모든 지표·차트는 일봉 종가 기준 자동 계산값이며 투자 권유가 아닙니다.<br>
    추세 분류는 이동평균·MACD·RSI 규칙에 따른 기계적 판정으로, 실제 매매 판단의 근거로 삼기 전 반드시 추가 확인이 필요합니다.
  </div>
</div>"""

    # ---------- 텍스트(폴백) ----------
    lines = [f"나스닥100 데일리 리포트 — {asof} 마감", "변동률: [1일/1주/1달/1년/3년/5년]", "", f"■ PER 최저 {len(picks_rows)}"]
    for rank, sym, name, per, ind in picks_rows:
        per_t = "—" if _isnan(per) else f"{float(per):.1f}"
        chg = f"{_pct(ind.get('chg_1d'))}/{_pct(ind.get('chg_1w'))}/{_pct(ind.get('chg_1m'))}/{_pct(ind.get('chg_1y'))}/{_pct(ind.get('chg_3y'))}/{_pct(ind.get('chg_5y'))}"
        meta = info_of(sym)
        lines.append(f"{rank:>2}. {sym:<6} PER {per_t:<5} {meta['sector']} | {meta['desc']}")
        lines.append(f"      {_money(ind.get('price'))}  [{chg}]")
    lines += ["", "■ 추세 전환(하락→상승)"]
    if reversal_rows:
        for _, sym, name, per, ind in reversal_rows:
            lines.append(f"  · {sym}: {ind.get('reversal_reason','')}")
    else:
        lines.append("  (해당 없음)")
    lines += ["", "■ 추세 굳힘(상승 고착)"]
    if solid_rows:
        for _, sym, name, per, ind in solid_rows:
            lines.append(f"  · {sym}: {ind.get('solidified_reason','')}")
    else:
        lines.append("  (해당 없음)")
    lines += ["", f"신규 편입: {', '.join(new_in) or '없음'} / 이탈: {', '.join(dropped) or '없음'}"]
    text = "\n".join(lines)

    return subject, html, text


# ------------------------- 오케스트레이션 ------------------------
def generate_report():
    """전체 파이프라인. (subject, html, text, images) 반환.
    images: {cid: png_bytes} — 이메일 임베드용.
    """
    asof = datetime.now(KST).strftime("%Y-%m-%d")
    universe = get_nasdaq100_symbols()

    quotes = get_quotes(universe)
    hist = download_histories(universe)

    # 전 종목 지표 계산
    ind_map: dict[str, dict] = {}
    for sym in universe:
        close = hist.get(sym)
        ind = compute_indicators(close) if close is not None else None
        if ind:
            # 3개월 수익률(추세 카드용)
            ind["chg_3m"] = _ret(close, 63)
            ind_map[sym] = ind

    # ---- PER 최저 N ----
    candidates = []
    for sym in universe:
        pe = quotes.get(sym, {}).get("pe")
        try:
            pe = float(pe)
        except (TypeError, ValueError):
            continue
        if pe > 0 and sym in ind_map:
            candidates.append((sym, pe))
    candidates.sort(key=lambda x: x[1])
    picked = [s for s, _ in candidates[:TOP_N]]

    prev = load_prev_list()
    new_in = [s for s in picked if s not in prev]
    dropped = [s for s in prev if s not in picked]

    picks_rows = []
    for rank, sym in enumerate(picked, start=1):
        q = quotes.get(sym, {})
        picks_rows.append((rank, sym, q.get("name", sym), q.get("pe"), ind_map[sym]))

    # ---- 추세 전환 / 굳힘 (나스닥100 전체) ----
    rev = [(0, sym, quotes.get(sym, {}).get("name", sym), quotes.get(sym, {}).get("pe"), ind)
           for sym, ind in ind_map.items() if ind.get("reversal")]
    rev.sort(key=lambda r: r[4].get("reversal_score", 0), reverse=True)
    reversal_rows = rev[:TREND_MAX]

    sol = [(0, sym, quotes.get(sym, {}).get("name", sym), quotes.get(sym, {}).get("pe"), ind)
           for sym, ind in ind_map.items() if ind.get("solidified")]
    sol.sort(key=lambda r: r[4].get("solidified_score", 0), reverse=True)
    solid_rows = sol[:TREND_MAX]

    # ---- 차트 생성 ----
    images: dict[str, bytes] = {}
    for _, sym, *_ in picks_rows:
        close = hist.get(sym)
        if close is not None:
            up = ind_map[sym].get("chg_1y", 0) >= 0 if not _isnan(ind_map[sym].get("chg_1y")) else True
            images[f"per_{sym}"] = make_chart_png(close, days=None, ma_windows=(50, 200), up=up, title=f"{sym} · 5Y")
    for _, sym, *_ in reversal_rows:
        close = hist.get(sym)
        if close is not None:
            images[f"rev_{sym}"] = make_chart_png(close, days=130, ma_windows=(20, 50), up=True, title=f"{sym} · 6M")
    for _, sym, *_ in solid_rows:
        close = hist.get(sym)
        if close is not None:
            images[f"sol_{sym}"] = make_chart_png(close, days=260, ma_windows=(20, 50, 200), up=True, title=f"{sym} · 1Y")

    subject, html, text = build_report(picks_rows, reversal_rows, solid_rows, new_in, dropped, asof, images)
    save_curr_list(picked)
    return subject, html, text, images


def html_with_inline_images(html: str, images: dict[str, bytes]) -> str:
    """cid: 참조를 data:base64로 치환해 브라우저에서 단독으로 열리는 미리보기 HTML 생성."""
    out = html
    for cid, png in images.items():
        b64 = base64.b64encode(png).decode("ascii")
        out = out.replace(f"cid:{cid}", f"data:image/png;base64,{b64}")
    return out


# ------------------------- 이메일 발송 ---------------------------
def send_email(subject: str, html: str, text: str, images: dict[str, bytes]) -> None:
    host   = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    port   = int(os.environ.get("SMTP_PORT", "465"))
    user   = os.environ.get("SMTP_USER", "").strip()
    pw     = os.environ.get("SMTP_PASS", "").strip()
    to     = os.environ.get("EMAIL_TO", user).strip()
    sender = os.environ.get("EMAIL_FROM", user).strip()
    if not (user and pw and to):
        raise RuntimeError("SMTP_USER / SMTP_PASS / EMAIL_TO 환경변수가 필요합니다.")

    root = MIMEMultipart("related")
    root["Subject"] = subject
    root["From"] = sender
    root["To"] = to

    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText(text, "plain", "utf-8"))
    alt.attach(MIMEText(html, "html", "utf-8"))
    root.attach(alt)

    for cid, png in images.items():
        img = MIMEImage(png, _subtype="png")
        img.add_header("Content-ID", f"<{cid}>")
        img.add_header("Content-Disposition", "inline", filename=f"{cid}.png")
        root.attach(img)

    recipients = [a.strip() for a in to.split(",") if a.strip()]
    ctx = ssl.create_default_context()
    with smtplib.SMTP_SSL(host, port, context=ctx) as s:
        s.login(user, pw)
        s.sendmail(sender, recipients, root.as_string())


def main():
    subject, html, text, images = generate_report()
    os.makedirs("output", exist_ok=True)
    # 단독 열람용 미리보기(이미지 인라인)
    with open("output/email.html", "w", encoding="utf-8") as f:
        f.write(html_with_inline_images(html, images))
    with open("output/email.txt", "w", encoding="utf-8") as f:
        f.write(text)
    with open("output/subject.txt", "w", encoding="utf-8") as f:
        f.write(subject)
    print("제목:", subject)
    print("-" * 60)
    print(text)
    print("-" * 60)
    print(f"차트 {len(images)}개 생성 · output/email.html 미리보기 저장")

    if os.environ.get("SMTP_USER") and os.environ.get("EMAIL_TO"):
        send_email(subject, html, text, images)
        print("✅ 이메일 발송 완료 →", os.environ.get("EMAIL_TO"))
    else:
        print("(SMTP 미설정: 파일만 생성. 메일 받으려면 SMTP_USER/SMTP_PASS/EMAIL_TO 설정)")


if __name__ == "__main__":
    main()
