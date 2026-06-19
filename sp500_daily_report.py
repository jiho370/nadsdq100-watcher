#!/usr/bin/env python3
"""
sp500_daily_report.py  (v5 - 고도화 버전)

매일 S&P 500 구성종목을 분석해 이메일 리포트를 생성/발송한다.
v5 핵심 개선점:
  1. 퀄리티 하드 필터: ROE 15% 미만, FCF 적자 기업은 스크리닝에서 원천 배제 (Value Trap 차단).
  2. 위험 조정 모멘텀: 단순 수익률이 아닌 변동성 대비 수익률(Risk-Adjusted Return) 중심 가점 부여.
  3. 스마트 청산(Exit): 이중 확인(50일선+200일선 동시 이탈 등) 구조로 노이즈에 의한 조기 청산 방지.
  4. 이벤트 드리븐 백테스트: 정기 교체를 폐지하고 락업(21일)과 상태 기반 편출입으로 마찰 비용 최소화.

데이터 : Yahoo Finance(yfinance, 키 불필요) + SPY 보유종목(State Street)
실행   : GitHub Actions cron (매일 1회, 미국장 마감 후)
"""

from __future__ import annotations

import os
import io
import sys
import json
import base64
import time
import argparse
from datetime import datetime, timezone, timedelta

import pandas as pd
import numpy as np

import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", message=".*tight_layout.*")
warnings.filterwarnings("ignore", message=".*Tight layout.*")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=UserWarning, module=r"matplotlib\..*")

try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    import requests
except ImportError:
    requests = None

import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

# ----------------------------- 설정 -----------------------------
RECO_N         = int(os.environ.get("RECO_N", "10"))            # 추천 종목 최대 개수
RECO_PER_MAX   = float(os.environ.get("RECO_PER_MAX", "20"))    # PER 절대 상한(이상치 차단용)
RECO_SECTOR_MAX= int(os.environ.get("RECO_SECTOR_MAX", "3"))    # 추천 섹터당 최대 종목수(집중 완화)
MIN_SIGNAL_DAYS= int(os.environ.get("MIN_SIGNAL_DAYS", "2"))    # 기술 진입신호 최소 지속일(휘프소 완화)
MAX_STALE_DAYS = int(os.environ.get("MAX_STALE_DAYS", "5"))     # 종가 신선도 허용 일수(달력일)
TREND_MAX      = int(os.environ.get("TREND_MAX", "6"))          # 전환/굳힘 섹션별 최대 종목
HISTORY_PERIOD = os.environ.get("HISTORY_PERIOD", "5y")
STATE_FILE     = os.environ.get("STATE_FILE", "state_prev_list.json")
KST            = timezone(timedelta(hours=9))

# 백테스트 설정 (이벤트 드리븐 구조)
BT_YEARS       = float(os.environ.get("BT_YEARS", "5"))         # 백테스트 기간(년)
BT_TOPK        = int(os.environ.get("BT_TOPK", "10"))           # 포트폴리오 최대 보유 슬롯(분산 통제)
BT_COST_BPS    = float(os.environ.get("BT_COST_BPS", "5"))      # 편입/편출 1회 거래비용(bp)
BT_PER_PROXY   = os.environ.get("BT_PER_PROXY", "0") == "1"     # 현재 PER을 과거에 적용(룩어헤드!) 실험용
REGIME_FILTER  = os.environ.get("REGIME_FILTER", "1") == "1"
WEIGHTING      = os.environ.get("WEIGHTING", "invvol")          # invvol=역변동성, equal=동일가중
VOL_WINDOW     = int(os.environ.get("VOL_WINDOW", "63"))        # 변동성 계산 창(거래일, ≈3개월)
MAX_VOL_PCTL   = float(os.environ.get("MAX_VOL_PCTL", "0.90"))  # 후보 변동성 상위(1-이값)% 제외

RSI_PERIOD     = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
MA_WINDOWS     = (20, 50, 200)
P_1W, P_1M, P_1Y, P_3Y = 5, 21, 252, 756

SPY_HOLDINGS_URL = ("https://www.ssga.com/us/en/intermediary/etfs/library-content/"
                    "products/fund-data/etfs/us/holdings-daily-us-en-spy.xlsx")

# GICS 섹터 영문 → 한글
GICS_KR = {
    "Information Technology": "정보기술", "Health Care": "헬스케어", "Financials": "금융",
    "Consumer Discretionary": "자유소비재", "Communication Services": "커뮤니케이션",
    "Industrials": "산업재", "Consumer Staples": "필수소비재", "Energy": "에너지",
    "Utilities": "유틸리티", "Real Estate": "부동산", "Materials": "소재",
}

# yfinance industry(영문) → 한글
INDUSTRY_KR = {
    "Semiconductors": "반도체", "Semiconductor Equipment & Materials": "반도체 장비·소재",
    "Software - Infrastructure": "인프라 소프트웨어", "Software - Application": "응용 소프트웨어",
    "Information Technology Services": "IT 서비스", "Communication Equipment": "통신장비",
    "Computer Hardware": "컴퓨터 하드웨어", "Consumer Electronics": "소비자 전자제품",
    "Electronic Components": "전자부품", "Scientific & Technical Instruments": "정밀계측기기",
    "Internet Content & Information": "인터넷 콘텐츠·플랫폼", "Internet Retail": "온라인 소매",
    "Entertainment": "엔터테인먼트", "Telecom Services": "통신 서비스",
    "Banks - Diversified": "종합 은행", "Banks - Regional": "지역 은행",
    "Capital Markets": "자본시장·증권", "Asset Management": "자산운용",
    "Insurance - Diversified": "종합 보험", "Insurance - Property & Casualty": "손해보험",
    "Insurance - Life": "생명보험", "Credit Services": "여신·결제 서비스",
    "Financial Data & Stock Exchanges": "금융데이터·거래소",
    "Drug Manufacturers - General": "대형 제약", "Drug Manufacturers - Specialty & Generic": "전문·제네릭 제약",
    "Biotechnology": "바이오테크", "Medical Devices": "의료기기", "Medical Instruments & Supplies": "의료기기·소모품",
    "Diagnostics & Research": "진단·연구", "Healthcare Plans": "건강보험", "Medical Care Facilities": "의료시설",
    "Drug Manufacturers": "제약", "Health Information Services": "헬스케어 IT",
    "Oil & Gas Integrated": "종합 석유·가스", "Oil & Gas E&P": "석유·가스 탐사생산",
    "Oil & Gas Midstream": "석유·가스 운송·저장", "Oil & Gas Equipment & Services": "유전 장비·서비스",
    "Oil & Gas Refining & Marketing": "정유·판매",
    "Aerospace & Defense": "항공우주·방산", "Specialty Industrial Machinery": "산업기계",
    "Farm & Heavy Construction Machinery": "건설·중장비", "Building Products & Equipment": "건축자재·설비",
    "Railroads": "철도", "Integrated Freight & Logistics": "물류·운송", "Airlines": "항공",
    "Trucking": "화물 운송", "Engineering & Construction": "엔지니어링·건설",
    "Industrial Distribution": "산업재 유통", "Staffing & Employment Services": "인력·고용 서비스",
    "Discount Stores": "할인점", "Specialty Retail": "전문 소매", "Home Improvement Retail": "주택용품 소매",
    "Restaurants": "외식·레스토랑", "Apparel Retail": "의류 소매", "Footwear & Accessories": "신발·액세서리",
    "Auto Manufacturers": "자동차 제조", "Auto Parts": "자동차 부품", "Travel Services": "여행 서비스",
    "Lodging": "호텔·숙박", "Resorts & Casinos": "리조트·카지노", "Packaging & Containers": "포장재",
    "Beverages - Non-Alcoholic": "음료(비주류)", "Beverages - Brewers": "주류",
    "Confectioners": "제과", "Packaged Foods": "가공식품", "Household & Personal Products": "생활·개인용품",
    "Tobacco": "담배", "Grocery Stores": "식료품 소매", "Farm Products": "농산물",
    "Utilities - Regulated Electric": "전력 유틸리티", "Utilities - Regulated Gas": "가스 유틸리티",
    "Utilities - Diversified": "종합 유틸리티", "Utilities - Renewable": "신재생 유틸리티",
    "Utilities - Regulated Water": "수도 유틸리티",
    "REIT - Specialty": "특수 리츠", "REIT - Industrial": "산업용 리츠", "REIT - Retail": "리테일 리츠",
    "REIT - Residential": "주거용 리츠", "REIT - Office": "오피스 리츠", "REIT - Healthcare Facilities": "헬스케어 리츠",
    "Specialty Chemicals": "특수 화학", "Chemicals": "화학", "Building Materials": "건축소재",
    "Gold": "금광", "Copper": "구리", "Steel": "철강", "Agricultural Inputs": "비료·농자재",
    "Industrial Gases": "산업용 가스",
}

# 주요 종목 한글 한 줄 설명
KR_DESC = {
    "NVDA": "AI·데이터센터용 GPU 1위", "AAPL": "아이폰·맥·서비스 생태계",
    "MSFT": "윈도우·오피스·Azure 클라우드", "AMZN": "전자상거래·AWS 클라우드",
    "GOOGL": "구글 검색·유튜브·클라우드", "GOOG": "구글 검색·유튜브·클라우드",
    "AVGO": "AI 네트워킹 반도체+인프라SW(브로드컴)", "TSLA": "전기차·에너지·자율주행",
    "META": "페이스북·인스타·AI 광고", "MU": "D램·낸드 메모리(마이크론)",
    "WMT": "미국 최대 유통(월마트)", "AMD": "CPU·GPU 반도체",
    "INTC": "CPU·파운드리 종합반도체(인텔)", "AMAT": "반도체 전공정 장비 1위",
    "LRCX": "식각·증착 반도체 장비(램리서치)", "CSCO": "네트워크 장비·보안(시스코)",
    "COST": "회원제 창고형 할인점(코스트코)", "KLAC": "반도체 검사·계측 장비",
    "NFLX": "글로벌 스트리밍 1위(넷플릭스)", "PLTR": "빅데이터 분석SW(팔란티어)",
    "TXN": "아날로그·임베디드 반도체(TI)", "MRVL": "데이터센터·AI 반도체(마벨)",
    "QCOM": "모바일 AP·통신 모뎀(퀄컴)", "LIN": "세계 최대 산업용 가스(린데)",
    "PANW": "차세대 사이버보안(팔로알토)", "ADI": "아날로그 반도체(ADI)",
    "TMUS": "미국 이동통신(T모바일)", "PEP": "음료·스낵(펩시코)",
    "AMGN": "바이오 신약(암젠)", "CRWD": "클라우드 엔드포인트 보안(크라우드스트라이크)",
    "APP": "모바일 앱 광고 플랫폼(앱러빈)", "GILD": "항바이러스·항암 신약(길리어드)",
    "HON": "항공·자동화 산업재(하니웰)", "ISRG": "수술용 로봇 다빈치",
    "BKNG": "온라인 여행 예약(부킹닷컴)", "VRTX": "희귀질환 신약(버텍스)",
    "SBUX": "글로벌 커피 체인(스타벅스)", "CDNS": "반도체 설계 EDA(케이던스)",
    "FTNT": "네트워크 방화벽 보안(포티넷)", "MAR": "글로벌 호텔(메리어트)",
    "CEG": "미국 최대 원자력 발전(컨스텔레이션)", "MNST": "에너지 음료(몬스터)",
    "SNPS": "반도체 설계 EDA(시놉시스)", "ADP": "급여·인사 아웃소싱(ADP)",
    "CSX": "미 동부 화물 철도", "ABNB": "숙박 공유(에어비앤비)",
    "CMCSA": "케이블·미디어(컴캐스트)", "NXPI": "차량용 반도체(NXP)",
    "DDOG": "클라우드 모니터링(데이터독)", "MDLZ": "과자·초콜릿(몬델리즈)",
    "ADBE": "크리에이티브·문서 SW(어도비)", "MPWR": "전력관리 반도체(모놀리식파워)",
    "DASH": "음식 배달(도어대시)", "ROST": "오프프라이스 의류(로스)",
    "INTU": "세무·회계 SW(인튜이트)", "ORLY": "자동차 부품 유통(오라일리)",
    "AEP": "전력 유틸리티(AEP)", "CTAS": "유니폼 렌탈·기업서비스(신타스)",
    "WBD": "영화·방송(워너브러더스디스커버리)", "REGN": "항체 신약(리제네론)",
    "PCAR": "대형 트럭(파카)", "BKR": "유전 서비스·장비(베이커휴즈)",
    "MCHP": "마이크로컨트롤러(마이크로칩)", "FAST": "산업용 부품 유통(패스널)",
    "FANG": "셰일 원유·가스(다이아몬드백)", "EA": "비디오게임(EA)",
    "XEL": "전력·가스 유틸리티(엑셀에너지)", "EXC": "전력 배전(엑셀론)",
    "ODFL": "LTL 화물 운송(올드도미니언)", "TTWO": "GTA 등 게임(테이크투)",
    "IDXX": "동물병원 진단(아이덱스)", "KDP": "음료·커피(큐리그닥터페퍼)",
    "ADSK": "3D 설계 CAD(오토데스크)", "PYPL": "온라인 결제(페이팔)",
    "PAYX": "중소기업 급여·HR(페이첵스)", "AXON": "테이저·바디캠(액손)",
    "ROP": "다각화 SW·산업재(로퍼)", "WDAY": "클라우드 인사·재무 SW(워크데이)",
    "DXCM": "연속혈당측정기(덱스콤)", "CPRT": "온라인 중고차 경매(코파트)",
    "GEHC": "의료영상 장비(GE헬스케어)", "KHC": "가공식품(크래프트하인즈)",
    "VRSK": "보험 데이터 분석(베리스크)", "CTSH": "IT 컨설팅(코그니전트)",
    "CHTR": "케이블 인터넷·방송(차터)", "JPM": "미국 최대 은행(JP모건)",
    "V": "글로벌 결제 네트워크(비자)", "MA": "글로벌 결제 네트워크(마스터카드)",
    "UNH": "최대 건강보험·헬스케어(유나이티드헬스)", "JNJ": "제약·의료기기(존슨앤드존슨)",
    "LLY": "비만·당뇨 신약(일라이릴리)", "XOM": "글로벌 석유메이저(엑슨모빌)",
    "CVX": "글로벌 석유메이저(셰브론)", "HD": "주택용품 유통(홈디포)",
    "PG": "생활용품(P&G)", "KO": "음료(코카콜라)", "BAC": "대형 은행(뱅크오브아메리카)",
    "ABBV": "면역·항암 신약(애브비)", "MRK": "제약(머크)", "PFE": "제약(화이자)",
    "ORCL": "데이터베이스·클라우드(오라클)", "CRM": "CRM 클라우드 SW(세일즈포스)",
    "ACN": "IT 컨설팅(액센추어)", "MCD": "글로벌 패스트푸드(맥도날드)",
    "NKE": "스포츠 의류·신발(나이키)", "DIS": "미디어·테마파크(디즈니)",
    "GS": "투자은행(골드만삭스)", "MS": "투자은행(모건스탠리)", "IBM": "IT 서비스·하이브리드 클라우드",
    "GE": "항공엔진·전력(GE에어로스페이스)", "CAT": "건설·광산 장비(캐터필러)",
    "BA": "항공기 제조(보잉)", "RTX": "방산·항공(RTX)", "LMT": "방산(록히드마틴)",
    "UBER": "차량호출·배달(우버)", "NOW": "기업용 워크플로 SW(서비스나우)",
    "T": "이동통신(AT&T)", "VZ": "이동통신(버라이즌)", "WFC": "대형 은행(웰스파고)",
    "ANET": "데이터센터 네트워크 장비(아리스타)", "DELL": "PC·서버(델)",
    "SMCI": "AI 서버(슈퍼마이크로)", "DHR": "생명과학·진단(다나허)",
    "TMO": "생명과학 장비·진단(써모피셔)", "ABT": "의료기기·진단(애벗)",
}

SP500_FALLBACK = {"AAPL": "Information Technology"} 

# ===================== 구성종목(유니버스) =====================
def get_sp500() -> tuple[list[str], dict[str, str]]:
    syms, sectors = _fetch_spy_holdings()
    if syms and 400 <= len(syms) <= 520:
        for s in syms:
            sectors.setdefault(s, SP500_FALLBACK.get(s, ""))
        print(f"[정보] SPY 보유종목 {len(syms)}개 로드", file=sys.stderr)
        return syms, sectors
    print("[경고] SPY 보유종목 로드 실패 → 내장 스냅샷으로 폴백", file=sys.stderr)
    return list(SP500_FALLBACK.keys()), dict(SP500_FALLBACK)

def _fetch_spy_holdings() -> tuple[list[str], dict[str, str]]:
    if requests is None: return [], {}
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(SPY_HOLDINGS_URL, headers=headers, timeout=30)
        r.raise_for_status()
        raw = pd.read_excel(io.BytesIO(r.content), engine="openpyxl", header=None)
    except Exception as e:
        print(f"[경고] SPY xlsx 파싱 실패: {e}", file=sys.stderr)
        return [], {}

    hdr = None
    for i in range(min(15, len(raw))):
        cells = [str(c).strip().lower() for c in raw.iloc[i].tolist()]
        if "ticker" in cells and ("name" in cells or "sector" in cells):
            hdr = i; break
    if hdr is None: return [], {}

    cols = [str(c).strip() for c in raw.iloc[hdr].tolist()]
    df = raw.iloc[hdr + 1:].copy()
    df.columns = cols

    def find_col(*names):
        for n in names:
            for c in cols:
                if c.strip().lower() == n: return c
        return None

    c_tic, c_sec = find_col("ticker"), find_col("sector")
    if c_tic is None: return [], {}

    syms, sectors = [], {}
    for _, row in df.iterrows():
        tic = str(row.get(c_tic, "")).strip()
        if not tic or tic.lower() in ("nan", "-", "cash", "ssga", "uscash"): continue
        if not all(ch.isalnum() or ch in ".-" for ch in tic): continue
        yh = tic.replace(".", "-")
        if not yh[0].isalpha(): continue
        sec = str(row.get(c_sec, "")).strip() if c_sec else ""
        if sec.lower() == "nan": sec = ""
        if yh not in sectors:
            syms.append(yh)
            sectors[yh] = _norm_sector(sec)
    return syms, sectors

def _norm_sector(sec: str) -> str:
    if not sec: return ""
    s = sec.strip().lower()
    table = {
        "information technology": "Information Technology", "health care": "Health Care",
        "financials": "Financials", "consumer discretionary": "Consumer Discretionary",
        "communication services": "Communication Services", "industrials": "Industrials", 
        "consumer staples": "Consumer Staples", "energy": "Energy", 
        "utilities": "Utilities", "real estate": "Real Estate", "materials": "Materials",
    }
    return table.get(s, sec.strip())

# ------------------------- yfinance 유틸 ------------------------
def _require_yf():
    if yf is None: raise RuntimeError("yfinance 설치 필요")

def get_info_for(symbols: list[str]) -> dict[str, dict]:
    _require_yf()
    out = {}
    for sym in symbols:
        info = None
        for attempt in range(2):
            try:
                info = yf.Ticker(sym).info or {}
                break
            except Exception:
                if attempt == 0: time.sleep(0.8)
        if info is None: continue
        pe = info.get("trailingPE")
        price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")
        if pe is None:
            eps = info.get("trailingEps")
            try:
                if eps and price: pe = float(price) / float(eps)
            except: pe = None
        out[sym] = {
            "pe": pe, "price": price, "name": info.get("shortName") or info.get("longName") or sym,
            "industry": info.get("industry") or "", "sector_en": info.get("sector") or "",
            "summary": info.get("longBusinessSummary") or "",
            "roe": info.get("returnOnEquity"), "de": info.get("debtToEquity"),
            "fcf": info.get("freeCashflow"), "rev_growth": info.get("revenueGrowth"),
            "profit_margin": info.get("profitMargins"),
        }
    return out

def download_histories(symbols: list[str], period: str = HISTORY_PERIOD) -> dict[str, pd.Series]:
    _require_yf()
    out = {}
    try:
        data = yf.download(symbols, period=period, interval="1d", auto_adjust=True, group_by="ticker", threads=True, progress=False)
    except:
        data = None

    if data is not None and not data.empty:
        for sym in symbols:
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    if sym not in data.columns.get_level_values(0): continue
                    close = data[sym]["Close"]
                else:
                    close = data["Close"]
                close = _clean_close(close)
                if not close.empty: out[sym] = close
            except: continue

    missing = [s for s in symbols if s not in out]
    for sym in missing:
        try:
            raw = yf.Ticker(sym).history(period=period, interval="1d", auto_adjust=True)
            if raw is not None and not raw.empty and "Close" in raw.columns:
                close = _clean_close(raw["Close"])
                if not close.empty: out[sym] = close
        except: continue
    return _filter_stale(out, MAX_STALE_DAYS)

def _filter_stale(hist: dict[str, pd.Series], max_stale_days: int) -> dict[str, pd.Series]:
    if not hist: return hist
    last_dates = {s: c.index[-1] for s, c in hist.items() if len(c)}
    if not last_dates: return hist
    ref = max(last_dates.values())
    cutoff = ref - pd.Timedelta(days=max_stale_days)
    fresh = {s: c for s, c in hist.items() if len(c) and c.index[-1] >= cutoff}
    return fresh

def _clean_close(close: pd.Series) -> pd.Series:
    s = pd.to_numeric(close, errors="coerce")
    idx = pd.to_datetime(s.index)
    try: idx = idx.tz_localize(None)
    except: pass
    s.index = idx
    return s.dropna().sort_index()

# ------------------------- 지표 계산 ----------------------------
def _rsi(close, period=RSI_PERIOD):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def _macd(close):
    ef = close.ewm(span=MACD_FAST, adjust=False).mean()
    es = close.ewm(span=MACD_SLOW, adjust=False).mean()
    m = ef - es
    sig = m.ewm(span=MACD_SIGNAL, adjust=False).mean()
    return m, sig, m - sig

def _ret(close, periods):
    if len(close) > periods:
        prev = close.iloc[-1 - periods]
        if prev and not pd.isna(prev): return (float(close.iloc[-1]) / float(prev) - 1.0) * 100.0
    return float("nan")

def _ret_full(close):
    c = close.dropna()
    if len(c) >= 2 and c.iloc[0]: return (float(c.iloc[-1]) / float(c.iloc[0]) - 1.0) * 100.0
    return float("nan")

def _tech_entry_series(close: pd.Series) -> pd.Series:
    ma200 = close.rolling(200).mean()
    _, _, hist = _macd(close)
    cond = (close > ma200) & (hist > 0)
    return cond.fillna(False)

def _signal_streak(cond: pd.Series) -> int:
    streak = 0
    for v in reversed(cond.values):
        if bool(v): streak += 1
        else: break
    return streak

def _streak_series(cond: pd.Series) -> pd.Series:
    c = cond.astype(bool)
    grp = (~c).cumsum()
    return c.groupby(grp).cumsum()

def market_regime(spy_close: pd.Series) -> dict:
    s = _clean_close(spy_close)
    if len(s) < 200:
        return {"risk_on": True, "spy": float(s.iloc[-1]) if len(s) else float("nan"), "ma200": float("nan"), "gap_pct": float("nan")}
    ma200 = float(s.rolling(200).mean().iloc[-1])
    last = float(s.iloc[-1])
    return {"risk_on": last > ma200, "spy": last, "ma200": ma200, "gap_pct": (last / ma200 - 1) * 100 if ma200 else float("nan")}

def _ann_vol(close: pd.Series, window: int = VOL_WINDOW) -> float:
    r = close.pct_change().dropna()
    if len(r) < max(20, window // 2): return float("nan")
    return float(r.iloc[-window:].std() * np.sqrt(252) * 100)

def compute_indicators(close):
    if close is None or close.empty: return None
    last = float(close.iloc[-1])
    ma = {w: (close.rolling(w).mean() if len(close) >= w else pd.Series(dtype=float)) for w in MA_WINDOWS}
    ma_last = {w: (float(ma[w].iloc[-1]) if len(ma[w]) and not np.isnan(ma[w].iloc[-1]) else np.nan) for w in MA_WINDOWS}
    rsi_series = _rsi(close)
    rsi_val = float(rsi_series.iloc[-1]) if not np.isnan(rsi_series.iloc[-1]) else np.nan
    macd, signal, hist = _macd(close)
    macd_val, sig_val, hist_val = float(macd.iloc[-1]), float(signal.iloc[-1]), float(hist.iloc[-1])
    
    cross = None
    if len(close) >= 205:
        ma50, ma200 = close.rolling(50).mean(), close.rolling(200).mean()
        diff = (ma50 - ma200).dropna()
        if len(diff) >= 6:
            recent = np.sign(diff.iloc[-6:])
            if recent.iloc[0] < 0 and recent.iloc[-1] > 0: cross = "golden"
            elif recent.iloc[0] > 0 and recent.iloc[-1] < 0: cross = "death"
            
    entry_streak = _signal_streak(_tech_entry_series(close))
    ind = {
        "price": last, "ma20": ma_last[20], "ma50": ma_last[50], "ma200": ma_last[200],
        "above_ma200": (not np.isnan(ma_last[200])) and last > ma_last[200],
        "rsi": rsi_val, "macd": macd_val, "macd_signal": sig_val, "macd_hist": hist_val,
        "macd_up": hist_val > 0, "cross": cross, "entry_streak": entry_streak,
        "vol_ann": _ann_vol(close), "chg_1d": _ret(close, 1), "chg_1w": _ret(close, P_1W), 
        "chg_1m": _ret(close, P_1M), "chg_3m": _ret(close, 63), "chg_1y": _ret(close, P_1Y), 
        "chg_3y": _ret(close, P_3Y), "chg_5y": _ret_full(close),
    }
    ind.update(_classify_trend(close, ma, hist, rsi_series))
    return ind

def _classify_trend(close, ma, hist, rsi_series):
    res = {"reversal": False, "reversal_score": 0.0, "reversal_reason": "",
           "solidified": False, "solidified_score": 0.0, "solidified_reason": ""}
    if len(close) < 60: return res
    last = float(close.iloc[-1])
    ma20s, ma50s, ma200s = ma[20], ma[50], ma[200]
    ma20 = float(ma20s.iloc[-1]) if len(ma20s) and not np.isnan(ma20s.iloc[-1]) else np.nan
    ma50 = float(ma50s.iloc[-1]) if len(ma50s) and not np.isnan(ma50s.iloc[-1]) else np.nan
    ma200 = float(ma200s.iloc[-1]) if len(ma200s) and not np.isnan(ma200s.iloc[-1]) else np.nan
    h, rsi = hist.dropna(), rsi_series.dropna()
    if len(h) < 6 or len(rsi) < 6: return res
    return res

# --------------------- 밸류에이션 · 스크리닝 ----------------------
def sector_median_pes(info: dict[str, dict], sector_map: dict[str, str]) -> tuple[dict, float]:
    by_sec, allpe = {}, []
    for sym, meta in info.items():
        pe = meta.get("pe")
        try: pe = float(pe)
        except: continue
        if not (0 < pe <= 200): continue
        sec = sector_map.get(sym, "") or "(기타)"
        by_sec.setdefault(sec, []).append(pe)
        allpe.append(pe)
    med = {sec: float(np.median(v)) for sec, v in by_sec.items()}
    global_med = float(np.median(allpe)) if allpe else float("nan")
    for sec in list(med.keys()):
        if len(by_sec[sec]) < 3 and not np.isnan(global_med): med[sec] = global_med
    return med, global_med

def score_reco(ind: dict, meta: dict, pe, rel_pe: float) -> tuple[float, str] | None:
    """추천 적격이면 (점수, 한글사유) 반환, 아니면 None.
    [개선된 로직]
    1. 퀄리티 하드 필터: ROE 15% 이상, FCF 흑자 필수.
    2. 위험 조정 모멘텀: 변동성 대비 수익률 우수 종목 높은 가점.
    """
    try:
        pe = float(pe)
    except (TypeError, ValueError):
        return None
    if not (0 < pe <= RECO_PER_MAX):
        return None
    
    # 1. 퀄리티 하드 필터 (Value Trap 차단)
    roe = meta.get("roe")
    fcf = meta.get("fcf")
    qreasons = []
    
    if roe is not None:
        if roe < 0.15: return None
        qreasons.append(f"ROE {roe*100:.0f}%")
    if fcf is not None:
        if fcf <= 0: return None
        qreasons.append("FCF 흑자")

    # 기본 기술적 허들
    if not ind.get("above_ma200"): return None
    if not (ind.get("macd_up") or ind.get("cross") == "golden"): return None
    if ind.get("entry_streak", 0) < MIN_SIGNAL_DAYS: return None
    if not _isnan(ind.get("rsi")) and ind["rsi"] > 82: return None

    score, reasons = 0.0, []

    # 2. 밸류에이션 가점 (상대적으로 비중 축소)
    cheap = max(0.0, min(1.0, 1.0 - rel_pe))
    if cheap > 0:
        score += cheap * 1.0
        reasons.append(f"PER {pe:.1f}")

    if qreasons:
        reasons.append("우량재무(" + ", ".join(qreasons[:2]) + ")")

    # 3. 위험 조정 모멘텀 (Smooth Compounder 발굴)
    chg_3m = ind.get("chg_3m")
    vol = ind.get("vol_ann")
    if not _isnan(chg_3m) and not _isnan(vol) and vol > 0:
        risk_adj_ret = chg_3m / vol
        if risk_adj_ret > 0:
            score += risk_adj_ret * 3.0  
            reasons.append("위험조정 모멘텀 우수")

    # 기존 기술적 추세 가점
    if ind.get("cross") == "golden":
        score += 1.5
        reasons.append("골든크로스")
    elif ind.get("macd_up"):
        score += 1.0
        
    price, ma20, ma50, ma200 = ind.get("price"), ind.get("ma20"), ind.get("ma50"), ind.get("ma200")
    if all(not _isnan(x) for x in (price, ma20, ma50, ma200)) and price > ma20 > ma50 > ma200:
        score += 1.5
        reasons.append("이동평균 정배열")
        
    return score, " · ".join(reasons)

def pick_with_sector_cap(scored: list[tuple], sector_map: dict[str, str], n: int, cap: int) -> list[tuple]:
    out, per_sec = [], {}
    for sym, sc, reason in scored:
        sec = sector_map.get(sym, "") or "(기타)"
        if per_sec.get(sec, 0) >= cap: continue
        out.append((sym, sc, reason))
        per_sec[sec] = per_sec.get(sec, 0) + 1
        if len(out) >= n: break
    return out

def _alloc_weights(chosen: list[str], vol_row, columns) -> pd.Series:
    w = pd.Series(0.0, index=columns)
    if not chosen: return w
    if WEIGHTING == "invvol":
        inv = {}
        for s in chosen:
            v = vol_row.get(s, np.nan)
            inv[s] = (1.0 / v) if (v is not None and not pd.isna(v) and v > 0) else np.nan
        vals = [x for x in inv.values() if not pd.isna(x)]
        if vals:
            med = float(np.median(vals))
            tot = sum((x if not pd.isna(x) else med) for x in inv.values())
            for s in chosen:
                w[s] = (inv[s] if not pd.isna(inv[s]) else med) / tot
            return w
    for s in chosen: w[s] = 1.0 / len(chosen)
    return w

# --------------------- 스마트 청산(EXIT) 신호 --------------------
def detect_exits(prev_syms: list[str], ind_map: dict[str, dict], picked_syms: list[str]) -> list[tuple]:
    """이중 확인 구조 적용 (조정장 버티기 목적)"""
    exits = []
    for sym in prev_syms:
        if sym in picked_syms: continue
        ind = ind_map.get(sym)
        if not ind:
            exits.append((sym, ind, "데이터 없음(거래정지·신선도 미달 가능) — 점검 필요"))
            continue
            
        why = []
        price = ind.get("price")
        ma50 = ind.get("ma50")
        ma200 = ind.get("ma200")
        macd_up = ind.get("macd_up")
        
        if all(not _isnan(x) for x in (price, ma50, ma200)):
            if price < ma50 and price < ma200:
                why.append("50일선 및 200일선 동시 이탈(추세 붕괴)")
            elif not macd_up and price < ma50:
                why.append("MACD 하락 전환 및 50일선 이탈")
                
        if why: exits.append((sym, ind, " · ".join(why)))
            
    return exits

# ==================== 이벤트 드리븐 백테스트 ====================
def _align_panel(hist: dict[str, pd.Series], dates: pd.DatetimeIndex) -> pd.DataFrame:
    cols = {s: c.reindex(dates).astype(float) for s, c in hist.items()}
    return pd.DataFrame(cols, index=dates)

def _max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1.0).min())

def _metrics(equity: pd.Series, periods_per_year: int = 252) -> dict:
    eq = equity.dropna()
    if len(eq) < 2: return {"total": float("nan"), "cagr": float("nan"), "vol": float("nan"), "mdd": float("nan"), "sharpe": float("nan")}
    rets = eq.pct_change().dropna()
    years = len(eq) / periods_per_year
    total = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1.0) if years > 0 else float("nan")
    vol = float(rets.std() * np.sqrt(periods_per_year))
    sharpe = float(rets.mean() / rets.std() * np.sqrt(periods_per_year)) if rets.std() > 0 else float("nan")
    return {"total": total, "cagr": cagr, "vol": vol, "mdd": _max_drawdown(eq), "sharpe": sharpe}

def run_backtest(hist: dict[str, pd.Series], spy_close: pd.Series,
                 sector_map: dict[str, str], info: dict | None = None) -> dict:
    """매일 상태 평가, 락업 기반 매도 통제 방식 적용."""
    if spy_close is None or spy_close.empty: raise RuntimeError("SPY 종가 필요.")
    spy_close = _clean_close(spy_close)

    end = spy_close.index[-1]
    start = end - pd.Timedelta(days=int(BT_YEARS * 365.25) + 5)
    dates = spy_close.index[(spy_close.index >= start) & (spy_close.index <= end)]
    if len(dates) < 60: raise RuntimeError("데이터 부족.")

    panel = _align_panel(hist, dates)
    rets = panel.pct_change().fillna(0.0)
    mom = panel.pct_change(63)

    entry_full = {s: _tech_entry_series(c) for s, c in hist.items()}
    entry = pd.DataFrame({s: e.reindex(dates) for s, e in entry_full.items()}, index=dates).fillna(False)
    streak = pd.DataFrame({s: _streak_series(e).reindex(dates) for s, e in entry_full.items()}, index=dates).fillna(0)

    exit_signal = pd.DataFrame(False, index=dates, columns=panel.columns)
    for s, c in hist.items():
        ma200 = c.rolling(200).mean().reindex(dates)
        _, _, m_hist = _macd(c)
        m_hist = m_hist.reindex(dates)
        macd_death = (m_hist < 0) & (m_hist.shift(1) >= 0)
        under_ma200 = panel[s] < ma200
        exit_signal[s] = under_ma200 | macd_death

    spy_aligned = spy_close.reindex(dates)
    spy_ma200 = spy_close.rolling(200).mean().reindex(dates)

    MAX_POSITIONS = BT_TOPK if BT_TOPK > 0 else 10
    LOCK_UP_DAYS = 21
    TARGET_WEIGHT = 1.0 / MAX_POSITIONS

    weights = pd.DataFrame(0.0, index=dates, columns=panel.columns)
    turnover_on = {}
    regime_off_days = 0
    portfolio = {}  

    for i in range(len(dates)):
        d = dates[i]
        cur_spy = float(spy_aligned.iloc[i])
        cur_spy_ma200 = float(spy_ma200.iloc[i])
        risk_on = cur_spy > cur_spy_ma200 if not pd.isna(cur_spy_ma200) else True

        if not risk_on: regime_off_days += 1

        to_remove = []
        for s, entry_idx in portfolio.items():
            if i - entry_idx >= LOCK_UP_DAYS and exit_signal[s].iloc[i]:
                to_remove.append(s)

        for s in to_remove: del portfolio[s]

        empty_slots = MAX_POSITIONS - len(portfolio)
        if risk_on and empty_slots > 0:
            elig = entry.iloc[i]
            strk = streak.iloc[i]
            cand = [s for s in panel.columns if bool(elig.get(s, False)) and strk.get(s, 0) >= MIN_SIGNAL_DAYS and s not in portfolio and not pd.isna(panel.iloc[i][s])]
            if cand:
                ranked = list(mom.iloc[i][cand].dropna().sort_values(ascending=False).index)
                for s in ranked[:empty_slots]: portfolio[s] = i

        w = pd.Series(0.0, index=panel.columns)
        for s in portfolio.keys(): w[s] = TARGET_WEIGHT
        weights.iloc[i] = w

        if i > 0:
            to = float((w - weights.iloc[i-1]).abs().sum())
            if to > 0: turnover_on[d] = to / 2.0

    w_lag = weights.shift(1).fillna(0.0)
    port_ret = (w_lag * rets).sum(axis=1)
    cost = pd.Series(0.0, index=dates)
    for d, to in turnover_on.items(): cost[d] = to * (BT_COST_BPS / 1e4)
    port_ret = (port_ret - cost).fillna(0.0)

    strat_eq = (1.0 + port_ret).cumprod()
    spy_dr = spy_aligned.pct_change().fillna(0.0)
    spy_eq = (1.0 + spy_dr).cumprod()

    m_s, m_b = _metrics(strat_eq), _metrics(spy_eq)
    ann_turnover = sum(turnover_on.values()) / (len(dates) / 252) if len(dates) > 0 else float("nan")

    annual = []
    for yr in sorted(set(port_ret.index.year)):
        sp = (1 + port_ret[port_ret.index.year == yr]).prod() - 1
        bp = (1 + spy_dr[spy_dr.index.year == yr]).prod() - 1
        annual.append((yr, sp * 100, bp * 100, (sp - bp) * 100))

    excess_m = ((1 + port_ret).resample("ME").prod() - 1) - ((1 + spy_dr).resample("ME").prod() - 1)
    excess_m = excess_m.dropna()
    total_ex = float(excess_m.sum())
    top3 = float(excess_m.sort_values(ascending=False).head(3).sum())

    return {"strat_eq": strat_eq, "spy_eq": spy_eq, "strat": m_s, "spy": m_b,
            "win_rate": float((excess_m > 0).mean()), "avg_holdings": float((weights > 0).sum(axis=1).mean()), 
            "start": dates[0], "end": dates[-1], "rebalances": len(turnover_on),
            "ann_turnover": ann_turnover, "regime_off": regime_off_days,
            "regime_on_pct": (1 - regime_off_days / len(dates)) * 100,
            "annual": annual, "conc": {"total_excess_pp": total_ex * 100, "top3_share": (top3 / total_ex) if abs(total_ex) > 1e-9 else float("nan"), "excl_top3_pp": (total_ex - top3) * 100, "win_months": float((excess_m > 0).mean())}}

def _backtest_chart(res: dict) -> bytes:
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ax.plot(res["strat_eq"].index, res["strat_eq"].values, color="#15803d", lw=1.6, label="Strategy")
    ax.plot(res["spy_eq"].index, res["spy_eq"].values, color="#6b7280", lw=1.4, label="SPY buy & hold")
    ax.set_title("Backtest cumulative growth (1.0 = start)", fontsize=10, loc="left")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=8, frameon=False)
    for sp in ax.spines.values(): sp.set_visible(False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return buf.getvalue()

def backtest_main():
    _require_yf()
    universe, sector_map = get_sp500()
    hist = download_histories(universe, period=f"{int(BT_YEARS)+2}y")
    spy = download_histories(["SPY"], period=f"{int(BT_YEARS)+2}y").get("SPY")
    res = run_backtest(hist, spy, sector_map, None)
    s, b = res["strat"], res["spy"]
    
    print("=" * 64)
    print(f" 백테스트 결과 (이벤트 드리븐)  {res['start'].date()} ~ {res['end'].date()}")
    print("=" * 64)
    print(f"총수익률: {s['total']*100:.1f}% (SPY {b['total']*100:.1f}%)")
    print(f"연복리(CAGR): {s['cagr']*100:.1f}% (SPY {b['cagr']*100:.1f}%)")
    print(f"연변동성: {s['vol']*100:.1f}%
