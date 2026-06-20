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
실행   : GitHub Actions cron (미국 거래일 다음날 아침 KST 10:00, 전일 종가 기준)
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
RECO_PER_MAX   = float(os.environ.get("RECO_PER_MAX", "20"))    # (참고용) 과거 저PER 캡 — 하이브리드에선 미사용
RECO_SECTOR_MAX= int(os.environ.get("RECO_SECTOR_MAX", "3"))    # 추천 섹터당 최대 종목수(집중 완화)
MIN_SIGNAL_DAYS= int(os.environ.get("MIN_SIGNAL_DAYS", "2"))    # 기술 진입신호 최소 지속일(휘프소 완화)
# ── 하이브리드(퀄리티 코어 + 모멘텀, 밸류 소프트 틸트) — 백테스트 검증 개선판 ──
PER_SANITY     = float(os.environ.get("PER_SANITY", "60"))      # 이상치 차단용 느슨한 PER 상한(절대캡 아님)
MOM_MIN_6M     = float(os.environ.get("MOM_MIN_6M", "0"))       # 6개월 모멘텀 최소(%) — 진입 코어 요건
MOM_WEIGHT     = float(os.environ.get("MOM_WEIGHT", "3.0"))     # 위험조정 모멘텀 주가중
VALUE_WEIGHT   = float(os.environ.get("VALUE_WEIGHT", "0.4"))   # 밸류 소프트 틸트(작게)
RSI_MAX        = float(os.environ.get("RSI_MAX", "90"))         # 과열 제외(모멘텀 허용 위해 82→90 완화)
EXIT_BUFFER    = float(os.environ.get("EXIT_BUFFER", "0.03"))   # 매도: 200일선 -3% 하향 버퍼
TRAIL_STOP     = float(os.environ.get("TRAIL_STOP", "0.25"))    # 매도: 52주 고점 대비 -25% 트레일링
# 리스크 다이얼(변동성 타게팅) — 0=off(숨김), 0.12=목표 연변동성 12%. 메일에 '권장 주식 노출'로 표시
VOL_TARGET     = float(os.environ.get("VOL_TARGET", "0"))       # 0=off / 0.12=연 12% 목표
VOLTGT_WINDOW  = int(os.environ.get("VOLTGT_WINDOW", "60"))     # 실현변동성 창(거래일): 20 또는 60
MAX_STALE_DAYS = int(os.environ.get("MAX_STALE_DAYS", "5"))     # 종가 신선도 허용 일수(달력일)
TREND_MAX      = int(os.environ.get("TREND_MAX", "6"))          # 전환/굳힘 섹션별 최대 종목
HISTORY_PERIOD = os.environ.get("HISTORY_PERIOD", "5y")
STATE_FILE     = os.environ.get("STATE_FILE", "state_prev_list.json")
PROFILES_FILE  = os.environ.get("PROFILES_FILE", "sp500_profiles.json")  # 분기 갱신 종목 프로필
_PROFILES_CACHE = None
KST            = timezone(timedelta(hours=9))

# 평일 4섹션 모드 설정 (각 섹션 종목 수 / 트리거 기준)
SECTION_N      = int(os.environ.get("SECTION_N", "3"))             # 각 섹션 추천 종목 수
TREND_HIGH_FRAC= float(os.environ.get("TREND_HIGH_FRAC", "0.75"))  # 1) 52주 고가 대비 하한(현재가>=고가x이값)
NOTABLE_RVOL   = float(os.environ.get("NOTABLE_RVOL", "2.0"))      # 3) 당일 상대거래량 트리거(20일 평균 대비)
NOTABLE_MOVE   = float(os.environ.get("NOTABLE_MOVE", "5.0"))      # 3) 당일 절대변동률(%) 트리거
NOTABLE_TOPN_MC= int(os.environ.get("NOTABLE_TOPN_MC", "120"))     # 3) 시총 상위 후보 풀 크기
HOLIDAY_SKIP_STREAK = int(os.environ.get("HOLIDAY_SKIP_STREAK", "4"))  # 휴장 streak 이 값 이상이면 메일 스킵
WEEKLY_TOP_N   = int(os.environ.get("WEEKLY_TOP_N", "10"))         # 휴장 TOP10/펀더멘탈 TOP10 종목 수
SECTOR_COVERAGE_MIN = float(os.environ.get("SECTOR_COVERAGE_MIN", "0.9"))  # 섹터 분류율 이 값 미만이면 경고 표시

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

KR_DESC.update({
    "WDC": "하드디스크·플래시 메모리 저장장치(웨스턴디지털)", "STX": "하드디스크 저장장치(씨게이트)",
    "MRNA": "mRNA 백신·치료제(모더나)", "GEV": "전력·에너지 설비(GE버노바)",
    "HOOD": "주식·코인 거래 앱(로빈후드)", "COIN": "암호화폐 거래소(코인베이스)",
    "MSTR": "비트코인 보유·BI 소프트웨어(마이크로스트래티지)", "VST": "전력 발전·소매(비스트라)",
    "CARR": "냉난방공조 설비(캐리어)", "PWR": "전력·통신 인프라 시공(콴타서비스)",
    "URI": "건설장비 렌탈(유나이티드렌탈스)", "CTVA": "농업 종자·작물보호제(코르테바)",
    "DOW": "기초 화학소재(다우)", "DD": "특수 화학소재(듀폰)", "PPG": "산업용 도료·코팅(PPG)",
    "SHW": "건축용 페인트(셔윈윌리엄스)", "NUE": "철강 제조(뉴코어)", "FCX": "구리·금 광산(프리포트맥모란)",
    "NEM": "금 채굴(뉴몬트)", "APD": "산업용 가스(에어프로덕츠)", "ECL": "위생·수처리 솔루션(에코랩)",
    "VMC": "건설용 골재·시멘트(벌컨머티리얼스)", "MLM": "건설용 골재(마틴마리에타)",
    "EMR": "산업 자동화 장비(에머슨)", "ETN": "전력관리 부품(이튼)", "PH": "유압·산업부품(파커하니핀)",
    "ITW": "산업용 부품·장비(일리노이툴웍스)", "DE": "농기계·건설장비(존디어)",
    "GD": "방산·항공(제너럴다이내믹스)", "NOC": "방산(노스럽그러먼)", "LHX": "방산 통신·전자(L3해리스)",
    "TDG": "항공기 부품(트랜스다임)", "TT": "냉난방공조(트레인테크놀로지스)", "JCI": "빌딩 설비·제어(존슨컨트롤스)",
    "CMI": "디젤 엔진·발전(커민스)", "PCAR": "대형 트럭(파카)", "GWW": "산업용 MRO 유통(그레인저)",
    "FDX": "특송·물류(페덱스)", "UPS": "특송·물류(UPS)", "UNP": "철도 화물(유니온퍼시픽)",
    "NSC": "동부 화물 철도(노퍽서던)", "LUV": "저가 항공(사우스웨스트)", "DAL": "대형 항공(델타)",
    "UAL": "대형 항공(유나이티드)", "WM": "폐기물 처리(웨이스트매니지먼트)", "RSG": "폐기물 처리(리퍼블릭서비스)",
    "EMR": "산업 자동화(에머슨)", "ROK": "공장 자동화(로크웰오토메이션)", "AME": "정밀 계측·전자(아메텍)",
    "DOV": "산업기계 다각화(도버)", "IR": "산업용 압축기·펌프(인거솔랜드)",
    "PG": "생활용품(P&G)", "CL": "치약·생활용품(콜게이트)", "KMB": "위생용품(킴벌리클라크)",
    "GIS": "시리얼·가공식품(제너럴밀스)", "K": "시리얼·스낵(켈라노바)", "HSY": "초콜릿·제과(허쉬)",
    "SYY": "식자재 유통(시스코)", "ADM": "곡물 가공·유통(ADM)", "STZ": "주류(컨스텔레이션브랜즈)",
    "MO": "담배(알트리아)", "PM": "글로벌 담배(필립모리스)", "CLX": "표백제·생활용품(클로락스)",
    "CHD": "생활용품(처치앤드와이트)", "EL": "화장품(에스티로더)", "KVUE": "소비자 건강용품(켄뷰)",
    "TGT": "종합 소매(타깃)", "DG": "달러 할인점(달러제너럴)", "DLTR": "달러 할인점(달러트리)",
    "KR": "식료품 소매(크로거)", "LOW": "주택용품 소매(로우스)", "TJX": "오프프라이스 의류(TJX)",
    "LULU": "운동복(룰루레몬)", "GM": "자동차 제조(GM)", "F": "자동차 제조(포드)",
    "APTV": "차량 전장·부품(앱티브)", "GPC": "자동차·산업부품 유통(제뉴인파츠)",
    "YUM": "패스트푸드 프랜차이즈(얌브랜즈)", "CMG": "패스트캐주얼 외식(치폴레)",
    "HLT": "글로벌 호텔(힐튼)", "RCL": "크루즈(로열캐리비안)", "CCL": "크루즈(카니발)",
    "EXPE": "온라인 여행(익스피디아)", "DHI": "주택 건설(DR호튼)", "LEN": "주택 건설(레나)",
    "PHM": "주택 건설(펄트그룹)", "NVR": "주택 건설(NVR)",
    "C": "대형 은행(씨티그룹)", "USB": "지역 은행(US뱅코프)", "PNC": "지역 은행(PNC)",
    "TFC": "지역 은행(트루이스트)", "COF": "신용카드·은행(캐피탈원)", "SCHW": "증권·자산관리(찰스슈왑)",
    "BLK": "세계 최대 자산운용(블랙록)", "BX": "사모펀드·대체투자(블랙스톤)", "KKR": "사모펀드(KKR)",
    "APO": "사모펀드·보험(아폴로)", "SPGI": "신용평가·지수(S&P글로벌)", "MCO": "신용평가(무디스)",
    "ICE": "거래소·데이터(인터컨티넨탈익스체인지)", "CME": "파생상품 거래소(CME)", "COIN": "암호화폐 거래소(코인베이스)",
    "AXP": "신용카드·결제(아메리칸익스프레스)", "PGR": "자동차 보험(프로그레시브)", "TRV": "손해보험(트래블러스)",
    "CB": "글로벌 손해보험(처브)", "AIG": "종합 보험(AIG)", "MET": "생명보험(메트라이프)",
    "PRU": "생명보험(푸르덴셜)", "AFL": "보험(애플랙)", "ALL": "자동차·주택 보험(올스테이트)",
    "AON": "보험 중개·컨설팅(에이온)", "MMC": "보험 중개(마쉬맥레넌)", "BRK-B": "버크셔 해서웨이(버핏 지주사)",
    "ABBV": "면역·항암 신약(애브비)", "BMY": "항암·면역 신약(BMS)", "ZTS": "동물 의약품(조에티스)",
    "BSX": "의료기기·심혈관(보스턴사이언티픽)", "MDT": "의료기기(메드트로닉)", "SYK": "정형외과 의료기기(스트라이커)",
    "BDX": "의료기기·진단(벡톤디킨슨)", "EW": "심장 판막(에드워즈라이프사이언스)", "HCA": "병원 운영(HCA헬스케어)",
    "CI": "건강보험(시그나)", "CVS": "약국·건강보험(CVS헬스)", "ELV": "건강보험(엘레번스헬스)",
    "MCK": "의약품 유통(맥케슨)", "COR": "의약품 유통(센코라)", "CAH": "의약품 유통(카디널헬스)",
    "BIIB": "신경질환 신약(바이오젠)", "MRNA": "mRNA 백신·치료제(모더나)",
    "A": "생명과학 분석장비(애질런트)", "RMD": "수면호흡기 의료기기(레스메드)", "WST": "의약품 포장재(웨스트파마)",
    "NOW": "기업용 워크플로 SW(서비스나우)", "SNOW": "클라우드 데이터 분석(스노우플레이크)",
    "MDB": "클라우드 데이터베이스(몽고DB)", "NET": "엣지 네트워크·보안(클라우드플레어)",
    "ZS": "클라우드 보안(지스케일러)", "TEAM": "협업 SW(아틀라시안)", "TTD": "디지털 광고 플랫폼(더트레이드데스크)",
    "HUBS": "마케팅·CRM SW(허브스팟)", "ZM": "화상회의(줌)", "DOCU": "전자서명(도큐사인)",
    "WDAY": "클라우드 인사·재무 SW(워크데이)", "TEL": "커넥터·센서(TE커넥티비티)", "GLW": "광섬유·디스플레이 유리(코닝)",
    "HPQ": "PC·프린터(HP)", "HPE": "기업용 서버·네트워크(HPE)", "NTAP": "데이터 스토리지(넷앱)",
    "STX": "하드디스크 저장장치(씨게이트)", "WDC": "저장장치(웨스턴디지털)", "KEYS": "전자 계측장비(키사이트)",
    "FICO": "신용평가 분석SW(파이코)", "GDDY": "도메인·웹호스팅(고대디)", "FSLR": "태양광 패널(퍼스트솔라)",
    "ENPH": "태양광 인버터(엔페이즈)", "ON": "전력·차량용 반도체(온세미)", "GFS": "반도체 파운드리(글로벌파운드리스)",
    "TER": "반도체 테스트 장비(테러다인)", "SWKS": "무선통신 반도체(스카이웍스)", "QRVO": "RF 반도체(코보)",
    "NEE": "전력·신재생 유틸리티(넥스트에라)", "DUK": "전력 유틸리티(듀크에너지)", "SO": "전력 유틸리티(서던컴퍼니)",
    "D": "전력·가스 유틸리티(도미니언)", "SRE": "전력·가스 유틸리티(셈프라)", "PEG": "전력 유틸리티(PSEG)",
    "ED": "전력 유틸리티(콘에디슨)", "PCG": "캘리포니아 전력(PG&E)", "EIX": "전력 유틸리티(에디슨인터내셔널)",
    "WEC": "전력·가스 유틸리티(WEC에너지)", "AWK": "수도 유틸리티(아메리칸워터웍스)",
    "VST": "전력 발전·소매(비스트라)", "GEV": "전력·에너지 설비(GE버노바)", "PWR": "전력·통신 시공(콴타서비스)",
    "COP": "석유·가스 탐사생산(코노코필립스)", "EOG": "셰일 원유·가스(EOG리소스)",
    "SLB": "유전 서비스(슐럼버거)", "HAL": "유전 서비스(핼리버튼)", "PSX": "정유·석유화학(필립스66)",
    "MPC": "정유(마라톤페트롤리엄)", "VLO": "정유(발레로)", "WMB": "천연가스 파이프라인(윌리엄스)",
    "KMI": "천연가스 파이프라인(킨더모건)", "OKE": "천연가스 미드스트림(원오크)", "OXY": "석유·가스 생산(옥시덴탈)",
    "PLD": "물류창고 리츠(프로로지스)", "AMT": "통신탑 리츠(아메리칸타워)", "EQIX": "데이터센터 리츠(에퀴닉스)",
    "CCI": "통신탑 리츠(크라운캐슬)", "PSA": "셀프스토리지 리츠(퍼블릭스토리지)", "O": "리테일 리츠(리얼티인컴)",
    "WELL": "헬스케어 리츠(웰타워)", "DLR": "데이터센터 리츠(디지털리얼티)", "SPG": "쇼핑몰 리츠(사이먼프로퍼티)",
    "CTVA": "농업 종자·작물보호제(코르테바)", "CSGP": "부동산 데이터(코스타)", "MOH": "메디케이드 건강보험(몰리나)",
    "ROK": "공장 자동화(로크웰)", "BR": "금융 IT 서비스(브로드리지)", "JKHY": "은행 IT 솔루션(잭헨리)",
})

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

# yfinance 자체 섹터 분류(11개) → GICS 표준명. yfinance info["sector"]는 GICS와 이름이 달라 매핑 필수.
YF_SECTOR_TO_GICS = {
    "Technology": "Information Technology",
    "Healthcare": "Health Care",
    "Financial Services": "Financials",
    "Consumer Cyclical": "Consumer Discretionary",
    "Consumer Defensive": "Consumer Staples",
    "Communication Services": "Communication Services",
    "Industrials": "Industrials",
    "Energy": "Energy",
    "Utilities": "Utilities",
    "Real Estate": "Real Estate",
    "Basic Materials": "Materials",
}

WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_WIKI_SECTOR_CACHE = {}   # {ticker(야후형, .→-): GICS 섹터}

def _norm_yf_sector(sec: str) -> str:
    """yfinance/임의 섹터 문자열을 GICS 표준명으로. 못 맞추면 빈 문자열."""
    if not sec:
        return ""
    s = sec.strip()
    g = _norm_sector(s)
    if g in GICS_KR:
        return g
    return YF_SECTOR_TO_GICS.get(s, "")

def fetch_wikipedia_sectors() -> dict:
    """위키피디아 S&P500 표에서 ticker→GICS 섹터를 긁어온다(가장 권위 있는 1차 소스).
    pandas.read_html 우선, 실패 시 requests+정규식 파싱. 티커는 야후형(.→-)으로 정규화.
    네트워크/파싱 실패 시 빈 dict(다른 폴백이 받음). 1회 캐시."""
    global _WIKI_SECTOR_CACHE
    if _WIKI_SECTOR_CACHE:
        return _WIKI_SECTOR_CACHE
    out = {}
    # 1) pandas.read_html (lxml 필요)
    try:
        tables = pd.read_html(WIKI_SP500_URL)
        for t in tables:
            cols = [str(c).strip() for c in t.columns]
            if "Symbol" in cols and "GICS Sector" in cols:
                for _, row in t.iterrows():
                    tic = str(row["Symbol"]).strip().replace(".", "-")
                    sec = _norm_sector(str(row["GICS Sector"]).strip())
                    if tic and sec in GICS_KR:
                        out[tic] = sec
                break
    except Exception as e:
        print(f"[섹터] 위키피디아 read_html 실패: {e}", file=sys.stderr)
    # 2) requests + 정규식 폴백
    if not out and requests is not None:
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            html = requests.get(WIKI_SP500_URL, headers=headers, timeout=30).text
            import re as _re
            # 표 행: <td>...TICKER...</td> ... 이어지는 셀들 중 GICS 섹터
            # 간단 파싱: 'TICKER' 링크 뒤 두 번째 <td> 가 회사명, 세 번째가 GICS 섹터
            for m in _re.finditer(
                r'<td[^>]*>\s*<a[^>]*>([A-Z][A-Z0-9.\-]{0,6})</a>.*?</td>\s*'
                r'<td[^>]*>.*?</td>\s*<td[^>]*>\s*([A-Za-z &]+?)\s*</td>',
                html, _re.S):
                tic = m.group(1).strip().replace(".", "-")
                sec = _norm_sector(m.group(2).strip())
                if tic and sec in GICS_KR:
                    out[tic] = sec
        except Exception as e:
            print(f"[섹터] 위키피디아 requests 파싱 실패: {e}", file=sys.stderr)
    if out:
        print(f"[섹터] 위키피디아에서 {len(out)}종목 GICS 섹터 확보", file=sys.stderr)
    _WIKI_SECTOR_CACHE = out
    return out


# ------------------------- yfinance 유틸 ------------------------
def load_profiles() -> dict:
    """분기 갱신 프로필(sp500_profiles.json) 로드. {meta, sector_briefings, tickers}.
    파일 없으면 빈 구조 반환(기존 KR_DESC/위키 폴백으로 정상 동작)."""
    global _PROFILES_CACHE
    if _PROFILES_CACHE is not None:
        return _PROFILES_CACHE
    data = {"meta": {}, "sector_briefings": {}, "tickers": {}}
    try:
        with open(PROFILES_FILE, encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            data["meta"] = loaded.get("meta", {}) or {}
            data["sector_briefings"] = loaded.get("sector_briefings", {}) or {}
            data["tickers"] = loaded.get("tickers", {}) or {}
            print(f"[프로필] {len(data['tickers'])}종목 로드 "
                  f"(생성 {data['meta'].get('generated','?')})", file=sys.stderr)
    except FileNotFoundError:
        print("[프로필] sp500_profiles.json 없음 → 기존 KR_DESC/위키 폴백 사용", file=sys.stderr)
    except Exception as e:
        print(f"[프로필] 로드 실패({e}) → 폴백 사용", file=sys.stderr)
    _PROFILES_CACHE = data
    return data

def _profile(sym) -> dict:
    return load_profiles()["tickers"].get(sym, {})

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
            "longName": info.get("longName") or info.get("shortName") or sym,
            "industry": info.get("industry") or "", "sector_en": info.get("sector") or "",
            "summary": info.get("longBusinessSummary") or "",
            "roe": info.get("returnOnEquity"), "de": info.get("debtToEquity"),
            "fcf": info.get("freeCashflow"), "rev_growth": info.get("revenueGrowth"),
            "profit_margin": info.get("profitMargins"),
            "marketCap": info.get("marketCap"),
        }
    return out

def fetch_news_flags(symbols: list[str]) -> dict[str, bool]:
    """yfinance 뉴스 필드 존재 여부만 수집(3섹션 트리거용). 실패는 조용히 False."""
    _require_yf()
    flags = {}
    for sym in symbols:
        try:
            news = yf.Ticker(sym).news or []
            flags[sym] = bool(news)
        except Exception:
            flags[sym] = False
    return flags

def download_histories(symbols: list[str], period: str = HISTORY_PERIOD,
                       with_volume: bool = False):
    """종가 시계열 수집. with_volume=True 이면 (종가맵, 거래량맵) 튜플 반환.
    거래량은 같은 yf.download 배치의 Volume 컬럼에서 확보(추가 API 콜 없음).
    기존 호출부(with_volume 미지정)는 종가맵만 반환하므로 동작 불변."""
    _require_yf()
    out = {}
    vol_out = {}
    try:
        data = yf.download(symbols, period=period, interval="1d", auto_adjust=True, group_by="ticker", threads=True, progress=False)
    except:
        data = None

    if data is not None and not data.empty:
        for sym in symbols:
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    if sym not in data.columns.get_level_values(0): continue
                    sub = data[sym]
                    close = sub["Close"]
                    vol = sub["Volume"] if "Volume" in sub.columns else None
                else:
                    close = data["Close"]
                    vol = data["Volume"] if "Volume" in data.columns else None
                close = _clean_close(close)
                if not close.empty: out[sym] = close
                if with_volume and vol is not None:
                    v = _clean_volume(vol)
                    if not v.empty: vol_out[sym] = v
            except: continue

    missing = [s for s in symbols if s not in out]
    for sym in missing:
        try:
            raw = yf.Ticker(sym).history(period=period, interval="1d", auto_adjust=True)
            if raw is not None and not raw.empty and "Close" in raw.columns:
                close = _clean_close(raw["Close"])
                if not close.empty: out[sym] = close
                if with_volume and "Volume" in raw.columns:
                    v = _clean_volume(raw["Volume"])
                    if not v.empty: vol_out[sym] = v
        except: continue
    fresh = _filter_stale(out, MAX_STALE_DAYS)
    if with_volume:
        vol_out = {s: v for s, v in vol_out.items() if s in fresh}
        return fresh, vol_out
    return fresh

def _clean_volume(vol: pd.Series) -> pd.Series:
    s = pd.to_numeric(vol, errors="coerce")
    idx = pd.to_datetime(s.index)
    try: idx = idx.tz_localize(None)
    except: pass
    s.index = idx
    return s.dropna().sort_index()

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
def _isnan(x) -> bool:
    """None 또는 NaN 이면 True (펀더멘털·지표 결측 안전 처리)."""
    if x is None: return True
    try: return bool(np.isnan(x))
    except (TypeError, ValueError): return False

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

FNG_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
_FNG_CACHE = {}

def _fng_rating_kr(score) -> str:
    if score is None or _isnan(score): return "—"
    if score < 25:  return "극단적 공포"
    if score < 45:  return "공포"
    if score <= 55: return "중립"
    if score <= 75: return "탐욕"
    return "극단적 탐욕"

def _fng_parse(data):
    """CNN 응답(JSON dict)에서 score 추출. 형식이 바뀌어도 최대한 견고하게."""
    if not isinstance(data, dict):
        return None
    fg = data.get("fear_and_greed")
    sc = fg.get("score") if isinstance(fg, dict) else None
    if sc is None:
        sc = data.get("score")
    if sc is None:
        return None
    try:
        sc = float(sc)
    except (TypeError, ValueError):
        return None
    return {"score": sc, "rating_kr": _fng_rating_kr(sc)}

def fetch_fear_greed() -> dict:
    """CNN 공포·탐욕 지수(0~100) 조회. {score, rating_kr} 반환.
    CNN dataviz 엔드포인트는 generic User-Agent를 403 차단하므로 '완전한 브라우저 헤더'가 필요하다.
    실패 시 {score: None} (UI에서 '탐욕지수 확보 실패'). HTTP 상태/사유를 로그로 남긴다. 1회 캐시."""
    global _FNG_CACHE
    if _FNG_CACHE:
        return _FNG_CACHE
    out = {"score": None, "rating_kr": "—"}
    if requests is None:
        _FNG_CACHE = out
        return out

    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://edition.cnn.com/markets/fear-and-greed",
        "Origin": "https://edition.cnn.com",
    }
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    urls = [FNG_URL, f"{FNG_URL}/{today}"]
    for url in urls:
        try:
            r = requests.get(url, headers=headers, timeout=20)
            if r.status_code != 200:
                print(f"[탐욕지수] {url} → HTTP {r.status_code}", file=sys.stderr)
                continue
            parsed = _fng_parse(r.json())
            if parsed:
                out = parsed
                print(f"[탐욕지수] {out['score']:.0f} ({out['rating_kr']})", file=sys.stderr)
                break
            print(f"[탐욕지수] {url} → 응답에 score 없음", file=sys.stderr)
        except Exception as e:
            print(f"[탐욕지수] {url} → 오류 {type(e).__name__}: {str(e)[:100]}", file=sys.stderr)
    if out["score"] is None:
        print("[탐욕지수] 확보 실패 — 메일엔 '탐욕지수 확보 실패'로 표시(다른 기능 정상)", file=sys.stderr)
    _FNG_CACHE = out
    return out

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
        "chg_1m": _ret(close, P_1M), "chg_3m": _ret(close, 63), "chg_6m": _ret(close, 126),
        "chg_1y": _ret(close, P_1Y), "chg_3y": _ret(close, P_3Y), "chg_5y": _ret_full(close),
        "high_52w": (float(close.rolling(252, min_periods=60).max().iloc[-1]) if len(close) >= 60 else np.nan),
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
    [하이브리드 — 백테스트 검증: '15~'26 CAGR 17.4%(SPY 13.8%)·MDD -26%·Sharpe 0.95·회전율 156%]
      · 퀄리티 코어(하드): ROE≥15%, FCF 흑자
      · 추세+모멘텀 코어(하드): 200일선 위, (MACD상승 OR 골든크로스), 6개월 모멘텀 양(+), 진입신호 지속
      · 점수: 위험조정 6개월 모멘텀을 '주(主)가중' + 퀄리티 + 밸류는 '소프트 틸트'(작게)
      · PER 절대캡 폐지(이상치만 PER_SANITY로 차단) → 고멀티플 우량 모멘텀주를 배제하지 않음
    """
    try:
        pe = float(pe)
    except (TypeError, ValueError):
        pe = None
    # 이상치만 차단(절대캡 아님). PER 없어도 통과(밸류는 소프트 틸트라 필수 아님).
    if pe is not None and not (0 < pe <= PER_SANITY):
        return None

    # 1. 퀄리티 코어 (하드) — Value Trap 차단
    roe = meta.get("roe")
    fcf = meta.get("fcf")
    qreasons = []
    if roe is not None:
        if roe < 0.15: return None
        qreasons.append(f"ROE {roe*100:.0f}%")
    if fcf is not None:
        if fcf <= 0: return None
        qreasons.append("FCF 흑자")

    # 2. 추세 + 모멘텀 코어 (하드)
    if not ind.get("above_ma200"): return None
    if not (ind.get("macd_up") or ind.get("cross") == "golden"): return None
    if ind.get("entry_streak", 0) < MIN_SIGNAL_DAYS: return None
    mom6 = ind.get("chg_6m")
    if _isnan(mom6) or mom6 <= MOM_MIN_6M: return None          # 6개월 모멘텀 양(+) 필수
    if not _isnan(ind.get("rsi")) and ind["rsi"] > RSI_MAX: return None

    score, reasons = 0.0, []

    # 3. 모멘텀 주가중 (위험조정 6개월 = 6개월수익률 / 연변동성)
    vol = ind.get("vol_ann")
    if not _isnan(vol) and vol > 0:
        rar = mom6 / vol
        if rar > 0:
            score += rar * MOM_WEIGHT
            reasons.append("위험조정 모멘텀 우수")

    # 4. 퀄리티 가점
    if qreasons:
        reasons.append("우량재무(" + ", ".join(qreasons[:2]) + ")")
    if roe is not None and roe > 0:
        score += min(roe, 0.5)

    # 5. 밸류 소프트 틸트(작게) — 싸면 약간 가점, 비싸도 배제하지 않음
    if pe is not None:
        cheap = max(0.0, min(1.0, 1.0 - rel_pe))
        if cheap > 0:
            score += cheap * VALUE_WEIGHT
            reasons.append(f"PER {pe:.1f}")

    # 6. 추세 가점
    if ind.get("cross") == "golden":
        score += 1.0
        reasons.append("골든크로스")
    price, ma20, ma50, ma200 = ind.get("price"), ind.get("ma20"), ind.get("ma50"), ind.get("ma200")
    if all(not _isnan(x) for x in (price, ma20, ma50, ma200)) and price > ma20 > ma50 > ma200:
        score += 1.0
        reasons.append("이동평균 정배열")

    return score, " · ".join(reasons)

# =============== 평일 4섹션: 거래량 지표 / 섹션 선정 ===============
def build_volume_stats(vol_map: dict[str, pd.Series], window: int = 20) -> dict[str, dict]:
    """종목별 거래량 통계: 당일 거래량 / 20일 평균(상대거래량) 계산.
    거래량 결손 종목은 제외(3섹션에서 변동률만으로 평가하는 이중 안전장치)."""
    stats = {}
    for sym, v in vol_map.items():
        v = v.dropna()
        if len(v) < 5:
            continue
        last = float(v.iloc[-1])
        ref = v.iloc[-(window + 1):-1] if len(v) > window else v.iloc[:-1]
        avg = float(ref.mean()) if len(ref) else float("nan")
        rvol = (last / avg) if (avg and not _isnan(avg) and avg > 0) else float("nan")
        stats[sym] = {"vol": last, "avg_vol": avg, "rvol": rvol}
    return stats


def _spy_ret_6m(spy_close: pd.Series) -> float:
    s = _clean_close(spy_close) if spy_close is not None else pd.Series(dtype=float)
    return _ret(s, 126) if len(s) else float("nan")


def pick_trend(ind_map: dict[str, dict], sector_map: dict[str, str],
               spy_close: pd.Series, n: int = SECTION_N) -> list[tuple]:
    """1) 추세 우수 (펀더멘탈 무관).
    조건: 정배열(price>ma20>ma50>ma200) + 현재가 >= 52주고가 x TREND_HIGH_FRAC + 6개월 모멘텀 양(+).
    정렬: 상대강도 = (1+종목6M수익률) / (1+SPY6M수익률).
    반환: [(sym, score=상대강도, reason)]"""
    spy6 = _spy_ret_6m(spy_close)
    scored = []
    for sym, ind in ind_map.items():
        price, ma20, ma50, ma200 = ind.get("price"), ind.get("ma20"), ind.get("ma50"), ind.get("ma200")
        if any(_isnan(x) for x in (price, ma20, ma50, ma200)):
            continue
        if not (price > ma20 > ma50 > ma200):  # 정배열
            continue
        high52 = ind.get("high_52w")
        if _isnan(high52) or high52 <= 0 or price < high52 * TREND_HIGH_FRAC:
            continue
        mom6 = ind.get("chg_6m")
        if _isnan(mom6) or mom6 <= 0:
            continue
        if not _isnan(spy6) and (1.0 + spy6 / 100.0) > 0:
            rs = (1.0 + mom6 / 100.0) / (1.0 + spy6 / 100.0)
        else:
            rs = 1.0 + mom6 / 100.0
        reason = f"정배열 / 52주고가 근접 / 상대강도 {rs:.2f}(SPY 대비)"
        scored.append((sym, float(rs), reason))
    scored.sort(key=lambda x: x[1], reverse=True)
    return pick_with_sector_cap(scored, sector_map, n, max(1, RECO_SECTOR_MAX))


def pick_fundamental(ind_map: dict[str, dict], info: dict[str, dict],
                     sector_map: dict[str, str], n: int = SECTION_N) -> list[tuple]:
    """2) 펀더멘탈 우수 (추세 양호).
    퀄리티 코어: ROE>=15%, FCF 흑자(둘 다 데이터 필수 - 결손 종목 제외).
    추세 요건은 '200일선 위'로만 완화(정배열 불요).
    정렬: ROE 중심 퀄리티 점수(+ 매출성장/이익률 소폭 가점)."""
    scored = []
    for sym, meta in info.items():
        ind = ind_map.get(sym)
        if not ind or not ind.get("above_ma200"):
            continue
        roe = meta.get("roe")
        fcf = meta.get("fcf")
        if _isnan(roe) or _isnan(fcf):      # 펀더멘탈 결손 제외(추측값 금지)
            continue
        if roe < 0.15 or fcf <= 0:
            continue
        score = min(roe, 0.6) * 100.0       # ROE 주가중
        rg = meta.get("rev_growth")
        pm = meta.get("profit_margin")
        bits = [f"ROE {roe*100:.0f}%", "FCF 흑자"]
        if not _isnan(rg):
            score += max(0.0, rg) * 30.0
            bits.append(f"매출성장 {rg*100:.0f}%")
        if not _isnan(pm):
            score += max(0.0, pm) * 20.0
            bits.append(f"이익률 {pm*100:.0f}%")
        reason = "우량재무(" + " / ".join(bits) + ") / 200일선 위"
        scored.append((sym, float(score), reason))
    scored.sort(key=lambda x: x[1], reverse=True)
    return pick_with_sector_cap(scored, sector_map, n, max(1, RECO_SECTOR_MAX))


def pick_notable(ind_map: dict[str, dict], info: dict[str, dict], vol_stats: dict[str, dict],
                 sector_map: dict[str, str], n: int = SECTION_N) -> list[tuple]:
    """3) 주목할 종목 (대형 이슈/변동성).
    트리거(하나 이상): 상대거래량 >= NOTABLE_RVOL / 당일 절대변동률 >= NOTABLE_MOVE / 뉴스 존재(meta['has_news']).
    대상: 시총 상위(NOTABLE_TOPN_MC) 위주. 거래량 결손 종목은 변동률만으로 평가."""
    mc = [(s, m.get("marketCap")) for s, m in info.items() if not _isnan(m.get("marketCap"))]
    mc.sort(key=lambda x: x[1], reverse=True)
    pool = {s for s, _ in mc[:NOTABLE_TOPN_MC]} if mc else set(info.keys())
    if not pool:
        pool = set(ind_map.keys())

    scored = []
    for sym in pool:
        ind = ind_map.get(sym)
        if not ind:
            continue
        meta = info.get(sym, {})
        vs = vol_stats.get(sym, {})
        rvol = vs.get("rvol", float("nan"))
        chg1d = ind.get("chg_1d")
        triggers, score = [], 0.0
        if not _isnan(rvol) and rvol >= NOTABLE_RVOL:
            triggers.append(f"상대거래량 {rvol:.1f}배")
            score += rvol
        if not _isnan(chg1d) and abs(chg1d) >= NOTABLE_MOVE:
            triggers.append(f"당일 {chg1d:+.1f}%")
            score += abs(chg1d) / 5.0
        if meta.get("has_news"):
            triggers.append("뉴스 발생")
            score += 1.0
        if not triggers:
            continue
        vol_note = "" if not _isnan(rvol) else " (거래량 데이터 확보 실패 - 변동률만 평가)"
        reason = " / ".join(triggers) + vol_note
        scored.append((sym, float(score), reason))
    scored.sort(key=lambda x: x[1], reverse=True)
    return pick_with_sector_cap(scored, sector_map, n, max(1, RECO_SECTOR_MAX))


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
    """[하이브리드] '진짜 추세이탈'에만 매도(점검) 신호 — 회전율 통제.
      · 200일선 -EXIT_BUFFER(기본 3%) 아래로 이탈   또는   52주 고점 대비 -TRAIL_STOP(기본 25%)
    사소한 눌림목·50일선 톱질로는 신호를 내지 않는다(원본 대비 회전율 664%→156%로 검증).
    ※ 락업(최소 보유)·연속일 확인은 상태파일(state_prev_list.json)에 보유 시작일을 기록해 적용 권장."""
    exits = []
    for sym in prev_syms:
        if sym in picked_syms: continue
        ind = ind_map.get(sym)
        if not ind:
            exits.append((sym, ind, "데이터 없음(거래정지·신선도 미달 가능) — 점검 필요"))
            continue

        why = []
        price = ind.get("price")
        ma200 = ind.get("ma200")
        high52 = ind.get("high_52w")

        if not _isnan(price) and not _isnan(ma200) and price < ma200 * (1 - EXIT_BUFFER):
            why.append(f"200일선 {EXIT_BUFFER*100:.0f}% 하향 이탈(추세 붕괴)")
        if not _isnan(price) and not _isnan(high52) and high52 > 0 and price < high52 * (1 - TRAIL_STOP):
            why.append(f"52주 고점 대비 -{TRAIL_STOP*100:.0f}% 트레일링 스톱")

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
    print(f"연변동성: {s['vol']*100:.1f}% (SPY {b['vol']*100:.1f}%)")
    print(f"MDD: {s['mdd']*100:.1f}% (SPY {b['mdd']*100:.1f}%)")
    print(f"Sharpe: {s['sharpe']:.2f} (SPY {b['sharpe']:.2f})")
    print(f"연평균 회전율: {res['ann_turnover']*100:.0f}%  평균보유: {res['avg_holdings']:.1f}종목")
    print("=" * 64)
    os.makedirs("output", exist_ok=True)
    try:
        with open("output/backtest.png", "wb") as f: f.write(_backtest_chart(res))
    except Exception as e:
        print(f"[경고] 백테스트 차트 저장 실패: {e}", file=sys.stderr)
    with open("output/backtest.txt", "w", encoding="utf-8") as f:
        f.write(f"{res['start'].date()} ~ {res['end'].date()}\n")
        f.write(f"total {s['total']*100:.1f}% (SPY {b['total']*100:.1f}%) / CAGR {s['cagr']*100:.1f}% "
                f"/ vol {s['vol']*100:.1f}% / MDD {s['mdd']*100:.1f}% / Sharpe {s['sharpe']:.2f}\n")
        for yr, sp, bp, ex in res["annual"]:
            f.write(f"{yr}  {sp:7.1f}  {bp:7.1f}  {ex:7.1f}\n")


# ====================== 일일 추천 파이프라인 ======================
LOCK_UP_DAYS  = int(os.environ.get("LOCK_UP_DAYS", "21"))   # 매도 확정 전 최소 보유(거래일)
MIN_EXIT_DAYS = int(os.environ.get("MIN_EXIT_DAYS", "5"))   # 매도조건 연속 충족 요구일

def load_state() -> dict:
    try:
        with open(STATE_FILE, encoding="utf-8") as f: return json.load(f)
    except Exception:
        return {}

def save_state(state: dict):
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[경고] 상태 저장 실패: {e}", file=sys.stderr)

def _bdays_between(d1: str, d2: str) -> int:
    try: return int(np.busday_count(d1, d2))
    except Exception: return 999

def _kr_sector(sec_en: str) -> str:
    return GICS_KR.get(sec_en, sec_en or "기타")

def _has_final_consonant(word: str) -> bool:
    """단어 마지막 글자에 받침이 있는지. 한글은 유니코드로, 영문/숫자는 발음 끝소리로 근사."""
    if not word:
        return False
    ch = word.strip()[-1]
    code = ord(ch)
    if 0xAC00 <= code <= 0xD7A3:          # 한글 음절: (code-0xAC00) % 28 != 0 이면 받침 있음
        return (code - 0xAC00) % 28 != 0
    low = ch.lower()
    if low.isdigit():
        # 0~9 끝소리: 0(영)2(이)4(사)5(오)9(구)=받침無, 1(일)3(삼)6(육)7(칠)8(팔)=받침有
        return low in "13678"
    if low.isalpha():
        # 영문 알파벳 이름 끝소리 기준(대략): 받침 없는 끝소리 모음/우/이 계열
        no_final = set("abeijko")  # 에이.비.이.제이.케이.오 등(받침 없음)
        return low not in no_final
    return False

def _josa(word: str, with_final: str, without_final: str) -> str:
    """받침 유무에 맞는 조사를 반환. 예: _josa('유지','이','가')='가', _josa('접근','이','가')='이'."""
    return with_final if _has_final_consonant(word) else without_final

def _fmt(x, suf="%") -> str:
    return "—" if (x is None or _isnan(x)) else f"{x:+.1f}{suf}"

def _stock_chart(close: pd.Series, days: int = 252) -> bytes:
    """가격선(굵게) + 50/200일 이동평균선 + 옅은 격자. 추세가 눈에 보이도록 개선.
    (이메일 임베드용 PNG. 백테스트 차트와 별개.)"""
    full = close.dropna()
    c = full.iloc[-days:]
    fig, ax = plt.subplots(figsize=(4.6, 1.55))
    # 가격선
    ax.plot(c.index, c.values, color="#15803d", lw=1.8, zorder=3, label="가격")
    # 이동평균(전체 시계열로 계산 후 표시 구간만 슬라이스 → 끝부분 왜곡 방지)
    if len(full) >= 50:
        ma50 = full.rolling(50).mean().iloc[-days:]
        ax.plot(ma50.index, ma50.values, color="#f59e0b", lw=1.0, alpha=0.9, zorder=2)
    if len(full) >= 200:
        ma200 = full.rolling(200).mean().iloc[-days:]
        ax.plot(ma200.index, ma200.values, color="#9ca3af", lw=1.0, alpha=0.9, zorder=1)
    # 시작/끝 대비 색으로 영역 채움(상승=연초록 / 하락=연빨강)
    try:
        up = float(c.iloc[-1]) >= float(c.iloc[0])
        ax.fill_between(c.index, c.values, c.min(), color="#15803d" if up else "#b91c1c", alpha=0.06, zorder=0)
    except Exception:
        pass
    ax.grid(True, axis="y", alpha=0.18, lw=0.6)
    ax.margins(x=0.01)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    fig.tight_layout(pad=0.2)
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=110); plt.close(fig)
    return buf.getvalue()

# ===================== 모드 분기 / 공통 렌더 =====================
MODE_LABELS = {
    "sections":     "평일 4섹션 추천",
    "top10":        "통합 TOP 10 (휴장 1일차)",
    "sector":       "섹터 브리핑 (휴장 2일차)",
    "fund_top10":   "펀더멘탈 TOP 10 (휴장 3일차)",
    "skip":         "메일 스킵 (휴장 4일+ )",
    "weekly":       "주간 결산 (토요일)",
    "strategy":     "차주 전략 (일요일)",
}

def _regime_line(regime: dict) -> str:
    """시장상태: 단정(위험선호/회피) 대신 객관 지표 2개만 — SPY 200일선 이격도 + 탐욕지수."""
    gap = regime.get("gap_pct")
    if gap is not None and not _isnan(gap):
        pos = "위" if gap >= 0 else "아래"
        spy_part = f"SPY 200일선 {pos}({_fmt(gap)})"
    else:
        spy_part = "SPY 200일선 데이터 확보 실패"
    sc = regime.get("fng_score")
    if sc is not None and not _isnan(sc):
        fng_part = f"탐욕지수 {sc:.0f}/100 ({regime.get('fng_rating','—')})"
    else:
        fng_part = "탐욕지수 확보 실패"
    return f"{spy_part} · {fng_part}"

def _vt_block(regime: dict) -> str:
    if not regime.get("vol_target"):
        return ""
    w = regime.get("eq_exposure", 1.0); rv = regime.get("realized_vol", float("nan"))
    rvs = f"{rv*100:.0f}%" if not _isnan(rv) else "-"
    return ('<div style="background:#fff7ed;border:1px solid #fdba74;padding:8px 10px;'
            'border-radius:8px;margin:8px 0;font-size:13px">'
            f'<b>리스크 다이얼</b> / 권장 주식 노출 <b style="color:#c2410c">{w*100:.0f}%</b> '
            f'(잔여 {(1-w)*100:.0f}% 현금/단기채) / 시장 실현변동성 {rvs} vs 목표 {regime["vol_target"]*100:.0f}%</div>')

def _html_head(mode: str, regime: dict, subtitle: str = "") -> str:
    today = datetime.now(KST).strftime("%Y-%m-%d")
    label = MODE_LABELS.get(mode, mode)
    sub = subtitle or label
    sc = regime.get("fng_score")
    # 알약 색: 탐욕지수 기준(공포=빨강 계열, 탐욕=초록 계열, 중립=회색). 단정 대신 지표 표시.
    if sc is None or _isnan(sc):
        pill_bg, pill_fg = "#f3f4f6", "#374151"
    elif sc < 45:
        pill_bg, pill_fg = "#fee2e2", "#b91c1c"
    elif sc <= 55:
        pill_bg, pill_fg = "#f3f4f6", "#374151"
    else:
        pill_bg, pill_fg = "#dcfce7", "#15803d"
    return (
        '<!doctype html><html><body style="font-family:-apple-system,BlinkMacSystemFont,'
        '\'Apple SD Gothic Neo\',Malgun Gothic,sans-serif;max-width:720px;margin:0 auto;'
        'color:#111827;background:#ffffff;padding:4px">'
        f'<a id="top"></a><h2 style="margin:0 0 4px">📊 S&amp;P 500 데일리 리포트</h2>'
        f'<div style="color:#6b7280;font-size:13px;margin-bottom:8px">{today} 마감 기준 · {sub}</div>'
        f'<div style="margin-bottom:12px"><span style="background:{pill_bg};color:{pill_fg};'
        f'border-radius:6px;padding:3px 10px;font-size:12px;font-weight:700">시장 지표: {_regime_line(regime)}</span></div>'
        + _vt_block(regime))

_HTML_FOOT = ('<div style="color:#999;font-size:11px;margin-top:16px;line-height:1.6">'
              '※ <b>시장 지표</b>는 두 가지 객관 수치만 표시합니다. '
              '(1) <b>SPY 200일선 이격도</b>: SPY 종가가 200일 이동평균선 대비 몇 % 위/아래인지(추세 위치). '
              '(2) <b>탐욕지수</b>: CNN Fear &amp; Greed Index(0~100, 낮을수록 공포·높을수록 탐욕). '
              '시장 방향을 단정하지 않으며 해석은 투자자 판단입니다.<br>'
              '규칙 기반 자동 산출이며 투자 권유가 아닙니다. 데이터: Yahoo Finance / SEC EDGAR / CNN.'
              '</div></body></html>')

SECTOR_TAG_BG = "#eef2ff"; SECTOR_TAG_FG = "#4338ca"

def _sector_tag(sec_kr) -> str:
    if not sec_kr or sec_kr == "기타":
        return ""
    return (f'<span style="background:{SECTOR_TAG_BG};color:{SECTOR_TAG_FG};border-radius:4px;'
            f'padding:1px 6px;font-size:11px;margin-left:4px">{sec_kr}</span>')

def _rank_badge(rank) -> str:
    return (f'<span style="background:#b45309;color:#fff;border-radius:50%;width:22px;height:22px;'
            f'display:inline-block;text-align:center;line-height:22px;font-size:12px;'
            f'margin-right:4px">{rank}</span>')

def _ret_chip(label, val) -> str:
    if val is None or _isnan(val):
        color = "#9ca3af"; txt = "—"
    else:
        color = "#15803d" if val >= 0 else "#b91c1c"; txt = f"{val:+.1f}%"
    return ('<span style="display:inline-block;margin:2px 6px 2px 0;font-size:12px">'
            f'<span style="color:#9ca3af">{label}</span> <b style="color:{color}">{txt}</b></span>')

def _ret_chips(ind) -> str:
    return "".join(_ret_chip(l, ind.get(k)) for l, k in
                   [("1일","chg_1d"),("1주","chg_1w"),("1달","chg_1m"),
                    ("6개월","chg_6m"),("1년","chg_1y"),("3년","chg_3y")])

def _company_name(sym, meta) -> str:
    """회사 풀네임. info의 longName/shortName 우선, 없으면 티커."""
    return meta.get("longName") or meta.get("name") or sym

def _one_liner(sym, meta) -> str:
    """뭐 하는 회사인지 '항상 한글' 한 줄. KR_DESC(엄선) -> 업종 한글명 순.
    영문 사업요약은 노출하지 않는다(가독성). KR_DESC에 없고 업종 한글도 없으면
    업종 영문 그대로(드묾) 또는 '데이터 확보 실패'."""
    prof = _profile(sym)
    if prof.get("one_liner"):
        return prof["one_liner"]
    if sym in KR_DESC:
        return KR_DESC[sym]
    industry = meta.get("industry", "") or ""
    ind_kr = INDUSTRY_KR.get(industry, "")
    if ind_kr:
        return f"{ind_kr} 분야 기업"
    sec_kr = _kr_sector(meta.get("sector_en", "") or "")
    if sec_kr and sec_kr != "기타":
        return f"{sec_kr} 섹터 기업"
    return industry or "데이터 확보 실패"

def _chart_img(hist, sym, images, inline_b64, days=252):
    c = hist.get(sym)
    if c is None or len(c.dropna()) < 20: return ""
    try: png = _stock_chart(c, days=days)
    except Exception: return ""
    style = "display:block;width:100%;max-width:300px;height:auto"
    if inline_b64:
        return (f'<img src="data:image/png;base64,{base64.b64encode(png).decode()}" '
                f'style="{style}">')
    cid = f"chart_{sym}"; images.append((cid, png))
    return f'<img src="cid:{cid}" style="{style}">'

def _badge(label, color):
    return (f'<span style="background:{color};color:#fff;padding:1px 7px;border-radius:8px;'
            f'font-size:11px">{label}</span>')

def _stock_row(sym, reason, ind_map, info, sector_map, hist, images, inline_b64,
               badge="", is_new=False, chart_days=252, rank=None, card_bg="#ffffff",
               card_border="#e5e7eb", reason_label="추천 이유"):
    """v3 카드 스타일: [순위][뱃지][NEW] 티커 (섹터태그) / 풀네임 / 한줄설명 /
    PER·종가 / 변동률 색상칩 / 추천이유 + 우측 추세선 차트(가격+MA)."""
    ind = ind_map.get(sym, {}); meta = info.get(sym, {})
    name = _company_name(sym, meta)
    sec_kr = _kr_sector(sector_map.get(sym, "") or meta.get("sector_en", ""))
    desc = _one_liner(sym, meta)
    pe = meta.get("pe"); pe_s = f"<b>{float(pe):.1f}</b>" if (pe and not _isnan(pe)) else "—"
    price = meta.get("price"); price_s = f" · 종가 <b>${float(price):,.2f}</b>" if (price and not _isnan(price)) else ""
    newb = ' <span style="color:#b91c1c;font-size:11px;font-weight:700">NEW</span>' if is_new else ''
    rb = _rank_badge(rank) if rank is not None else ""
    bd = (badge + " ") if badge else ""
    chart = _chart_img(hist, sym, images, inline_b64, days=chart_days)
    # 상세 설명: 카드 안에 직접 표시(이메일은 앵커 점프가 막혀 '자세히' 링크가 동작하지 않음).
    det = _profile(sym).get("detail", "")
    detail_html = ""
    if det:
        detail_html = (f'<div style="font-size:12px;color:#374151;background:#f8fafc;'
                       f'border-left:3px solid #cbd5e1;border-radius:0 6px 6px 0;'
                       f'padding:8px 10px;margin-top:6px;line-height:1.6">'
                       f'<b style="color:#475569">기업 설명</b> · {det}</div>')
    reason_html = ""
    if reason_label and reason:
        reason_html = (f'<div style="font-size:12px;color:#374151;background:#fef3c7;border-radius:6px;'
                       f'padding:8px;line-height:1.5"><b>{reason_label}</b> · {reason}</div>')
    return (
        f'<tr style="background:{card_bg}">'
        f'<td style="padding:12px 14px;vertical-align:top;width:54%;border-bottom:1px solid {card_border}">'
        f'<div style="font-size:15px">{rb}{bd}{newb}<b>{sym}</b>{_sector_tag(sec_kr)}</div>'
        f'<div style="font-size:12px;color:#6b7280;margin:3px 0 1px">{name}</div>'
        f'<div style="font-size:12px;color:#374151;margin-bottom:6px">{desc}</div>'
        f'<div style="font-size:13px;margin-bottom:6px">PER {pe_s}{price_s}</div>'
        f'<div style="margin-bottom:6px">{_ret_chips(ind)}</div>'
        f'{reason_html}'
        f'{detail_html}'
        f'</td><td style="padding:8px;vertical-align:middle;width:46%;border-bottom:1px solid {card_border}">'
        f'{chart}</td></tr>')

def _section_table(title, note, picks, ind_map, info, sector_map, hist, images, inline_b64,
                   badge="", chart_days=252, card_bg="#ffffff", card_border="#e5e7eb", numbered=True):
    if not picks:
        body = '<div style="color:#9ca3af;font-size:13px;padding:8px 0">조건을 충족하는 종목이 없습니다(데이터 확보 실패 가능).</div>'
    else:
        rows = "".join(_stock_row(s, r, ind_map, info, sector_map, hist, images, inline_b64,
                                  badge=badge, chart_days=chart_days,
                                  rank=(i + 1 if numbered else None),
                                  card_bg=card_bg, card_border=card_border)
                       for i, (s, _sc, r) in enumerate(picks))
        body = (f'<table style="width:100%;border-collapse:collapse;margin:8px 0;'
                f'border:1px solid {card_border};border-radius:8px;overflow:hidden">{rows}</table>')
    return (f'<h3 style="margin:18px 0 2px">{title}</h3>'
            f'<div style="color:#6b7280;font-size:12px;margin-bottom:2px">{note}</div>{body}')

def build_final_summary(sec1, sec2, sec3, regime) -> str:
    """4) 최종 분석/결론: 3섹션 종합 + 시장상태 + 리스크 다이얼 요약(3~5문장)."""
    parts = []
    gap = regime.get("gap_pct")
    sc = regime.get("fng_score")
    bits = []
    if gap is not None and not _isnan(gap):
        bits.append(f"SPY가 200일선 {'위' if gap >= 0 else '아래'}({_fmt(gap)})")
    if sc is not None and not _isnan(sc):
        bits.append(f"탐욕지수 {sc:.0f}({regime.get('fng_rating','—')})")
    parts.append("시장 지표는 " + (", ".join(bits) if bits else "데이터 확보 실패") + " 수준입니다.")
    if sec1:
        parts.append(f"추세 측면에서는 {', '.join(s for s,_,_ in sec1)} 등이 정배열·신고가 부근에서 상대강도 우위를 보입니다.")
    if sec2:
        _n2 = ', '.join(s for s,_,_ in sec2)
        parts.append(f"펀더멘탈 측면에서는 {_n2}{_josa(_n2, '이', '가')} 고ROE·FCF 흑자 등 재무 건전성에서 두드러집니다.")
    if sec3:
        _n3 = ', '.join(s for s,_,_ in sec3)
        parts.append(f"한편 {_n3}{_josa(_n3, '은', '는')} 거래량·변동성 급증으로 단기 이슈를 점검할 필요가 있습니다.")
    else:
        parts.append("당일 거래량·변동성 급증으로 주목할 만한 대형 이슈는 두드러지지 않았습니다.")
    if regime.get("vol_target"):
        w = regime.get("eq_exposure", 1.0)
        parts.append(f"리스크 다이얼 기준 권장 주식 노출은 약 {w*100:.0f}%이며, 나머지는 현금·단기채 비중으로 변동성을 통제하는 것을 권장합니다.")
    elif gap is not None and not _isnan(gap) and gap < 0:
        parts.append("SPY가 200일선 아래에 있어 추세상 신규 진입은 신중할 필요가 있습니다.")
    return " ".join(parts[:5])

# ---- 모드별 렌더 ----
def render_sections(payload, inline_b64=False):
    """평일 4섹션 모드."""
    ind_map, info, sector_map, regime, hist = (payload[k] for k in ("ind_map","info","sector_map","regime","hist"))
    sec1, sec2, sec3 = payload["sec1"], payload["sec2"], payload["sec3"]
    exits = payload.get("exits", [])
    images = []
    html = [_html_head("sections", regime, "4섹션 추천(추세 / 펀더멘탈 / 주목 / 결론)")]
    html.append(_market_overview_block(payload))
    exit_html = ""
    if exits:
        items = "".join(f'<li>{e[0]} - {e[2]}</li>' for e in exits)
        exit_html = (f'<h3 style="color:#b91c1c">매도 검토 {len(exits)}건</h3>'
                     f'<ul style="color:#444;font-size:13px">{items}</ul>')
    html.append(exit_html)
    html.append(_section_table("📈 ① 추세 우수 <span style=\"color:#9ca3af;font-size:12px\">(펀더멘탈 무관)</span>",
        "정배열 + 52주 고가 근접 + 6개월 상대강도(SPY 대비) 상위.",
        sec1, ind_map, info, sector_map, hist, images, inline_b64,
        badge=_badge("추세", "#15803d"), chart_days=252, card_bg="#f6fdf9", card_border="#bbf7d0"))
    html.append(_section_table("💎 ② 펀더멘탈 우수 <span style=\"color:#9ca3af;font-size:12px\">(추세 양호)</span>",
        "ROE≥15% · FCF 흑자 · 200일선 위. 펀더멘탈 결손 종목 제외.",
        sec2, ind_map, info, sector_map, hist, images, inline_b64,
        badge=_badge("펀더멘탈", "#1d4ed8"), chart_days=252, card_bg="#f5f8ff", card_border="#c7d7fe"))
    html.append(_section_table("🔔 ③ 주목할 종목 <span style=\"color:#9ca3af;font-size:12px\">(대형 이슈/변동성)</span>",
        "상대거래량 급증 · 당일 변동률 큼 · 뉴스 발생. 시총 상위 위주.",
        sec3, ind_map, info, sector_map, hist, images, inline_b64,
        badge=_badge("주목", "#c2410c"), chart_days=252, card_bg="#fffaf5", card_border="#fed7aa"))
    summary = build_final_summary(sec1, sec2, sec3, regime)
    html.append('<h3 style="margin-bottom:2px">4) 최종 분석 / 결론</h3>'
                f'<div style="color:#333;font-size:13px;line-height:1.6;background:#f8fafc;'
                f'border-left:3px solid #15803d;padding:8px 12px">{summary}</div>')
    html.append(_HTML_FOOT)
    return "".join(html), images

def _market_overview_block(payload) -> str:
    """분기 프로필의 시장 총평을 상단에 표시(있을 때만)."""
    prof = payload.get("profiles") or load_profiles()
    meta = prof.get("meta", {}) or {}
    ov = meta.get("market_overview", "")
    if not ov:
        return ""
    q = meta.get("quarter", "")
    qlabel = f' <span style="color:#9ca3af;font-size:11px">({q} 분기 총평)</span>' if q else ""
    return (f'<h3 style="margin:14px 0 2px">🧭 시장 총평{qlabel}</h3>'
            f'<div style="font-size:13px;color:#374151;line-height:1.7;background:#f8fafc;'
            f'border-left:3px solid #6b7280;padding:8px 12px;margin-bottom:6px">{ov}</div>')

def _detail_section(syms) -> str:
    """메일 하단 '종목 상세' — 각 종목 detail(주가 추이·원인·전망 한 문단) + 앵커."""
    profs = load_profiles()["tickers"]
    seen, blocks = set(), []
    for s in syms:
        if s in seen:
            continue
        seen.add(s)
        p = profs.get(s, {})
        det = p.get("detail", "")
        if not det:
            continue
        nm = p.get("name") or s
        blocks.append(
            f'<div id="d_{s}" style="padding:10px 0;border-bottom:1px solid #eee">'
            f'<div style="font-weight:700;font-size:14px">{s} · {nm}</div>'
            f'<div style="font-size:13px;color:#374151;line-height:1.7;margin-top:4px">{det}</div>'
            f'<div style="margin-top:4px"><a href="#top" style="color:#9ca3af;font-size:11px;'
            f'text-decoration:none">▲ 맨 위로</a></div></div>')
    if not blocks:
        return ""
    return ('<h3 style="margin:20px 0 4px">📚 종목 상세 <span style="color:#9ca3af;font-size:12px">'
            '(기업명을 누르면 여기로 이동 · 분기 갱신)</span></h3>' + "".join(blocks))

def _sector_missing_note(has_sector) -> str:
    if has_sector:
        return ""
    return ('<div style="background:#fff7ed;border:1px solid #fed7aa;color:#9a3412;'
            'font-size:12px;padding:8px 12px;border-radius:8px;margin:6px 0">'
            '⚠️ 섹터 데이터 확보 실패 — SPY 보유종목 공시에 섹터 정보가 없어 일부 섹터가 비어 있을 수 있습니다.</div>')

def _coverage_note(payload) -> str:
    """섹터 분류율이 임계치 미만일 때만 경고. 평소(높음)엔 빈 문자열."""
    cov = payload.get("coverage")
    if cov is None or cov >= SECTOR_COVERAGE_MIN:
        return ""
    nun = payload.get("unclassified_n", 0)
    return ('<div style="background:#fffbeb;border:1px solid #fcd34d;color:#92400e;'
            'font-size:12px;padding:8px 12px;border-radius:8px;margin:6px 0">'
            f'ℹ️ 섹터 분류율 {cov*100:.0f}% · 미분류 {nun}종목은 섹터 집계에서 제외되었습니다'
            f'(주간 최강 종목 등 다른 섹션엔 정상 포함). 분류율이 계속 낮으면 섹터 매핑 보강을 권장합니다.</div>')

def render_sector_briefing(payload, inline_b64=False):
    rows = payload["sector_rows"]; regime = payload["regime"]
    html = [_html_head("sector", regime, "11개 GICS 섹터 최근 주간 수익률·강약")]
    html.append(_sector_missing_note(payload.get("has_sector", True)))
    html.append(_coverage_note(payload))
    trs = []
    for kr, wk, mo, breadth in rows:
        color = "#15803d" if (not _isnan(wk) and wk >= 0) else ("#b91c1c" if not _isnan(wk) else "#9ca3af")
        bar = ""
        if not _isnan(wk):
            w = min(100, abs(wk) * 8)
            bar = (f'<div style="background:{"#dcfce7" if wk>=0 else "#fee2e2"};height:6px;'
                   f'border-radius:3px;width:{w:.0f}%;margin-top:3px"></div>')
        trs.append('<tr style="background:#ffffff">'
            f'<td style="padding:8px 12px;border-bottom:1px solid #eee">{kr}{bar}</td>'
            f'<td style="padding:8px 12px;border-bottom:1px solid #eee;text-align:right;color:{color};font-weight:700">{_fmt(wk)}</td>'
            f'<td style="padding:8px 12px;border-bottom:1px solid #eee;text-align:right;color:#6b7280">{_fmt(mo)}</td>'
            f'<td style="padding:8px 12px;border-bottom:1px solid #eee;text-align:right;color:#6b7280">{breadth}</td></tr>')
    html.append('<table style="width:100%;border-collapse:collapse;font-size:13px;'
                'border:1px solid #e5e7eb;border-radius:8px;overflow:hidden">'
                '<tr style="color:#6b7280;background:#f9fafb">'
                '<th style="text-align:left;padding:8px 12px">섹터(주간 강도 막대)</th>'
                '<th style="text-align:right;padding:8px 12px">주간</th>'
                '<th style="text-align:right;padding:8px 12px">1개월</th>'
                '<th style="text-align:right;padding:8px 12px">상승종목%</th></tr>'
                + "".join(trs) + '</table>')
    html.append(_HTML_FOOT)
    return "".join(html), []

def render_top10_table(payload, mode, subtitle, inline_b64=False):
    """통합 TOP10 / 펀더멘탈 TOP10 공통 표 렌더(종목 행 재사용)."""
    ind_map, info, sector_map, regime, hist = (payload[k] for k in ("ind_map","info","sector_map","regime","hist"))
    picks = payload["picks"]
    images = []
    html = [_html_head(mode, regime, subtitle)]
    html.append(_market_overview_block(payload))
    if not picks:
        html.append('<div style="color:#9ca3af;font-size:13px">조건을 충족하는 종목이 없습니다(데이터 확보 실패 가능).</div>')
    else:
        n_core = max(1, (len(picks) + 1) // 2)
        rrows = []
        for rank, (s, _sc, r) in enumerate(picks):
            bdg = _badge("코어", "#15803d") if rank < n_core else _badge("관찰", "#6b7280")
            cbg = "#fffdf7" if rank < n_core else "#ffffff"
            rrows.append(_stock_row(s, r, ind_map, info, sector_map, hist, images, inline_b64,
                                    badge=bdg, chart_days=252, rank=rank + 1,
                                    card_bg=cbg, card_border="#fde68a"))
        html.append(f'<h3 style="margin:18px 0 2px">⭐ 상위 {len(picks)}종목 '
                    f'<span style="color:#9ca3af;font-size:13px">(코어=확신 / 관찰=후보)</span></h3>'
                    f'<table style="width:100%;border-collapse:collapse;border:1px solid #fde68a;'
                    f'border-radius:8px;overflow:hidden">{"".join(rrows)}</table>')
    html.append(_HTML_FOOT)
    return "".join(html), images

def _spy_chart_block(spy, images, inline_b64, label="SPY 1년 추세"):
    if spy is None or len(getattr(spy, "dropna", lambda: spy)()) < 20:
        return ""
    img = _chart_img({"_SPY": spy}, "_SPY", images, inline_b64, days=252)
    if not img:
        return ""
    return (f'<div style="margin:8px 0"><div style="color:#666;font-size:12px;margin-bottom:2px">{label}</div>{img}</div>')

def render_weekly(payload, inline_b64=False):
    regime = payload["regime"]
    images = []
    html = [_html_head("weekly", regime, "그 주 SPY 등락 / 주도·부진 섹터 / 주간 최강 종목")]
    html.append(f'<div style="font-size:14px;margin:6px 0">SPY 주간 등락: <b>{_fmt(payload.get("spy_week"))}</b></div>')
    html.append(_spy_chart_block(payload.get("spy"), images, inline_b64))
    html.append(_sector_missing_note(payload.get("has_sector", True)))
    html.append(_coverage_note(payload))
    def _sec_chips(items, pos=True):
        return "".join(
            f'<span style="display:inline-block;background:{"#dcfce7" if pos else "#fee2e2"};'
            f'color:{"#15803d" if pos else "#b91c1c"};border-radius:4px;padding:2px 8px;'
            f'font-size:12px;margin:2px 6px 2px 0">{kr} {_fmt(v)}</span>' for kr, v in items)
    lead_body = _sec_chips(payload["lead"], True) if payload.get("lead") else '<span style="color:#9ca3af;font-size:12px">이번 주 상승 섹터 없음</span>'
    lag_body = _sec_chips(payload["lag"], False) if payload.get("lag") else '<span style="color:#9ca3af;font-size:12px">이번 주 하락 섹터 없음</span>'
    html.append('<div style="font-size:13px;margin:6px 0"><b>주도 섹터(상승)</b><br>' + lead_body + '</div>')
    html.append('<div style="font-size:13px;margin:6px 0"><b>부진 섹터(하락)</b><br>' + lag_body + '</div>')
    picks = payload.get("picks", [])
    if picks:
        rows = "".join(_stock_row(s, "", payload["ind_map"], payload["info"], payload["sector_map"],
                                  payload["hist"], images, inline_b64,
                                  badge=_badge("최강", "#15803d"), chart_days=252, rank=i + 1,
                                  card_bg="#fffdf7", card_border="#fde68a", reason_label=None)
                       for i, (s, _sc, r) in enumerate(picks))
        html.append(f'<h3 style="margin:18px 0 2px">🔝 주간 최강 종목 '
                    f'<span style="color:#9ca3af;font-size:12px">(추천이 아닌 안내)</span></h3>'
                    f'<table style="width:100%;border-collapse:collapse;border:1px solid #fde68a;'
                    f'border-radius:8px;overflow:hidden">{rows}</table>')
    html.append(_HTML_FOOT)
    return "".join(html), images

def render_strategy(payload, inline_b64=False):
    regime = payload["regime"]
    images = []
    html = [_html_head("strategy", regime, "차주 관찰 포인트 (데이터 기반)")]
    html.append(_spy_chart_block(payload.get("spy"), images, inline_b64))
    html.append(_sector_missing_note(payload.get("has_sector", True)))
    if payload.get("watch_sectors"):
        chips = "".join(
            f'<span style="display:inline-block;background:#dcfce7;color:#15803d;border-radius:4px;'
            f'padding:2px 8px;font-size:12px;margin:2px 6px 2px 0">{kr} {_fmt(v)}</span>'
            for kr, v in payload["watch_sectors"])
        html.append('<div style="font-size:13px;margin:6px 0"><b>주목 섹터(직전 주 상대강도)</b><br>' + chips + '</div>')
    picks = payload.get("picks", [])
    if picks:
        rows = "".join(_stock_row(s, "", payload["ind_map"], payload["info"], payload["sector_map"],
                                  payload["hist"], images, inline_b64,
                                  badge=_badge("관찰", "#1d4ed8"), chart_days=252, rank=i + 1,
                                  card_bg="#f5f8ff", card_border="#c7d7fe", reason_label=None)
                       for i, (s, _sc, r) in enumerate(picks))
        html.append(f'<h3 style="margin:18px 0 2px">👀 관찰 종목 '
                    f'<span style="color:#9ca3af;font-size:12px">(추천이 아닌 관찰 안내)</span></h3>'
                    f'<table style="width:100%;border-collapse:collapse;border:1px solid #c7d7fe;'
                    f'border-radius:8px;overflow:hidden">{rows}</table>')
    _gap = regime.get("gap_pct")
    _stance = ("추세 추종 유지" if (_gap is not None and not _isnan(_gap) and _gap >= 0) else "방어적 접근")
    _fng = regime.get("fng_score")
    _fng_s = f" · 탐욕지수 {_fng:.0f}({regime.get('fng_rating','—')})" if (_fng is not None and not _isnan(_fng)) else ""
    html.append('<div style="font-size:13px;color:#444;margin-top:6px">'
                f'관찰 포인트: SPY 200일선 대비 {_fmt(_gap)}{_fng_s} 수준이며, '
                f'추세상 {_stance}{_josa(_stance, "이", "가")} 기본 전제입니다.</div>')
    html.append(_HTML_FOOT)
    return "".join(html), images

def build_html(tiers, exits, ind_map, info, sector_map, regime, hist, inline_b64=False):
    """tiers: list[(sym, score, reason, tier, is_new)] / exits: list[(sym, ind, why)]"""
    images = []
    def chart_tag(sym):
        c = hist.get(sym)
        if c is None or len(c.dropna()) < 20: return ""
        try: png = _stock_chart(c)
        except Exception: return ""
        if inline_b64:
            return (f'<img src="data:image/png;base64,{base64.b64encode(png).decode()}" '
                    f'width="220" height="70" style="display:block">')
        cid = f"chart_{sym}"; images.append((cid, png))
        return f'<img src="cid:{cid}" width="220" height="70" style="display:block">'

    rows = []
    for sym, sc, reason, tier, is_new in tiers:
        ind = ind_map.get(sym, {}); meta = info.get(sym, {})
        name = meta.get("name") or sym
        sec = _kr_sector(sector_map.get(sym, "") or meta.get("sector_en", ""))
        desc = KR_DESC.get(sym, INDUSTRY_KR.get(meta.get("industry", ""), meta.get("industry", "")))
        if tier == "코어":
            badge = '<span style="background:#15803d;color:#fff;padding:1px 7px;border-radius:8px;font-size:11px">코어</span>'
        else:
            badge = '<span style="background:#6b7280;color:#fff;padding:1px 7px;border-radius:8px;font-size:11px">관찰</span>'
        newb = ' <span style="color:#b91c1c;font-size:11px;font-weight:700">NEW</span>' if is_new else ''
        pe = meta.get("pe"); pe_s = f"{float(pe):.1f}" if (pe and not _isnan(pe)) else "—"
        rets = (f"1주 {_fmt(ind.get('chg_1w'))} · 1개월 {_fmt(ind.get('chg_1m'))} · "
                f"6개월 {_fmt(ind.get('chg_6m'))} · 1년 {_fmt(ind.get('chg_1y'))}")
        rows.append(
            '<tr><td style="padding:10px;border-bottom:1px solid #eee;vertical-align:top">'
            f'<div style="font-weight:700">{badge}{newb} {sym} · {name}</div>'
            f'<div style="color:#555;font-size:13px">{sec} · {desc} · PER {pe_s}</div>'
            f'<div style="color:#15803d;font-size:13px;margin-top:3px">{reason}</div>'
            f'<div style="color:#444;font-size:12px;margin-top:3px">{rets}</div>'
            f'</td><td style="padding:10px;border-bottom:1px solid #eee">{chart_tag(sym)}</td></tr>')

    exit_html = ""
    if exits:
        items = "".join(f'<li>{e[0]} — {e[2]}</li>' for e in exits)
        exit_html = (f'<h3 style="color:#b91c1c">⚠️ 매도 검토 {len(exits)}건</h3>'
                     f'<ul style="color:#444;font-size:13px">{items}</ul>')

    reg = "위험선호(Risk-On)" if regime.get("risk_on") else "위험회피(Risk-Off) — 신규 추천 신중"
    gap = regime.get("gap_pct")
    reg_s = f"{reg} · SPY 200일선 대비 {_fmt(gap)}" if (gap is not None and not _isnan(gap)) else reg
    vt_html = ""
    if regime.get("vol_target"):
        w = regime.get("eq_exposure", 1.0); rv = regime.get("realized_vol", float("nan"))
        rvs = f"{rv*100:.0f}%" if not _isnan(rv) else "—"
        vt_html = ('<div style="background:#fff7ed;border:1px solid #fdba74;padding:8px 10px;'
                   'border-radius:8px;margin:8px 0;font-size:13px">'
                   f'<b>🎚️ 리스크 다이얼</b> · 권장 주식 노출 <b style="color:#c2410c">{w*100:.0f}%</b> '
                   f'(잔여 {(1-w)*100:.0f}% 현금/단기채) · 시장 실현변동성 {rvs} vs 목표 {regime["vol_target"]*100:.0f}%</div>')
    today = datetime.now(KST).strftime("%Y-%m-%d")
    html = (
        '<!doctype html><html><body style="font-family:Apple SD Gothic Neo,Malgun Gothic,sans-serif;'
        'max-width:680px;margin:0 auto;color:#222">'
        f'<h2 style="margin-bottom:2px">S&amp;P 500 데일리 추천 · {today}</h2>'
        f'<div style="color:#666;font-size:13px;margin-bottom:12px">하이브리드(퀄리티+모멘텀) · 시장상태: {reg_s}</div>'
        f'{vt_html}'
        f'{exit_html}'
        f'<h3>⭐ 추천 {len(tiers)}종목 <span style="color:#888;font-size:13px">(코어=확신 / 관찰=후보)</span></h3>'
        f'<table style="border-collapse:collapse;width:100%">{"".join(rows)}</table>'
        '<div style="color:#999;font-size:11px;margin-top:16px">규칙 기반 자동 산출 · 투자 권유가 아닙니다. '
        '데이터: Yahoo Finance / SEC EDGAR.</div></body></html>')
    return html, images

def send_email(subject: str, html: str, images) -> bool:
    user = os.environ.get("SMTP_USER"); pw = os.environ.get("SMTP_PASS"); to = os.environ.get("EMAIL_TO")
    if not (user and pw and to):
        print("[정보] SMTP 환경변수 미설정 → 메일 발송 생략(미리보기만 생성)", file=sys.stderr)
        return False
    msg = MIMEMultipart("related")
    msg["Subject"], msg["From"], msg["To"] = subject, user, to
    alt = MIMEMultipart("alternative"); msg.attach(alt)
    alt.attach(MIMEText("HTML 미리보기를 지원하는 메일 클라이언트로 확인하세요.", "plain", "utf-8"))
    alt.attach(MIMEText(html, "html", "utf-8"))
    for cid, png in images:
        img = MIMEImage(png, _subtype="png")
        img.add_header("Content-ID", f"<{cid}>"); img.add_header("Content-Disposition", "inline")
        msg.attach(img)
    try:
        ctx = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ctx) as srv:
            srv.login(user, pw)
            srv.sendmail(user, [x.strip() for x in to.split(",")], msg.as_string())
        print(f"[정보] 메일 발송 완료 → {to}", file=sys.stderr)
        return True
    except Exception as e:
        print(f"[오류] 메일 발송 실패: {e}", file=sys.stderr)
        return False

def suggested_exposure(spy_close):
    """리스크 다이얼: 시장(SPY) 실현변동성 기반 권장 주식 노출 W=min(1, σ_target/σ_realized).
    잔여(1-W)는 현금/단기채 권장. VOL_TARGET<=0 이면 100%(off)."""
    s = pd.to_numeric(spy_close, errors="coerce").dropna()
    r = s.pct_change().dropna()
    if VOL_TARGET <= 0 or len(r) < max(10, VOLTGT_WINDOW // 2):
        return 1.0, float("nan")
    rv = float(r.iloc[-VOLTGT_WINDOW:].std() * np.sqrt(252))
    if not (rv > 0):
        return 1.0, rv
    return float(min(1.0, VOL_TARGET / rv)), rv

def gather_universe_data(with_volume: bool = False) -> dict:
    """전 모드 공용 데이터 수집(1회). 종가·(옵션)거래량·지표·SPY·레짐을 반환.
    info는 비싸므로 여기서 받지 않고 호출부에서 필요한 종목만 조회한다."""
    universe, sector_map = get_sp500()
    # 위키피디아 GICS 섹터로 sector_map 보강(가장 권위 있는 분류). 없으면 SPY 공시값 유지.
    wiki = fetch_wikipedia_sectors()
    if wiki:
        for s in universe:
            w = wiki.get(s)
            if w in GICS_KR:
                sector_map[s] = w
    if with_volume:
        hist, vol = download_histories(universe, with_volume=True)
    else:
        hist, vol = download_histories(universe), {}
    spy = download_histories(["SPY"]).get("SPY")
    regime = market_regime(spy) if spy is not None and not spy.empty else {"risk_on": True, "gap_pct": float("nan")}
    if VOL_TARGET and VOL_TARGET > 0 and spy is not None and not spy.empty:
        w, rv = suggested_exposure(spy)
        regime.update({"vol_target": VOL_TARGET, "eq_exposure": w, "realized_vol": rv})
    ind_map = {}
    for s, c in hist.items():
        ind = compute_indicators(c)
        if ind: ind_map[s] = ind
    fng = fetch_fear_greed()
    regime["fng_score"] = fng.get("score")
    regime["fng_rating"] = fng.get("rating_kr")
    profiles = load_profiles()
    return {"universe": universe, "sector_map": sector_map, "hist": hist, "vol": vol,
            "spy": spy, "regime": regime, "ind_map": ind_map, "wiki_sectors": wiki,
            "profiles": profiles}

def _tech_ok(ind) -> bool:
    return (ind.get("above_ma200") and (ind.get("macd_up") or ind.get("cross") == "golden")
            and ind.get("entry_streak", 0) >= MIN_SIGNAL_DAYS
            and not _isnan(ind.get("chg_6m")) and ind["chg_6m"] > MOM_MIN_6M)

def build_recommendations(data: dict | None = None):
    """기존 하이브리드 통합 추천(휴장 1일차 TOP10 모드에서 재사용).
    data 가 주어지면 공용 데이터를 재사용, 아니면 직접 수집(하위호환)."""
    if data is None:
        data = gather_universe_data()
    sector_map, hist, regime, ind_map = data["sector_map"], data["hist"], data["regime"], data["ind_map"]
    tech_pass = [s for s, ind in ind_map.items() if _tech_ok(ind)]
    print(f"[정보] 기술 사전필터 통과 {len(tech_pass)}종목 -> 펀더멘털 조회", file=sys.stderr)
    info = get_info_for(tech_pass)
    sec_med, gmed = sector_median_pes(info, sector_map)
    scored = []
    for s in tech_pass:
        meta = info.get(s, {}); pe = meta.get("pe")
        ref = sec_med.get(sector_map.get(s, ""), gmed)
        try: rel_pe = float(pe) / ref if (pe and ref and not _isnan(ref) and ref > 0) else 1.0
        except (TypeError, ValueError): rel_pe = 1.0
        r = score_reco(ind_map[s], meta, pe, rel_pe)
        if r: scored.append((s, r[0], r[1]))
    scored.sort(key=lambda x: x[1], reverse=True)
    picked = pick_with_sector_cap(scored, sector_map, RECO_N, RECO_SECTOR_MAX)
    return picked, ind_map, info, sector_map, regime, hist

def _last_data_date(hist: dict) -> str | None:
    """수집된 종가의 최신 거래일(YYYY-MM-DD). 휴장 판정 기준."""
    dates = [c.index[-1] for c in hist.values() if len(c)]
    if not dates: return None
    return max(dates).date().isoformat()

def decide_mode(state: dict, data_date: str | None, now_kst=None, force: str | None = None) -> dict:
    """모드 분기: KST 요일 + 휴장 streak.
    - 토/일 요일 모드가 휴장 streak 보다 우선.
    - 데이터가 직전 실행과 동일하면 휴장(streak+1), 갱신되면 streak=0.
    반환: {mode, streak, reason, data_updated}"""
    now = now_kst or datetime.now(KST)
    wd = now.weekday()  # 0=월 ... 5=토 6=일
    prev_date = state.get("last_data_date")
    data_updated = (data_date is not None and data_date != prev_date)
    streak = 0 if data_updated else int(state.get("holiday_streak", 0)) + 1

    if force:
        return {"mode": force, "streak": streak, "data_updated": data_updated,
                "reason": f"강제 모드 지정(--force-mode {force})"}

    if wd == 5:   # 토요일
        return {"mode": "weekly", "streak": streak, "data_updated": data_updated,
                "reason": "한국시간 토요일 -> 주간 결산"}
    if wd == 6:   # 일요일
        return {"mode": "strategy", "streak": streak, "data_updated": data_updated,
                "reason": "한국시간 일요일 -> 차주 전략"}

    if data_updated:
        return {"mode": "sections", "streak": 0, "data_updated": True,
                "reason": "평일 / 데이터 갱신됨 -> 4섹션 추천"}
    # 휴장(데이터 동일)
    if streak == 1:
        m = "top10";       why = "통합 TOP 10"
    elif streak == 2:
        m = "sector";      why = "섹터 브리핑"
    elif streak == 3:
        m = "fund_top10";  why = "펀더멘탈 TOP 10"
    else:
        m = "skip";        why = "메일 스킵(같은 데이터 반복)"
    return {"mode": m, "streak": streak, "data_updated": False,
            "reason": f"휴장 {streak}일차 -> {why}"}

# ---- 휴장/주말 모드 콘텐츠 빌더 ----
def _resolve_sector(sym, sector_map, info) -> str:
    """종목의 GICS 영문 섹터를 확정(3단 폴백):
      1) sector_map (SPY 공시 + 위키피디아 GICS, gather 단계에서 병합됨)
      2) 위키피디아 캐시 직접 조회(안전망)
      3) yfinance info sector → GICS 매핑(YF_SECTOR_TO_GICS)
    못 맞추면 빈 문자열(섹터 집계에서 제외)."""
    pg = _profile(sym).get("gics_sector", "")
    if pg in GICS_KR:
        return pg
    sec = sector_map.get(sym, "") or ""
    if sec in GICS_KR:
        return sec
    w = _WIKI_SECTOR_CACHE.get(sym, "")
    if w in GICS_KR:
        return w
    if info and sym in info:
        g = _norm_yf_sector(info[sym].get("sector_en", "") or "")
        if g in GICS_KR:
            return g
    return ""

def build_sector_briefing(data: dict, info: dict | None = None) -> dict:
    """11개 GICS 섹터별 최근 주간/1개월 수익률·상승종목 비율.
    sector_map 이 비어 있으면(SPY sector 컬럼 결손) yfinance info sector 로 폴백.
    섹터를 하나도 못 구하면 has_sector=False 로 UI에서 '데이터 확보 실패' 안내."""
    sector_map, ind_map = data["sector_map"], data["ind_map"]
    info = info if info is not None else data.get("info")
    agg = {}
    matched = 0
    total = 0
    unclassified = []
    for sym, ind in ind_map.items():
        total += 1
        sec = _resolve_sector(sym, sector_map, info)
        if sec not in GICS_KR:
            unclassified.append(sym)   # 11개 GICS 어디에도 못 들어간 종목(조용히 제외)
            continue
        matched += 1
        wk, mo = ind.get("chg_1w"), ind.get("chg_1m")
        d = agg.setdefault(sec, {"wk": [], "mo": [], "up": 0, "n": 0})
        if not _isnan(wk): d["wk"].append(wk); d["n"] += 1; d["up"] += 1 if wk > 0 else 0
        if not _isnan(mo): d["mo"].append(mo)
    rows = []
    for sec_en, kr in GICS_KR.items():
        d = agg.get(sec_en)
        if not d or d["n"] == 0:
            rows.append((kr, float("nan"), float("nan"), "-"))
            continue
        wk = float(np.mean(d["wk"])) if d["wk"] else float("nan")
        mo = float(np.mean(d["mo"])) if d["mo"] else float("nan")
        breadth = f"{d['up']/d['n']*100:.0f}%" if d["n"] else "-"
        rows.append((kr, wk, mo, breadth))
    rows.sort(key=lambda r: (-r[1]) if not _isnan(r[1]) else 1e9)
    coverage = (matched / total) if total else 0.0
    if total:
        print(f"[섹터] 분류 {matched}/{total}종목 (분류율 {coverage*100:.0f}%) · 미분류 {len(unclassified)}종목",
              file=sys.stderr)
        if coverage < SECTOR_COVERAGE_MIN and unclassified:
            ex = ", ".join(unclassified[:12]) + ("…" if len(unclassified) > 12 else "")
            print(f"[섹터][경고] 분류율이 낮습니다(<{SECTOR_COVERAGE_MIN*100:.0f}%). 섹터 매핑 점검 필요. 미분류 예: {ex}",
                  file=sys.stderr)
    return {"sector_rows": rows, "regime": data["regime"], "has_sector": matched > 0,
            "total": total, "matched": matched, "unclassified_n": len(unclassified),
            "coverage": coverage}

def build_fundamental_top10(data: dict) -> dict:
    """추세 가중 제외, ROE·FCF·매출성장·이익률 종합점수 줄세우기(가치주 관점)."""
    sector_map, ind_map = data["sector_map"], data["ind_map"]
    # info 후보: 200일선 위 종목으로 한정해 API 콜 절감(가치주도 추세 붕괴주는 제외)
    cand = [s for s, ind in ind_map.items() if ind.get("above_ma200")]
    info = get_info_for(cand)
    scored = []
    for s, meta in info.items():
        roe, fcf = meta.get("roe"), meta.get("fcf")
        if _isnan(roe) or _isnan(fcf) or roe < 0.15 or fcf <= 0:
            continue
        score = min(roe, 0.6) * 100.0
        bits = [f"ROE {roe*100:.0f}%", "FCF 흑자"]
        rg, pm = meta.get("rev_growth"), meta.get("profit_margin")
        if not _isnan(rg): score += max(0.0, rg) * 40.0; bits.append(f"매출성장 {rg*100:.0f}%")
        if not _isnan(pm): score += max(0.0, pm) * 30.0; bits.append(f"이익률 {pm*100:.0f}%")
        pe = meta.get("pe")
        if pe and not _isnan(pe) and 0 < pe <= PER_SANITY:  # 저PER 가치주 소폭 가점
            score += max(0.0, (30.0 - pe)) * 0.3; bits.append(f"PER {float(pe):.1f}")
        scored.append((s, float(score), "가치 종합(" + " / ".join(bits) + ")"))
    scored.sort(key=lambda x: x[1], reverse=True)
    picks = pick_with_sector_cap(scored, sector_map, WEEKLY_TOP_N, RECO_SECTOR_MAX)
    return {"picks": picks, "ind_map": ind_map, "info": info,
            "sector_map": sector_map, "regime": data["regime"], "hist": data["hist"]}

def _sector_match_count(sector_map, ind_map, info=None) -> int:
    return sum(1 for s in ind_map if _resolve_sector(s, sector_map, info) in GICS_KR)

def ensure_sector_info(data: dict, sample: int = 200) -> dict:
    """SPY 보유종목 sector_map 으로 GICS 매칭이 충분치 않으면(컬럼 결손 등)
    yfinance info의 sector를 표본(거래대금/모멘텀 무관, 알파벳 순 상위 sample)으로 보강.
    이미 충분하면 추가 조회 없음(비용 절감)."""
    sector_map, ind_map = data["sector_map"], data["ind_map"]
    info = data.get("info") or {}
    have = _sector_match_count(sector_map, ind_map, info)
    if have >= max(30, len(ind_map) // 4):
        return info  # 이미 충분
    targets = [s for s in sorted(ind_map.keys()) if s not in info][:sample]
    if targets:
        fetched = get_info_for(targets)
        info = {**info, **fetched}
        data["info"] = info
    return info

def _split_lead_lag(valid, n=3):
    """주도(상승)·부진(하락) 섹터를 부호로 분리하고 서로 겹치지 않게 한다.
    - 주도: 수익률 > 0 인 섹터만, 높은 순 최대 n개.
    - 부진: 수익률 < 0 인 섹터만, 낮은 순 최대 n개.
    - 양수/음수가 모자라면 그만큼만(빈 리스트 가능). 0%는 어느 쪽에도 넣지 않음."""
    pos = sorted([(kr, v) for kr, v in valid if v > 0], key=lambda x: x[1], reverse=True)
    neg = sorted([(kr, v) for kr, v in valid if v < 0], key=lambda x: x[1])
    return pos[:n], neg[:n]

def build_weekly_review(data: dict) -> dict:
    """주간 결산: SPY 주간 등락, 주도/부진 섹터, 주간 최강 종목."""
    ind_map, sector_map = data["ind_map"], data["sector_map"]
    spy = data.get("spy")
    spy_week = _ret(_clean_close(spy), P_1W) if (spy is not None and len(spy)) else float("nan")
    sinfo = ensure_sector_info(data)           # 섹터 커버리지 확보(필요시에만 조회)
    brief_res = build_sector_briefing(data, info=sinfo)
    brief = brief_res["sector_rows"]; has_sector = brief_res["has_sector"]
    valid = [(kr, wk) for kr, wk, _mo, _b in brief if not _isnan(wk)]
    lead, lag = _split_lead_lag(valid, n=3)
    movers = [(s, ind.get("chg_1w")) for s, ind in ind_map.items() if not _isnan(ind.get("chg_1w"))]
    movers.sort(key=lambda x: x[1], reverse=True)
    top = movers[:5]
    info = get_info_for([s for s, _ in top])
    info = {**sinfo, **info}
    picks = [(s, v, f"주간 {_fmt(v)} · 주간 최강 종목") for s, v in top]
    return {"spy_week": spy_week, "lead": lead, "lag": lag, "picks": picks, "has_sector": has_sector,
            "coverage": brief_res.get("coverage", 1.0), "unclassified_n": brief_res.get("unclassified_n", 0),
            "sector_rows": brief, "ind_map": ind_map, "info": info, "sector_map": sector_map,
            "hist": data["hist"], "spy": data.get("spy"), "regime": data["regime"]}

def build_next_week_strategy(data: dict) -> dict:
    """차주 전략(데이터 기반): 주목 섹터(직전 주 상대강도), 관찰 종목."""
    wk = build_weekly_review(data)
    ind_map = data["ind_map"]
    # 관찰 종목: 정배열 + 6개월 모멘텀 상위 일부(추세 관점)
    trend = pick_trend(ind_map, data["sector_map"], data.get("spy"), n=5)
    info = get_info_for([s for s, _, _ in trend])
    picks = [(s, sc, r) for s, sc, r in trend]   # (sym, score, reason)
    return {"watch_sectors": wk["lead"], "watch_lag": wk["lag"], "has_sector": wk.get("has_sector", True),
            "picks": picks, "ind_map": ind_map, "info": {**wk.get("info", {}), **info},
            "sector_map": data["sector_map"], "hist": data["hist"],
            "spy": data.get("spy"), "regime": data["regime"]}

def build_sections_payload(data: dict):
    """평일 4섹션: 섹션별 후보 선정 + 펀더멘탈/주목용 info·뉴스·거래량 수집."""
    sector_map, ind_map, hist = data["sector_map"], data["ind_map"], data["hist"]
    spy = data.get("spy")
    vol_stats = build_volume_stats(data.get("vol", {}))

    # 1) 추세 (info 불필요)
    sec1 = pick_trend(ind_map, sector_map, spy)

    # info 대상: 200일선 위(펀더멘탈 후보) + 시총상위 주목 후보 풀. 한 번만 조회.
    above = [s for s, ind in ind_map.items() if ind.get("above_ma200")]
    info = get_info_for(above)

    # 3) 주목: 거래량/변동률/뉴스. 후보(트리거 가능)만 뉴스 조회로 콜 절감.
    pre = []
    for s in info:
        ind = ind_map.get(s, {})
        rvol = vol_stats.get(s, {}).get("rvol", float("nan"))
        chg1d = ind.get("chg_1d")
        if (not _isnan(rvol) and rvol >= NOTABLE_RVOL) or (not _isnan(chg1d) and abs(chg1d) >= NOTABLE_MOVE):
            pre.append(s)
    mc = [(s, info[s].get("marketCap")) for s in pre if not _isnan(info[s].get("marketCap"))]
    mc.sort(key=lambda x: x[1], reverse=True)
    news_targets = [s for s, _ in mc[:30]] or pre[:30]
    for s, has in fetch_news_flags(news_targets).items():
        if s in info: info[s]["has_news"] = has

    sec2 = pick_fundamental(ind_map, info, sector_map)
    sec3 = pick_notable(ind_map, info, vol_stats, sector_map)

    # 표시될 모든 종목(특히 추세 섹션)의 풀네임/설명 보장 — 누락분만 추가 조회
    shown = {s for s, _, _ in (sec1 + sec2 + sec3)}
    missing = [s for s in shown if s not in info]
    if missing:
        info.update(get_info_for(missing))
    return sec1, sec2, sec3, info

def _compute_exits_and_tiers(picked, ind_map, holdings, today):
    """기존 스마트 청산·티어·상태갱신 로직 보존(4섹션/TOP10 모드에서 재사용)."""
    picked_syms = [p[0] for p in picked]
    raw_exits = detect_exits(list(holdings.keys()), ind_map, picked_syms)
    raw_exit_syms = {e[0] for e in raw_exits}
    confirmed = []
    for sym in list(holdings.keys()):
        h = holdings[sym]
        h["exit_streak"] = h.get("exit_streak", 0) + 1 if sym in raw_exit_syms else 0
        held = _bdays_between(h.get("since", today), today)
        if sym not in picked_syms and h["exit_streak"] >= MIN_EXIT_DAYS and held >= LOCK_UP_DAYS:
            why = next((e[2] for e in raw_exits if e[0] == sym), "추세 이탈")
            confirmed.append((sym, ind_map.get(sym), why))
    n_core = max(1, (len(picked) + 1) // 2)
    tiers, new_holdings = [], {}
    for rank, (sym, sc, reason) in enumerate(picked):
        tier = "코어" if rank < n_core else "관찰"
        tiers.append((sym, sc, reason, tier, sym not in holdings))
        new_holdings[sym] = {"since": holdings.get(sym, {}).get("since", today), "exit_streak": 0}
    return confirmed, tiers, new_holdings

def daily_main(no_email: bool = False, force_mode: str | None = None):
    _require_yf()
    today = datetime.now(KST).date().isoformat()
    state = load_state(); holdings = state.get("holdings", {})

    # 4섹션·주목 모드는 거래량이 필요. 일단 거래량 포함으로 1회 수집(공용).
    data = gather_universe_data(with_volume=True)
    data_date = _last_data_date(data["hist"])
    decision = decide_mode(state, data_date, force=force_mode)
    mode, streak, reason = decision["mode"], decision["streak"], decision["reason"]
    print(f"[모드] {reason}", file=sys.stderr)  # 매 실행 첫 줄: 모드 판단 근거

    regime = data["regime"]; sector_map = data["sector_map"]
    ind_map = data["ind_map"]; hist = data["hist"]

    # ---- streak >= 4: 메일 스킵(state는 정상 갱신) ----
    if mode == "skip":
        print(f"[정보] 휴장 {streak}일차 -> 메일 발송 스킵(같은 데이터 반복). state만 갱신.", file=sys.stderr)
        new_state = dict(state)
        new_state.update({"date": today, "last_data_date": data_date, "holiday_streak": streak})
        save_state(new_state)
        return

    os.makedirs("output", exist_ok=True)
    images = []

    if mode == "sections":
        sec1, sec2, sec3, info = build_sections_payload(data)
        merged = [(s, sc, r) for s, sc, r in (sec1 + sec2 + sec3)]
        confirmed, _tiers, new_holdings = _compute_exits_and_tiers(merged, ind_map, holdings, today)
        payload = {"ind_map": ind_map, "info": info, "sector_map": sector_map,
                   "regime": regime, "hist": hist, "sec1": sec1, "sec2": sec2, "sec3": sec3,
                   "exits": confirmed}
        preview, _ = render_sections(payload, inline_b64=True)
        subject = f"[S&P500] {today} 4섹션 추천(추세/펀더멘탈/주목)"
        body_count = len(sec1) + len(sec2) + len(sec3)

    elif mode == "top10":
        picked, ind_map, info, sector_map, regime, hist = build_recommendations(data)
        confirmed, tiers, new_holdings = _compute_exits_and_tiers(picked, ind_map, holdings, today)
        payload = {"picks": picked, "ind_map": ind_map, "info": info,
                   "sector_map": sector_map, "regime": regime, "hist": hist}
        preview, _ = render_top10_table(payload, "top10", "하이브리드(퀄리티+모멘텀) 통합 점수 상위", inline_b64=True)
        subject = f"[S&P500] {today} 통합 TOP{len(picked)} (휴장 1일차)"
        body_count = len(picked)

    elif mode == "fund_top10":
        payload = build_fundamental_top10(data)
        preview, _ = render_top10_table(payload, "fund_top10", "ROE·FCF·매출성장·이익률 종합(가치주 관점)", inline_b64=True)
        subject = f"[S&P500] {today} 펀더멘탈 TOP{len(payload['picks'])} (휴장 3일차)"
        new_holdings = holdings; body_count = len(payload["picks"])

    elif mode == "sector":
        ensure_sector_info(data)
        payload = build_sector_briefing(data, info=data.get("info"))
        preview, _ = render_sector_briefing(payload, inline_b64=True)
        subject = f"[S&P500] {today} 섹터 브리핑 (휴장 2일차)"
        new_holdings = holdings; body_count = len(payload["sector_rows"])

    elif mode == "weekly":
        payload = build_weekly_review(data)
        preview, _ = render_weekly(payload, inline_b64=True)
        subject = f"[S&P500] {today} 주간 결산"
        new_holdings = holdings; body_count = len(payload.get("top_movers", []))

    elif mode == "strategy":
        payload = build_next_week_strategy(data)
        preview, _ = render_strategy(payload, inline_b64=True)
        subject = f"[S&P500] {today} 차주 전략"
        new_holdings = holdings; body_count = len(payload.get("watch_names", []))

    else:
        print(f"[경고] 알 수 없는 모드 {mode} -> 4섹션으로 대체", file=sys.stderr)
        sec1, sec2, sec3, info = build_sections_payload(data)
        payload = {"ind_map": ind_map, "info": info, "sector_map": sector_map,
                   "regime": regime, "hist": hist, "sec1": sec1, "sec2": sec2, "sec3": sec3, "exits": []}
        preview, _ = render_sections(payload, inline_b64=True)
        subject = f"[S&P500] {today} 4섹션 추천"
        new_holdings = holdings; body_count = len(sec1+sec2+sec3); mode = "sections"

    with open("output/email.html", "w", encoding="utf-8") as f: f.write(preview)
    print(f"[정보] 모드={mode} 항목={body_count} 미리보기 output/email.html", file=sys.stderr)

    # 상태 갱신(holdings 로직 보존 + 휴장 streak/마지막 데이터 날짜 기록)
    new_state = {"date": today, "holdings": new_holdings,
                 "last_data_date": data_date, "holiday_streak": streak}
    save_state(new_state)

    if not no_email:
        # 메일용은 CID 인라인(가벼움) — 모드별 렌더 재호출
        html, images = _render_for_email(mode, payload if mode != "sections" else None, data, holdings, today, force_mode)
        send_email(subject, html, images)

def _render_for_email(mode, payload, data, holdings, today, force_mode):
    """메일 발송용 CID 인라인 렌더(미리보기와 동일 내용, inline_b64=False)."""
    if mode == "sections":
        sec1, sec2, sec3, info = build_sections_payload(data)
        merged = [(s, sc, r) for s, sc, r in (sec1 + sec2 + sec3)]
        confirmed, _t, _nh = _compute_exits_and_tiers(merged, data["ind_map"], holdings, today)
        p = {"ind_map": data["ind_map"], "info": info, "sector_map": data["sector_map"],
             "regime": data["regime"], "hist": data["hist"], "sec1": sec1, "sec2": sec2,
             "sec3": sec3, "exits": confirmed}
        return render_sections(p, inline_b64=False)
    if mode == "top10":
        return render_top10_table(payload, "top10", "하이브리드(퀄리티+모멘텀) 통합 점수 상위", inline_b64=False)
    if mode == "fund_top10":
        return render_top10_table(payload, "fund_top10", "ROE·FCF·매출성장·이익률 종합(가치주 관점)", inline_b64=False)
    if mode == "sector":
        ensure_sector_info(data)
        payload = build_sector_briefing(data, info=data.get("info"))
        return render_sector_briefing(payload, inline_b64=False)
    if mode == "weekly":
        return render_weekly(payload, inline_b64=False)
    if mode == "strategy":
        return render_strategy(payload, inline_b64=False)
    return render_sector_briefing(payload, inline_b64=False)

def main():
    ap = argparse.ArgumentParser(description="S&P500 일일 추천 리포트(하이브리드 / 다모드)")
    ap.add_argument("--backtest", action="store_true", help="리포트 대신 백테스트 실행")
    ap.add_argument("--no-email", action="store_true", help="메일 발송 없이 미리보기만 생성")
    ap.add_argument("--force-mode", default=None,
                    choices=["sections", "top10", "sector", "fund_top10", "skip", "weekly", "strategy"],
                    help="휴장/주말을 기다리지 않고 특정 모드를 강제 실행(검증용)")
    args = ap.parse_args()
    if args.backtest:
        backtest_main()
    else:
        daily_main(no_email=args.no_email, force_mode=args.force_mode)

if __name__ == "__main__":
    main()