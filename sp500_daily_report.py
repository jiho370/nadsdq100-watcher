#!/usr/bin/env python3
"""
sp500_daily_report.py  (v4)

매일 S&P 500 구성종목을 분석해 이메일 리포트를 생성/발송한다.
v4 개선점(요청 반영):
  · 백테스트 모드 추가 (--backtest): 기술적 전략을 룩어헤드 없이 과거 검증, SPY 대비 성과 비교
  · 밸류에이션을 '섹터 상대 PER'로 전환 + 퀄리티(ROE·부채·FCF·성장) 필터 결합
  · 추천 종목 섹터 상한(RECO_SECTOR_MAX)으로 집중 위험 완화
  · 신호 지속일(MIN_SIGNAL_DAYS) 요건으로 휘프소(잦은 뒤집힘) 완화
  · 데이터 신선도 검증: 마지막 종가가 오래된 종목 자동 제외
  · 보유/추천 청산 신호 섹션(🚪 EXIT) 추가
  · 상태파일(state) CI 영속화 안내 강화

데이터 : Yahoo Finance(yfinance, 키 불필요) + SPY 보유종목(State Street)
실행   : GitHub Actions cron (매일 1회, 미국장 마감 후)
지표·차트·추천·백테스트는 일봉 종가 기준 자동 계산값이며 투자 권유가 아니다.
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

# 백테스트 설정
BT_REBALANCE   = int(os.environ.get("BT_REBALANCE", "21"))      # 리밸런싱 주기(거래일)
BT_YEARS       = float(os.environ.get("BT_YEARS", "5"))         # 백테스트 기간(년)
BT_TOPK        = int(os.environ.get("BT_TOPK", "20"))           # 매 리밸런싱 보유 종목수(0=전체)
BT_COST_BPS    = float(os.environ.get("BT_COST_BPS", "5"))      # 편입/편출 1회 거래비용(bp)
BT_PER_PROXY   = os.environ.get("BT_PER_PROXY", "0") == "1"     # 현재 PER을 과거에 적용(룩어헤드!) 실험용

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

# yfinance industry(영문) → 한글. KR_DESC에 없는 종목의 한 줄 설명 폴백용.
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

# 주요 종목 한글 한 줄 설명(없으면 INDUSTRY_KR → 야후 영문 industry/summary로 폴백)
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
    "M08": "", "SMCI": "AI 서버(슈퍼마이크로)", "DHR": "생명과학·진단(다나허)",
    "TMO": "생명과학 장비·진단(써모피셔)", "ABT": "의료기기·진단(애벗)",
}

# 내장 폴백 스냅샷 (ticker -> GICS sector). SPY 다운로드 실패 시에만 사용.
SP500_FALLBACK = {
    "MMM":"Industrials", "AOS":"Industrials", "ABT":"Health Care", "ABBV":"Health Care", "ACN":"Information Technology", "ADBE":"Information Technology",
    "AMD":"Information Technology", "AES":"Utilities", "AFL":"Financials", "A":"Health Care", "APD":"Materials", "ABNB":"Consumer Discretionary",
    "AKAM":"Information Technology", "ALB":"Materials", "ARE":"Real Estate", "ALGN":"Health Care", "ALLE":"Industrials", "LNT":"Utilities",
    "ALL":"Financials", "GOOGL":"Communication Services", "GOOG":"Communication Services", "MO":"Consumer Staples", "AMZN":"Consumer Discretionary", "AMCR":"Materials",
    "AEE":"Utilities", "AEP":"Utilities", "AXP":"Financials", "AIG":"Financials", "AMT":"Real Estate", "AWK":"Utilities",
    "AMP":"Financials", "AME":"Industrials", "AMGN":"Health Care", "APH":"Information Technology", "ADI":"Information Technology", "AON":"Financials",
    "APA":"Energy", "APO":"Financials", "AAPL":"Information Technology", "AMAT":"Information Technology", "APP":"Information Technology", "APTV":"Consumer Discretionary",
    "ACGL":"Financials", "ADM":"Consumer Staples", "ARES":"Financials", "ANET":"Information Technology", "AJG":"Financials", "AIZ":"Financials",
    "T":"Communication Services", "ATO":"Utilities", "ADSK":"Information Technology", "ADP":"Industrials", "AZO":"Consumer Discretionary", "AVB":"Real Estate",
    "AVY":"Materials", "AXON":"Industrials", "BKR":"Energy", "BALL":"Materials", "BAC":"Financials", "BAX":"Health Care",
    "BDX":"Health Care", "BRK-B":"Financials", "BBY":"Consumer Discretionary", "TECH":"Health Care", "BIIB":"Health Care", "BLK":"Financials",
    "BX":"Financials", "XYZ":"Financials", "BNY":"Financials", "BA":"Industrials", "BKNG":"Consumer Discretionary", "BSX":"Health Care",
    "BMY":"Health Care", "AVGO":"Information Technology", "BR":"Industrials", "BRO":"Financials", "BF-B":"Consumer Staples", "BLDR":"Industrials",
    "BG":"Consumer Staples", "BXP":"Real Estate", "CHRW":"Industrials", "CDNS":"Information Technology", "CPT":"Real Estate", "CPB":"Consumer Staples",
    "COF":"Financials", "CAH":"Health Care", "CCL":"Consumer Discretionary", "CARR":"Industrials", "CVNA":"Consumer Discretionary", "CASY":"Consumer Staples",
    "CAT":"Industrials", "CBOE":"Financials", "CBRE":"Real Estate", "CDW":"Information Technology", "COR":"Health Care", "CNC":"Health Care",
    "CNP":"Utilities", "CF":"Materials", "CRL":"Health Care", "SCHW":"Financials", "CHTR":"Communication Services", "CVX":"Energy",
    "CMG":"Consumer Discretionary", "CB":"Financials", "CHD":"Consumer Staples", "CIEN":"Information Technology", "CI":"Health Care", "CINF":"Financials",
    "CTAS":"Industrials", "CSCO":"Information Technology", "C":"Financials", "CFG":"Financials", "CLX":"Consumer Staples", "CME":"Financials",
    "CMS":"Utilities", "KO":"Consumer Staples", "CTSH":"Information Technology", "COHR":"Information Technology", "COIN":"Financials", "CL":"Consumer Staples",
    "CMCSA":"Communication Services", "FIX":"Industrials", "CAG":"Consumer Staples", "COP":"Energy", "ED":"Utilities", "STZ":"Consumer Staples",
    "CEG":"Utilities", "COO":"Health Care", "CPRT":"Industrials", "GLW":"Information Technology", "CPAY":"Financials", "CTVA":"Materials",
    "CSGP":"Real Estate", "COST":"Consumer Staples", "CRH":"Materials", "CRWD":"Information Technology", "CCI":"Real Estate", "CSX":"Industrials",
    "CMI":"Industrials", "CVS":"Health Care", "DHR":"Health Care", "DRI":"Consumer Discretionary", "DDOG":"Information Technology", "DVA":"Health Care",
    "DECK":"Consumer Discretionary", "DE":"Industrials", "DELL":"Information Technology", "DAL":"Industrials", "DVN":"Energy", "DXCM":"Health Care",
    "FANG":"Energy", "DLR":"Real Estate", "DG":"Consumer Staples", "DLTR":"Consumer Staples", "D":"Utilities", "DPZ":"Consumer Discretionary",
    "DASH":"Consumer Discretionary", "DOV":"Industrials", "DOW":"Materials", "DHI":"Consumer Discretionary", "DTE":"Utilities", "DUK":"Utilities",
    "DD":"Materials", "ETN":"Industrials", "EBAY":"Consumer Discretionary", "SATS":"Communication Services", "ECL":"Materials", "EIX":"Utilities",
    "EW":"Health Care", "EA":"Communication Services", "ELV":"Health Care", "EME":"Industrials", "EMR":"Industrials", "ETR":"Utilities",
    "EOG":"Energy", "EPAM":"Information Technology", "EQT":"Energy", "EFX":"Industrials", "EQIX":"Real Estate", "EQR":"Real Estate",
    "ERIE":"Financials", "ESS":"Real Estate", "EL":"Consumer Staples", "EG":"Financials", "EVRG":"Utilities", "ES":"Utilities",
    "EXC":"Utilities", "EXE":"Energy", "EXPE":"Consumer Discretionary", "EXPD":"Industrials", "EXR":"Real Estate", "XOM":"Energy",
    "FFIV":"Information Technology", "FDS":"Financials", "FICO":"Information Technology", "FAST":"Industrials", "FRT":"Real Estate", "FDX":"Industrials",
    "FIS":"Financials", "FITB":"Financials", "FSLR":"Information Technology", "FE":"Utilities", "FISV":"Financials", "F":"Consumer Discretionary",
    "FTNT":"Information Technology", "FTV":"Industrials", "FOXA":"Communication Services", "FOX":"Communication Services", "BEN":"Financials", "FCX":"Materials",
    "GRMN":"Consumer Discretionary", "IT":"Information Technology", "GE":"Industrials", "GEHC":"Health Care", "GEV":"Industrials", "GEN":"Information Technology",
    "GNRC":"Industrials", "GD":"Industrials", "GIS":"Consumer Staples", "GM":"Consumer Discretionary", "GPC":"Consumer Discretionary", "GILD":"Health Care",
    "GPN":"Financials", "GL":"Financials", "GDDY":"Information Technology", "GS":"Financials", "HAL":"Energy", "HIG":"Financials",
    "HAS":"Consumer Discretionary", "HCA":"Health Care", "DOC":"Real Estate", "HSIC":"Health Care", "HSY":"Consumer Staples", "HPE":"Information Technology",
    "HLT":"Consumer Discretionary", "HD":"Consumer Discretionary", "HON":"Industrials", "HRL":"Consumer Staples", "HST":"Real Estate", "HWM":"Industrials",
    "HPQ":"Information Technology", "HUBB":"Industrials", "HUM":"Health Care", "HBAN":"Financials", "HII":"Industrials", "IBM":"Information Technology",
    "IEX":"Industrials", "IDXX":"Health Care", "ITW":"Industrials", "INCY":"Health Care", "IR":"Industrials", "PODD":"Health Care",
    "INTC":"Information Technology", "IBKR":"Financials", "ICE":"Financials", "IFF":"Materials", "IP":"Materials", "INTU":"Information Technology",
    "ISRG":"Health Care", "IVZ":"Financials", "INVH":"Real Estate", "IQV":"Health Care", "IRM":"Real Estate", "JBHT":"Industrials",
    "JBL":"Information Technology", "JKHY":"Financials", "J":"Industrials", "JNJ":"Health Care", "JCI":"Industrials", "JPM":"Financials",
    "KVUE":"Consumer Staples", "KDP":"Consumer Staples", "KEY":"Financials", "KEYS":"Information Technology", "KMB":"Consumer Staples", "KIM":"Real Estate",
    "KMI":"Energy", "KKR":"Financials", "KLAC":"Information Technology", "KHC":"Consumer Staples", "KR":"Consumer Staples", "LHX":"Industrials",
    "LH":"Health Care", "LRCX":"Information Technology", "LVS":"Consumer Discretionary", "LDOS":"Industrials", "LEN":"Consumer Discretionary", "LII":"Industrials",
    "LLY":"Health Care", "LIN":"Materials", "LYV":"Communication Services", "LMT":"Industrials", "L":"Financials", "LOW":"Consumer Discretionary",
    "LULU":"Consumer Discretionary", "LITE":"Information Technology", "LYB":"Materials", "MTB":"Financials", "MPC":"Energy", "MAR":"Consumer Discretionary",
    "MRSH":"Financials", "MLM":"Materials", "MAS":"Industrials", "MA":"Financials", "MKC":"Consumer Staples", "MCD":"Consumer Discretionary",
    "MCK":"Health Care", "MDT":"Health Care", "MRK":"Health Care", "META":"Communication Services", "MET":"Financials", "MTD":"Health Care",
    "MGM":"Consumer Discretionary", "MCHP":"Information Technology", "MU":"Information Technology", "MSFT":"Information Technology", "MAA":"Real Estate", "MRNA":"Health Care",
    "TAP":"Consumer Staples", "MDLZ":"Consumer Staples", "MPWR":"Information Technology", "MNST":"Consumer Staples", "MCO":"Financials", "MS":"Financials",
    "MOS":"Materials", "MSI":"Information Technology", "MSCI":"Financials", "NDAQ":"Financials", "NTAP":"Information Technology", "NFLX":"Communication Services",
    "NEM":"Materials", "NWSA":"Communication Services", "NWS":"Communication Services", "NEE":"Utilities", "NKE":"Consumer Discretionary", "NI":"Utilities",
    "NDSN":"Industrials", "NSC":"Industrials", "NTRS":"Financials", "NOC":"Industrials", "NCLH":"Consumer Discretionary", "NRG":"Utilities",
    "NUE":"Materials", "NVDA":"Information Technology", "NVR":"Consumer Discretionary", "NXPI":"Information Technology", "ORLY":"Consumer Discretionary", "OXY":"Energy",
    "ODFL":"Industrials", "OMC":"Communication Services", "ON":"Information Technology", "OKE":"Energy", "ORCL":"Information Technology", "OTIS":"Industrials",
    "PCAR":"Industrials", "PKG":"Materials", "PLTR":"Information Technology", "PANW":"Information Technology", "PSKY":"Communication Services", "PH":"Industrials",
    "PAYX":"Industrials", "PYPL":"Financials", "PNR":"Industrials", "PEP":"Consumer Staples", "PFE":"Health Care", "PCG":"Utilities",
    "PM":"Consumer Staples", "PSX":"Energy", "PNW":"Utilities", "PNC":"Financials", "POOL":"Consumer Discretionary", "PPG":"Materials",
    "PPL":"Utilities", "PFG":"Financials", "PG":"Consumer Staples", "PGR":"Financials", "PLD":"Real Estate", "PRU":"Financials",
    "PEG":"Utilities", "PTC":"Information Technology", "PSA":"Real Estate", "PHM":"Consumer Discretionary", "PWR":"Industrials", "QCOM":"Information Technology",
    "DGX":"Health Care", "Q":"Information Technology", "RL":"Consumer Discretionary", "RJF":"Financials", "RTX":"Industrials", "O":"Real Estate",
    "REG":"Real Estate", "REGN":"Health Care", "RF":"Financials", "RSG":"Industrials", "RMD":"Health Care", "RVTY":"Health Care",
    "HOOD":"Financials", "ROK":"Industrials", "ROL":"Industrials", "ROP":"Information Technology", "ROST":"Consumer Discretionary", "RCL":"Consumer Discretionary",
    "SPGI":"Financials", "CRM":"Information Technology", "SNDK":"Information Technology", "SBAC":"Real Estate", "SLB":"Energy", "STX":"Information Technology",
    "SRE":"Utilities", "NOW":"Information Technology", "SHW":"Materials", "SPG":"Real Estate", "SWKS":"Information Technology", "SJM":"Consumer Staples",
    "SW":"Materials", "SNA":"Industrials", "SOLV":"Health Care", "SO":"Utilities", "LUV":"Industrials", "SWK":"Industrials",
    "SBUX":"Consumer Discretionary", "STT":"Financials", "STLD":"Materials", "STE":"Health Care", "SYK":"Health Care", "SMCI":"Information Technology",
    "SYF":"Financials", "SNPS":"Information Technology", "SYY":"Consumer Staples", "TMUS":"Communication Services", "TROW":"Financials", "TTWO":"Communication Services",
    "TPR":"Consumer Discretionary", "TRGP":"Energy", "TGT":"Consumer Staples", "TEL":"Information Technology", "TDY":"Information Technology", "TER":"Information Technology",
    "TSLA":"Consumer Discretionary", "TXN":"Information Technology", "TPL":"Energy", "TXT":"Industrials", "TMO":"Health Care", "TJX":"Consumer Discretionary",
    "TKO":"Communication Services", "TTD":"Communication Services", "TSCO":"Consumer Discretionary", "TT":"Industrials", "TDG":"Industrials", "TRV":"Financials",
    "TRMB":"Information Technology", "TFC":"Financials", "TYL":"Information Technology", "TSN":"Consumer Staples", "USB":"Financials", "UBER":"Industrials",
    "UDR":"Real Estate", "ULTA":"Consumer Discretionary", "UNP":"Industrials", "UAL":"Industrials", "UPS":"Industrials", "URI":"Industrials",
    "UNH":"Health Care", "UHS":"Health Care", "VLO":"Energy", "VEEV":"Health Care", "VTR":"Real Estate", "VLTO":"Industrials",
    "VRSN":"Information Technology", "VRSK":"Industrials", "VZ":"Communication Services", "VRTX":"Health Care", "VRT":"Industrials", "VTRS":"Health Care",
    "VICI":"Real Estate", "V":"Financials", "VST":"Utilities", "VMC":"Materials", "WRB":"Financials", "GWW":"Industrials",
    "WAB":"Industrials", "WMT":"Consumer Staples", "DIS":"Communication Services", "WBD":"Communication Services", "WM":"Industrials", "WAT":"Health Care",
    "WEC":"Utilities", "WFC":"Financials", "WELL":"Real Estate", "WST":"Health Care", "WDC":"Information Technology", "WY":"Real Estate",
    "WSM":"Consumer Discretionary", "WMB":"Energy", "WTW":"Financials", "WDAY":"Information Technology", "WYNN":"Consumer Discretionary", "XEL":"Utilities",
    "XYL":"Industrials", "YUM":"Consumer Discretionary", "ZBRA":"Information Technology", "ZBH":"Health Care", "ZTS":"Health Care",
}


# ===================== 구성종목(유니버스) =====================
def get_sp500() -> tuple[list[str], dict[str, str]]:
    """SPY ETF 공식 일일 보유종목을 받아 (티커목록, {티커:GICS섹터})를 반환.
    실패하면 내장 스냅샷(SP500_FALLBACK)으로 폴백한다.
    """
    syms, sectors = _fetch_spy_holdings()
    if syms and 400 <= len(syms) <= 520:
        for s in syms:
            sectors.setdefault(s, SP500_FALLBACK.get(s, ""))
        print(f"[정보] SPY 보유종목 {len(syms)}개 로드(공식 일일 데이터)", file=sys.stderr)
        return syms, sectors
    print("[경고] SPY 보유종목 로드 실패/이상 → 내장 스냅샷으로 폴백", file=sys.stderr)
    return list(SP500_FALLBACK.keys()), dict(SP500_FALLBACK)


def _fetch_spy_holdings() -> tuple[list[str], dict[str, str]]:
    if requests is None:
        return [], {}
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                 "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
        r = requests.get(SPY_HOLDINGS_URL, headers=headers, timeout=30)
        r.raise_for_status()
        raw = pd.read_excel(io.BytesIO(r.content), engine="openpyxl", header=None)
    except Exception as e:  # noqa: BLE001
        print(f"[경고] SPY xlsx 다운로드/파싱 실패: {e}", file=sys.stderr)
        return [], {}

    hdr = None
    for i in range(min(15, len(raw))):
        cells = [str(c).strip().lower() for c in raw.iloc[i].tolist()]
        if "ticker" in cells and ("name" in cells or "sector" in cells):
            hdr = i
            break
    if hdr is None:
        return [], {}

    cols = [str(c).strip() for c in raw.iloc[hdr].tolist()]
    df = raw.iloc[hdr + 1:].copy()
    df.columns = cols

    def find_col(*names):
        for n in names:
            for c in cols:
                if c.strip().lower() == n:
                    return c
        return None

    c_tic = find_col("ticker")
    c_sec = find_col("sector")
    if c_tic is None:
        return [], {}

    syms, sectors = [], {}
    for _, row in df.iterrows():
        tic = str(row.get(c_tic, "")).strip()
        if not tic or tic.lower() in ("nan", "-", "cash", "ssga", "uscash"):
            continue
        if not all(ch.isalnum() or ch in ".-" for ch in tic):
            continue
        yh = tic.replace(".", "-")
        if not yh[0].isalpha():
            continue
        sec = str(row.get(c_sec, "")).strip() if c_sec else ""
        if sec.lower() == "nan":
            sec = ""
        if yh not in sectors:
            syms.append(yh)
            sectors[yh] = _norm_sector(sec)
    return syms, sectors


def _norm_sector(sec: str) -> str:
    """SPY 파일의 섹터 표기를 GICS 영문 명칭으로 정규화."""
    if not sec:
        return ""
    s = sec.strip().lower()
    table = {
        "information technology": "Information Technology", "technology": "Information Technology",
        "health care": "Health Care", "healthcare": "Health Care",
        "financials": "Financials", "financial": "Financials",
        "consumer discretionary": "Consumer Discretionary",
        "communication services": "Communication Services", "communication": "Communication Services",
        "industrials": "Industrials", "consumer staples": "Consumer Staples",
        "energy": "Energy", "utilities": "Utilities", "real estate": "Real Estate",
        "materials": "Materials",
    }
    return table.get(s, sec.strip())


# ------------------------- yfinance 유틸 ------------------------
def _require_yf():
    if yf is None:
        raise RuntimeError("yfinance가 설치되어 있지 않습니다. `pip install yfinance` 후 실행하세요.")


def get_info_for(symbols: list[str]) -> dict[str, dict]:
    """PER/가격/이름/업종/요약 + 퀄리티 지표(ROE·부채·FCF·성장)를 .info로 조회(종목당 1콜)."""
    _require_yf()
    out: dict[str, dict] = {}
    for sym in symbols:
        info = None
        for attempt in range(2):
            try:
                info = yf.Ticker(sym).info or {}
                break
            except Exception:  # noqa: BLE001
                if attempt == 0:
                    time.sleep(0.8)
        if info is None:
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
            "industry": info.get("industry") or "",
            "sector_en": info.get("sector") or "",
            "summary": info.get("longBusinessSummary") or "",
            # 퀄리티 원천(없을 수 있음)
            "roe": info.get("returnOnEquity"),
            "de": info.get("debtToEquity"),
            "fcf": info.get("freeCashflow"),
            "rev_growth": info.get("revenueGrowth"),
            "profit_margin": info.get("profitMargins"),
        }
    return out


def download_histories(symbols: list[str], period: str = HISTORY_PERIOD) -> dict[str, pd.Series]:
    """모든 종목 일봉 종가를 배치로 받아 {sym: close}로 반환. 실패 시 개별 폴백.
    이후 _filter_stale()로 신선도가 떨어지는 종목을 제거한다.
    """
    _require_yf()
    out: dict[str, pd.Series] = {}
    try:
        data = yf.download(symbols, period=period, interval="1d", auto_adjust=True,
                           group_by="ticker", threads=True, progress=False)
    except Exception as e:  # noqa: BLE001
        print(f"[경고] 배치 다운로드 실패({e}) → 개별 폴백", file=sys.stderr)
        data = None

    if data is not None and not data.empty:
        for sym in symbols:
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    if sym not in data.columns.get_level_values(0):
                        continue
                    close = data[sym]["Close"]
                else:
                    close = data["Close"]
                close = _clean_close(close)
                if not close.empty:
                    out[sym] = close
            except Exception:  # noqa: BLE001
                continue

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
    return _filter_stale(out, MAX_STALE_DAYS)


def _filter_stale(hist: dict[str, pd.Series], max_stale_days: int) -> dict[str, pd.Series]:
    """유니버스 최신 거래일 기준으로 마지막 종가가 너무 오래된 종목을 제거한다.
    (상장폐지·거래정지 등으로 데이터가 멈춘 종목이 추천에 섞이는 것을 차단)
    """
    if not hist:
        return hist
    last_dates = {s: c.index[-1] for s, c in hist.items() if len(c)}
    if not last_dates:
        return hist
    ref = max(last_dates.values())
    cutoff = ref - pd.Timedelta(days=max_stale_days)
    fresh, dropped = {}, []
    for s, c in hist.items():
        if len(c) and c.index[-1] >= cutoff:
            fresh[s] = c
        else:
            dropped.append(s)
    if dropped:
        print(f"[정보] 신선도 필터: {len(dropped)}개 종목 제외(마지막 거래일이 {max_stale_days}일 이상 과거): "
              f"{', '.join(dropped[:12])}{'...' if len(dropped) > 12 else ''}", file=sys.stderr)
    return fresh


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
def _rsi(close, period=RSI_PERIOD):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    ag = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    al = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = ag / al.replace(0, np.nan)
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
        if prev and not pd.isna(prev):
            return (float(close.iloc[-1]) / float(prev) - 1.0) * 100.0
    return float("nan")


def _ret_full(close):
    c = close.dropna()
    if len(c) >= 2 and c.iloc[0]:
        return (float(c.iloc[-1]) / float(c.iloc[0]) - 1.0) * 100.0
    return float("nan")


def _tech_entry_series(close: pd.Series) -> pd.Series:
    """매 시점의 '기술적 진입 조건' 불리언 시리즈(룩어헤드 없음).
    조건: 종가 > 200일선  AND  MACD 히스토그램 > 0.
    백테스트와 신호 지속일 계산에 공통으로 사용한다.
    """
    ma200 = close.rolling(200).mean()
    _, _, hist = _macd(close)
    cond = (close > ma200) & (hist > 0)
    return cond.fillna(False)


def _signal_streak(cond: pd.Series) -> int:
    """불리언 시리즈 끝에서부터 연속 True 개수."""
    streak = 0
    for v in reversed(cond.values):
        if bool(v):
            streak += 1
        else:
            break
    return streak


def compute_indicators(close):
    if close is None or close.empty:
        return None
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
            if recent.iloc[0] < 0 and recent.iloc[-1] > 0:
                cross = "golden"
            elif recent.iloc[0] > 0 and recent.iloc[-1] < 0:
                cross = "death"
    entry_streak = _signal_streak(_tech_entry_series(close))
    ind = {
        "price": last, "ma20": ma_last[20], "ma50": ma_last[50], "ma200": ma_last[200],
        "above_ma200": (not np.isnan(ma_last[200])) and last > ma_last[200],
        "rsi": rsi_val, "macd": macd_val, "macd_signal": sig_val, "macd_hist": hist_val,
        "macd_up": hist_val > 0, "cross": cross, "entry_streak": entry_streak,
        "chg_1d": _ret(close, 1), "chg_1w": _ret(close, P_1W), "chg_1m": _ret(close, P_1M),
        "chg_3m": _ret(close, 63), "chg_1y": _ret(close, P_1Y), "chg_3y": _ret(close, P_3Y),
        "chg_5y": _ret_full(close),
    }
    ind.update(_classify_trend(close, ma, hist, rsi_series))
    return ind


def _classify_trend(close, ma, hist, rsi_series):
    res = {"reversal": False, "reversal_score": 0.0, "reversal_reason": "",
           "solidified": False, "solidified_score": 0.0, "solidified_reason": ""}
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

    rev_reasons, rev_score = [], 0.0
    if (h.iloc[-1] > 0) and (h.iloc[-5:-1] <= 0).any():
        rev_reasons.append("MACD 히스토그램이 음(-)에서 양(+)으로 전환(상승 모멘텀 발생)")
        rev_score += 2.0
    if not np.isnan(ma20):
        if last > ma20 and (close.iloc[-6:-1].values < ma20s.iloc[-6:-1].values).any():
            rev_reasons.append("종가가 20일 이동평균선을 아래에서 위로 돌파")
            rev_score += 2.0
    if len(rsi) > 6 and rsi_now >= 50 and float(rsi.iloc[-6]) < 50:
        rev_reasons.append("RSI가 50선을 상향 돌파(매수 우위로 전환)")
        rev_score += 1.0
    prior_down = False
    if not np.isnan(ma20) and not np.isnan(ma50) and len(ma20s) > 6 and len(ma50s) > 6:
        prior_down = float(ma20s.iloc[-6]) < float(ma50s.iloc[-6])
    mom_1m = _ret(close, P_1M)
    if not prior_down and not np.isnan(mom_1m) and mom_1m < 0:
        prior_down = True
    if rev_score >= 3.0 and prior_down:
        res["reversal"] = True
        res["reversal_score"] = rev_score
        res["reversal_reason"] = " · ".join(rev_reasons)

    sol_reasons, sol_score = [], 0.0
    aligned = (not np.isnan(ma20) and not np.isnan(ma50) and not np.isnan(ma200)
               and last > ma20 > ma50 > ma200)
    if aligned:
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
        if not np.isnan(ma200) and ma200 > 0:
            sol_reasons.append(f"200일선 대비 +{(last / ma200 - 1) * 100:.0f}%")
        if sol_score >= 3.0 and sol_reasons:
            res["solidified"] = True
            res["solidified_score"] = sol_score
            res["solidified_reason"] = " · ".join(sol_reasons)
    return res


# --------------------- 밸류에이션 · 퀄리티 ----------------------
def sector_median_pes(info: dict[str, dict], sector_map: dict[str, str]) -> tuple[dict, float]:
    """후보들의 PER을 섹터별로 모아 중앙값을 구한다. (섹터 median, 전체 median) 반환.
    표본이 3개 미만인 섹터는 호출부에서 전체 median으로 폴백한다.
    """
    by_sec: dict[str, list[float]] = {}
    allpe: list[float] = []
    for sym, meta in info.items():
        pe = meta.get("pe")
        try:
            pe = float(pe)
        except (TypeError, ValueError):
            continue
        if not (0 < pe <= 200):       # 명백한 이상치 제외
            continue
        sec = sector_map.get(sym, "") or "(기타)"
        by_sec.setdefault(sec, []).append(pe)
        allpe.append(pe)
    med = {sec: float(np.median(v)) for sec, v in by_sec.items()}
    counts = {sec: len(v) for sec, v in by_sec.items()}
    global_med = float(np.median(allpe)) if allpe else float("nan")
    # 표본 부족 섹터는 전체 median으로
    for sec in list(med.keys()):
        if counts[sec] < 3 and not np.isnan(global_med):
            med[sec] = global_med
    return med, global_med


def quality_score(meta: dict) -> tuple[float, list[str]]:
    """ROE·부채·FCF·매출성장으로 0~3점 안팎의 퀄리티 점수와 사유 산출.
    데이터가 없으면 해당 항목은 건너뛴다(불이익 없음).
    """
    s, reasons = 0.0, []
    roe = meta.get("roe")
    de = meta.get("de")
    fcf = meta.get("fcf")
    rg = meta.get("rev_growth")
    pm = meta.get("profit_margin")
    if isinstance(roe, (int, float)) and roe is not None:
        if roe >= 0.20:
            s += 1.2; reasons.append(f"ROE {roe*100:.0f}%(우수)")
        elif roe >= 0.12:
            s += 0.6; reasons.append(f"ROE {roe*100:.0f}%")
        elif roe < 0:
            s -= 0.8; reasons.append("ROE 적자")
    if isinstance(de, (int, float)) and de is not None:
        # yfinance debtToEquity는 보통 % 단위(예: 80 = 0.8배)
        if de < 80:
            s += 0.5; reasons.append("저부채")
        elif de > 200:
            s -= 0.5; reasons.append("고부채 주의")
    if isinstance(fcf, (int, float)) and fcf is not None:
        if fcf > 0:
            s += 0.6; reasons.append("FCF 흑자")
        else:
            s -= 0.6; reasons.append("FCF 적자")
    if isinstance(rg, (int, float)) and rg is not None:
        if rg >= 0.10:
            s += 0.5; reasons.append(f"매출성장 +{rg*100:.0f}%")
        elif rg < 0:
            s -= 0.3
    if isinstance(pm, (int, float)) and pm is not None and pm >= 0.15:
        s += 0.3; reasons.append("고마진")
    return s, reasons


# --------------------- 추천 종목 스코어링 -----------------------
def score_reco(ind: dict, pe, rel_pe: float, qscore: float, qreasons: list[str]) -> tuple[float, str] | None:
    """추천 적격이면 (점수, 한글사유) 반환, 아니면 None.
    필수:
      · PER ∈ (0, RECO_PER_MAX]  (이상치 차단용 절대 상한)
      · 섹터 상대 PER ≤ 1.0  (같은 섹터 중앙값 이하 = 동종 대비 저평가)
      · 200일선 위  AND  (MACD 상승 OR 골든크로스)
      · 기술 진입신호 {MIN_SIGNAL_DAYS}일 이상 지속  (휘프소 방지)
      · 퀄리티 점수 ≥ 0  (적자/고부채 트랩 제거)
    """
    try:
        pe = float(pe)
    except (TypeError, ValueError):
        return None
    if not (0 < pe <= RECO_PER_MAX):
        return None
    if not (rel_pe <= 1.0):
        return None
    if not ind.get("above_ma200"):
        return None
    if not (ind.get("macd_up") or ind.get("cross") == "golden"):
        return None
    if ind.get("entry_streak", 0) < MIN_SIGNAL_DAYS:
        return None
    if qscore < 0:                                   # 밸류 트랩 차단
        return None
    if not _isnan(ind.get("rsi")) and ind["rsi"] > 82:
        return None

    score, reasons = 0.0, []
    # 밸류에이션(섹터 대비 쌀수록 가점)
    cheap = max(0.0, min(1.0, 1.0 - rel_pe))
    score += cheap * 2.5
    reasons.append(f"PER {pe:.1f}(섹터 중앙 대비 {rel_pe:.0%} 수준)")
    # 퀄리티
    if qscore > 0:
        score += min(qscore, 2.5)
        if qreasons:
            reasons.append("퀄리티: " + ", ".join(qreasons[:3]))
    # 추세/모멘텀
    if ind.get("cross") == "golden":
        score += 2.5
        reasons.append("최근 골든크로스(50일선이 200일선 상향 돌파)")
    if ind.get("macd_up"):
        score += 1.5
        reasons.append("MACD가 시그널선 위(상승 모멘텀)")
    price, ma20, ma50, ma200 = ind.get("price"), ind.get("ma20"), ind.get("ma50"), ind.get("ma200")
    if all(not _isnan(x) for x in (price, ma20, ma50, ma200)) and price > ma20 > ma50 > ma200:
        score += 1.5
        reasons.append("20·50·200일선 정배열(상승추세 견고)")
    else:
        reasons.append("주가가 200일선 위(중장기 상승 흐름)")
    es = ind.get("entry_streak", 0)
    if es >= MIN_SIGNAL_DAYS:
        reasons.append(f"진입신호 {es}일 지속")
    rsi = ind.get("rsi")
    if not _isnan(rsi):
        if 50 <= rsi <= 70:
            score += 1.0
            reasons.append(f"RSI {rsi:.0f} 건강한 강세 구간")
        elif 70 < rsi <= 80:
            score += 0.3
            reasons.append(f"RSI {rsi:.0f}(다소 과열)")
        elif rsi > 80:
            score -= 0.6
            reasons.append(f"RSI {rsi:.0f} 과열 주의")
    if not _isnan(ind.get("chg_1m")) and ind["chg_1m"] > 0:
        score += 0.5
    return score, " · ".join(reasons)


def pick_with_sector_cap(scored: list[tuple], sector_map: dict[str, str],
                         n: int, cap: int) -> list[tuple]:
    """점수 내림차순으로 정렬된 [(sym, score, reason), ...]에서
    섹터당 최대 cap개까지만 골라 상위 n개를 반환(집중 위험 완화)."""
    out, per_sec = [], {}
    for sym, sc, reason in scored:
        sec = sector_map.get(sym, "") or "(기타)"
        if per_sec.get(sec, 0) >= cap:
            continue
        out.append((sym, sc, reason))
        per_sec[sec] = per_sec.get(sec, 0) + 1
        if len(out) >= n:
            break
    return out


# --------------------- 청산(EXIT) 신호 --------------------------
def detect_exits(prev_syms: list[str], ind_map: dict[str, dict],
                 picked_syms: list[str]) -> list[tuple]:
    """직전 추천 종목 중 이번에 기술적 조건이 무너진 종목을 EXIT로 표시.
    무너짐 = 200일선 이탈  OR  데드크로스  OR  MACD 하락 전환.
    이번에도 추천에 다시 든 종목은 EXIT에서 제외.
    """
    exits = []
    for sym in prev_syms:
        if sym in picked_syms:
            continue
        ind = ind_map.get(sym)
        if not ind:
            exits.append((sym, ind, "데이터 없음(거래정지·신선도 미달 가능) — 보유 시 점검 필요"))
            continue
        why = []
        if not ind.get("above_ma200"):
            why.append("200일선 하향 이탈")
        if ind.get("cross") == "death":
            why.append("데드크로스(50일선이 200일선 하향 돌파)")
        if not ind.get("macd_up"):
            why.append("MACD 하락 전환")
        if why:
            exits.append((sym, ind, " · ".join(why)))
    return exits


# ============================ 백테스트 ===========================
# 주의: 이 백테스트는 '기술적' 전략(200일선 위 + MACD 히스토그램>0)만 인과적으로 검증한다.
# 롤링평균·EWM·과거수익률은 모두 과거 데이터만 쓰므로 룩어헤드가 없다.
# 반면 PER/퀄리티는 '과거 그 시점'의 값이 필요한데 yfinance 무료 데이터로는
# 현재값만 제공된다. 현재 PER을 과거에 적용하면 룩어헤드 편향이 생기므로 기본 비활성.
# (BT_PER_PROXY=1로 켤 수 있으나, 결과는 실제보다 좋게 나오는 '참고용'임을 명시한다.)

def _align_panel(hist: dict[str, pd.Series], dates: pd.DatetimeIndex) -> pd.DataFrame:
    cols = {}
    for s, c in hist.items():
        cols[s] = c.reindex(dates).astype(float)
    return pd.DataFrame(cols, index=dates)


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def _metrics(equity: pd.Series, periods_per_year: int = 252) -> dict:
    eq = equity.dropna()
    if len(eq) < 2:
        return {"total": float("nan"), "cagr": float("nan"), "vol": float("nan"),
                "mdd": float("nan"), "sharpe": float("nan")}
    rets = eq.pct_change().dropna()
    years = len(eq) / periods_per_year
    total = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1.0) if years > 0 else float("nan")
    vol = float(rets.std() * np.sqrt(periods_per_year))
    sharpe = float(rets.mean() / rets.std() * np.sqrt(periods_per_year)) if rets.std() > 0 else float("nan")
    return {"total": total, "cagr": cagr, "vol": vol, "mdd": _max_drawdown(eq), "sharpe": sharpe}


def run_backtest(hist: dict[str, pd.Series], spy_close: pd.Series,
                 sector_map: dict[str, str], info: dict | None = None) -> dict:
    """기술 전략 vs SPY 매수후보유 백테스트.
    리밸런싱마다 진입조건을 만족하는 종목을 (3개월 모멘텀 상위 BT_TOPK개) 동일가중 보유.
    """
    if spy_close is None or spy_close.empty:
        raise RuntimeError("백테스트에는 SPY 종가가 필요합니다.")
    spy_close = _clean_close(spy_close)

    # 공통 달력 = SPY 거래일 중 최근 BT_YEARS년
    end = spy_close.index[-1]
    start = end - pd.Timedelta(days=int(BT_YEARS * 365.25) + 5)
    dates = spy_close.index[(spy_close.index >= start) & (spy_close.index <= end)]
    if len(dates) < BT_REBALANCE * 3:
        raise RuntimeError("백테스트 기간이 너무 짧습니다(데이터 부족).")

    panel = _align_panel(hist, dates)                 # 가격
    rets = panel.pct_change()                         # 일간 수익률
    mom = panel.pct_change(63)                        # 3개월 모멘텀(인과적)

    # 진입조건 패널(인과적): 종가>200MA AND MACD hist>0  — 전체기간에서 계산 후 윈도우 슬라이스
    entry_full = {}
    for s, c in hist.items():
        entry_full[s] = _tech_entry_series(c)
    entry = pd.DataFrame({s: e.reindex(dates) for s, e in entry_full.items()},
                         index=dates).fillna(False)

    # 선택적 PER 프록시(룩어헤드 — 참고용)
    per_ok = None
    if BT_PER_PROXY and info:
        med, gmed = sector_median_pes(info, sector_map)
        cheap_syms = set()
        for s, meta in info.items():
            try:
                pe = float(meta.get("pe"))
            except (TypeError, ValueError):
                continue
            sec = sector_map.get(s, "") or "(기타)"
            m = med.get(sec, gmed)
            if 0 < pe <= RECO_PER_MAX and not np.isnan(m) and pe <= m:
                cheap_syms.add(s)
        per_ok = cheap_syms
        print("[경고] BT_PER_PROXY=1: 현재 PER을 과거에 적용 — 룩어헤드 편향(결과 과대평가) 참고용",
              file=sys.stderr)

    rebal_idx = list(range(0, len(dates), BT_REBALANCE))
    weights = pd.DataFrame(0.0, index=dates, columns=panel.columns)
    cur_w = pd.Series(0.0, index=panel.columns)
    turnover_on = {}

    for k, i in enumerate(rebal_idx):
        d = dates[i]
        elig = entry.iloc[i]
        cand = [s for s in panel.columns if bool(elig.get(s, False)) and not pd.isna(panel.iloc[i][s])]
        if per_ok is not None:
            cand = [s for s in cand if s in per_ok]
        # 3개월 모멘텀 상위 BT_TOPK
        if cand:
            mser = mom.iloc[i][cand].dropna()
            ranked = list(mser.sort_values(ascending=False).index)
            if BT_TOPK and len(ranked) > BT_TOPK:
                ranked = ranked[:BT_TOPK]
            w = pd.Series(0.0, index=panel.columns)
            if ranked:
                w[ranked] = 1.0 / len(ranked)
        else:
            w = pd.Series(0.0, index=panel.columns)   # 현금
        turnover_on[d] = float((w - cur_w).abs().sum())
        cur_w = w
        nxt = rebal_idx[k + 1] if k + 1 < len(rebal_idx) else len(dates)
        weights.iloc[i:nxt] = w.values

    # 전일 가중치로 당일 수익 실현(룩어헤드 방지)
    w_lag = weights.shift(1).fillna(0.0)
    port_ret = (w_lag * rets).sum(axis=1)
    # 리밸런싱일 거래비용 차감
    cost = pd.Series(0.0, index=dates)
    for d, to in turnover_on.items():
        cost[d] = to * (BT_COST_BPS / 1e4)
    port_ret = port_ret - cost
    port_ret = port_ret.fillna(0.0)

    strat_eq = (1.0 + port_ret).cumprod()
    spy_eq = spy_close.reindex(dates).pct_change().fillna(0.0).add(1).cumprod()

    m_s = _metrics(strat_eq)
    m_b = _metrics(spy_eq)
    # 기간(리밸런싱 주기) 단위 벤치 대비 승률
    rb_dates = [dates[i] for i in rebal_idx]
    s_lv = strat_eq.reindex(rb_dates).values
    b_lv = spy_eq.reindex(rb_dates).values
    wins = 0
    for j in range(1, len(s_lv)):
        sp = s_lv[j] / s_lv[j-1] - 1 if s_lv[j-1] else 0
        bp = b_lv[j] / b_lv[j-1] - 1 if b_lv[j-1] else 0
        if sp > bp:
            wins += 1
    win_rate = wins / max(1, len(s_lv) - 1)
    avg_hold = float((weights > 0).sum(axis=1).reindex(rb_dates).mean())

    return {"strat_eq": strat_eq, "spy_eq": spy_eq, "strat": m_s, "spy": m_b,
            "win_rate": win_rate, "avg_holdings": avg_hold,
            "start": dates[0], "end": dates[-1], "rebalances": len(rebal_idx)}


def _backtest_chart(res: dict) -> bytes:
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ax.plot(res["strat_eq"].index, res["strat_eq"].values, color="#15803d", lw=1.6, label="전략(기술)")
    ax.plot(res["spy_eq"].index, res["spy_eq"].values, color="#6b7280", lw=1.4, label="SPY 매수후보유")
    ax.set_title("백테스트 누적성과 (1.0=시작)", fontsize=10, loc="left")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=8, frameon=False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return buf.getvalue()


def backtest_main():
    _require_yf()
    print(f"[백테스트] 기간≈{BT_YEARS}년 · 리밸런싱 {BT_REBALANCE}거래일 · "
          f"보유 {BT_TOPK or '전체'}종목 · 비용 {BT_COST_BPS:.0f}bp · PER프록시={'ON' if BT_PER_PROXY else 'OFF'}")
    universe, sector_map = get_sp500()
    period = f"{int(BT_YEARS)+2}y"
    hist = download_histories(universe, period=period)
    spy = download_histories(["SPY"], period=period).get("SPY")
    if spy is None:
        raise RuntimeError("SPY 데이터 다운로드 실패")
    info = get_info_for(universe) if BT_PER_PROXY else None
    res = run_backtest(hist, spy, sector_map, info)

    s, b = res["strat"], res["spy"]
    lines = [
        "=" * 64,
        f" 백테스트 결과  {res['start'].date()} ~ {res['end'].date()}  (리밸런싱 {res['rebalances']}회)",
        "=" * 64,
        f"{'지표':<14}{'전략(기술)':>16}{'SPY 보유':>16}",
        "-" * 64,
        f"{'총수익률':<14}{s['total']*100:>14.1f}%{b['total']*100:>15.1f}%",
        f"{'연복리(CAGR)':<14}{s['cagr']*100:>14.1f}%{b['cagr']*100:>15.1f}%",
        f"{'연변동성':<14}{s['vol']*100:>14.1f}%{b['vol']*100:>15.1f}%",
        f"{'최대낙폭(MDD)':<14}{s['mdd']*100:>14.1f}%{b['mdd']*100:>15.1f}%",
        f"{'샤프(rf=0)':<14}{s['sharpe']:>15.2f}{b['sharpe']:>16.2f}",
        "-" * 64,
        f"리밸런싱 단위 벤치 대비 승률: {res['win_rate']*100:.0f}%   평균 보유종목: {res['avg_holdings']:.0f}개",
        "=" * 64,
    ]
    excess = (s["cagr"] - b["cagr"]) * 100 if not (np.isnan(s["cagr"]) or np.isnan(b["cagr"])) else float("nan")
    verdict = ("✅ 벤치(SPY) 대비 초과수익" if excess > 0 else "⚠️ 벤치(SPY)에 미달")
    lines.append(f"판정: {verdict}  (CAGR 차이 {excess:+.1f}%p)")
    if not BT_PER_PROXY:
        lines.append("주의: PER/퀄리티는 과거 시점 데이터 부재로 미검증(기술 전략만 검증). 실제 추천엔 PER·퀄리티가 추가됨.")
    else:
        lines.append("주의: PER프록시는 룩어헤드 편향 — 실제보다 좋게 나오는 참고용 결과.")
    report = "\n".join(lines)
    print(report)

    os.makedirs("output", exist_ok=True)
    with open("output/backtest.txt", "w", encoding="utf-8") as f:
        f.write(report + "\n")
    with open("output/backtest.png", "wb") as f:
        f.write(_backtest_chart(res))
    print("\n저장: output/backtest.txt · output/backtest.png")
    return res


# --------------------- 차트 이미지 생성 -------------------------
def make_chart_png(close, days, ma_windows=(50, 200), figsize=(6.0, 1.85), up=True, title=None):
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
# [중요] GitHub Actions 러너는 매 실행마다 초기화된다. 이 상태파일을 영속화하지 않으면
#        load_prev_list()가 항상 빈 리스트를 반환해 '신규/이탈/EXIT'가 무의미해진다.
#        워크플로에 actions/cache 또는 commit-back 단계를 반드시 추가할 것(파일 하단 안내 참조).
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
def _isnan(x):
    return x is None or (isinstance(x, float) and np.isnan(x))


def _pct(x):
    return "—" if _isnan(x) else f"{x:+.2f}%"


def _money(x):
    return "—" if _isnan(x) else f"${x:,.2f}"


def _rsi_zone(r):
    if _isnan(r):
        return ""
    if r >= 70:
        return "과매수"
    if r <= 30:
        return "과매도"
    return ""


def _color_chg(c):
    return "#9ca3af" if _isnan(c) else ("#15803d" if c >= 0 else "#b91c1c")


def sector_kr(sym, sector_map, info):
    en = sector_map.get(sym) or (info.get(sym, {}) or {}).get("sector_en") or ""
    return GICS_KR.get(en, en or "—")


def desc_of(sym, info, sector_map=None):
    """종목 한 줄 설명. KR_DESC → INDUSTRY_KR(업종 한글) → 섹터 한글 폴백 순."""
    if sym in KR_DESC and KR_DESC[sym]:
        return KR_DESC[sym]
    meta = info.get(sym, {}) or {}
    ind = meta.get("industry") or ""
    summ = meta.get("summary") or ""
    if ind and ind in INDUSTRY_KR:
        return INDUSTRY_KR[ind] + " 기업"
    if ind:
        return ind  # 매핑 없으면 영문 업종이라도
    if summ:
        return summ.split(". ")[0][:80]
    if sector_map is not None:
        sec_en = sector_map.get(sym) or meta.get("sector_en") or ""
        sec_kr = GICS_KR.get(sec_en, sec_en)
        if sec_kr:
            return f"{sec_kr} 관련 기업"
    return ""


# ------------------------- 카드/본문 ----------------------------
def _ret_chip(label, val):
    color = _color_chg(val)
    txt = "—" if _isnan(val) else f"{val:+.1f}%"
    return (f'<span style="display:inline-block;margin:2px 6px 2px 0;font-size:12px">'
            f'<span style="color:#9ca3af">{label}</span> <b style="color:{color}">{txt}</b></span>')


def _badges(ind):
    out = []
    if ind.get("above_ma200") is not None:
        out.append('<span style="background:#dcfce7;color:#15803d;border-radius:4px;padding:1px 6px;font-size:11px">200일선 위</span>'
                   if ind["above_ma200"] else
                   '<span style="background:#fee2e2;color:#b91c1c;border-radius:4px;padding:1px 6px;font-size:11px">200일선 아래</span>')
    rsi = ind.get("rsi")
    if not _isnan(rsi):
        z = _rsi_zone(rsi)
        zt = f" {z}" if z else ""
        out.append(f'<span style="background:#f3f4f6;color:#374151;border-radius:4px;padding:1px 6px;font-size:11px">RSI {rsi:.0f}{zt}</span>')
    if ind.get("macd_up") is not None:
        out.append('<span style="background:#dcfce7;color:#15803d;border-radius:4px;padding:1px 6px;font-size:11px">MACD ＋</span>'
                   if ind["macd_up"] else
                   '<span style="background:#fee2e2;color:#b91c1c;border-radius:4px;padding:1px 6px;font-size:11px">MACD －</span>')
    if ind.get("cross") == "golden":
        out.append('<span style="background:#fef9c3;color:#a16207;border-radius:4px;padding:1px 6px;font-size:11px">골든크로스</span>')
    elif ind.get("cross") == "death":
        out.append('<span style="background:#fee2e2;color:#b91c1c;border-radius:4px;padding:1px 6px;font-size:11px">데드크로스</span>')
    return " ".join(out)


def _reco_card(rank, sym, name, per, ind, cid, sector, desc, reason):
    per_txt = "—" if _isnan(per) else f"{float(per):.1f}"
    rank_html = (f'<span style="background:#b45309;color:#fff;border-radius:50%;width:22px;height:22px;'
                 f'display:inline-block;text-align:center;line-height:22px;font-size:12px">{rank}</span> ')
    rets = (_ret_chip("1일", ind.get("chg_1d")) + _ret_chip("1주", ind.get("chg_1w"))
            + _ret_chip("1달", ind.get("chg_1m")) + _ret_chip("1년", ind.get("chg_1y"))
            + _ret_chip("3년", ind.get("chg_3y")) + _ret_chip("5년", ind.get("chg_5y")))
    return f"""
    <table style="width:100%;border-collapse:collapse;margin:10px 0;border:1px solid #fde68a;border-radius:8px;background:#fffdf7"><tr>
      <td style="padding:12px 14px;vertical-align:top;width:52%">
        <div style="font-size:15px">{rank_html}<b>{sym}</b>
          <span style="background:#eef2ff;color:#4338ca;border-radius:4px;padding:1px 6px;font-size:11px;margin-left:4px">{sector}</span></div>
        <div style="font-size:12px;color:#6b7280;margin:3px 0 1px">{name}</div>
        <div style="font-size:12px;color:#374151;margin-bottom:6px">{desc}</div>
        <div style="font-size:13px;margin-bottom:6px">PER <b>{per_txt}</b> <span style="color:#9ca3af">·</span> 종가 <b>{_money(ind.get('price'))}</b></div>
        <div style="margin-bottom:6px">{rets}</div>
        <div style="margin-bottom:6px">{_badges(ind)}</div>
        <div style="font-size:12px;color:#374151;background:#fef3c7;border-radius:6px;padding:8px;line-height:1.6"><b>추천 이유</b> · {reason}</div>
      </td>
      <td style="padding:8px;vertical-align:middle;width:48%">
        <img src="cid:{cid}" alt="{sym}" style="width:100%;max-width:340px;height:auto;display:block"/></td>
    </tr></table>"""


def _trend_card(sym, name, ind, cid, sector, desc, reason):
    rets = (_ret_chip("1주", ind.get("chg_1w")) + _ret_chip("1달", ind.get("chg_1m"))
            + _ret_chip("3달", ind.get("chg_3m")) + _ret_chip("1년", ind.get("chg_1y")))
    return f"""
    <table style="width:100%;border-collapse:collapse;margin:10px 0;border:1px solid #eef0f3;border-radius:8px"><tr>
      <td style="padding:10px;vertical-align:middle;width:46%">
        <img src="cid:{cid}" alt="{sym}" style="width:100%;max-width:320px;height:auto;display:block"/></td>
      <td style="padding:12px 14px;vertical-align:top;width:54%">
        <div style="font-size:15px"><b>{sym}</b>
          <span style="background:#eef2ff;color:#4338ca;border-radius:4px;padding:1px 6px;font-size:11px;margin-left:4px">{sector}</span></div>
        <div style="font-size:12px;color:#6b7280;margin:2px 0">{name} · {desc}</div>
        <div style="margin:6px 0">{rets}</div>
        <div style="font-size:12px;color:#374151;background:#f9fafb;border-radius:6px;padding:8px;line-height:1.6"><b>분석</b> · {reason}</div>
      </td></tr></table>"""


def _exit_row_html(sym, name, sector, desc, reason):
    desc_part = f' <span style="color:#374151;font-size:12px">— {desc}</span>' if desc else ""
    return (f'<tr><td style="padding:8px 10px;border-bottom:1px solid #f3f4f6;font-size:13px">'
            f'<b>{sym}</b> <span style="color:#9ca3af;font-size:11px">{sector}</span> '
            f'<span style="color:#6b7280;font-size:12px">{name}</span>{desc_part}</td>'
            f'<td style="padding:8px 10px;border-bottom:1px solid #f3f4f6;font-size:12px;color:#b91c1c">{reason}</td></tr>')


def build_commentary(reco_rows, reversal_rows, solid_rows, exit_rows,
                     ind_map, sector_map, info, asof):
    """그날 종목들을 데이터로 분석하는 2문단 총평(HTML, 텍스트) 생성."""
    from collections import Counter

    # 시장 폭(breadth): 유니버스 중 200일선 위 비중
    n_uni = len(ind_map)
    above = sum(1 for i in ind_map.values() if i.get("above_ma200"))
    breadth = (above / n_uni * 100) if n_uni else float("nan")
    if _isnan(breadth):
        tone = "시장 폭을 계산하기 어려운 상태"
    elif breadth >= 70:
        tone = f"구성종목의 {breadth:.0f}%가 200일선 위에 있어 시장 전반이 강세 국면"
    elif breadth >= 50:
        tone = f"구성종목의 {breadth:.0f}%가 200일선 위로 중립~강세 구간"
    elif breadth >= 30:
        tone = f"구성종목의 {breadth:.0f}%만 200일선 위에 있어 혼조세"
    else:
        tone = f"구성종목의 {breadth:.0f}%만 200일선 위에 있어 약세 국면"

    # ── 1문단: 오늘의 추천 요약 ──
    if reco_rows:
        secs = Counter(sector_kr(r[1], sector_map, info) for r in reco_rows)
        sec_txt = ", ".join(f"{k} {v}종목" for k, v in secs.most_common(3))
        pers = [float(r[3]) for r in reco_rows if not _isnan(r[3])]
        rsis = [r[4].get("rsi") for r in reco_rows if not _isnan(r[4].get("rsi"))]
        gc = sum(1 for r in reco_rows if r[4].get("cross") == "golden")
        avg_per = sum(pers) / len(pers) if pers else float("nan")
        avg_rsi = sum(rsis) / len(rsis) if rsis else float("nan")
        top = reco_rows[0]
        top_sym, top_desc = top[1], desc_of(top[1], info, sector_map)
        top_reason = (top[5].split(" · ")[0] if top[5] else "")

        p1 = f"오늘 추천 {len(reco_rows)}종목은 {sec_txt} 등에 분포합니다. "
        if not _isnan(avg_per):
            p1 += (f"이들은 각 섹터 중앙값 이하의 PER에서 선별돼 동종 대비 저평가 영역에 있으며"
                   f"(평균 PER 약 {avg_per:.1f}배), ")
        if not _isnan(avg_rsi):
            heat = "과열 없이 " if avg_rsi < 70 else "다소 과열된 가운데 "
            p1 += f"평균 RSI는 {avg_rsi:.0f} 수준으로 {heat}상승 모멘텀이 살아 있는 구간입니다. "
        if gc:
            p1 += f"이 중 {gc}종목은 최근 50일선이 200일선을 상향 돌파한 골든크로스 종목입니다. "
        if top_desc:
            p1 += f"점수가 가장 높은 종목은 {top_sym}({top_desc})으로, {top_reason} 점이 부각됐습니다."
        else:
            p1 += f"점수가 가장 높은 종목은 {top_sym}입니다."
    else:
        p1 = ("오늘은 섹터 상대 저평가와 기술적 강세(200일선 위·상승 모멘텀 지속)를 동시에 "
              "만족하는 추천 종목이 없었습니다. 가치와 추세가 서로 어긋나 있거나 시장이 단기 "
              "과열된 구간일 수 있어, 무리한 신규 진입보다 관망이 합리적일 수 있는 날입니다.")

    # ── 2문단: 시장 맥락 + 추세/청산 + 주의 ──
    p2 = tone + "입니다. "
    if solid_rows:
        names = ", ".join(r[0] for r in solid_rows[:3])
        p2 += f"상승 추세가 굳어진 종목으로는 {names} 등이 정배열을 유지하고 있고, "
    if reversal_rows:
        names = ", ".join(r[0] for r in reversal_rows[:3])
        p2 += f"하락을 멈추고 반등 신호가 나온 {names} 등은 추세 전환 초기 후보로 관찰할 만합니다. "
    elif solid_rows:
        p2 += "추세 전환 초기 후보는 오늘 두드러지지 않았습니다. "
    if exit_rows:
        names = ", ".join(r[0] for r in exit_rows[:3])
        p2 += (f"반면 직전 추천 중 {names} 등 {len(exit_rows)}종목은 200일선 이탈·데드크로스 등으로 "
               f"기술적 조건이 무너져, 보유 중이라면 비중 점검이 필요합니다. ")
    p2 += ("모든 수치는 규칙 기반 자동 계산값이며 특정 종목 매매 권유가 아니므로, 실제 매매 전 "
           "기업 펀더멘털과 최신 뉴스를 반드시 함께 확인하세요.")

    html = (f'<div style="margin:24px 0 8px;background:#f8fafc;border:1px solid #e5e7eb;'
            f'border-radius:10px;padding:14px 16px">'
            f'<div style="font-size:15px;font-weight:700;margin-bottom:6px">📝 오늘의 총평</div>'
            f'<p style="font-size:13px;line-height:1.8;color:#374151;margin:0 0 10px">{p1}</p>'
            f'<p style="font-size:13px;line-height:1.8;color:#374151;margin:0">{p2}</p></div>')
    text = "■ 오늘의 총평\n" + p1 + "\n\n" + p2
    return html, text


def build_report(reco_rows, reversal_rows, solid_rows, exit_rows, new_in, dropped, asof, sector_map, info, ind_map):
    subject = (f"[S&P500] {asof} · 추천 {len(reco_rows)} · 전환 {len(reversal_rows)} · "
               f"굳힘 {len(solid_rows)} · EXIT {len(exit_rows)}")

    reco_cards = []
    for rank, sym, name, per, ind, reason in reco_rows:
        reco_cards.append(_reco_card(rank, sym, name, per, ind, f"reco_{sym}",
                                     sector_kr(sym, sector_map, info), desc_of(sym, info, sector_map), reason))
    rev_cards = []
    for sym, name, ind in reversal_rows:
        rev_cards.append(_trend_card(sym, name, ind, f"rev_{sym}",
                                     sector_kr(sym, sector_map, info), desc_of(sym, info, sector_map), ind.get("reversal_reason", "")))
    sol_cards = []
    for sym, name, ind in solid_rows:
        sol_cards.append(_trend_card(sym, name, ind, f"sol_{sym}",
                                     sector_kr(sym, sector_map, info), desc_of(sym, info, sector_map), ind.get("solidified_reason", "")))

    def chips(items, color):
        if not items:
            return '<span style="color:#9ca3af">없음</span>'
        return " ".join(f'<span style="background:{color};color:#fff;border-radius:4px;padding:1px 6px;font-size:12px;margin-right:4px">{s}</span>' for s in items)

    reco_block = ("".join(reco_cards) if reco_cards else
                  '<div style="color:#9ca3af;font-size:13px">오늘은 조건을 충족한 추천 종목이 없습니다.</div>')
    rev_block = (f'<div style="color:#6b7280;font-size:12px;margin-bottom:6px">최근 며칠 사이 하락세를 멈추고 상승 신호(MACD 전환·20일선 돌파·RSI 50 회복)가 나타난 종목입니다.</div>{"".join(rev_cards)}'
                 if rev_cards else '<div style="color:#9ca3af;font-size:13px">오늘은 조건을 충족한 종목이 없습니다.</div>')
    sol_block = (f'<div style="color:#6b7280;font-size:12px;margin-bottom:6px">종가가 20·50·200일선 위 정배열로 며칠째 유지되며 상승 추세가 굳어진 종목입니다.</div>{"".join(sol_cards)}'
                 if sol_cards else '<div style="color:#9ca3af;font-size:13px">오늘은 조건을 충족한 종목이 없습니다.</div>')

    if exit_rows:
        rows = "".join(_exit_row_html(sym, info.get(sym, {}).get("name", sym),
                                      sector_kr(sym, sector_map, info), desc_of(sym, info, sector_map), reason)
                       for sym, ind, reason in exit_rows)
        exit_block = (f'<div style="color:#6b7280;font-size:12px;margin-bottom:6px">직전 추천 종목 중 기술적 조건이 무너진 종목입니다. 보유 중이라면 청산/비중축소를 점검하세요.</div>'
                      f'<table style="width:100%;border-collapse:collapse;border:1px solid #fee2e2;border-radius:8px">{rows}</table>')
    else:
        exit_block = '<div style="color:#9ca3af;font-size:13px">청산 신호가 발생한 직전 추천 종목이 없습니다.</div>'

    commentary_html, commentary_text = build_commentary(
        reco_rows, reversal_rows, solid_rows, exit_rows, ind_map, sector_map, info, asof)

    html = f"""\
<div style="font-family:-apple-system,'Segoe UI',Roboto,'Apple SD Gothic Neo',sans-serif;max-width:720px;margin:0 auto;color:#111827">
  <h2 style="margin:0 0 4px">📊 S&amp;P 500 데일리 리포트</h2>
  <div style="color:#6b7280;font-size:13px;margin-bottom:8px">{asof} 마감 기준 · 변동률 1일/1주/1달/1년/3년/5년 · 차트는 가격+이동평균</div>

  <h3 style="margin:18px 0 2px">⭐ 추천 종목 {len(reco_rows)} <span style="font-size:12px;color:#9ca3af;font-weight:400">(섹터 상대 저PER + 퀄리티 + 기술 지표)</span></h3>
  <div style="color:#6b7280;font-size:12px;margin-bottom:4px">조건: 섹터중앙 이하 PER · 퀄리티(ROE·부채·FCF) 양호 · 200일선 위 · (MACD↑ 또는 골든크로스) · 진입신호 {MIN_SIGNAL_DAYS}일+ 지속 · 섹터당 최대 {RECO_SECTOR_MAX}개 · 상위 {RECO_N}개</div>
  {reco_block}

  <div style="margin:12px 0;font-size:13px;line-height:1.7">
    <div>신규 진입: {chips(new_in, "#b45309")}</div>
    <div>목록 이탈: {chips(dropped, "#6b7280")}</div>
  </div>

  <h3 style="margin:26px 0 4px">🚪 청산(EXIT) 신호 — 직전 추천 중 조건 붕괴</h3>
  {exit_block}

  <h3 style="margin:26px 0 4px">🔄 추세 전환 — 하락에서 상승으로</h3>
  {rev_block}

  <h3 style="margin:26px 0 4px">📈 추세 굳힘 — 상승 흐름 고착</h3>
  {sol_block}

  {commentary_html}

  <div style="margin-top:22px;font-size:11px;color:#9ca3af;border-top:1px solid #eee;padding-top:8px;line-height:1.6">
    구성종목: SPDR S&amp;P500 ETF(SPY) 공식 일일 보유종목 · 지표 데이터: Yahoo Finance.<br>
    모든 지표·차트·추천·청산신호는 규칙 기반 자동 계산값이며 <b>투자 권유가 아닙니다.</b> 매매 전 반드시 추가 확인이 필요합니다.<br>
    전략 검증은 <code>python sp500_daily_report.py --backtest</code>로 확인하세요(기술 전략 기준).
  </div>
</div>"""

    # ---------- 텍스트 폴백 ----------
    lines = [f"S&P500 데일리 리포트 — {asof} 마감", "변동률: [1일/1주/1달/1년/3년/5년]", "", f"■ 추천 종목 {len(reco_rows)}"]
    for rank, sym, name, per, ind, reason in reco_rows:
        per_t = "—" if _isnan(per) else f"{float(per):.1f}"
        lines.append(f"{rank:>2}. {sym:<6} PER {per_t:<5} {sector_kr(sym, sector_map, info)} | {desc_of(sym, info, sector_map)}")
        lines.append(f"      이유: {reason}")
    lines += ["", "■ 청산(EXIT) 신호"]
    lines += ([f"  · {sym}: {reason}" for sym, ind, reason in exit_rows] or ["  (해당 없음)"])
    lines += ["", "■ 추세 전환(하락→상승)"]
    lines += ([f"  · {sym}: {ind.get('reversal_reason','')}" for sym, name, ind in reversal_rows] or ["  (해당 없음)"])
    lines += ["", "■ 추세 굳힘(상승 고착)"]
    lines += ([f"  · {sym}: {ind.get('solidified_reason','')}" for sym, name, ind in solid_rows] or ["  (해당 없음)"])
    lines += ["", f"신규 진입: {', '.join(new_in) or '없음'} / 이탈: {', '.join(dropped) or '없음'}"]
    lines += ["", commentary_text]
    text = "\n".join(lines)
    return subject, html, text


# ------------------------- 오케스트레이션 ------------------------
def generate_report():
    asof = datetime.now(KST).strftime("%Y-%m-%d")
    universe, sector_map = get_sp500()

    hist = download_histories(universe)
    ind_map = {}
    for sym in universe:
        close = hist.get(sym)
        ind = compute_indicators(close) if close is not None else None
        if ind:
            ind_map[sym] = ind

    # 기술 사전필터(.info 호출 축소): 200일선 위 AND (MACD↑ OR 골든크로스) AND 신호 지속
    tech_cand = [s for s, ind in ind_map.items()
                 if ind.get("above_ma200") and (ind.get("macd_up") or ind.get("cross") == "golden")
                 and ind.get("entry_streak", 0) >= MIN_SIGNAL_DAYS]
    info = get_info_for(tech_cand)

    # ---- 추천 종목 (섹터 상대 PER + 퀄리티) ----
    sec_med, glob_med = sector_median_pes(info, sector_map)
    scored = []
    for sym in tech_cand:
        meta = info.get(sym, {})
        pe = meta.get("pe")
        try:
            pe_f = float(pe)
        except (TypeError, ValueError):
            continue
        sec = sector_map.get(sym, "") or "(기타)"
        med = sec_med.get(sec, glob_med)
        if _isnan(med) or med <= 0:
            continue
        rel_pe = pe_f / med
        qscore, qreasons = quality_score(meta)
        r = score_reco(ind_map[sym], pe, rel_pe, qscore, qreasons)
        if r is not None:
            scored.append((sym, r[0], r[1]))
    scored.sort(key=lambda x: x[1], reverse=True)
    picked = pick_with_sector_cap(scored, sector_map, RECO_N, RECO_SECTOR_MAX)
    reco_syms = {s for s, _, _ in picked}

    reco_rows = []
    for rank, (sym, _score, reason) in enumerate(picked, start=1):
        meta = info.get(sym, {})
        reco_rows.append((rank, sym, meta.get("name", sym), meta.get("pe"), ind_map[sym], reason))

    # ---- 추세 전환/굳힘 (추천과 중복 제거) ----
    rev = [(s, ind) for s, ind in ind_map.items() if ind.get("reversal") and s not in reco_syms]
    rev.sort(key=lambda x: x[1].get("reversal_score", 0), reverse=True)
    rev = rev[:TREND_MAX]
    rev_syms = {s for s, _ in rev}

    sol = [(s, ind) for s, ind in ind_map.items()
           if ind.get("solidified") and s not in reco_syms and s not in rev_syms]
    sol.sort(key=lambda x: x[1].get("solidified_score", 0), reverse=True)
    sol = sol[:TREND_MAX]

    # ---- 청산(EXIT): 직전 추천 종목 중 조건 붕괴 ----
    prev = load_prev_list()
    picked_syms = [s for s, _, _ in picked]
    exit_rows = detect_exits(prev, ind_map, picked_syms)

    # 표시될 추세/EXIT 종목 info 보충
    need = [s for s, _ in (rev + sol) if s not in info] + [s for s, _, _ in exit_rows if s not in info]
    if need:
        info.update(get_info_for(list(dict.fromkeys(need))))

    reversal_rows = [(s, info.get(s, {}).get("name", s), ind) for s, ind in rev]
    solid_rows = [(s, info.get(s, {}).get("name", s), ind) for s, ind in sol]

    # 편입/이탈(추천 명단 기준)
    new_in = [s for s in picked_syms if s not in prev]
    dropped = [s for s in prev if s not in picked_syms]

    # ---- 차트 ----
    images = {}
    for rank, sym, *_ in reco_rows:
        close = hist.get(sym)
        if close is not None:
            images[f"reco_{sym}"] = make_chart_png(close, None, (50, 200), up=True, title=f"{sym} · 5Y")
    for sym, _name, _ind in reversal_rows:
        close = hist.get(sym)
        if close is not None:
            images[f"rev_{sym}"] = make_chart_png(close, 130, (20, 50), up=True, title=f"{sym} · 6M")
    for sym, _name, _ind in solid_rows:
        close = hist.get(sym)
        if close is not None:
            images[f"sol_{sym}"] = make_chart_png(close, 260, (20, 50, 200), up=True, title=f"{sym} · 1Y")

    subject, html, text = build_report(reco_rows, reversal_rows, solid_rows, exit_rows,
                                       new_in, dropped, asof, sector_map, info, ind_map)
    save_curr_list(picked_syms)
    return subject, html, text, images


def html_with_inline_images(html, images):
    out = html
    for cid, png in images.items():
        b64 = base64.b64encode(png).decode("ascii")
        out = out.replace(f"cid:{cid}", f"data:image/png;base64,{b64}")
    return out


# ------------------------- 이메일 발송 ---------------------------
def send_email(subject, html, text, images):
    host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    port = int(os.environ.get("SMTP_PORT", "465"))
    user = os.environ.get("SMTP_USER", "").strip()
    pw = os.environ.get("SMTP_PASS", "").strip()
    to = os.environ.get("EMAIL_TO", user).strip()
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


def report_main():
    subject, html, text, images = generate_report()
    os.makedirs("output", exist_ok=True)
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
    print(f"차트 {len(images)}개 · output/email.html 미리보기 저장")

    if os.environ.get("SMTP_USER") and os.environ.get("EMAIL_TO"):
        send_email(subject, html, text, images)
        print("✅ 이메일 발송 완료 →", os.environ.get("EMAIL_TO"))
    else:
        print("(SMTP 미설정: 파일만 생성. 메일 받으려면 SMTP_USER/SMTP_PASS/EMAIL_TO 설정)")


def main():
    parser = argparse.ArgumentParser(description="S&P500 데일리 리포트 / 백테스트")
    parser.add_argument("--backtest", action="store_true",
                        help="기술 전략을 과거 데이터로 검증(SPY 대비)")
    args = parser.parse_args()
    if args.backtest or os.environ.get("MODE", "").lower() == "backtest":
        backtest_main()
    else:
        report_main()


if __name__ == "__main__":
    main()
