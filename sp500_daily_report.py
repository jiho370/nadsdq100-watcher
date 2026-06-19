#!/usr/bin/env python3
"""
sp500_daily_report.py  (v3)

매일 S&P 500 구성종목을 분석해 이메일 리포트를 생성/발송한다.

리포트 구성
  1) ⭐ 추천 종목 (최대 10개) — 저PER이면서 기술적 지표가 우수한 종목
     · 필수 조건: 200일선 위  AND  (MACD 상승  OR  최근 골든크로스)  AND  PER 적정범위
     · 복합 점수(밸류에이션 + 추세 강도)로 정렬해 상위 N개. 조건 충족이 적으면 그만큼만.
  2) 🔄 추세 전환(하락→상승) — 며칠 사이 하락을 멈추고 상승 신호가 나타난 종목
  3) 📈 추세 굳힘(상승 고착) — 정배열이 며칠째 유지되는 종목
     ※ 추천에 이미 든 종목은 2)·3)에서 제외(중복 표시 방지)

구성종목은 매 실행 시 SPDR S&P500 ETF(SPY)의 공식 일일 보유종목을 받아 항상 최신으로 유지하고,
다운로드 실패 시에만 내장 스냅샷(SP500_FALLBACK)으로 폴백한다.

데이터 : Yahoo Finance(yfinance, 키 불필요) + SPY 보유종목(State Street)
실행   : GitHub Actions cron (매일 1회, 미국장 마감 후)
지표·차트는 일봉 종가 기준 자동 계산값이며 투자 권유가 아니다.
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
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
RECO_PER_MAX   = float(os.environ.get("RECO_PER_MAX", "20"))    # '저PER' 기준 상한
TREND_MAX      = int(os.environ.get("TREND_MAX", "6"))          # 전환/굳힘 섹션별 최대 종목
HISTORY_PERIOD = os.environ.get("HISTORY_PERIOD", "5y")
STATE_FILE     = os.environ.get("STATE_FILE", "state_prev_list.json")
KST            = timezone(timedelta(hours=9))

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

# 주요 종목 한글 한 줄 설명(없으면 야후 영문 industry/summary로 폴백)
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
        # 폴백 섹터로 빈 곳 보충
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

    # 'Ticker'와 'Name'을 포함한 헤더 행을 찾는다
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
        # 알파벳/점 위주의 정상 티커만
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
    """주어진 종목들의 PER/가격/이름/업종/요약을 .info로 조회(종목당 1콜)."""
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
        }
    return out


def download_histories(symbols: list[str], period: str = HISTORY_PERIOD) -> dict[str, pd.Series]:
    """모든 종목 일봉 종가를 배치로 받아 {sym: close}로 반환. 실패 시 개별 폴백."""
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
    ind = {
        "price": last, "ma20": ma_last[20], "ma50": ma_last[50], "ma200": ma_last[200],
        "above_ma200": (not np.isnan(ma_last[200])) and last > ma_last[200],
        "rsi": rsi_val, "macd": macd_val, "macd_signal": sig_val, "macd_hist": hist_val,
        "macd_up": hist_val > 0, "cross": cross,
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


# --------------------- 추천 종목 스코어링 -----------------------
def score_reco(ind: dict, pe) -> tuple[float, str] | None:
    """추천 적격이면 (점수, 한글사유) 반환, 아니면 None.
    필수: PER∈(0, RECO_PER_MAX]  AND  200일선 위  AND  (MACD 상승 OR 최근 골든크로스).
    """
    try:
        pe = float(pe)
    except (TypeError, ValueError):
        return None
    if not (0 < pe <= RECO_PER_MAX):
        return None
    if not ind.get("above_ma200"):
        return None
    if not (ind.get("macd_up") or ind.get("cross") == "golden"):
        return None
    # 극단적 과열(RSI>82)은 추천에서 제외(추격매수 방지)
    if not _isnan(ind.get("rsi")) and ind["rsi"] > 82:
        return None

    score, reasons = 0.0, []
    # 밸류에이션(저PER일수록 가점)
    score += (RECO_PER_MAX - pe) / RECO_PER_MAX * 2.5
    reasons.append(f"PER {pe:.1f}로 저평가 구간")
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


def desc_of(sym, info):
    if sym in KR_DESC and KR_DESC[sym]:
        return KR_DESC[sym]
    meta = info.get(sym, {}) or {}
    ind = meta.get("industry") or ""
    summ = meta.get("summary") or ""
    if ind:
        first = summ.split(". ")[0] if summ else ""
        return f"{ind}" + (f" — {first[:60]}" if first else "")
    if summ:
        return summ.split(". ")[0][:80]
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


def build_report(reco_rows, reversal_rows, solid_rows, new_in, dropped, asof, sector_map, info):
    subject = (f"[S&P500] {asof} · 추천 {len(reco_rows)} · 추세전환 {len(reversal_rows)} · 굳힘 {len(solid_rows)}")

    reco_cards = []
    for rank, sym, name, per, ind, reason in reco_rows:
        reco_cards.append(_reco_card(rank, sym, name, per, ind, f"reco_{sym}",
                                     sector_kr(sym, sector_map, info), desc_of(sym, info), reason))
    rev_cards = []
    for sym, name, ind in reversal_rows:
        rev_cards.append(_trend_card(sym, name, ind, f"rev_{sym}",
                                     sector_kr(sym, sector_map, info), desc_of(sym, info), ind.get("reversal_reason", "")))
    sol_cards = []
    for sym, name, ind in solid_rows:
        sol_cards.append(_trend_card(sym, name, ind, f"sol_{sym}",
                                     sector_kr(sym, sector_map, info), desc_of(sym, info), ind.get("solidified_reason", "")))

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

    html = f"""\
<div style="font-family:-apple-system,'Segoe UI',Roboto,'Apple SD Gothic Neo',sans-serif;max-width:720px;margin:0 auto;color:#111827">
  <h2 style="margin:0 0 4px">📊 S&amp;P 500 데일리 리포트</h2>
  <div style="color:#6b7280;font-size:13px;margin-bottom:8px">{asof} 마감 기준 · 변동률 1일/1주/1달/1년/3년/5년 · 차트는 가격+이동평균</div>

  <h3 style="margin:18px 0 2px">⭐ 추천 종목 {len(reco_rows)} <span style="font-size:12px;color:#9ca3af;font-weight:400">(저PER + 우수한 기술적 지표)</span></h3>
  <div style="color:#6b7280;font-size:12px;margin-bottom:4px">조건: 200일선 위 · (MACD 상승 또는 골든크로스) · PER ≤ {RECO_PER_MAX:.0f} · 점수 상위 {RECO_N}개(미달 시 그만큼)</div>
  {reco_block}

  <div style="margin:12px 0;font-size:13px;line-height:1.7">
    <div>신규 진입: {chips(new_in, "#b45309")}</div>
    <div>목록 이탈: {chips(dropped, "#6b7280")}</div>
  </div>

  <h3 style="margin:26px 0 4px">🔄 추세 전환 — 하락에서 상승으로</h3>
  {rev_block}

  <h3 style="margin:26px 0 4px">📈 추세 굳힘 — 상승 흐름 고착</h3>
  {sol_block}

  <div style="margin-top:22px;font-size:11px;color:#9ca3af;border-top:1px solid #eee;padding-top:8px;line-height:1.6">
    구성종목: SPDR S&amp;P500 ETF(SPY) 공식 일일 보유종목 · 지표 데이터: Yahoo Finance.<br>
    모든 지표·차트·추천은 규칙 기반 자동 계산값이며 <b>투자 권유가 아닙니다.</b> 매매 전 반드시 추가 확인이 필요합니다.
  </div>
</div>"""

    # ---------- 텍스트 폴백 ----------
    lines = [f"S&P500 데일리 리포트 — {asof} 마감", "변동률: [1일/1주/1달/1년/3년/5년]", "", f"■ 추천 종목 {len(reco_rows)}"]
    for rank, sym, name, per, ind, reason in reco_rows:
        per_t = "—" if _isnan(per) else f"{float(per):.1f}"
        lines.append(f"{rank:>2}. {sym:<6} PER {per_t:<5} {sector_kr(sym, sector_map, info)} | {desc_of(sym, info)}")
        lines.append(f"      이유: {reason}")
    lines += ["", "■ 추세 전환(하락→상승)"]
    lines += ([f"  · {sym}: {ind.get('reversal_reason','')}" for sym, name, ind in reversal_rows] or ["  (해당 없음)"])
    lines += ["", "■ 추세 굳힘(상승 고착)"]
    lines += ([f"  · {sym}: {ind.get('solidified_reason','')}" for sym, name, ind in solid_rows] or ["  (해당 없음)"])
    lines += ["", f"신규 진입: {', '.join(new_in) or '없음'} / 이탈: {', '.join(dropped) or '없음'}"]
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

    # 기술 사전필터(.info 호출 축소): 200일선 위 AND (MACD 상승 OR 골든크로스)
    tech_cand = [s for s, ind in ind_map.items()
                 if ind.get("above_ma200") and (ind.get("macd_up") or ind.get("cross") == "golden")]
    info = get_info_for(tech_cand)

    # ---- 추천 종목 ----
    scored = []
    for sym in tech_cand:
        pe = info.get(sym, {}).get("pe")
        r = score_reco(ind_map[sym], pe)
        if r is not None:
            scored.append((sym, r[0], r[1]))
    scored.sort(key=lambda x: x[1], reverse=True)
    picked = scored[:RECO_N]
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

    # 표시될 추세 종목 info 보충
    need = [s for s, _ in (rev + sol) if s not in info]
    if need:
        info.update(get_info_for(need))

    reversal_rows = [(s, info.get(s, {}).get("name", s), ind) for s, ind in rev]
    solid_rows = [(s, info.get(s, {}).get("name", s), ind) for s, ind in sol]

    # 편입/이탈(추천 명단 기준)
    prev = load_prev_list()
    picked_syms = [s for s, _, _ in picked]
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

    subject, html, text = build_report(reco_rows, reversal_rows, solid_rows, new_in, dropped, asof, sector_map, info)
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


def main():
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


if __name__ == "__main__":
    main()
