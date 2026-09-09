# 주간 자산배분 결산 및 조언 알고리즘 리서치 명세

## A. 핵심 결론

**결론부터 말하면, 개인 투자자용 주간 자산배분 알고리즘의 1차 백테스트 후보는 `가격 기반 모멘텀·추세`, `실현변동성·하방위험`, `낙폭·최대낙폭`, `자산군 간 상대강도`, `단순 상관관계·분산효과 경고`, `금리·인플레이션·환율의 저빈도 거시 보조지표` 순서로 두는 것이 가장 현실적이다.** 다만 이는 “채택”이 아니라 **백테스트 우선순위**다. 특정 이동평균 기간, 모멘텀 기간, 기준 배분, 리밸런싱 주기, 자산군 비중은 모두 후보 범위로 두고, 비용·세금·환율·최근 구간 검증을 통과한 뒤에만 채택해야 한다.

**우선 백테스트해야 할 팩터**는 무료·저비용 가격 데이터로 계산 가능한 것들이다. 구체적으로는 다중기간 수익률, 최근 구간 제외 모멘텀, 가격-이동평균 괴리율, 절대 모멘텀, 상대강도, 변동성 조정 모멘텀, 실현변동성, downside deviation, drawdown, 자산군별 위험기여도, 단순 rolling correlation이다. 최근 연구도 다중자산 모멘텀·추세추종, 변동성·거래비용 통제, 국면별 자산배분을 반복적으로 다루지만, 그 자체가 정답이라는 뜻은 아니다.

**데이터 접근성 때문에 후순위인 팩터**는 밸류에이션, 이익 전망, 퀄리티, 배당성장, 리츠 펀더멘털, 부동산 경기 세부지표, 옵션 기반 내재변동성 구조, 실시간 뉴스·NLP, 고빈도 미시구조 데이터다. 이들은 설명력은 있을 수 있지만 개인 프로젝트에서 자동화·장기 백테스트·약관 준수·데이터 품질 관리가 어렵다.

**리포트 설명용으로만 적합한 팩터**는 FOMC·CPI·고용 발표 같은 이벤트 캘린더, 단기 뉴스성 위험, 해석 중심의 거시 코멘트, 최근 수익률 순위, VIX 단독 신호, 단기 환율 급등락 설명, 자산군별 라벨이다. 이들은 사람이 리포트를 이해하는 데 도움은 되지만, 백테스트 없이 비중 조정 규칙에 넣으면 사후해석 편향이 커진다.

**MVP에서 제외해야 할 팩터·모델**은 고빈도 데이터, 옵션 체인 전체를 쓰는 모델, 대체데이터, 대규모 딥러닝, 실시간 뉴스 NLP, 선물 레버리지·숏 포지션 전제 전략, 복잡한 강화학습, 설명 불가능한 블랙박스 모델이다.

**최종 팩터 채택 기준**은 단순 수익률이 아니라 `거래비용·세금 차감 후 위험조정성과`, `MDD 감소`, `위기 구간 방어`, `최근 5년·10년 붕괴 여부`, `파라미터 민감도`, `turnover`, `원화 기준 성과`, `구현·유지 가능성`, `설명 가능성`이어야 한다.

**주간 리포트와 실제 매매 알고리즘은 반드시 분리해야 한다.** 주간 리포트는 더 많은 정보를 보여줄 수 있지만, 실제 비중 조정은 백테스트를 통과한 소수의 신호만 사용해야 한다. 예를 들어 “주식-채권 상관관계 상승”, “금 변동성 급등”, “CPI 발표 예정”은 리포트에는 표시할 수 있지만, 검증 전에는 목표 비중을 바꾸는 직접 신호가 아니다.

**개인 투자자용 MVP의 적정 복잡도**는 자산군 3~7개, 팩터 1~3개, 월간 또는 임계값 기반 리밸런싱, 거래비용·세금·환율 반영, 원화 기준 성과 확인, 단순 규칙 또는 점수화 모델 수준이다. 복잡한 모델이 단순 모델 대비 비용 차감 후 Sharpe, Sortino, Calmar, MDD, turnover에서 명확히 개선하지 못하면 단순 모델을 선택해야 한다.

**오래된 이론은 배경으로만 사용한다.** 평균-분산 최적화, 60/40, 리스크 패리티, 모멘텀, 추세추종, CAPM·Fama-French류 팩터는 출발점으로 의미가 있지만, 현재 알고리즘의 직접 채택 근거가 될 수 없다. 현대 시장에서는 ETF 세금, 환율, 거래비용, 주식-채권 동반 하락, 고금리·인플레이션 구간, 최근 데이터로 다시 검증해야 한다.

---

## B. 최신 자료 기반 후보 팩터 요약

| 자료 유형 | 출처 또는 연구 방향 | 최신성 | 핵심 시사점 | 알고리즘 후보로의 의미 | 주의점 |
|---|---:|---:|---|---|---|
| 다중자산 모멘텀·추세 | TrendFolios, 2025 | 최근 5년 | 다중자산·자산군 위험요인에 모멘텀과 추세 신호를 적용하는 구조를 제시 | 가격 기반 모멘텀, 절대 모멘텀, 상대강도, 추세 강도를 후보로 둠 | 논문 구조를 그대로 채택하지 말고 단순화 필요 |
| 동적 자산배분·국면 예측 | Dynamic Asset Allocation with Asset-Specific Regime Forecasts, 2024 | 최근 5년 | 자산별 강세·약세 국면을 예측해 다중자산 배분에 반영 | 국면 라벨, 단순 ML, bull/bear 확률을 고급 후보로 분류 | 개인 MVP에서는 GBDT·최적화 전체 구현은 과도할 수 있음 |
| 거시 국면 기반 TAA | Tactical Asset Allocation with Macroeconomic Regime Detection, 2025 | 최근 5년 | FRED-MD 같은 대형 거시 데이터셋으로 국면을 분류하고 배분에 연결 | 거시 국면은 고급 후보 또는 리포트 설명용 후보 | 발표 지연·수정치·과최적화 위험이 큼 |
| cross-asset time-series momentum | Accounting and Finance, 2025 | 최근 5년 | 1990~2023년 데이터에서 개선형 cross-asset time-series momentum 전략을 검토 | 자산군 간 모멘텀 신호, 산업금속·원자재 신호의 보조 역할을 후보화 | 원자재 데이터 접근성·거래 가능성·세금 확인 필요 |
| 변동성 관리·리밸런싱 경계 | Target volatility strategies, 2025 | 최근 5년 | 변동성 타깃팅에서 거래비용을 줄이기 위해 리밸런싱 경계를 추가 | 변동성 타깃팅, 임계값 리밸런싱, turnover 제한 후보 | 레버리지 없이 구현 가능한지 별도 검증 필요 |
| 변동성-모멘텀 상호작용 | Market volatility, momentum, and reversal, 2024 | 최근 5년 | 높은 변동성 이후 모멘텀 붕괴 또는 반전 가능성을 검토 | 변동성 급등 시 모멘텀 신호 약화 여부를 후보로 테스트 | 복잡한 반전 전략은 MVP가 아니라 고급 후보 |
| 모멘텀 팩터 리뷰 | Momentum factor investing, 2025 | 최근 5년 | 모멘텀이 장기·다시장 검증 대상이며 crash risk와 risk-managed momentum을 함께 고려 | 단순 모멘텀뿐 아니라 변동성 조정 모멘텀 후보 필요 | 개별주식 팩터 자료를 ETF 자산배분에 직접 일반화하면 안 됨 |
| 주식-채권 상관관계 | CFA Institute, 2025 | 최근 5년 | 주식-채권 상관관계가 거시 조건에 따라 바뀌며, 음의 상관관계 가정은 불안정 | rolling correlation은 매매 신호보다 위험 경고 후보 | 상관관계 추정은 불안정하므로 직접 비중 조정은 신중해야 함 |
| 60/40 역사 검토 | CFA Institute, 2025 | 최근 5년 | 60/40은 역사적 기준점이지만 국가·세대·상관관계에 따라 성과가 다름 | 60/40은 벤치마크·기준 배분 후보일 뿐 정답 아님 | 한국 투자자 원화·세금 기준으로 재검증 필요 |
| 실무 포트폴리오 자료 | Vanguard, BlackRock, Morningstar | 최근 1~2년 | 고금리, 인플레이션, 양의 주식-채권 상관관계, 금·국제분산·실물자산의 보조 역할을 강조 | 금, TIPS, 단기채, 국제주식, 달러 노출을 후보 자산군에 포함 | 실무 자료는 판매·마케팅 성격도 있으므로 백테스트로 검증 필요 |
| 데이터 편향·수정치 | FRED/ALFRED, St. Louis Fed | 지속 업데이트 | FRED는 최신값, ALFRED는 과거 특정 시점에 이용 가능했던 vintage 데이터를 제공 | 거시 백테스트는 ALFRED 또는 발표일 lag 적용 필요 | 사후 수정치 사용 시 look-ahead bias 발생 가능 |
| 고전 문헌 | 평균-분산, 60/40, 모멘텀, 리스크 패리티 | 배경 | 포트폴리오 이론과 후보 팩터의 출발점 | 벤치마크·후보 설계에만 사용 | 최신 데이터 검증 없이 직접 채택 금지 |

---

## C. 후보 팩터 우선순위표

아래 표의 “후보 기간/계산법”은 **고정값이 아니라 탐색 범위**다. 백테스트에서는 구간별 walk-forward, 파라미터 민감도, 비용·세금 차감 후 성과로 평가해야 한다.

| 우선순위 | 팩터 | 후보 기간/계산법 | 데이터 소스 | 예상 장점 | 예상 위험 | 백테스트 필요성 | MVP 적합성 |
|---:|---|---|---|---|---|---|---|
| 1 | 다중기간 가격 모멘텀 | 주간~12개월 이상 후보 범위, 최근 일부 기간 제외 버전 포함 | Yahoo/yfinance, Stooq, ETF 발행사, KRX | 계산 쉽고 자산군 공통 적용 가능 | 단기 노이즈, momentum crash | 매우 높음 | 높음 |
| 2 | 절대 모멘텀·추세 | 가격이 장기 이동평균 또는 누적수익 기준 위/아래인지, 기간은 후보 범위 | 가격 데이터 | 하락장 방어 후보 | 횡보장 whipsaw | 매우 높음 | 높음 |
| 3 | 상대강도 | 자산군별 수익률 랭킹, 변동성 조정 랭킹 | 가격 데이터 | 자산군 선택에 직관적 | turnover 증가 | 매우 높음 | 높음 |
| 4 | 실현변동성 | 일간/주간 수익률 rolling volatility, 후보 기간 범위 | 가격 데이터 | 위험 축소·비중 조정 후보 | 변동성 추정 불안정 | 매우 높음 | 높음 |
| 5 | downside deviation | 음수 수익률만 사용한 하방 변동성 | 가격 데이터 | Sortino·손실관리와 연결 | 표본 부족 시 불안정 | 높음 | 중간~높음 |
| 6 | drawdown 기반 위험 | 최근 고점 대비 낙폭, MDD, 회복 여부 | 가격 데이터 | 리포트 이해도 높음, 손실 경고 | 매매 신호로 쓰면 늦을 수 있음 | 높음 | 중간 |
| 7 | 변동성 조정 모멘텀 | 수익률 / 변동성, 수익률 / downside risk | 가격 데이터 | 수익·위험 동시 반영 | 이중계산·과최적화 | 높음 | 높음 |
| 8 | 변동성 역가중 | 자산별 inverse volatility weight | 가격 데이터 | 단순 위험균형 후보 | 저변동 자산 과대비중 | 높음 | 높음 |
| 9 | 변동성 타깃팅 | 포트폴리오 변동성 목표 범위, 레버리지 없는 축소형 우선 | 가격 데이터, 현금 수익률 | 위기 방어 가능성 | turnover·세금 증가 | 높음 | 중간 |
| 10 | rolling correlation | 주식-채권, 주식-금, 주식-리츠 등 | 가격 데이터 | 분산효과 약화 경고 | 불안정·후행성 | 높음 | 리포트/위험경고 우선 |
| 11 | 금리 수준·변화 | 정책금리, 국채금리, yield curve, 실질금리 후보 | FRED, ECOS, KRX, 증권사 | 채권·금·주식 설명력 후보 | 발표·시차·해석 복잡 | 높음 | 보조/리포트 우선 |
| 12 | 인플레이션 | CPI, 기대인플레이션, real yield proxy | FRED, ECOS | 금·채권·TIPS 설명 후보 | 수정치·발표지연 | 높음 | 설명/위험경고 우선 |
| 13 | 환율·달러 | USD/KRW, DXY, 원화 환산 수익률 | ECOS, Yahoo, Stooq, FRED | 한국 투자자 성과에 필수 | 환헤지 상품별 차이 | 매우 높음 | 높음 |
| 14 | 신용 스프레드 | IG/HY spread, TED/OAS 후보 | FRED | 위험국면 경고 | 한국 투자자 ETF 매매와 직접 연결 약함 | 중간~높음 | 위험경고 |
| 15 | VIX·위험심리 | VIX 수준, 변화율, percentile | Yahoo, Stooq, CBOE 계열 | 단기 위험 경고 | 단독 매매 신호로 불안정 | 중간 | 리포트/위험경고 |
| 16 | 밸류에이션 | CAPE, P/E, earnings yield, dividend yield | ETF 발행사, FRED, 유료/부분 무료 | 장기 기대수익 설명 | 자동화·데이터 품질 어려움 | 중간 | 후순위 |
| 17 | 이익 전망·퀄리티 | EPS revision, ROE, margin, quality score | 유료 데이터, 일부 공개 | 주식 자산군 세분화 가능 | 무료 장기 데이터 부족 | 낮음~중간 | 고급/보류 |
| 18 | 이벤트 캘린더 | CPI, FOMC, 금통위, 고용 발표 | 공식 발표 일정 | 리포트 가독성 | 성과 예측력 불명확 | 낮음 | 설명용 |
| 19 | 고빈도·옵션·뉴스 NLP | intraday, 옵션 smile, 뉴스 감성 | 고가/복잡 | 이론상 정보량 큼 | 구현·약관·과최적화 | 낮음 | MVP 제외 |

### C-2. 팩터 채택·분류 평가표

백테스트 전이므로 “단독 성과”와 “조합 기여도”는 **미검증**으로 둔다. 아래 분류는 **검증 전 초기 분류**다.

| 팩터 | 데이터 접근성 | 계산 난이도 | 단독 성과 | 조합 기여도 | turnover 영향 | 최근 구간 유효성 | 위기 구간 유효성 | 과최적화 위험 | 최종 분류 |
|---|---|---:|---|---|---|---|---|---|---|
| 다중기간 가격 모멘텀 | 높음 | 낮음 | 미검증 | 미검증 | 중간 | 검증 필요 | 검증 필요 | 중간 | MVP 채택 후보 |
| 절대 모멘텀·추세 | 높음 | 낮음 | 미검증 | 미검증 | 중간 | 검증 필요 | 검증 필요 | 중간 | MVP 채택 후보 |
| 상대강도 | 높음 | 낮음 | 미검증 | 미검증 | 중간~높음 | 검증 필요 | 검증 필요 | 중간 | MVP/기본 후보 |
| 실현변동성 | 높음 | 낮음 | 미검증 | 미검증 | 중간 | 검증 필요 | 검증 필요 | 낮음~중간 | MVP 채택 후보 |
| 변동성 조정 모멘텀 | 높음 | 낮음~중간 | 미검증 | 미검증 | 중간 | 검증 필요 | 검증 필요 | 중간 | MVP/기본 후보 |
| drawdown | 높음 | 낮음 | 미검증 | 미검증 | 낮음~중간 | 검증 필요 | 검증 필요 | 낮음 | 위험 경고용 또는 보조 후보 |
| 변동성 역가중 | 높음 | 낮음 | 미검증 | 미검증 | 중간 | 검증 필요 | 검증 필요 | 낮음~중간 | 기본 버전 후보 |
| 변동성 타깃팅 | 높음 | 중간 | 미검증 | 미검증 | 중간~높음 | 검증 필요 | 검증 필요 | 중간 | 기본/고급 후보 |
| rolling correlation | 높음 | 낮음 | 미검증 | 미검증 | 낮음 | 검증 필요 | 검증 필요 | 중간 | 위험 경고용 우선 |
| 금리·yield curve | 중간~높음 | 낮음~중간 | 미검증 | 미검증 | 낮음 | 검증 필요 | 검증 필요 | 중간 | 리포트/보조 후보 |
| 인플레이션·실질금리 | 중간 | 중간 | 미검증 | 미검증 | 낮음 | 검증 필요 | 검증 필요 | 높음 | 리포트 설명용 우선 |
| 환율·달러 | 높음 | 낮음 | 미검증 | 미검증 | 낮음~중간 | 검증 필요 | 검증 필요 | 낮음 | 필수 성과 환산·보조 후보 |
| 신용 스프레드 | 중간 | 낮음 | 미검증 | 미검증 | 낮음 | 검증 필요 | 검증 필요 | 중간 | 위험 경고용 |
| VIX | 중간 | 낮음 | 미검증 | 미검증 | 낮음~중간 | 검증 필요 | 검증 필요 | 중간 | 위험 경고용 |
| 밸류에이션 | 낮음~중간 | 중간 | 미검증 | 미검증 | 낮음 | 검증 필요 | 검증 필요 | 중간 | 고급/보류 |
| 이익 전망·퀄리티 | 낮음 | 중간~높음 | 미검증 | 미검증 | 낮음 | 검증 필요 | 검증 필요 | 높음 | 데이터 문제로 후순위 |
| 고빈도·옵션·뉴스 NLP | 낮음 | 높음 | 미검증 | 미검증 | 높음 | 검증 필요 | 검증 필요 | 높음 | MVP 제외 |

---

## D. 데이터 접근성 평가표

| 데이터 | 무료 접근 가능성 | 자동화 가능성 | 장기 데이터 가능성 | 한국 투자자 적용성 | 총수익률 가능성 | 문제점 | 최종 판단 |
|---|---|---|---|---|---|---|---|
| Yahoo Finance / yfinance | 높음 | 높음 | ETF 상장 이후 중심 | 해외 ETF·환율 확인에 유용 | 수정주가로 근사 가능 | 비공식 도구, 개인용·약관 확인 필요 | MVP 보조 소스, 단독 신뢰 금지 |
| Stooq | 높음 | CSV 다운로드 가능 | 지수·ETF·환율 일부 장기 | 글로벌 가격 데이터 보조 | 제한적 | 개인 사용 조건, 커버리지 확인 필요 | 가격 교차검증 소스 |
| FRED | 높음 | 공식 API | 매우 장기 가능 | 미국 금리·물가·스프레드·DXY 등 | 가격 총수익률보다는 거시 | 최신값 중심, 수정치 문제 | 거시·금리 기본 소스 |
| ALFRED | 높음 | API 가능 | vintage 데이터 | 거시 백테스트 편향 통제 | 해당 없음 | 모든 지표가 완전하지 않을 수 있음 | 거시 백테스트 필수 보조 |
| 한국은행 ECOS | 높음 | 공식 Open API | 한국 거시·환율 장기 가능 | 매우 높음 | 해당 없음 | 시계열 코드 관리 필요 | 한국 금리·환율·물가 핵심 소스 |
| KRX Data Marketplace | 높음~중간 | 웹·Open API/다운로드 구조 확인 필요 | 한국 주식·ETF·지수 가능 | 매우 높음 | 상품별 확인 필요 | API 승인·로그인·약관 확인 필요 | 한국 ETF·지수 필수 후보 |
| Investing.com | 중간 | 비공식 라이브러리 의존 | 폭넓음 | 보조 가능 | 불확실 | 스크래핑·약관·안정성 위험 | MVP 핵심 소스로 부적합 |
| Nasdaq Data Link 무료 데이터 | 중간 | 공식 API 가능 | 데이터셋별 다름 | 일부 유용 | 데이터셋별 다름 | 무료 데이터 제한, 유료 전환 가능 | 보조·특정 데이터셋 후보 |
| ETF 발행사 데이터 | 높음~중간 | 수동/CSV/페이지별 상이 | ETF 출시 이후 | 실제 투자상품 검증에 중요 | NAV·분배금 확인 가능 | 자동화 표준화 어려움 | 최종 검증용 필수 보조 |
| 국내 증권사 Open API | 중간 | 가능 | 증권사별 상이 | 실제 매매 연결에 유리 | 제한적 | 계좌·인증·호출제한 | 매매 자동화 단계에서 검토 |
| MSCI, S&P, Bloomberg, Refinitiv | 낮음 | 높음이나 유료 | 매우 우수 | 높음 | 우수 | 비용·라이선스 | MVP 제외, 연구용 참고 |

**데이터 원칙:** 가격 기반 팩터는 Yahoo/yfinance, Stooq, KRX, ETF 발행사 데이터를 교차검증한다. 거시지표는 FRED/ECOS를 쓰되, 과거 백테스트에는 ALFRED vintage 또는 발표일 lag를 적용한다. 세금은 하드코딩하지 말고 `tax_profile` 설정값으로 분리한다.

---

## E. 자산군 후보 평가표

| 자산군 | 대표 데이터 후보 | 장점 | 단점 | 한국 투자자 이슈 | 백테스트 우선순위 | 최종 후보 여부 |
|---|---|---|---|---|---:|---|
| 미국 주식 | SPY, IVV, VOO, S&P500 TR proxy | 장기 데이터·유동성·대표성 | 미국 집중 위험 | 환율·해외주식 세금 | 매우 높음 | 최소/기본 후보 |
| 글로벌 주식 | VT, ACWI, MSCI ACWI proxy | 분산 효과 | ETF 출시 이후 데이터 제한 | 원화 환산 필수 | 매우 높음 | 최소/기본 후보 |
| 선진국 ex-US | EFA, IEFA, MSCI EAFE proxy | 미국 집중 완화 | 장기 저성과 구간 가능 | 환율 다중 노출 | 높음 | 기본 후보 |
| 신흥국 주식 | EEM, IEMG, MSCI EM proxy | 성장·분산 후보 | 변동성·정치위험 | 환율·세금 | 중간 | 기본/고급 후보 |
| 한국 주식 | KOSPI200 ETF, KOSPI TR proxy | 원화 자산, 국내 접근성 | 글로벌 중복·반도체 집중 | 국내 ETF 과세 구조 확인 | 높음 | 기본 후보 |
| 미국 단기채 | SHY, BIL, SGOV 등 | 현금 대체, 낮은 변동성 | 환율 영향 | 해외 ETF 세금·환전 | 높음 | 최소/기본 후보 |
| 미국 중기채 | IEF, AGG, BND 등 | 방어·이자수익 후보 | 금리 상승기 손실 | 환율 효과 큼 | 높음 | 기본 후보 |
| 미국 장기채 | TLT, EDV 등 | 금리 하락기 방어 후보 | 금리 상승기 MDD 큼 | 환율·세금 | 중간~높음 | 기본/고급 후보 |
| 한국 국채·현금성 | KOSEF/TIGER 국채, MMF proxy | 원화 안정자산 | 데이터·상품별 차이 | 세금·분배금 | 높음 | 최소/기본 후보 |
| 금 | GLD, IAU, KRX 금, 금 ETF | 위기·인플레 후보 | 변동성, 시기별 부진 | 원화 기준 성과 중요 | 높음 | 최소/기본 후보 |
| 원자재 | DBC, GSG, 개별 원자재 proxy | 인플레이션 헤지 후보 | 롤오버·세금·상품복잡도 | ETF 구조 확인 | 중간 | 고급 후보 |
| 리츠 | VNQ, 국내 리츠 ETF | 인컴·부동산 노출 | 주식과 상관 높을 수 있음 | 금리 민감도 | 중간 | 고급/검증 후 |
| TIPS | TIP, SCHP | 인플레이션 연동 후보 | 실질금리 상승기 손실 | 환율 | 중간 | 고급 후보 |
| 달러·환율 노출 | USD/KRW, 달러 ETF, 현금 | 원화 투자자 위험관리 | 수익자산 아님 | 환전비용·세금 | 높음 | 리포트/보조 후보 |
| 배당·저변동·퀄리티 주식 | SCHD, USMV, QUAL 등 | 스타일 분산 후보 | 팩터 사이클 | 세금·중복 | 중간 | 고급 후보 |

### 자산군 버전 제안

| 버전 | 후보 자산군 | 목적 |
|---|---|---|
| 최소 버전, 3~4개 | 글로벌 또는 미국 주식, 한국/미국 단기채 또는 현금, 중기채, 금 또는 달러 노출 | 데이터 수집·백테스트 MVP |
| 기본 버전, 5~7개 | 미국 주식, 글로벌/선진국 주식, 한국 주식 또는 신흥국, 단기채/현금, 중기채, 장기채, 금 | 주식·채권·금·원화/달러 분산 검증 |
| 고급 버전, 8개 이상 | 기본 버전 + 리츠, TIPS, 원자재, 배당/저변동/퀄리티, 달러, 한국 국채 세분화 | 분산효과와 중복성 검증 |

---

## F. 기준 배분 후보

| 배분 방식 | 예시 | 장점 | 단점 | 필요한 데이터 | 백테스트 우선순위 |
|---|---|---|---|---|---:|
| 1/N 동일가중 | 선택 자산군 균등 | 단순, 과최적화 낮음 | 위험이 균등하지 않음 | 모든 자산 가격 | 매우 높음 |
| 60/40류 | 주식 60, 채권 40의 변형 후보 | 강력한 벤치마크 | 주식-채권 동반 하락 취약 | 주식·채권 장기 데이터 | 매우 높음 |
| 공격형 | 주식 비중 높음 | 장기 성장성 | MDD 큼 | 주식·채권·금 | 높음 |
| 중립형 | 주식·채권·금 균형 | 비교 기준으로 적합 | 자산군 정의에 민감 | 전체 가격 데이터 | 높음 |
| 보수형 | 채권·현금 비중 높음 | 변동성 낮음 | 인플레이션·기회비용 | 단기채·현금 수익률 | 높음 |
| 올웨더류 | 주식, 장기채, 중기채, 금, 원자재 후보 | 국면 분산 아이디어 | 레버리지 없는 구현 시 성격 달라짐 | 채권·금·원자재 | 중간~높음 |
| 리스크 패리티류 | 위험기여도 균등 | 리스크 균형 | 추정오차·레버리지 문제 | 변동성·상관관계 | 중간 |
| 변동성 역가중 | inverse volatility | 단순 위험 조정 | 저변동 자산 쏠림 | 가격 데이터 | 높음 |
| 목표 변동성 | 목표 변동성 범위로 위험자산 축소 | 손실관리 후보 | turnover·현금비중 증가 | 가격·현금 수익률 | 중간~높음 |
| 한국 투자자 원화 기준 | 원화 환산 수익률 기반 | 실제 체감성과 반영 | 환헤지 여부 복잡 | 환율·ETF 가격 | 매우 높음 |
| 사용자 직접 입력 | 사용자가 기준 비중 지정 | 개인화 | 비교 어려움 | 사용자 설정 | 높음 |
| 연령·위험성향 기반 | 공격/중립/보수 preset | UX에 유리 | 임의성 | 사용자 프로필 | 중간 |

---

## G. 백테스트 실험 설계

| 실험 | 목적 | 입력 데이터 | 비교 대상 | 성과 지표 | 통과 기준 | 실패 시 조치 |
|---|---|---|---|---|---|---|
| 자산군 선택 실험 | 어떤 자산군이 실제 분산효과를 주는지 확인 | 자산군별 총수익률 또는 수정가격, 환율 | 최소/기본/고급 universe | CAGR, vol, Sharpe, MDD, correlation, crisis return | 추가 자산군이 비용 후 MDD 또는 위험조정성과 개선 | 중복 자산군 제거 또는 리포트 전용 |
| 기준 배분 비교 | 정적 기준 배분 후보 검증 | 동일 universe | 1/N, 60/40, 공격형, 중립형, 보수형, risk-based | CAGR, Sharpe, Sortino, Calmar, MDD | 여러 구간에서 안정적이고 설명 가능 | 사용자 선택형으로 유지 |
| 단일 팩터 실험 | 팩터 자체 효용 확인 | 가격·거시·환율 | buy and hold, 정기 리밸런싱 | 비용 전후 CAGR, MDD, turnover, crisis return | 비용 후 개선, 파라미터 민감도 낮음 | 제외·리포트 전용 |
| 팩터 조합 실험 | 소수 팩터 결합 효과 확인 | 단일 팩터 통과 후보 | 단일 팩터, 기준 배분 | Sharpe, Sortino, MDD, turnover, ablation | 단순 모델 대비 명확한 개선 | 조합 축소 |
| 리밸런싱 주기 실험 | 주간 보고와 실제 매매 빈도 분리 | 신호·가격 | 주간, 격주, 월간, 분기, 임계값 | 성과, MDD, turnover, 세금 | 성과 개선이 turnover 증가를 정당화 | 더 낮은 빈도 선택 |
| 거래비용 민감도 | 실전 가능성 확인 | 매매내역, 스프레드·수수료 가정 | 비용 0, 낮음, 중간, 높음 | 비용 후 CAGR, turnover | 보수적 비용에서도 유효 | 신호 축소·임계값 확대 |
| 세금 민감도 | 한국 투자자 세후 성과 확인 | 계좌·상품별 세금 프로필 | 세전, 기본 세후, 보수 세후 | 세후 CAGR, after-tax MDD | 세후 우위 유지 | 저회전 전략 우선 |
| 원화 vs 달러 기준 | 한국 투자자 체감성과 확인 | USD 가격, USD/KRW | USD 성과, KRW 성과 | CAGR, MDD, vol, 환율기여도 | 원화 기준에서도 붕괴하지 않음 | 환헤지/원화자산 추가 검토 |
| 최근 구간 실험 | 시장 구조 변화 반영 | 최근 5년·10년·20년 | 전체 구간 성과 | rolling Sharpe, 최근 MDD | 최근 구간 완전 붕괴 없음 | 보류 또는 리포트 전용 |
| 위기 구간 실험 | 방어력 확인 | 2008, 2020, 2022, 고금리·인플레, 달러강세 등 | 벤치마크 | crisis return, drawdown, recovery time | 위기 손실 또는 회복 개선 | 위험관리 팩터 재검토 |
| 파라미터 민감도 | 사후 최적화 방지 | 모멘텀·MA·vol 기간 grid | 인접 파라미터 | heatmap, dispersion, worst decile | 넓은 범위에서 안정 | 특정값 의존 전략 제외 |
| 팩터 제거 실험 | 팩터 중요도 평가 | 조합 모델 | full vs factor removed | ΔSharpe, ΔMDD, Δturnover | 제거 시 성과 악화가 명확 | 중요도 낮은 팩터 제거 |
| 데이터 소스 비교 | 데이터 오류·편향 확인 | Yahoo, Stooq, ETF 발행사, KRX | 소스별 결과 | 성과 차이, 누락일, 배당처리 | 결과 차이 허용범위 내 | 소스 교체·정제 |
| ETF vs 지수 대체 | ETF 출시 전 proxy 타당성 확인 | ETF, index proxy | ETF-only vs spliced | tracking error, 성과 차이 | proxy가 왜곡 작음 | 출시 이후 구간만 사용 |
| 자산군 축소 vs 확장 | MVP 단순성 검증 | 최소/기본/고급 universe | 3~4개 vs 5~7개 vs 8개 이상 | CAGR, MDD, Sharpe, turnover | 확장 복잡도가 성과 개선을 정당화 | 단순 universe 선택 |

---

## H. 팩터 선택 알고리즘

```python
# 목적:
# 백테스트 전에는 어떤 팩터도 확정 채택하지 않는다.
# 모든 팩터는 데이터 접근성, 계산 가능성, 비용, 세금, 환율, robust out-of-sample 성과로 분류한다.

candidate_factors = collect_candidate_factors_from_recent_research_and_practice(
    include_categories=[
        "price_momentum",
        "trend",
        "relative_strength",
        "volatility",
        "downside_risk",
        "drawdown",
        "correlation",
        "macro",
        "asset_specific"
    ],
    exclude_predefined_rules=True
)

for factor in candidate_factors:
    data_check = evaluate_data_accessibility(
        factor=factor,
        requirements={
            "retail_accessible": True,
            "free_or_low_cost": True,
            "automatable": True,
            "sufficient_history": True,
            "point_in_time_available_if_macro": True,
            "terms_acceptable": True
        }
    )

    if not data_check.accessible:
        classify(factor, "exclude_or_report_only")
        continue

    if not is_computable_with_limited_resources(factor):
        classify(factor, "exclude_or_advanced_only")
        continue

    if factor.requires_high_frequency_data or factor.requires_expensive_proprietary_data:
        classify(factor, "exclude_for_mvp")
        continue

    if factor.is_macro:
        factor = apply_publication_lag_or_vintage_data(factor)
        if not factor.point_in_time_safe:
            classify(factor, "report_only_or_retest_later")
            continue

    single_result = backtest_single_factor(
        factor=factor,
        asset_universes=["minimal", "basic", "advanced"],
        base_allocations=["equal_weight", "stock_bond_static", "risk_based", "user_selected"],
        rebalancing_candidates=["weekly", "biweekly", "monthly", "quarterly", "threshold"],
        signal_execution_lag="next_tradable_close_or_next_open",
        avoid_same_close_execution=True
    )

    robustness_result = run_robustness_tests(
        result=single_result,
        tests=[
            "walk_forward",
            "rolling_window",
            "recent_5y",
            "recent_10y",
            "crisis_periods",
            "parameter_sensitivity",
            "data_source_comparison",
            "krw_vs_usd",
            "cost_sensitivity",
            "tax_sensitivity"
        ]
    )

    cost_adjusted_result = apply_costs_taxes_and_fx(
        result=single_result,
        transaction_cost_scenarios=["low", "base", "high"],
        tax_profiles=["domestic_etf", "overseas_etf", "tax_deferred_account_if_applicable"],
        fx_basis=["USD", "KRW"]
    )

    if passes_core_threshold(
        cost_adjusted_result,
        robustness_result,
        required_properties={
            "improves_risk_adjusted_return": True,
            "reduces_or_controls_mdd": True,
            "does_not_explode_turnover": True,
            "not_parameter_fragile": True,
            "recent_period_not_collapsed": True,
            "explainable": True
        }
    ):
        classify(factor, "candidate_for_combination")
    elif useful_for_reporting(factor):
        classify(factor, "report_only_or_risk_warning_only")
    else:
        classify(factor, "reject_or_retest_later")


candidate_for_combination = get_factors_by_label("candidate_for_combination")

factor_combinations = build_simple_combinations(
    factors=candidate_for_combination,
    max_number_of_factors=3,
    allowed_models=[
        "rule_based_score",
        "rank_based_score",
        "volatility_weighted_score",
        "simple_logistic_regression_optional"
    ],
    avoid_black_box=True
)

for combo in factor_combinations:
    result = backtest_combination(
        combo=combo,
        asset_universes=["minimal", "basic", "advanced"],
        base_allocations=["equal_weight", "stock_bond_static", "risk_based", "user_selected"],
        rebalance_rules=["monthly", "quarterly", "threshold"],
        execution_lag="next_tradable_close_or_next_open"
    )

    result_after_cost = apply_costs_taxes_and_fx(result)

    robustness_result = run_robustness_tests(
        result_after_cost,
        tests=[
            "walk_forward",
            "out_of_sample",
            "parameter_sensitivity",
            "factor_ablation",
            "recent_period",
            "crisis_period",
            "krw_vs_usd",
            "data_source_comparison"
        ]
    )

    if (
        improves_risk_adjusted_return(result_after_cost)
        and improves_or_controls_drawdown(result_after_cost)
        and does_not_increase_turnover_excessively(result_after_cost)
        and is_robust(robustness_result)
        and is_explainable(combo)
        and is_maintainable_by_retail_investor(combo)
    ):
        classify(combo, "implementation_candidate")
    else:
        classify(combo, "reject_or_retest")


final_model = choose_simplest_model_among_candidates(
    candidates=get_combinations_by_label("implementation_candidate"),
    objective_order=[
        "after_tax_risk_adjusted_return",
        "mdd_control",
        "turnover_control",
        "parameter_robustness",
        "recent_period_survival",
        "simplicity",
        "explainability"
    ]
)

if final_model is None:
    final_model = choose_static_or_minimal_dynamic_baseline(
        reason="no factor combination survived robustness and cost tests"
    )

export_research_spec(
    final_model_candidate=final_model,
    rejected_factors=get_rejected_factors(),
    report_only_factors=get_report_only_factors(),
    risk_warning_factors=get_risk_warning_factors(),
    assumptions="all parameters remain candidates until validated"
)
```

---

## I. 리포트 생성 알고리즘

**핵심 구조:** 리포트는 많은 정보를 표시할 수 있지만, 실제 매매는 검증된 신호만 사용한다.

| 표시 항목 | 실제 매매 영향 여부 | 설명용 여부 | 위험 경고 여부 | 백테스트 필요 여부 | 데이터 필요 |
|---|---|---|---|---|---|
| 자산군별 최근 수익률 | 직접 영향 없음, 검증된 모멘텀 모델에 포함될 때만 영향 | 예 | 아니오 | 매매 신호로 쓰려면 필요 | 가격 데이터 |
| 자산군별 중장기 모멘텀 점수 | 검증 후 가능 | 예 | 아니오 | 필요 | 가격 데이터 |
| 자산군별 추세 상태 | 검증 후 가능 | 예 | 예 | 필요 | 가격 데이터 |
| 자산군별 변동성 | 검증 후 가능 | 예 | 예 | 필요 | 가격 데이터 |
| downside deviation | 검증 후 가능 | 예 | 예 | 필요 | 가격 데이터 |
| drawdown | 직접 영향은 보류, 위험 경고 우선 | 예 | 예 | 필요 | 가격 데이터 |
| rolling correlation | 직접 영향 보류 | 예 | 예 | 필요 | 가격 데이터 |
| 주식-채권 상관관계 상승 경고 | 직접 영향 없음 | 예 | 예 | 경고 신뢰도 검증 | 가격 데이터 |
| 금·달러·채권 방어력 | 검증 후 가능 | 예 | 예 | 필요 | 가격·환율 |
| 기준 배분 대비 현재 목표 비중 | 예, 최종 모델 산출값일 때만 | 예 | 아니오 | 필요 | 포트폴리오·가격 |
| 리밸런싱 필요 여부 | 예 | 예 | 아니오 | 필요 | 현재 비중·목표 비중 |
| 거래비용 추정 | 예 | 예 | 아니오 | 필요 | 수수료·스프레드 |
| 세금 추정 | 예 | 예 | 아니오 | 필요 | 계좌·상품 세금 프로필 |
| 원화 기준 성과 | 직접 영향보다는 평가 기준 | 예 | 예 | 필요 | 환율 |
| CPI, FOMC, 금통위 일정 | 직접 영향 없음 | 예 | 참고 경고 | 이벤트 전략이면 필요 | 공식 일정 |
| VIX 급등 | 직접 영향 보류 | 예 | 예 | 필요 | VIX 가격 |
| 다음 주 확인 항목 | 직접 영향 없음 | 예 | 예 | 불필요 또는 이벤트 테스트 | 일정·거시 |
| 투자 유의 문구 | 직접 영향 없음 | 예 | 예 | 불필요 | 고정 문구 |

```python
def generate_weekly_report(as_of_date, portfolio, market_data, factor_outputs, model_outputs):
    report = {}

    report["market_summary"] = summarize_recent_market_moves(
        returns=market_data.asset_returns,
        fx=market_data.fx_rates,
        macro_events=market_data.upcoming_events
    )

    report["asset_table"] = []
    for asset in portfolio.asset_universe:
        row = {
            "asset": asset,
            "recent_returns": compute_recent_returns(asset, candidate_windows="report_only_windows"),
            "trend_state": factor_outputs.get(asset, "trend_state_report"),
            "volatility": factor_outputs.get(asset, "realized_volatility"),
            "drawdown": factor_outputs.get(asset, "drawdown"),
            "correlation_warning": factor_outputs.get(asset, "correlation_warning"),
            "is_trading_signal": False
        }

        if model_outputs.final_model_uses(asset):
            row["target_weight"] = model_outputs.target_weights[asset]
            row["rebalance_needed"] = check_rebalance_threshold(
                current_weight=portfolio.current_weights[asset],
                target_weight=model_outputs.target_weights[asset],
                threshold=model_outputs.rebalance_threshold
            )
            row["is_trading_signal"] = True
        else:
            row["target_weight"] = None
            row["rebalance_needed"] = "not_applicable"

        report["asset_table"].append(row)

    report["trading_section"] = {
        "uses_only_backtested_signals": True,
        "target_weights": model_outputs.target_weights,
        "orders_required": model_outputs.orders,
        "estimated_transaction_cost": estimate_transaction_cost(model_outputs.orders),
        "estimated_tax_impact": estimate_tax_impact(model_outputs.orders, portfolio.tax_profile),
        "execution_note": "signals generated after market close; execute only at next tradable price"
    }

    report["explanatory_section"] = {
        "report_only_indicators": factor_outputs.report_only_indicators,
        "risk_warnings": factor_outputs.risk_warnings,
        "macro_events": market_data.upcoming_events
    }

    report["disclaimer"] = create_investment_risk_disclaimer()

    return report
```

---

## J. 최종 MVP 후보 구조

백테스트 전 단계이므로 아래는 **확정 알고리즘이 아니라 MVP 후보 구조**다.

| 구성 요소 | 후보안 | 확정 여부 | 백테스트 필요 항목 |
|---|---|---|---|
| 자산군 | 3~7개: 글로벌/미국 주식, 한국 또는 미국 단기채·현금, 중기채, 장기채 후보, 금, 한국 주식 또는 선진국/신흥국 후보 | 미확정 | 자산군 선택 실험, 원화 기준 성과 |
| 가격 데이터 | Yahoo/yfinance + Stooq + ETF 발행사 + KRX 교차검증 | 미확정 | 데이터 소스별 결과 차이 |
| 거시 데이터 | FRED, ALFRED, ECOS | 미확정 | 발표 지연·빈티지 데이터 처리 |
| 기준 배분 | 1/N, 60/40류, 공격/중립/보수, 사용자 입력 | 미확정 | 기준 배분 비교 |
| 팩터 수 | 1~3개: 모멘텀/추세, 변동성, drawdown 또는 상대강도 | 미확정 | 단일 팩터·조합 실험 |
| 리밸런싱 | 월간, 분기, 임계값, 주간 리포트+월간 매매 | 미확정 | turnover·세금 민감도 |
| 매매 신호 | 백테스트 통과 신호만 사용 | 원칙 확정 | 신호별 비용 후 성과 |
| 리포트 지표 | 수익률, 추세, 변동성, 낙폭, 상관관계, 이벤트, 위험경고 | 미확정 | 표시 항목과 매매 신호 분리 |
| 비용 | 수수료, 스프레드, 환전비용 | 미확정 | 비용 민감도 |
| 세금 | 상품·계좌별 tax_profile | 미확정 | 세후 성과 |
| 환율 | USD 성과와 KRW 성과 모두 산출 | 원칙 확정 | 원화 vs 달러 비교 |
| 모델 형태 | 규칙 기반, 점수화, 랭킹, 변동성 역가중, 단순 로지스틱 회귀 후보 | 미확정 | 단순 모델 대비 개선 |
| 제외 모델 | 고빈도, 대규모 딥러닝, 옵션 의존, 실시간 NLP, 레버리지·숏 전제 | MVP 제외 | 고급 연구에서만 별도 |

---

## K. 구현 우선순위

| 단계 | 목표 | 구현 내용 | 완료 기준 |
|---:|---|---|---|
| 1단계 | 데이터 수집 | 가격, 환율, 금리, 물가, ETF 분배금, KRX/ECOS/FRED 연동 | 동일 자산에 대해 최소 2개 소스 교차검증 |
| 2단계 | 자산군 후보 백테스트 | 최소/기본/고급 universe 생성 | 자산군별 USD·KRW 성과표 산출 |
| 3단계 | 기준 배분 후보 비교 | 1/N, 60/40류, 공격/중립/보수, risk-based | 비용 전 정적 벤치마크 성과표 산출 |
| 4단계 | 단일 팩터 테스트 | 모멘텀, 추세, 변동성, drawdown, correlation, macro | 단일 팩터별 성과·turnover·민감도 산출 |
| 5단계 | 팩터 조합 테스트 | 1~3개 팩터 조합, 점수화·랭킹 | ablation 및 walk-forward 통과 후보 선별 |
| 6단계 | 비용·세금·환율 반영 | 거래비용, 환전비용, 세금 profile, 원화 환산 | 세후·원화 기준 결과 산출 |
| 7단계 | MVP 알고리즘 선택 | 가장 단순하고 robust한 후보 선택 | 최근·위기·비용·세금 테스트 통과 |
| 8단계 | 주간 리포트 생성기 연결 | 매매 신호와 설명용 지표 분리 출력 | 리포트가 “매매 영향 여부”를 명시 |

---

## L. 기계 판독용 JSON 요약

```json
{
  "research_goal": "Select practical asset allocation factors through backtesting under retail data and computation constraints",
  "html_role": "output_format_reference_only",
  "do_not_predefine": [
    "specific_base_allocation",
    "specific_moving_average_period",
    "specific_momentum_period",
    "specific_macro_indicator",
    "specific_asset_weight",
    "specific_rebalancing_frequency",
    "specific_academic_paper_as_final_authority"
  ],
  "candidate_factor_categories": {
    "price_based": [
      "multi_horizon_returns",
      "moving_average_distance",
      "price_to_recent_high",
      "new_high_new_low",
      "trend_duration",
      "trend_strength"
    ],
    "momentum_based": [
      "absolute_momentum",
      "relative_strength",
      "skip_recent_period_momentum",
      "cross_asset_time_series_momentum",
      "risk_adjusted_momentum"
    ],
    "trend_based": [
      "price_vs_moving_average_candidate_ranges",
      "trend_score",
      "multi_window_trend_consensus",
      "drawdown_recovery_trend"
    ],
    "volatility_based": [
      "realized_volatility",
      "relative_volatility",
      "volatility_spike",
      "inverse_volatility_weighting",
      "volatility_targeting_without_leverage",
      "downside_deviation"
    ],
    "drawdown_based": [
      "current_drawdown",
      "maximum_drawdown",
      "drawdown_duration",
      "recovery_time",
      "tail_loss_proxy"
    ],
    "correlation_based": [
      "stock_bond_rolling_correlation",
      "stock_gold_rolling_correlation",
      "stock_reit_rolling_correlation",
      "diversification_breakdown_warning",
      "portfolio_risk_contribution"
    ],
    "macro_based": [
      "policy_rate",
      "yield_curve",
      "inflation",
      "real_rate_proxy",
      "credit_spread",
      "dollar_index",
      "usd_krw",
      "macro_event_calendar",
      "macro_regime_candidate"
    ],
    "asset_specific": [
      "equity_momentum",
      "equity_volatility",
      "bond_duration_sensitivity",
      "bond_momentum",
      "gold_momentum",
      "gold_real_rate_sensitivity",
      "reit_rate_sensitivity",
      "reit_equity_correlation",
      "cash_yield"
    ]
  },
  "candidate_asset_universe": {
    "minimal_version": [
      "global_or_us_equity",
      "short_term_bond_or_cash",
      "intermediate_bond",
      "gold_or_usd_cash_candidate"
    ],
    "basic_version": [
      "us_equity",
      "global_or_developed_equity",
      "korea_equity_or_emerging_equity",
      "short_term_bond_or_cash",
      "intermediate_bond",
      "long_term_bond_candidate",
      "gold"
    ],
    "advanced_version": [
      "us_equity",
      "global_equity",
      "developed_ex_us_equity",
      "emerging_equity",
      "korea_equity",
      "us_short_term_bond",
      "us_intermediate_bond",
      "us_long_term_bond",
      "korea_government_bond",
      "cash",
      "gold",
      "commodities",
      "reits",
      "tips",
      "dividend_equity",
      "low_volatility_equity",
      "quality_equity",
      "usd_exposure"
    ]
  },
  "candidate_base_allocations": {
    "equal_weight": {
      "status": "candidate",
      "test_required": true
    },
    "stock_bond_static": {
      "status": "candidate_family",
      "examples": [
        "60_40_like",
        "stock_heavy",
        "bond_heavy"
      ],
      "test_required": true
    },
    "risk_based": {
      "status": "candidate_family",
      "examples": [
        "risk_parity_like",
        "risk_contribution_control"
      ],
      "test_required": true
    },
    "volatility_weighted": {
      "status": "candidate",
      "test_required": true
    },
    "user_selected": {
      "status": "candidate",
      "test_required": true
    }
  },
  "data_constraints": {
    "must_be_accessible_to_retail_investor": true,
    "prefer_free_or_low_cost_data": true,
    "must_be_automatable": true,
    "avoid_high_frequency_data": true,
    "avoid_expensive_proprietary_data_for_mvp": true,
    "consider_krw_based_performance": true,
    "consider_taxes_and_transaction_costs": true
  },
  "backtest_required_experiments": [
    "asset_universe_test",
    "base_allocation_test",
    "single_factor_test",
    "factor_combination_test",
    "rebalancing_frequency_test",
    "transaction_cost_sensitivity",
    "tax_sensitivity",
    "krw_vs_usd_test",
    "recent_period_test",
    "crisis_period_test",
    "parameter_sensitivity_test",
    "factor_ablation_test",
    "data_source_comparison_test"
  ],
  "factor_classification_labels": [
    "mvp_candidate",
    "basic_model_candidate",
    "advanced_model_candidate",
    "report_only",
    "risk_warning_only",
    "exclude",
    "retest_later"
  ],
  "model_selection_principle": {
    "primary": "choose_the_simplest_model_that_survives_robustness_tests",
    "secondary": "prefer_factors_with_accessible_data_low_turnover_and_clear_explainability",
    "avoid": [
      "overfitting",
      "excessive_turnover",
      "unavailable_data",
      "high_computation_cost",
      "unexplainable_complexity",
      "using_old_literature_as_final_authority",
      "predefining_specific_rules_without_backtesting"
    ]
  },
  "final_output_needed_for_next_ai": {
    "factor_priority_table": true,
    "data_accessibility_table": true,
    "asset_universe_candidates": true,
    "base_allocation_candidates": true,
    "backtest_design": true,
    "factor_selection_algorithm": true,
    "weekly_report_generation_rules": true,
    "mvp_candidate_structure": true,
    "implementation_priority": true
  }
}
```
