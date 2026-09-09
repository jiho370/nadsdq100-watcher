---
title: "개인 투자자용 주간 자산배분 결산 및 조언 알고리즘"
source: "uploaded text"
---

# 개인 투자자용 주간 자산배분 결산 및 조언 알고리즘: 백테스트 설계 및 리서치 명세


## A. 핵심 결론

본 리서치는 일반 개인 투자자가 무료 또는 저비용 데이터 인프라와 제한된 연산 환경을 활용하여 주간 자산배분 리포트를 생성하고, 실제 매매에 활용할 수 있는 실용적 알고리즘의 후보군과 백테스트 명세를 도출하는 것을 목적으로 한다. 문헌 조사, 최신 실무 사례, 그리고 한국 개인 투자자의 제약 조건(데이터 접근성, 거래비용, 세금, 환율)을 종합하여 다음의 핵심 질문에 대한 결론을 도출하였다.
첫째, 우선적으로 백테스트해야 할 팩터는 단순성과 강건성이 입증된 가격 기반의 지표들이다. 구체적으로는 복수 기간을 조합한 '절대 모멘텀' 및 '상대 모멘텀', 그리고 '추세 추종(이동평균 기반)'과 '변동성' 팩터가 최우선 검토 대상이다. 과거 수익률 기반의 가격 지표는 계산 복잡도가 낮으면서도 추세 지속성을 추종하고 하방 위험을 방어하는 데 효과적임이 다수의 연구에서 입증되었다. 특히, 최근 연구인 BAA(Bold Asset Allocation)나 DAA(Defensive Asset Allocation) 등에서 제안된 '카나리아 자산군(Canary Universe)'을 활용한 시장 폭(Breadth) 모멘텀은 하락장 방어를 위한 혁신적인 조기 경보 시스템으로, 최우선 백테스트 후보로 분류된다.
둘째, 데이터 접근성 및 지연 문제로 인해 고빈도(High-frequency) 마이크로스트럭처 데이터, 복잡한 옵션 내재 변동성(Implied Volatility) 데이터, 실시간 자연어 처리(NLP) 기반의 뉴스 센티먼트 팩터 등은 개인 투자자용 MVP(Minimum Viable Product) 모델에서 후순위로 미루거나 완전히 제외해야 한다. 일반 개인 투자자는 Yahoo Finance, Stooq, FRED와 같은 무료 API에 의존해야 하므로, 종가 기준의 일간/주간/월간 시계열 데이터로 연산 가능한 지표에 역량을 집중해야 한다.
셋째, 거시경제 지표(예: 기준금리, CPI, 고용지표, 장단기 금리차)와 기업의 밸류에이션 지표(PER, PBR 등)는 실제 매매 신호가 아닌 주간 리포트의 '설명용 지표' 및 '위험 경고용 지표'로만 활용하는 것이 적합하다. 거시 지표는 발표 지연(Publication lag)과 사후 수정(Revision)이 빈번하게 발생하여, 이를 매매 신호로 사용할 경우 과거의 시점에 미래의 데이터를 참조하게 되는 치명적인 '미래 참조 편향(Look-ahead bias)'을 유발할 위험이 매우 높다. 따라서 팩터로서의 엄밀한 통제가 불가능하다면 리포트의 내러티브를 구성하는 용도로만 한정해야 한다.
넷째, 주간 리포트의 생성 알고리즘과 실제 매매(리밸런싱) 알고리즘은 엄격하게 분리되어야 한다. 주간 리포트는 투자자의 심리적 안정과 시장 이해를 돕기 위해 단기적인 시장의 노이즈(일간 변동, 단기 매크로 이벤트 등)를 모두 표시해야 한다. 그러나 이러한 단기 정보를 실제 포트폴리오 리밸런싱에 매주 직접 반영하면 회전율(Turnover)과 거래비용, 세금 부담이 급증하여 장기 복리 수익률을 심각하게 훼손한다. 특히 한국 개인 투자자의 경우 해외 상장 ETF 매매 시 250만 원 공제 후 22%의 양도소득세가 부과되며, 국내 상장 해외 ETF의 경우 15.4%의 배당소득세가 발생하므로 잦은 매매는 성과에 치명적이다. 따라서 실제 매매는 월간 주기로 제한하거나 특정 괴리율 임계값(Threshold)을 초과할 때만 수행하도록 분리 설계해야 한다.
다섯째, 최종 팩터 및 파라미터(이동평균 기간 등)의 채택은 특정 문헌이나 고전적 경험칙을 절대적 근거로 삼지 않고, 데이터 스누핑(Data-snooping) 편향을 통제한 전진 분석(Walk-forward optimization)과 거래비용 차감 후의 위험 조정 수익률(예: GT-Score, Deflated Sharpe Ratio)을 기준으로만 판단해야 한다. 특정 이동평균선(예: 200일선)을 정답으로 고정하지 않고 파라미터 민감도 분석을 통해 성과가 붕괴하지 않는 평탄한 구간(Robust plateau)을 선택하는 과정 자체가 알고리즘화되어야 한다.
여섯째, 개인 투자자용 MVP 알고리즘의 복잡도는 철저히 통제되어야 한다. 최신 딥러닝 모델이나 대규모 앙상블 기계학습 모델은 비선형적 패턴을 찾을 수 있다는 장점이 있으나, 금융 시계열 특유의 낮은 신호 대 잡음비(Signal-to-noise ratio) 환경에서는 과최적화(Overfitting)의 늪에 빠지기 쉽다. 구현과 유지보수가 용이하며 '설명 가능성(Explainability)'이 높은 규칙 기반의 팩터 랭킹 모델이나 단순 점수화 모델이, 거래비용과 세금을 고려한 현실적인 수익률 측면에서 복잡한 블랙박스 모델보다 우월할 가능성이 높다.
일곱째, 최신 연구와 실무 자료에서 공통적으로 중요하게 보는 후보군은 꼬리 위험(Tail-risk) 관리와 시장 국면(Regime)에 대한 동적 대응이다. 전통적인 자산배분이 금리와 주식의 음의 상관관계를 가정한 반면, 최근 연구들은 금리 급등기나 인플레이션 상승기에 주식과 채권이 동반 하락하는 리스크를 방어하기 위해 특정 자산(원자재, 달러) 또는 절대 모멘텀 필터를 적극적으로 활용하는 추세이다.
마지막으로, 마코위츠의 현대 포트폴리오 이론(MPT)이나 전통적인 60/40 포트폴리오와 같은 오래된 고전 문헌은 분산 투자의 수학적 원리와 역사적 벤치마크로서의 훌륭한 배경 이론을 제공한다. 하지만 자산 간 상관관계가 동조화되고 변동성이 군집화되는 현대 금융 시장의 구조적 변화를 고려할 때, 이를 특정 배분 방식의 절대적 근거로 삼거나 현대 알고리즘에 표본 외 검증 없이 직접 채택하는 것은 지양해야 한다.

## B. 최신 자료 기반 후보 팩터 요약

특정 논문을 정답으로 제시하지 않으며, 최근 5~10년 이내의 동적 자산배분, 팩터 투자 및 과최적화 검증 연구를 폭넓게 조사하여 백테스트 설계를 위한 방법론적 힌트를 추출하였다. 과거의 고전 문헌은 이론적 출발점으로만 참조하며, 최신 문헌이 제시하는 시장의 구조적 변화와 위험 관리 기법을 알고리즘 후보로 수집한다.
자료 유형출처 또는 연구 방향최신성핵심 시사점알고리즘 후보로의 의미주의점실증 연구VAA, DAA, BAA 등 다중 모멘텀 및 동적 자산배분 연구 (W. Keller 등)최근 5년 이내단순 12개월 모멘텀보다 단기 수익률에 고가중치를 부여하는 빠른 모멘텀(1, 3, 6, 12개월 혼합)이 하락 방어와 추세 추종에 모두 유리함. 카나리아 자산의 시장 폭을 위험 회피 스위치로 사용함.공격 자산, 방어 자산, 카나리아 자산으로 유니버스를 분리하고 모멘텀에 따라 비중을 조절하는 동적 배분 팩터 후보로 채택 가능.
회전율이 극단적으로 높아질 수 있으므로, 한국의 양도소득세 및 배당소득세(15.4%) 차감 후 순수익률을 필수 검증해야 함.
실증 연구자산 클래스 트렌드 추종 (Asset Class Trend-following) 메타 연구최근 5년 이내트렌드 추종 및 절대 모멘텀은 장기 성과를 높이기보다는 최대낙폭(MDD)과 변동성을 획기적으로 줄이는 위험 관리 도구로서 작동함.
단순 이동평균(SMA) 및 지수 이동평균(EMA) 돌파 여부를 개별 자산의 투자 여부(Risk-on/off)를 결정하는 이진 필터로 활용할 후보.특정 기준선(예: 200일, 10개월)을 고정하지 말고, 범위 내에서 파라미터 민감도 및 견고성(Robustness)을 평가해야 함.계량 방법론데이터 스누핑 방지 및 과최적화 통제 (GT-Score, Deflated Sharpe 등)최근 3년 이내다수의 파라미터를 테스트하여 사후적으로 가장 좋은 결과를 선택하는 다중 검정(Multiple testing)은 실전에서 반드시 붕괴함. 수익률의 비정규성과 꼬리 위험을 반영한 새로운 목적 함수가 필요함.
팩터 선택 및 조합 단계에서 단순 누적 수익률(CAGR) 대신 워크포워드 검증 결과와 페널티가 부여된 스코어(GT-Score 등)를 평가 기준으로 도입해야 함.
이는 매매 신호 자체가 아니라 백테스트 엔진 내 모델 평가의 코어 프레임워크로 기능해야 함.학술/실무거래비용 및 세금 환경에서의 최적화 전략최근 10년 이내빈번한 리밸런싱은 거래비용으로 인해 성과를 갉아먹음. 목표 비중과 실제 비중 간에 '노-트레이드 밴드(No-trade band)'를 설정하는 것이 비용 차감 후 최적임.
매월 기계적 리밸런싱을 수행할지, 신호가 임계값을 넘을 때만 매매할지를 결정하는 리밸런싱 주기 백테스트의 이론적 근거.
한국 투자자의 ISA, IRP 등 과세 이연 계좌 환경과 일반 계좌 환경을 분리하여 비용을 모델링해야 함.
실무 자료환율 민감도 및 국내 상장 해외 ETF (환노출/환헤지) 성과최근 2년 이내위기 구간에서 원달러 환율 상승은 포트폴리오의 방어막 역할을 하나, 한미 금리 역전기에는 환헤지 비용이 막대하게 발생하여 장기 성과를 훼손할 수 있음.
환노출(UH) 지수와 환헤지(H) 지수를 독립적인 자산군 후보로 상정하여 각각의 백테스트를 수행. 환율 모멘텀 팩터를 별도로 고려.벤치마크 평가 시 달러 기준 성과와 원화 기준 성과를 명확히 분리하여, 환차익/환차손의 영향을 통제해야 함.


## C. 후보 팩터 우선순위표

어떤 팩터도 사전에 정답으로 간주하지 않는다. 다음은 논문 및 실무에서 널리 활용되는 팩터들을 백테스트를 위해 체계적으로 정리한 풀(Pool)이다. 각 팩터의 기간이나 계산 방식은 특정 상수로 고정하지 않고, 파라미터 최적화 범위로 설정한다.
우선순위팩터후보 기간/계산법데이터 소스예상 장점예상 위험백테스트 필요성MVP 적합성1순위복수 기간 가중 절대/상대 모멘텀1개월~12개월 수익률의 가중합 (예: 1M, 3M, 6M, 12M 변수 스윕)Yahoo Finance, Stooq최근 시장 변화에 민감하게 반응하면서 장기 추세를 추종하여, 대세 하락장 이전에 방어 자산으로 회전 가능.
횡보장에서 잦은 진입/청산 신호를 발생시켜 휩소(Whipsaw)로 인한 비용 및 세금 누수 유발.
파라미터 민감도 테스트 필수. 거래비용/세금 차감 후 벤치마크 초과 성과 검증.기본 버전 채택 후보1순위이동평균선(MA) 이격도 및 교차단기(20~60일)와 장기(120~250일) SMA/EMA 비율 또는 가격과의 괴리율Yahoo Finance, Stooq직관적인 추세 필터로, 설명 가능성이 높으며 거시적 하락 곡선을 방어하는 데 강력함.
시차(Lag)가 존재하여 위기 발생 초기 손실을 온전히 맞은 후 사후적으로 신호가 발생할 가능성.고정된 200일선이 아닌 파라미터 스윕을 통한 평탄한 수익 구간(Robustness) 탐색.기본 버전 채택 후보2순위변동성 및 변동성 역가중과거 20~120일 롤링 표준편차, ATR, 하방편차(Downside deviation)Yahoo Finance, KRX특정 고변동성 자산이 포트폴리오 위험을 독식하는 것을 막고 리스크 패리티를 추구.
위기 발생 직전의 고요한 장세(저변동성)에서 위험 자산 비중을 과도하게 늘릴 위험.단순 동일가중(1/N) 대비 MDD 개선 및 샤프 비율 상승 여부.고급 버전 채택 후보2순위상관관계 스코어링60~252일 간 자산 페어별 롤링 상관계수 평균Yahoo Finance실질적 분산 효과가 작동하는 저상관 자산군 위주로 자금을 배분하여 비체계적 위험 감소.
위기 시 자산 간 상관관계가 1로 수렴(동조화)하는 경향이 있어, 역사적 상관관계의 미래 예측력이 떨어짐.상관관계 페널티를 추가했을 때 연산 복잡도 대비 성과 개선폭(Calmar Ratio 등) 검증.고급 버전 또는 보류3순위카나리아 자산 시장 폭 (Breadth)지정 위험 감지 자산(예: VWO, BND, 고수익채 등)의 모멘텀 음수 개수Yahoo Finance위험 자산 전체의 비중을 일괄 축소하는 효율적이고 기민한 조기 경보 스위치 역할.
카나리아 자산 자체의 노이즈로 인해 포트폴리오 전체가 불필요하게 현금화되는 기회비용 발생.카나리아 자산의 수(1~4개) 및 임계값 변화에 따른 수익률/MDD 궤적 변화 관찰.리포트 설명용 또는 보조 신호4순위매크로 및 거시 경제 지표장단기 금리차, CPI 추세, 실업률 추세, FRED 금융환경지수FRED, 한국은행 ECOS근본적인 경기 국면(Regime) 판단을 통해 매크로 사이클에 부합하는 자산 선택 지원.
데이터 발표 지연(Lag) 및 사후 수정치(Revision)로 인해 과거 백테스트 시 미래 참조 편향 발생 위험.
발표 시차를 최소 1~2개월 지연 적용한 후 실증적 방어력 검증. 복잡성 증가.위험 경고 및 리포트 설명용제외고빈도 실현 변동성 / 옵션 내재 데이터분봉 데이터 기준 변동성 터미널 구조, VIX 파생 지표 등제한됨 (유료 API 필요)단기 미시 구조의 즉각적인 리스크 감지.개인 투자자 접근 불가. 연산 비용 막대. 주간/월간 리밸런싱 모델에 부적합.불필요. 연산 자원의 한계로 배제.제외 확정


## D. 데이터 접근성 평가표

알고리즘 시스템이 개인 투자자의 일반 PC나 저사양 클라우드 환경에서 영속적으로 구동되기 위해서는 데이터 소스의 안정성과 무료/저비용 접근성이 필수적이다.
데이터무료 접근 가능성자동화 가능성장기 데이터 가능성한국 투자자 적용성총수익률 가능성문제점최종 판단Yahoo Finance (yfinance)높음 (Rate limit 주의)매우 높음 (Python API)미국 중심 데이터는 1990년대 이전 데이터도 풍부함원/달러 환율 제공. 국내 지수 및 일부 주식 제공.
Adjusted Close 필드를 통해 배당 및 분할이 모두 반영된 총수익률(TR) 시계열 획득 가능.
비공식 API의 특성상 크롤링 차단이나 구조 변경 리스크가 상존함.
1차 핵심 소스 (MVP 채택)Stooq.com (pandas-datareader)높음높음글로벌 지수, 통화, 상품 등에 특화되어 풍부함Yahoo 대비 누락된 국내 ETF가 있을 수 있으나 매크로 프록시로 유용함.
배당 반영 여부가 자산별로 상이할 수 있어 데이터 클렌징 단계에서 교차 검증 요망.다운로드 횟수 제한 및 희귀 자산 ETF의 시계열 단절 가능성 존재.
2차 보조 소스 (교차 검증용)FRED (St. Louis Fed)높음 (무료 API 키)매우 높음1950년대 이전 과거 매크로 지표까지 포괄적 제공글로벌 거시 지표 포함, 미국 경제 상황 파악에 필수적(가격 데이터가 아닌 매크로 지표용)발표 시점 및 사후 데이터 수정 문제로 인해 시점 정렬(Point-in-time) 주의 요망.거시 설명용 (MVP 채택)한국은행 ECOS / KRX높음 (Open API 발급)보통 (일일 요청 한도 존재)한국 고유의 금리, 지수, 환율 데이터 획득에 유리완벽한 국소화 데이터로 한국 투자자 세팅에 필수적.KRX 지수 데이터의 경우 배당 반영 총수익(TR) 지수를 별도로 매핑해야 함.API 스펙이 다소 복잡하고 속도 지연 이슈가 발생할 수 있음.한국 환경 팩터 테스트용Alpha Vantage / Investing.com제한적 (일일/월간 무료 한도)보통 (Alpha Vantage는 공식, Investing은 크롤러)자산 및 요금제에 따라 상이한국 자산 일부 지원, 데이터는 양호한도 제약으로 인해 긴 과거 시계열을 일괄 다운로드하기 어려움.
배치 잡(Batch job) 실행 도중 한도 초과로 파이프라인이 멈출 리스크가 높음.제외 또는 후순위 보류


## E. 자산군 후보 평가표

백테스트를 통해 그 효용이 입증되어야 할 자산군의 후보 명단이다. 자산군 역시 미리 고정하지 않으며, 각각의 자산이 포트폴리오의 수익률 및 분산 효과에 기여하는 바를 데이터로 판단해야 한다.
자산군대표 데이터 후보 (Ticker 예시)장점단점한국 투자자 이슈 (세금/환율)백테스트 우선순위최종 후보 여부미국 대형주SPY, IVV, VOO세계 최고 유동성, 장기 우상향 팩터의 핵심, 장기 시계열 풍부.
2000, 2008, 2022년 등 굵직한 위기 시 심각한 MDD 발생.달러 노출 시 위기 방어 효과. 매매 차익에 대해 양도소득세 22% 부과.
최상MVP 핵심 자산군미국 장기 국채TLT, VGLT디플레이션 및 금융 위기 발생 시 주식과의 뚜렷한 역의 상관관계로 강력한 헷지.
금리 급등기 및 구조적 인플레이션 발작 시 주식과 동반 붕괴 위험.
환노출 전략 시 방어력 극대화되나, 한미 금리 역전 시 환헤지 ETF의 비용 부담 큼.
최상MVP 핵심 자산군미국 단기 국채 / 현금SHY, BIL, IEF위기 시 및 현금 비중 확대 시 변동성 없는 대피처 역할 수행.
이자 수익이 낮아 장기 보유 시 인플레이션을 상회하지 못함.달러 단기채는 한국 투자자에게 환율 변동성 자산으로 변모하므로 원화 예금과 분리 고려.높음방어 자산군 후보글로벌 주식 (미국 외)VEA, VWO, EEM미국 증시 횡보 구간(Lost decade) 대비책, 카나리아 자산으로 폭넓게 사용됨.
최근 10년 이상 미국 대형주 대비 극심한 언더퍼폼.신흥국 주식은 환율 리스크가 겹쳐 변동성이 증폭될 우려.중간기본 버전 후보금 및 원자재GLD, DBC, GSG인플레이션 발작, 지정학적 리스크, 달러 가치 훼손 시 독립적인 헷지 자산.
배당이나 이자가 없는 Negative carry. 변동성이 주식 수준으로 높음.금 ETF 투자 시 연금저축이나 ISA 등 절세 계좌 활용 시 배당소득세 과세 대상.
높음인플레이션 헷지 후보국내 상장 해외 ETFTIGER 미국S&P500, KODEX 미국채울트라30년원화로 손쉽게 거래. 연금저축/IRP/ISA 활용 시 과세 이연 및 저율 과세(3.3~5.5%) 혜택 막대.
대부분 2020년 이후 상장되어 자체 데이터로 장기 백테스트가 불가능. 대체 프록시 필수.일반 계좌에서 매매 차익이 배당소득세(15.4%)로 종합과세에 합산될 위험 존재.
최상한국 맞춤형 구현 핵심


## F. 기준 배분 후보

동적 배분 알고리즘(모멘텀, 추세 등)의 효용을 평가하기 위해서는 비교군이 되는 '기준 배분(Base Allocation)'의 성능이 먼저 벤치마킹되어야 한다. 하나의 배분을 강제하지 않고 다음 후보군들을 시뮬레이션하여 가장 견고한 닻(Anchor)을 찾는다.
배분 방식예시 구조장점단점필요한 데이터백테스트 우선순위동일 가중 (1/N)주식 25%, 장기채 25%, 금 25%, 현금 25%최적화 과정이 없으므로 과최적화 리스크 제로. 때로는 복잡한 모델을 압도함.
고변동성 자산(주식)이 전체 포트폴리오의 리스크를 사실상 장악함.대상 자산군 모두최우선 벤치마크전통적 60/40주식 60%, 중장기채 40% (재조정)역사적으로 가장 오랜 기간 검증된 자산배분 표준 모델. 설명이 직관적.
2022년과 같이 주식-채권 상관관계가 동조화(양의 상관)될 때 방어력 상실.
주식, 채권필수 벤치마크리스크 패리티 (Risk Parity)변동성 역가중 기반 비중 배분각 자산이 포트폴리오 리스크에 기여하는 비중을 균등하게 맞추어 꼬리 위험 축소.
현금성이나 채권 비중이 과도하게 커져 무위험수익률을 소폭 상회하는 데 그칠 수 있음.일간 변동성(표준편차)고급 백테스트 후보올웨더 변형 (All-weather)성장, 인플레 국면에 대응 (주식 30%, 장기채 40%, 중기채 15%, 금/원자재 15%)4가지 매크로 국면에 모두 대비하여 심리적 편안함을 제공.정적 배분이므로 추세 변화를 전혀 활용하지 못하며 장기 수익률이 비교적 낮음.4개 이상의 상이한 자산기본 버전 기준 후보


## G. 백테스트 실험 설계

모든 후보 팩터와 자산군이 과거 데이터에 편향(Overfitting)되지 않도록 체계적인 실험 프레임워크가 요구된다. 다음은 반드시 수행해야 할 15가지 핵심 백테스트 실험의 명세이다. 백테스트 과정에서 생존 편향, 미래 참조 편향, 데이터 스누핑 오류를 철저히 통제한다.
실험 목적입력 데이터비교 대상 (벤치마크)성과 지표통과 기준실패 시 조치1. 자산군 선택 실험장기 자산 시계열 (1990~)100% 주식 (SPY)샤프 비율, 자산 간 상관관계 매트릭스결합 시 벤치마크 대비 MDD 20% 이상 개선상관성이 높은 자산군 제거/통합2. 기준 배분 비교 실험고정 비중 포트폴리오1/N 동일가중 포트폴리오CAGR, MDD, Calmar Ratio단순 1/N 대비 위험조정 수익률 유의미 개선복잡한 고정 배분 방식 폐기3. 단일 팩터 검증 실험모멘텀, 이격도 등 개별 신호해당 팩터가 적용된 자산의 Buy & Hold팩터 성과 기여도, 초과 수익률단일 적용 시 벤치마크 대비 성과 향상
해당 팩터를 매매 신호에서 배제4. 팩터 조합 및 소거(Ablation)복수 팩터 로직 교차 적용단일 팩터 최상위 성과 모델결합 팩터의 Information Ratio조합 시 회전율 증가폭 대비 샤프 비율 개선 우위시너지가 없는 팩터 조합 해체5. 거래비용 민감도 실험신호에 따른 매매 내역, 슬리피지(0.2%)비용 반영 전 시뮬레이션Turnover, 비용 차감 후 CAGR비용 차감 후에도 초과 수익(알파) 유지
회전율 높은 팩터 제외6. 세금 민감도 실험 (한국형)수익 실현 건별 양도세(22%) 및 배당세(15.4%)
세금 없는 해외 논문 벤치마크실질 세후 CAGR, 세금 이연 효과세후 수익률이 Buy & Hold 세후 수익률 능가이익 실현 주기 하향 조정7. 환율 기준 통화 실험환율(USD/KRW) 합산 시계열순수 달러(USD) 기준 성과원화 기준 CAGR, 원화 기준 MDD달러 자산이 원화 환산 시 하락장 방어력 증대 확인환헤지(H) ETF 추가 고려
8. 리밸런싱 주기 실험주간, 월간, 분기, 임계값 트리거매월 말 정기 리밸런싱마찰 비용 대비 성과 향상
비용을 고려한 최적 주기가 신호 생성 주기와 부합주기를 고정 임계값(Threshold)으로 변경9. 최근 구간 (Post-2010) 실험2010년~현재 데이터 분할과거 장기 (1990~2009) 성과최근 10년/5년 수익률 및 승률과거 평균 대비 최근 5년 성과 하락폭 30% 이내구조적 변화로 인한 팩터 탈락10. 위기 구간 꼬리 위험 실험닷컴버블, 금융위기, 코로나, 2022년 발작기 데이터Buy & Hold위기 구간 최대낙폭(MDD), 회복 기간위기 시 MDD를 벤치마크 절반 이하로 방어
방어 자산군 스위칭 모듈 강화11. 파라미터 민감도 실험모멘텀(1~12개월), SMA(20~250일) 스윕최고 성과를 낸 특정 단일 파라미터파라미터 주변부 성과 평탄도파라미터를 ±10% 변형해도 성과 급락이 없어야 함과최적화로 간주하고 팩터 탈락12. 데이터 스누핑 (전진 분석)롤링 윈도우 기반 In-Sample / Out-of-Sample전체 기간 대상 In-Sample 최적화 모델Generalization Ratio, GT-Score
OOS 성과가 IS 성과 대비 심각하게 붕괴하지 않음모델 복잡도 대폭 하향 축소13. 데이터 소스 정합성 비교Yahoo vs Stooq vs FRED 데이터 교차기준 데이터 소스 백테스트소스 간 최종 수익률 오차율오차율 2% 이내로 동일 신호 도출결측치 및 배당(TR) 수정 로직 보완14. 프록시 대체 실험짧은 역사의 국내 상장 ETF vs 과거 지수 프록시원본 지수 데이터 적용 모델합성 프록시의 성과 궤적 동일성역사적 프록시가 현재 ETF의 움직임을 95% 이상 설명프록시 합성 로직(보수, 환율) 재수정15. 복잡도 (Occam's Razor) 실험팩터 1개 단순 모델 vs 팩터 3개 다중 모델최소 기능 팩터 모델복잡도 대비 초과 수익다중 모델이 복잡성을 감수할 만큼의 통계적 우위 증명가장 단순한 모델로 롤백


## H. 팩터 선택 알고리즘 (Pseudo-code)

실제 AI 또는 퀀트 개발자가 시스템 구축에 직접 참고할 수 있도록, 팩터를 수집하고 필터링하여 최종 채택하는 일련의 과정(Data Snooping 편향 차단 포함)을 Python 스타일 의사코드(Pseudo-code)로 명세한다.

```python
def factor_selection_pipeline():
    # 1. 문헌 연구 및 실무에서 제안된 넓은 범위의 후보 팩터 수집
    candidate_factors = collect_candidate_factors_from_recent_research_and_practice()
    implementation_candidates = []
    report_only_indicators = []

    for factor in candidate_factors:
        # 2. 데이터 접근성 필터 적용 (개인 투자자 환경 기준)
        data_check = evaluate_data_accessibility(factor, sources=["yfinance", "stooq", "fred"])

        if not data_check.accessible_with_free_api:
            classify(factor, "exclude_or_report_only")
            report_only_indicators.append(factor)
            continue

        if not is_computable_with_limited_resources(factor):
            classify(factor, "exclude_or_advanced_only")
            continue

        # 3. 단일 팩터의 기초 백테스트 (벤치마크 대비 성과 측정)
        single_result = backtest_single_factor(factor, benchmark=["buy_and_hold", "equal_weight"])

        # 4. 강건성 및 과최적화 검증 (전진 분석 및 GT-Score 평가)
        # IS(In-Sample)와 OOS(Out-of-Sample)를 비교하여 Data Snooping 편향 차단
        robustness_result = run_walk_forward_tests(factor)
        gt_score = calculate_gt_score(single_result, robustness_result)

        # 5. 거래비용 및 세금 모델링 (한국형 세제: 양도세 22%, 배당세 15.4% 등 고려)
        cost_adjusted_result = apply_korean_taxes_and_trading_costs(single_result, slippage=0.002)

        # 6. 단일 팩터 통과 기준 심사
        if passes_core_threshold(cost_adjusted_result, gt_score) and not is_turnover_excessive(cost_adjusted_result):
            classify(factor, "candidate_for_combination")
            implementation_candidates.append(factor)
        else:
            classify(factor, "reject_or_report_only")
            report_only_indicators.append(factor)

    # 7. 통과된 팩터들의 조합 구축 (Occam's razor: 복잡도 최소화)
    factor_combinations = build_simple_combinations(implementation_candidates, max_factors=3)
    final_mvp_candidates = []

    for combo in factor_combinations:
        result = backtest_combination(combo)
        result_after_cost = apply_korean_taxes_and_trading_costs(result)
        robustness_result = run_walk_forward_tests(combo)

        # 8. 최종 조합 평가 (단일 팩터 대비 시너지 확인)
        if improves_risk_adjusted_return_significantly(result_after_cost, implementation_candidates) and \
           does_not_increase_turnover_excessively(result_after_cost) and \
           is_robust(robustness_result):
            classify(combo, "implementation_candidate")
            final_mvp_candidates.append(combo)
        else:
            classify(combo, "reject_or_retest")

    # 9. 최종 채택: 생존한 모델 중 구동 비용이 가장 낮고 설명 가능한 '가장 단순한 모델' 채택
    final_model = choose_simplest_model_among_candidates(final_mvp_candidates)

    return final_model, report_only_indicators
```

## I. 리포트 생성 알고리즘 (표시 항목과 매매 신호 분리)
사용자의 행동 경제학적 측면을 고려할 때, 주간 리포트는 시장의 내러티브를 풍부하게 제공하여 투자자가 시스템을 이탈하지 않도록 돕는 역할을 한다. 반면, 실제 계좌의 비중을 조절하는 백엔드 매매 로직은 노이즈를 걸러낸 정제된 신호에만 반응해야 한다. 리포트 생성기와 트레이딩 엔진의 결합도를 낮추기 위해 데이터를 분리 설계한다.
표시 항목실제 매매 영향 여부설명용 여부위험 경고 여부백테스트 필요 여부데이터 필요 소스기준 배분 대비 현재 목표 비중예 (핵심 배분 신호)아니오아니오필수 (시뮬레이션 통과분)검증된 자산 가격 데이터리밸런싱 필요 여부 (Yes/No)예 (괴리율 임계값 초과 등)아니오아니오필수 (비용 최적화)포트폴리오 장고 및 가격자산군별 최근 수익률 (1w, 1m, 1y)아니오 (모멘텀 연산용 기초 재료)예아니오불필요Yahoo / Stooq 가격 데이터자산군별 추세 라벨 (상승/하락)아니오 (팩터가 채택되었을 때만 예)예아니오필수 (필터 채택 시)단순 이동평균(SMA) 연산 결과자산군별 낙폭 (MDD) 및 변동성아니오 (변동성 역가중일 때만 예)예예채택 시 필수롤링 표준편차 및 고점 괴리주요 매크로 경제 이벤트 (CPI 등)아니오 (미래 참조 편향 방지 목적)예예불필요 (표시 전용)FRED API / 경제 캘린더위험 경고 (카나리아 지표 발동 등)아니오 (팩터 융합 시에만 간접 영향)예예 (조기 경보)필요 (스코어링용)카나리아 자산군 모멘텀
J. 최종 MVP 후보 구조
수많은 팩터를 욱여넣은 과최적화 모델을 지양하고, 무료 데이터만으로 매주 백테스트 및 구동이 가능한 단순하면서도 강력한 MVP(Minimum Viable Product)의 후보 구조를 아래와 같이 제안한다. 모든 수치는 가변적이며 앞서 정의된 실험 설계에 의해 확정된다.
구성 요소후보안 (사전 결정되지 않음, 향후 확정)확정 여부백테스트 필요 항목자산군 (3~5개)대형 주식(SPY), 장기 채권(TLT), 단기 채권(SHY), 금(GLD)미확정각 자산군의 MDD 방어 및 상관계수(포트폴리오 다각화 기여) 검증기준 배분동일 가중(1/N) 대비 동적 배분 스코어 변동폭미확정정적 동일 가중 대비 OOS 샤프 비율 및 꼬리 위험(Tail risk) 방어 기여적용 팩터 1모멘텀 팩터 (예: 최근 1, 3, 6, 12개월의 비례 가중 합을 통한 상대적 우위 랭킹)미확정기간 파라미터의 민감도(±10% 변동 시 성과 평탄도), 휩소 발생 비율적용 팩터 2추세 필터 (예: 특정 자산이 N일 이동평균 이하일 경우, 해당 비중을 방어 자산으로 도피)미확정하락장 회피 효과 vs. 횡보장 잦은 매매에 따른 거래비용 증가분 비교리밸런싱 빈도가격 체크는 매주 하되, 목표 비중과 실제 비중 괴리가 특정 임계값(예: 5%)을 초과할 때만 매매미확정매주 기계적 리밸런싱 vs. 임계값(No-trade band) 적용 시의 순이익(Net CAGR) 비교
비용 및 환율 반영세금(해외ETF 양도소득세 22% 또는 국내상장 해외ETF 배당소득세 15.4%) 및 원화 환산 성과 확인미확정세금 및 거래비용 누락 시 발생하는 착시 현상 제거를 위한 철저한 시뮬레이션

K. 구현 우선순위
실제 퀀트 파이프라인 및 백테스트 환경을 코드 레벨에서 완성하기 위한 우선순위를 8단계로 제안한다.
단계목표구현 내용완료 기준1단계데이터 수집 파이프라인yfinance, pandas-datareader 연동. 결측치 및 배당 반영 수정주가(Adjusted Close) 20년 치 확보.
과거 20년 일간 가격/환율 시계열 무결성 달성2단계자산군 및 기준 배분 세팅1/N, 60/40 등 정적 포트폴리오의 과거 수익률 시뮬레이터 구축.벤치마크 모델의 20년 시계열 리포트 및 지표 출력3단계단일 팩터 엔진 코딩SMA 이격도, 1/3/6/12M 절대 모멘텀, 변동성 등 개별 신호 벡터화 연산 모듈 구현.각 팩터별 샤프 비율 및 단독 MDD 도출4단계비용·세금·환율 시뮬레이터회전율에 따른 슬리피지(0.2%) 차감 및 한국 세법(수익 실현 시 15.4% 또는 22%)을 반영한 Net-return 산출.세금/비용 적용 전후의 Equity Curve 비교 검증 통과5단계전진 분석 및 과최적화 검열롤링 윈도우 기반 Walk-forward 최적화 코딩. In-Sample vs OOS 성능 편차 스코어링(GT-Score 도출).
OOS 성과가 붕괴하지 않는 평탄한 파라미터 대역(Plateau) 확보6단계팩터 조합 테스트시너지를 내는 1~3개 팩터 모듈 결합. 앙상블 신호 기반의 동적 비중 산출 로직 완성.가장 단순하고 강건한 팩터 조합 1개 선정 (Occam's Razor)7단계MVP 알고리즘 최종 선택백테스트 통과 결과를 종합하여 시스템에 하드코딩될 최종 모델(규칙) 락인(Lock-in).시스템 명세서(Json 형식) 최종 출력8단계주간 리포트 프런트엔드 연결트레이딩 신호(매매용)와 거시 설명 데이터(HTML 렌더링용)를 분리하여 JSON API 형태로 제공.주간 결산 HTML과 매매 지시서 파일 동시 생성

L. 기계 판독용 JSON 요약
이후의 AI 또는 시스템 모듈이 개발 지침과 파라미터 탐색 범위를 즉시 파싱하여 활용할 수 있도록 작성된 기계 판독용 JSON 요약 명세이다. 내부 값은 고정값이 아니라 검증해야 할 '후보 범위 및 절차' 중심으로 기술되었다.
JSON

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
      "moving_average_crossover",
      "price_distance_from_ma"
    ],
    "momentum_based": [
      "absolute_momentum_multi_period",
      "relative_momentum_ranking"
    ],
    "trend_based": [
      "trend_duration",
      "trend_strength"
    ],
    "volatility_based": [
      "inverse_volatility_weighting",
      "rolling_standard_deviation",
      "downside_deviation"
    ],
    "drawdown_based": [
      "distance_from_high"
    ],
    "correlation_based": [
      "rolling_asset_correlation"
    ],
    "macro_based": [
      "yield_curve_spread",
      "cpi_trend",
      "unemployment_rate"
    ],
    "asset_specific": [
      "breadth_momentum_canary_universe",
      "currency_hedged_vs_unhedged_spread"
    ]
  },
  "candidate_asset_universe": {
    "minimal_version": ["US_Large_Cap", "US_Long_Treasury", "US_Inter_Treasury", "Gold", "Cash"],
    "basic_version": ["US_Large_Cap", "Developed_Ex_US", "Emerging_Markets", "US_Long_Treasury", "US_Inter_Treasury", "Commodities", "Gold"],
    "advanced_version": ["US_Large_Cap", "Developed_Ex_US", "Emerging_Markets", "US_Long_Treasury", "US_Inter_Treasury", "Commodities", "Gold", "REITs", "TIPS"]
  },
  "candidate_base_allocations": {
    "equal_weight": null,
    "stock_bond_static": null,
    "risk_based": null,
    "volatility_weighted": null,
    "user_selected": null
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