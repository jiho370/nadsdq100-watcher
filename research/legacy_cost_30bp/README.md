# legacy_cost_30bp/

이 폴더의 8개 스크립트는 코인(BTC/ETH) 편도 거래비용으로 **30bp**를 가정한다.
실제 업비트 시장가 수수료는 **5bp**(2026-09-06 실계정 확인) — 6배 과대 계상이다.
발견 경위와 판정 재검토는 [`docs/playbook/03-method/EXECUTION.md` §3-1](../../docs/playbook/03-method/EXECUTION.md#3-1-코인-비용이-실제-계정-수수료의-6배-)에 있다.

**결론: 상수는 고치지 않고 여기 격리만 한다.** 이 8개가 만든 판정은 전부 "기각"이고,
5bp로 재계산해도 DSR이 채택기준(0.95)에서 멀고 OOS 샤프가 음수/0이라 뒤집히지 않는다 —
비용이 아니라 과최적화가 기각 사유였다(§3-1 표 참고). 30bp는 보수적 방향(비용을 과대
계상하면 전략이 실제보다 불리하게 나온다)이라 이 기각들은 실제로 안전하다.

새로 코인 백테스트를 짤 때는 여기 있는 스크립트를 복사하지 말 것 — `COST_BPS_MARKET = 5.0`
(`research/crypto/circuit_breaker_validation.py` 참고)이 맞는 값이다.

| 스크립트 | 대상 | 원래 위치 |
|---|---|---|
| `btc_eth_vs_random_baseline.py` | BTC/ETH vs 무작위매매 | `research/crypto/` |
| `eth_ma30_verification.py` | ETH MA30 | `research/crypto/` |
| `run_eth_grid.py` | ETH 레짐 그리드 | `research/crypto/` |
| `vol_target_dense_grid.py` | 변동성타겟 촘촘 그리드 | `research/crypto/` |
| `vol_target_fast_response.py` | 변동성타겟 급변동 대응 | `research/crypto/` |
| `vol_target_validation.py` | 변동성타겟 검증(BTC/ETH/SPX) | `research/crypto/` |
| `filter_vs_bh_bootstrap.py` | 레짐필터 vs buy&hold | `research/regime/` |
| `regime_era_split.py` | 레짐필터 시대구분 | `research/regime/` |

`research/regime/backtest_regime_assets.py`는 옮기지 않았다 — gold(`COST_BPS["gold"]=5bp`,
이미 정답)를 포함해 21개 파일이 함께 쓰는 공용 모듈이라, 여기로 옮기면 30bp와 무관한
스크립트까지 잘못 라벨링된다.
