#!/usr/bin/env python3
"""
backtest_costs.py — STRATEGY_UPGRADE_PROPOSAL.md 6장 로드맵 1~2단계.

기존 backtest_weights.py(수정하지 않음)의 로직을 임포트해 재사용하며, 두 가지를 추가:
  1) 거래비용·세금: 시장별 비용모델 —
       미국: 매도 시 SEC fee 0.00278% (+수수료·슬리피지 파라미터)
       한국: 코스피 매도 0.20%(거래세 0.05%+농특세 0.15%), 코스닥 매도 0.20%(거래세)
     이벤트별 forward-return에 왕복비용을 반영해 gross/net을 항상 병기.
     (이벤트 = "이 바스켓을 사서 h개월 보유" 이므로 이벤트당 왕복 1회가 정확한 모델.
      실전에서 리밸런싱 간 유지되는 종목은 재매매하지 않으므로 실비용은 회전율만큼 낮음.)
  2) 생존편향 제거: point-in-time S&P500 구성종목(fja05680/sp500 공개 CSV)으로
     각 리밸런싱 시점 유니버스를 '그 시점 실제 구성종목'으로 제한.
     한계: 상장폐지 종목은 야후에 시세가 없어 잔존편향이 일부 남음 → 커버리지 통계로 보고.

비교(동일 패널·동일 이벤트, 나란히 출력):
  [legacy]      현재 구성종목 유니버스 + 일괄 10bp — 기존 backtest_weights 방식 재현
  [pit_legacyw] PIT 유니버스 + 비용모델, 가중치는 legacy 최적치 고정 — "숫자가 얼마나 깎이나"
  [pit]         PIT 유니버스 + 비용모델, IC·가중치 재탐색 — 로드맵 2단계("다시 계산")

실행(PC):
  python fundamentals_edgar.py                              # (권장) 펀더멘탈 수집
  python backtest_costs.py --export-universe                # PIT 합집합 티커 출력(펀더멘탈 보강용)
  python backtest_costs.py --years 10 --topn 30 --oos 0.4
  python backtest_costs.py --self-test
결과: output/backtest_costs_compare.json (비교표)
      output/trial_returns.json         (조합별 이벤트 수익률 행렬 — overfit_stats.py 입력)
"""
from __future__ import annotations
import os, re, sys, json, csv, time, argparse, bisect, warnings
import numpy as np
import pandas as pd

# 상수 컬럼 상관계산에서 나오는 무해한 divide 경고 억제(결과는 NaN 처리로 걸러짐)
warnings.filterwarnings("ignore", message="invalid value encountered in divide")

import backtest_weights as BW

PIT_CACHE = "output/sp500_pit.csv"
PIT_API = "https://api.github.com/repos/fja05680/sp500/contents/"
SEC_FEE = 0.0000278                                  # 매도금액 대비(2026년 기준 0.00278%)
KR_SELL_TAX = {"kospi": 0.0020, "kosdaq": 0.0020}    # 손실이어도 부과

# 티커 변경(동일 법인 존속·주식 연속) 매핑 — 야후는 새 티커 아래 과거 시세를 이어서 제공.
# 피인수·합병으로 소멸한 종목(TWTR, CELG, SIVB 등)은 절대 매핑하지 않는다(남의 시세가 섞임).
TICKER_ALIASES = {
    "UTX": "RTX", "ANTM": "ELV", "BLL": "BALL", "MYL": "VTRS", "CTL": "LUMN",
    "HFC": "DINO", "FBHS": "FBIN", "WLTW": "WTW", "TMK": "GL", "JEC": "J",
    "HRS": "LHX", "ABC": "COR", "FLT": "CPAY", "ARNC": "HWM", "PKI": "RVTY",
    "SYMC": "GEN", "KORS": "CPRI", "GPS": "GAP", "ADS": "BFH", "RE": "EG",
    "WYND": "TNL", "BHGE": "BKR", "COG": "CTRA", "DISCA": "WBD",
    "MMC": "MRSH",   # 2026-01 Marsh 리브랜딩(티커 변경, 동일 법인)
}
# fja 데이터셋의 'XXX-YYYYMM' 접미사 = 나중에 티커가 재사용된 과거 회사(예: JCI-201609).
# 야후에 존재하지 않으므로 다운로드는 건너뛰되, 멤버십(커버리지 분모)에는 남긴다.
_SUFFIXED = re.compile(r"-\d{6}$")


def _log(m): print(m, file=sys.stderr)


# ------------------------- 비용 모델 -------------------------
class CostModel:
    """왕복 거래비용. net = (1+r)·(1−c_sell)/(1+c_buy) − 1
    market: us | kospi | kosdaq | flat(기존 방식 재현: net = r − 0.001)"""
    def __init__(self, market="us", commission_bps=0.0, slippage_bps=5.0):
        self.market = market
        c, s = commission_bps / 1e4, slippage_bps / 1e4
        self.buy = c + s
        self.sell = c + s + (SEC_FEE if market == "us" else KR_SELL_TAX.get(market, 0.0))

    def net(self, r: float) -> float:
        if self.market == "flat":
            return r - BW.COST
        return (1.0 + r) * (1.0 - self.sell) / (1.0 + self.buy) - 1.0

    def describe(self) -> str:
        if self.market == "flat":
            return f"flat {BW.COST*1e4:.0f}bp(기존 방식)"
        return (f"{self.market}: 매수 {self.buy*1e4:.1f}bp + 매도 {self.sell*1e4:.2f}bp "
                f"(왕복 {(self.buy+self.sell)*1e4:.2f}bp)")


# ------------------------- PIT 유니버스 -------------------------
def _norm(t: str) -> str:
    t = t.strip().upper().replace(".", "-")
    return TICKER_ALIASES.get(t, t)


def _download_pit(dest=PIT_CACHE):
    """fja05680/sp500 저장소에서 'S&P 500 Historical Components ...csv' 자동 탐색·다운로드.

    2026-07-13 버그 수정: 이 저장소엔 두 후보가 있다 —
      "...Changes (Updated).csv"(계속 최신화되는 파일, 2026년까지 데이터 있음)
      "...Changes.csv"(2019-01-11에서 멈춘 옛 파일).
    기존 코드는 `sorted(cand, key=name)[-1]`로 '알파벳상 마지막'을 골랐는데, 공백(0x20)이
    마침표(0x2E)보다 아스키값이 작아 " (Updated).csv" < ".csv"로 정렬되어 옛 파일이 뽑히고
    있었다 — 그 결과 PIT 유니버스가 7년째 2019년에 멈춰 있었고(DELL·VRT 등 이후 편입 종목이
    전부 누락), 최근 구간을 다루는 모든 백테스트(backtest_exec 등)가 이 영향을 받았다.
    이제 "(Updated)"가 포함된 파일을 명시적으로 우선한다."""
    import requests
    _log("[PIT] fja05680/sp500 구성종목 이력 다운로드 중…")
    items = requests.get(PIT_API, timeout=30).json()
    cand = [i for i in items if i.get("name", "").startswith("S&P 500 Historical Components")]
    if not cand:
        raise RuntimeError("PIT CSV를 저장소에서 찾지 못함 — --pit-file 로 직접 지정하세요.")
    updated = [i for i in cand if "(Updated)" in i.get("name", "")]
    chosen = (sorted(updated, key=lambda i: i["name"])[-1] if updated
              else sorted(cand, key=lambda i: i["name"])[-1])
    _log(f"[PIT] 선택: {chosen['name']}")
    url = chosen["download_url"]
    text = requests.get(url, timeout=60).text
    os.makedirs("output", exist_ok=True)
    with open(dest, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    _log(f"[PIT] 저장: {dest}")


def load_pit(pit_file=None) -> list[tuple[str, frozenset]]:
    """[(date_iso, frozenset(tickers)), …] 날짜 오름차순. 컬럼: date,tickers(쉼표구분)."""
    path = pit_file or PIT_CACHE
    if not os.path.exists(path):
        _download_pit(path if pit_file else PIT_CACHE)
        path = pit_file or PIT_CACHE
    out = []
    with open(path, encoding="utf-8", newline="") as f:
        rd = csv.reader(f)
        next(rd)                                     # header: date,tickers
        for row in rd:
            if len(row) < 2 or not row[0]:
                continue
            out.append((row[0][:10],
                        frozenset(_norm(t) for t in row[1].split(",") if t.strip())))
    out.sort(key=lambda x: x[0])
    if not out:
        raise RuntimeError(f"PIT 파일이 비었음: {path}")
    return out


def membership_asof(pit, date_iso: str) -> frozenset:
    dates = [d for d, _ in pit]
    i = bisect.bisect_right(dates, date_iso) - 1
    return pit[max(i, 0)][1]


def pit_union(pit, start_iso: str) -> list[str]:
    """백테스트 구간에 한 번이라도 편입됐던 티커 합집합(시작 직전 멤버십 포함)."""
    dates = [d for d, _ in pit]
    i0 = max(bisect.bisect_right(dates, start_iso) - 1, 0)
    u = set()
    for _, m in pit[i0:]:
        u |= m
    return sorted(u)


# ------------------------- 패널 구축 (PIT 합집합 유니버스) -------------------------
def _purge_yf_cache():
    """yfinance는 조회 실패 티커를 로컬 캐시(tkr-tz.db)에 저장해, 첫 실행에서
    rate-limit로 실패한 멀쩡한 티커(예: MMC, CTRA)를 이후 실행에서 재조회 없이
    '상장폐지'로 처리한다. 재시도 전에 캐시를 삭제해 재조회를 강제한다(자동 재생성됨)."""
    import shutil
    cands = []
    try:
        import appdirs
        cands.append(os.path.join(appdirs.user_cache_dir(), "py-yfinance"))
    except Exception:
        pass
    home = os.path.expanduser("~")
    cands += [os.path.join(home, ".cache", "py-yfinance"),
              os.path.join(os.environ.get("LOCALAPPDATA", ""), "py-yfinance")]
    for d in dict.fromkeys(cands):
        if d and os.path.isdir(d):
            try:
                shutil.rmtree(d)
                _log(f"[PIT] yfinance 실패-캐시 삭제: {d}")
            except Exception:
                pass


def build_panel_pit(years, pit):
    import sp500_daily_report as R
    R._require_yf()
    start = (pd.Timestamp.today() - pd.DateOffset(years=int(years))).date().isoformat()
    universe = pit_union(pit, start)
    bad = ("-W", "-WI", "-WS", "-U", "-RT", "-R", ".W", ".U")
    n_suf = sum(1 for s in universe if _SUFFIXED.search(s))
    universe = [s for s in universe if not any(s.endswith(x) for x in bad)
                and not _SUFFIXED.search(s)]
    _log(f"[PIT] 구간 내 편입 이력 합집합 {len(universe)}종목 시세 다운로드 "
         f"(재사용-접미사 티커 {n_suf}개 제외 — 야후에 없음)…")
    hist = R.download_histories(universe, period=f"{int(years)}y")
    panel = pd.DataFrame({s: c for s, c in hist.items() if c is not None and len(c)}).sort_index()
    missing = [s for s in universe if s not in panel.columns]
    if missing:                                    # 야후 대량요청 rate-limit 오탐 구제(1회 재시도)
        _purge_yf_cache()                          # 실패가 로컬 캐시에 박제되는 것 방지
        _log(f"[PIT] 실패 {len(missing)}종목 15초 후 재시도(rate-limit 오탐 구제)…")
        time.sleep(15)
        hist2 = R.download_histories(missing, period=f"{int(years)}y")
        add = {s: c for s, c in (hist2 or {}).items() if c is not None and len(c)}
        if add:
            panel = pd.concat([panel, pd.DataFrame(add)], axis=1).sort_index()
            _log(f"[PIT] 재시도로 {len(add)}종목 추가 확보: {sorted(add)}")
    spy = R.download_histories(["SPY"], period=f"{int(years)}y").get("SPY")
    opens = None
    try:
        import yfinance as yf
        od = yf.download(list(panel.columns), period=f"{int(years)}y", auto_adjust=True,
                         progress=False, threads=True)
        opens = od["Open"].reindex(panel.index) if "Open" in od else None
    except Exception:
        opens = None
    _log(f"[PIT] 시세 확보 {panel.shape[1]}/{len(universe)}종목 "
         f"(누락 {len(universe)-panel.shape[1]} = 대부분 상장폐지 → 잔존편향 잔여분)")
    return panel, spy, opens


# ------------------------- 스냅샷(이벤트) 생성 -------------------------
def build_snaps(panel, spy, funds, opens, rebal_days):
    import tech_factors as T
    use_fund = bool(funds)
    spy = spy.reindex(panel.index).ffill()
    n = len(panel); max_h = max(BW.TD.values())
    ps = list(range(BW.LOOKBACK, n - max_h - 1, rebal_days))
    if not ps:
        raise RuntimeError("기간이 짧아 리밸런싱 시점 없음.")
    cross = T.build_panels(panel)
    if opens is not None:
        opens = opens.reindex_like(panel)
        on_cum = (1 + (opens / panel.shift(1) - 1).fillna(0)).cumprod()
        cross["overnight_mom"] = on_cum.shift(21) / on_cum.shift(252) - 1
    snaps = []
    for p in ps:
        raw = BW._raw_frame(panel, p, funds, use_fund, cross)
        if raw is None or raw.empty:
            continue
        e = p + 1
        fwd = {h: (panel.iloc[e + hd][raw.index] / panel.iloc[e][raw.index] - 1)
               for h, hd in BW.TD.items()}
        bench = {h: float(spy.iloc[e + hd] / spy.iloc[e] - 1) for h, hd in BW.TD.items()}
        snaps.append({"date": panel.index[p].date().isoformat(), "raw": raw,
                      "fwd": fwd, "bench": bench})
    return snaps


# ------------------------- 시나리오 평가 -------------------------
def _filter_snaps(snaps, pit, mode):
    """mode: current(마지막 멤버십 고정=생존편향 재현) | pit(시점별 멤버십). 커버리지 통계 포함."""
    latest = pit[-1][1]
    out, covs = [], []
    for s in snaps:
        members = latest if mode == "current" else membership_asof(pit, s["date"])
        idx = s["raw"].index.intersection(members)
        if len(members):
            covs.append(len(idx) / len(members))
        if len(idx) < 10:
            continue
        raw = s["raw"].loc[idx]
        out.append({**s, "raw": raw, "z": raw.apply(BW._z).fillna(0.0)})
    cov = {"mean": round(100 * float(np.mean(covs)), 1),
           "min": round(100 * float(np.min(covs)), 1)} if covs else None
    return out, cov


def _agg_ic(fsnaps, idxs):
    acc = {}
    for i in idxs:
        raw = fsnaps[i]["raw"]
        f6r = fsnaps[i]["fwd"]["6m"].reindex(raw.index).rank()
        for f in raw.columns:
            v = raw[f].rank().corr(f6r)
            if pd.notna(v):
                acc.setdefault(f, []).append(v)
    return sorted(((f, round(float(np.mean(v)), 4)) for f, v in acc.items() if v),
                  key=lambda kv: kv[1], reverse=True)


def _pick(ic_sorted, keep):
    sel = [f for f, ic in ic_sorted if ic > 0][:keep]
    if "mom12_1" not in sel:
        sel = (["mom12_1"] + sel)[:max(keep, 1)]
    return sel if len(sel) >= 2 else [f for f, _ in ic_sorted[:2]]


def eval_config(w, fsnaps, idxs, cost: CostModel, topn, collect_6m=False):
    """gross/net 병기 평가. collect_6m=True면 이벤트별 6m 순초과수익 리스트도 반환(PBO용)."""
    wv = pd.Series(w); cols = list(w)
    gv = {h: [] for h in BW.TD}; nv = {h: [] for h in BW.TD}; ex = {h: [] for h in BW.TD}
    sels, ev6 = [], []
    for i in idxs:
        s = fsnaps[i]; z = s["z"]
        top = (z[cols] * wv).sum(axis=1).sort_values(ascending=False).index[:topn]
        sels.append(set(top))
        for h in BW.TD:
            r = s["fwd"][h].reindex(top).dropna()
            if len(r):
                gross = float(r.mean())
                net = float(np.mean([cost.net(x) for x in r]))
                gv[h].append(gross); nv[h].append(net); ex[h].append(net - s["bench"][h])
                if h == "6m" and collect_6m:
                    ev6.append(round(net - s["bench"][h], 6))
    row = {"weights": w}
    turns = [1 - len(sels[j] & sels[j-1]) / max(len(sels[j]), 1) for j in range(1, len(sels))]
    row["turnover"] = round(100 * float(np.mean(turns)), 1) if turns else None
    for h in BW.TD:
        if nv[h]:
            g, a, e2 = np.array(gv[h]), np.array(nv[h]), np.array(ex[h])
            row[f"ret_{h}_gross"] = round(100 * g.mean(), 2)
            row[f"ret_{h}"] = round(100 * a.mean(), 2)                 # net
            row[f"excess_{h}"] = round(100 * e2.mean(), 2)             # net − bench
            row[f"win_{h}"] = round(100 * float((a > 0).mean()), 1)
            row[f"worst_{h}"] = round(100 * float(a.min()), 1)
            if e2.std() > 0:
                row[f"sharpe_{h}"] = round(float(e2.mean() / e2.std())
                                           * (252.0 / BW.TD[h]) ** 0.5, 2)
    return (row, ev6) if collect_6m else row


def run_scenario(fsnaps, cost, topn, keep, levels, oos_frac=0.0, fixed_weights=None):
    """IC 선별→가중치 탐색(또는 fixed_weights 고정 평가). 반환: (best행, ic_sorted, oos, grid)"""
    allidx = list(range(len(fsnaps)))
    if fixed_weights:
        w = {k: v for k, v in fixed_weights.items() if v}
        return eval_config(w, fsnaps, allidx, cost, topn), None, None, None
    if oos_frac and 0 < oos_frac < 0.9:
        cut = int(len(fsnaps) * (1 - oos_frac))
        train, test = allidx[:cut], allidx[cut:]
    else:
        train, test = allidx, None
    ic_sorted = _agg_ic(fsnaps, train)
    selected = _pick(ic_sorted, keep)
    grid = [eval_config(w, fsnaps, train, cost, topn) for w in BW._weight_grid(selected, levels)]
    best = max(grid, key=BW.score_config)
    oos = None
    if test:
        o = eval_config(best["weights"], fsnaps, test, cost, topn)
        oos = {"train": {k: best.get(k) for k in ("excess_6m", "excess_12m", "sharpe_6m")},
               "test": {k: o.get(k) for k in ("excess_6m", "excess_12m", "sharpe_6m")},
               "n_train": len(train), "n_test": len(test)}
        # 비교표는 세 시나리오 모두 '전체 이벤트' 기준이어야 비교 가능(학습구간만 쓰면
        # fixed_weights 열과 기간이 달라짐). IS/OOS 수치는 oos 딕셔너리로 따로 보고.
        best = eval_config(best["weights"], fsnaps, allidx, cost, topn)
    return best, ic_sorted, oos, grid


# ------------------------- 비교 리포트 -------------------------
def compare(snaps, pit, args, cost: CostModel):
    levels = tuple(int(x) for x in args.levels.split(","))
    flat = CostModel("flat")

    leg_snaps, _ = _filter_snaps(snaps, pit, "current")
    pit_snaps, cov = _filter_snaps(snaps, pit, "pit")
    if cov:
        _log(f"\n[유니버스] legacy(현재 구성종목) 이벤트 {len(leg_snaps)} · "
             f"pit 이벤트 {len(pit_snaps)} · PIT 커버리지 평균 {cov['mean']}% (최저 {cov['min']}%)")
    _log(f"[비용] legacy: {flat.describe()} · pit: {cost.describe()}")

    legacy, leg_ic, leg_oos, _ = run_scenario(leg_snaps, flat, args.topn, args.keep, levels, args.oos)
    pit_legw = run_scenario(pit_snaps, cost, args.topn, args.keep, levels,
                            fixed_weights=legacy["weights"])[0]
    pit_best, pit_ic, pit_oos, _ = run_scenario(pit_snaps, cost, args.topn, args.keep,
                                                levels, args.oos)

    # PBO 입력: pit 유니버스·비용 기준, 전체 이벤트에 대한 조합별 6m 순초과수익 행렬
    selected = _pick(pit_ic, args.keep)
    allidx = list(range(len(pit_snaps)))
    trials, matrix = [], []
    for w in BW._weight_grid(selected, levels):
        row, ev6 = eval_config(w, pit_snaps, allidx, cost, args.topn, collect_6m=True)
        trials.append(BW._wstr(w)); matrix.append(ev6)
    n_ev = min(len(m) for m in matrix) if matrix else 0
    os.makedirs("output", exist_ok=True)
    with open("output/trial_returns.json", "w", encoding="utf-8") as f:
        json.dump({"horizon": "6m", "universe": "pit", "cost": cost.describe(),
                   "rebal_days": args.rebal_days if hasattr(args, "rebal_days") else 63,
                   "hold_days": BW.TD["6m"],      # overfit_stats의 embargo·T_eff 계산용
                   "dates": [pit_snaps[i]["date"] for i in range(n_ev)],
                   "trials": trials, "excess_returns": [m[:n_ev] for m in matrix]},
                  f, ensure_ascii=False)
    _log(f"[PBO 입력] output/trial_returns.json — 조합 {len(trials)}개 × 이벤트 {n_ev}회")

    # ---- 나란히 비교표 ----
    def fmt(row, key):
        v = row.get(key); return "-" if v is None else str(v)
    rows = [("6M수익 gross %", lambda r: fmt(r, "ret_6m_gross")),
            ("6M수익 net %", lambda r: fmt(r, "ret_6m")),
            ("6M초과 net %p", lambda r: fmt(r, "excess_6m")),
            ("12M초과 net %p", lambda r: fmt(r, "excess_12m")),
            ("승률 6M %", lambda r: fmt(r, "win_6m")),
            ("최악 12M %", lambda r: fmt(r, "worst_12m")),
            ("6M 순샤프", lambda r: fmt(r, "sharpe_6m")),
            ("회전율 %", lambda r: fmt(r, "turnover"))]
    _log("\n=== 비교: 기존 방식 vs PIT+비용 (세 열 모두 전체 이벤트 기준 · IS/OOS는 아래 별도) ===")
    _log(f"  legacy 가중치     : {BW._wstr(legacy['weights'])}")
    _log(f"  pit 재탐색 가중치 : {BW._wstr(pit_best['weights'])}")
    hdr = f"{'':16s}{'legacy(기존)':>16s}{'pit(기존가중치)':>16s}{'pit(재탐색)':>16s}"
    _log(hdr); _log("-" * 70)
    for name, get in rows:
        _log(f"{name:16s}{get(legacy):>16s}{get(pit_legw):>16s}{get(pit_best):>16s}")
    for tag, oos in (("legacy", leg_oos), ("pit", pit_oos)):
        if oos:
            t, o = oos["train"], oos["test"]
            _log(f"  [{tag} 워크포워드] IS 6M초과 {t['excess_6m']}%p → OOS {o['excess_6m']}%p "
                 f"(학습 {oos['n_train']} / 검증 {oos['n_test']}이벤트)")

    payload = {"as_of": pd.Timestamp.today().date().isoformat(),
               "events": {"legacy": len(leg_snaps), "pit": len(pit_snaps)},
               "pit_coverage_pct": cov,
               "cost_model": {"legacy": flat.describe(), "pit": cost.describe()},
               "legacy": legacy, "pit_legacy_weights": pit_legw, "pit_best": pit_best,
               "ic": {"legacy": dict(leg_ic), "pit": dict(pit_ic)},
               "oos": {"legacy": leg_oos, "pit": pit_oos},
               "note": ("이벤트당 왕복비용 1회 반영(바스켓 h개월 보유 가정). "
                        "PIT 커버리지 미달분(상장폐지 종목)은 잔존 생존편향으로 남음.")}
    with open("output/backtest_costs_compare.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    _log("\n>>> 저장: output/backtest_costs_compare.json")
    return payload


# ------------------------- 가중치 발행 (라이브 선정 연결) -------------------------
def publish_weights():
    """PBO/DSR 게이트 통과 시에만 pit_best 가중치를 output/best_weights.json 으로 발행.
    export_data.select_pool 이 이 파일을 읽어 일일 종목 선정에 사용한다(없으면 모멘텀 폴백).
    검증 안 된 가중치가 라이브로 새는 것을 코드로 차단하는 것이 목적."""
    with open("output/backtest_costs_compare.json", encoding="utf-8") as f:
        cmp_ = json.load(f)
    try:
        with open("output/pbo_report.json", encoding="utf-8") as f:
            rep = json.load(f)
    except Exception:
        rep = {}
    if not rep.get("passed"):
        _log("[발행 거부] pbo_report.json passed=true 아님 — 검증 통과 전 가중치는 발행 불가")
        sys.exit(1)
    best = cmp_.get("pit_best") or {}
    w = {k: v for k, v in (best.get("weights") or {}).items() if v}
    if not w:
        _log("[발행 거부] pit_best 가중치 없음 — 먼저 backtest_costs.py 본 실행"); sys.exit(1)
    payload = {"weights": w,
               "metrics": {k: v for k, v in best.items() if k != "weights"},
               "selected_factors": list(w),
               "recommended_hold": "6m",
               "self_test": False,
               "published_from": "backtest_costs(pit_best) — overfit_stats passed 게이트 통과분",
               "published_at": pd.Timestamp.today().date().isoformat(),
               "gate": {"pbo": rep["pbo"]["pbo"], "dsr_teff": rep["dsr"].get("dsr")},
               "criteria": "PIT 유니버스+거래비용 재탐색 최적 · PBO/DSR(T_eff) 통과 시에만 발행"}
    with open("output/best_weights.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    _log(f"[발행] output/best_weights.json ← {BW._wstr(w)} "
         f"(PBO {rep['pbo']['pbo']:.1%} · DSR {rep['dsr'].get('dsr')}) — 다음 리포트부터 라이브 반영")


# ------------------------- self-test -------------------------
def _synthetic_pit(panel, seed=7):
    """합성 PIT: 6개월마다 90종목 중 ~78%가 멤버(무작위 교체) + 마지막 멤버십."""
    rng = np.random.default_rng(seed)
    syms = list(panel.columns); pit = []
    for d in pd.date_range(panel.index[0], panel.index[-1], freq="126D"):
        pit.append((d.date().isoformat(),
                    frozenset(rng.choice(syms, size=int(len(syms) * 0.78), replace=False))))
    return pit


def self_test():
    _log("[self-test] 합성 데이터로 비용·PIT 로직 점검")
    panel, spy, funds, opens = BW._synthetic()
    pit = _synthetic_pit(panel)
    args = argparse.Namespace(topn=15, keep=4, levels="0,1,2", oos=0.4)
    cost = CostModel("us", commission_bps=0.0, slippage_bps=5.0)
    snaps = build_snaps(panel, spy, funds, opens, rebal_days=63)
    payload = compare(snaps, pit, args, cost)
    # 검증 1: net ≤ gross (모든 시나리오·전 기간)
    for tag in ("pit_legacy_weights", "pit_best"):
        for h in BW.TD:
            g, n = payload[tag].get(f"ret_{h}_gross"), payload[tag].get(f"ret_{h}")
            assert g is None or n is None or n <= g + 1e-9, f"{tag} {h}: net({n}) > gross({g})"
    # 검증 2: 비용모델 수치 (한국 매도세 20bp 포함 확인)
    kr = CostModel("kospi", commission_bps=1.5, slippage_bps=0.0)
    assert abs(kr.sell - (0.0020 + 0.00015)) < 1e-9 and abs(kr.buy - 0.00015) < 1e-9
    us = CostModel("us", commission_bps=0.0, slippage_bps=0.0)
    assert abs(us.net(0.10) - (1.1 * (1 - SEC_FEE) - 1)) < 1e-12
    # 검증 3: PBO 입력 행렬 저장·형태
    with open("output/trial_returns.json", encoding="utf-8") as f:
        tr = json.load(f)
    assert len(tr["trials"]) == len(tr["excess_returns"]) >= 2
    assert all(len(r) == len(tr["dates"]) for r in tr["excess_returns"])
    _log("[self-test] 통과: net≤gross · 비용모델 수치 · PBO 입력 행렬 OK")


def main():
    ap = argparse.ArgumentParser(description="거래비용+PIT 유니버스 백테스트(기존 방식과 나란히 비교)")
    ap.add_argument("--years", type=float, default=10)
    ap.add_argument("--topn", type=int, default=30)
    ap.add_argument("--rebal-days", type=int, default=63)
    ap.add_argument("--keep", type=int, default=6)
    ap.add_argument("--levels", default="0,1,2")
    ap.add_argument("--oos", type=float, default=0.4, help="워크포워드 표본외 비율")
    ap.add_argument("--market", default="us", choices=["us", "kospi", "kosdaq"])
    ap.add_argument("--commission-bps", type=float, default=0.0, help="편도 수수료(bp)")
    ap.add_argument("--slippage-bps", type=float, default=5.0, help="편도 슬리피지(bp)")
    ap.add_argument("--pit-file", default=None, help="PIT CSV 직접 지정(date,tickers)")
    ap.add_argument("--export-universe", action="store_true",
                    help="PIT 합집합 티커만 출력(fundamentals_edgar --tickers 용)")
    ap.add_argument("--publish-weights", action="store_true",
                    help="검증(PBO/DSR) 통과 가중치를 best_weights.json 으로 발행(라이브 반영)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test(); return
    if args.publish_weights:
        publish_weights(); return
    pit = load_pit(args.pit_file)
    if args.export_universe:
        start = (pd.Timestamp.today() - pd.DateOffset(years=int(args.years))).date().isoformat()
        print(",".join(pit_union(pit, start)))
        return
    panel, spy, opens = build_panel_pit(args.years, pit)
    funds = BW.load_funds()
    _log(f"[백테스트] 패널 {panel.shape[1]}종목 × {panel.shape[0]}일 · "
         f"펀더멘탈 {'있음(' + str(len(funds)) + ')' if funds else '없음'}")
    if funds:
        miss = [s for s in panel.columns if s not in funds]
        if miss:
            _log(f"  ※ 펀더멘탈 누락 {len(miss)}종목(과거 편입분) — 보강하려면 "
                 f"--export-universe 출력 티커로 fundamentals_edgar.py --tickers 실행")
    cost = CostModel(args.market, args.commission_bps, args.slippage_bps)
    snaps = build_snaps(panel, spy, funds, opens, args.rebal_days)
    compare(snaps, pit, args, cost)


if __name__ == "__main__":
    main()
