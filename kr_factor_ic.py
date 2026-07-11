#!/usr/bin/env python3
"""
kr_factor_ic.py — STRATEGY_UPGRADE_PROPOSAL.md 4.7절: DART 재무데이터 기반 한국 팩터 IC 검증.

현재 한국 랭킹은 z(12-1 모멘텀)×0.6 + z(52주 고점 근접도)×0.4 뿐이고 펀더멘털은 필터로만
쓰인다. 미국과 동일한 gross_margin·accruals류 팩터를 DART(전자공시) 재무데이터로 계산해
IC(6개월 forward-return Spearman)를 검증하고, 유의한 팩터만 향후 랭킹 편입 후보로 보고한다.

시점정합성: OpenDART 접수번호(rcept_no) 앞 8자리 = 실제 공시일. 공시일 +1거래일 지연을
적용해 저장하므로 look-ahead 없음 (fundamentals_edgar.asof 와 동일한 방식으로 조회).

생존편향: pykrx 지수 구성종목을 각 시점 기준으로 조회(point-in-time). 과거 시점 조회가
실패하면 현재 구성종목으로 폴백하되 결과 JSON에 survivorship_caveat 로 기록.

준비물(PC):
  1) https://opendart.fss.or.kr 무료 가입 → API 키 발급 → 환경변수 DART_API_KEY
  2) pip install pykrx  (구성종목·시세 폴백)
실행(PC):
  python kr_factor_ic.py --collect --years 10          # DART 수집(캐시·재개 가능, 연 1회)
  python kr_factor_ic.py --years 10                    # IC 검증
  python kr_factor_ic.py --self-test
결과: output/kr_fundamentals_dart.json (수집 캐시) · output/kr_ic_report.json (IC 리포트)
"""
from __future__ import annotations
import os, sys, json, time, argparse, datetime as dt
import numpy as np
import pandas as pd

DART_CACHE = "output/kr_fundamentals_dart.json"
IC_REPORT = "output/kr_ic_report.json"
DART_URL = "https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json"
CORP_URL = "https://opendart.fss.or.kr/api/corpCode.xml"
REPRT = {"11011": "사업보고서", "11012": "반기보고서", "11013": "1분기", "11014": "3분기"}
# account_id(우선) / account_nm(폴백) → fundamentals_edgar 스키마 키
ACCOUNTS = {
    "revenue": (["ifrs-full_Revenue"], ["수익(매출액)", "매출액"]),
    "cogs":    (["ifrs-full_CostOfSales"], ["매출원가"]),
    "gross":   (["ifrs-full_GrossProfit"], ["매출총이익"]),
    "opinc":   (["dart_OperatingIncomeLoss", "ifrs-full_OperatingIncomeLoss"], ["영업이익"]),
    "ni":      (["ifrs-full_ProfitLoss"], ["당기순이익"]),
    "assets":  (["ifrs-full_Assets"], ["자산총계"]),
    "equity":  (["ifrs-full_Equity"], ["자본총계"]),
    "debt":    (["ifrs-full_Liabilities"], ["부채총계"]),
    "ocf":     (["ifrs-full_CashFlowsFromUsedInOperatingActivities"], ["영업활동현금흐름"]),
}
# 검증 대상 팩터(가격 불필요 — fundamentals_edgar.factor_values 가 계산하는 것 중 부분집합)
KR_FACTORS = ["gross_margin", "gp_assets", "op_margin", "net_margin", "roe", "roa",
              "accruals", "leverage", "rev_growth", "ni_growth"]
HOLD_DAYS = 126          # 6m forward-return
LOOKBACK = 260


def _log(m): print(f"[KR-IC] {m}", file=sys.stderr)


# ------------------------- DART 수집 -------------------------
def _corp_map(key: str) -> dict:
    """stock_code(6자리) → corp_code. corpCode.xml(zip) 1회 다운로드."""
    import requests, zipfile, io
    import xml.etree.ElementTree as ET
    r = requests.get(CORP_URL, params={"crtfc_key": key}, timeout=60)
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    root = ET.fromstring(zf.read(zf.namelist()[0]))
    out = {}
    for el in root.iter("list"):
        stock = (el.findtext("stock_code") or "").strip()
        if stock:
            out[stock] = el.findtext("corp_code").strip()
    _log(f"corp_code 매핑 {len(out)}개(상장사)")
    return out


def _parse_report(js: dict) -> dict:
    """fnlttSinglAcntAll 응답 → {키: 값}. 연결(CFS) 우선."""
    rows = js.get("list") or []
    got = {}
    for fs in ("CFS", "OFS"):
        for row in rows:
            if row.get("fs_div") != fs:
                continue
            aid, anm = row.get("account_id", ""), (row.get("account_nm") or "").strip()
            for k, (ids, nms) in ACCOUNTS.items():
                if k in got:
                    continue
                if aid in ids or any(anm == n for n in nms):
                    try:
                        got[k] = float(str(row.get("thstrm_amount", "")).replace(",", ""))
                    except ValueError:
                        pass
        if got:
            break
    if "gross" not in got and "revenue" in got and "cogs" in got:
        got["gross"] = got["revenue"] - got["cogs"]
    got.pop("cogs", None)
    return got


def collect(tickers: list[str], years: int, reports=("11011",), sleep_s=0.15):
    key = os.environ.get("DART_API_KEY")
    if not key:
        _log("환경변수 DART_API_KEY 필요 — opendart.fss.or.kr 무료 발급"); sys.exit(1)
    import requests
    cache = {}
    if os.path.exists(DART_CACHE):
        with open(DART_CACHE, encoding="utf-8") as f:
            cache = json.load(f)
    cmap = _corp_map(key)
    y_now = dt.date.today().year
    n_req = 0
    for i, t in enumerate(tickers):
        corp = cmap.get(t)
        if not corp:
            continue
        rec = cache.setdefault(t, {})
        done = set(rec.get("_done", []))
        for y in range(y_now - years, y_now + 1):
            for rc in reports:
                tag = f"{y}-{rc}"
                if tag in done:
                    continue
                try:
                    js = requests.get(DART_URL, timeout=30, params={
                        "crtfc_key": key, "corp_code": corp, "bsns_year": str(y),
                        "reprt_code": rc, "fs_div": "CFS"}).json()
                    n_req += 1
                except Exception as e:
                    _log(f"{t} {tag} 요청 실패({e}) — 다음 실행 시 재개"); continue
                if js.get("status") == "000" and js.get("list"):
                    vals = _parse_report(js)
                    rcept = str(js["list"][0].get("rcept_no", ""))[:8]
                    if vals and len(rcept) == 8:
                        # 공시일 +1일 지연(look-ahead 방지, 제안서 3.3절)
                        filed = (dt.date(int(rcept[:4]), int(rcept[4:6]), int(rcept[6:8]))
                                 + dt.timedelta(days=1)).isoformat()
                        end = js["list"][0].get("thstrm_dt", f"{y}.12.31").split("~")[-1]
                        end = end.strip().replace(".", "-")
                        for k, v in vals.items():
                            rec.setdefault(k, []).append({"end": end, "filed": filed, "val": v})
                done.add(tag)
                time.sleep(sleep_s)
        rec["_done"] = sorted(done)
        for k in rec:
            if k != "_done":
                rec[k].sort(key=lambda p: p["filed"])
        if (i + 1) % 20 == 0 or i == len(tickers) - 1:
            os.makedirs("output", exist_ok=True)
            with open(DART_CACHE, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False)
            _log(f"저장 {i+1}/{len(tickers)}종목 (요청 누적 {n_req})")
    return cache


# ------------------------- 유니버스·시세 -------------------------
def kospi200_members(date_yyyymmdd: str):
    """시점별 코스피200 구성종목(pykrx). 실패 시 None."""
    try:
        from pykrx import stock as K
        m = list(K.get_index_portfolio_deposit_file("1028", date_yyyymmdd))
        return m or None
    except Exception:
        return None


def build_kr_panel(tickers: list[str], years: int) -> pd.DataFrame:
    """yfinance .KS 배치 다운로드 → 종가 패널(6자리 티커 컬럼)."""
    import yfinance as yf
    syms = [f"{t}.KS" for t in tickers]
    df = yf.download(syms, period=f"{years}y", auto_adjust=True, progress=False, threads=True)
    close = df["Close"] if "Close" in df else df
    close.columns = [str(c).replace(".KS", "") for c in close.columns]
    return close.dropna(axis=1, how="all").sort_index()


# ------------------------- IC 검증 -------------------------
def factor_frame(funds: dict, tickers, date_iso: str) -> pd.DataFrame:
    import fundamentals_edgar as F
    rows = {}
    for t in tickers:
        rec = funds.get(t)
        if not rec:
            continue
        vals = F.factor_values({k: v for k, v in rec.items() if k != "_done"}, date_iso, 0.0)
        row = {f: vals[f] for f in KR_FACTORS if f in vals}
        if row:
            rows[t] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def run_ic(panel: pd.DataFrame, funds: dict, rebal_days=63, pit_lookup=kospi200_members):
    n = len(panel)
    ics, n_pit_ok = {f: [] for f in KR_FACTORS + ["mom12_1"]}, 0
    events = 0
    for p in range(LOOKBACK, n - HOLD_DAYS - 1, rebal_days):
        date = panel.index[p]
        members = pit_lookup(date.strftime("%Y%m%d"))
        if members:
            n_pit_ok += 1
            cols = [c for c in panel.columns if c in set(members)]
        else:
            cols = list(panel.columns)                    # 폴백: 현재 구성(캐비앳 기록)
        valid = [c for c in cols if not np.isnan(panel.iloc[p][c])
                 and not np.isnan(panel.iloc[p - 252][c])]
        if len(valid) < 20:
            continue
        date_iso = date.date().isoformat()
        ff = factor_frame(funds, valid, date_iso)
        mom = (panel.iloc[p - 21][valid] / panel.iloc[p - 252][valid] - 1).rename("mom12_1")
        e = p + 1
        fwd = (panel.iloc[e + HOLD_DAYS][valid] / panel.iloc[e][valid] - 1)
        fr = fwd.rank()
        for f in KR_FACTORS:
            if f in ff.columns and ff[f].notna().sum() >= 15:
                v = ff[f].rank().corr(fr.reindex(ff.index))
                if pd.notna(v):
                    ics[f].append(float(v))
        v = mom.rank().corr(fr)
        if pd.notna(v):
            ics["mom12_1"].append(float(v))
        events += 1
    rows = []
    for f, arr in ics.items():
        if not arr:
            continue
        a = np.array(arr)
        t = float(a.mean() / (a.std(ddof=1) / np.sqrt(len(a)))) if len(a) > 2 and a.std() > 0 else 0.0
        rows.append({"factor": f, "ic_mean": round(float(a.mean()), 4),
                     "ic_std": round(float(a.std(ddof=1)), 4) if len(a) > 1 else None,
                     "t_stat": round(t, 2), "n_events": len(a),
                     # Harvey-Liu-Zhu(2016): 다중검정 감안 t≥3.0 요구
                     "significant_hlz": bool(t >= 3.0 and a.mean() > 0)})
    rows.sort(key=lambda r: r["ic_mean"], reverse=True)
    return rows, events, n_pit_ok


def report(rows, events, n_pit_ok, save=True):
    _log(f"\n=== 한국 팩터 IC (6개월 forward, 이벤트 {events}회, PIT 조회 성공 {n_pit_ok}회) ===")
    _log(f"{'팩터':16s}{'IC':>9s}{'t':>7s}{'이벤트':>7s}{'HLZ(t≥3)':>10s}")
    for r in rows:
        _log(f"{r['factor']:16s}{r['ic_mean']:>+9.4f}{r['t_stat']:>7.2f}{r['n_events']:>7d}"
             f"{'★유의' if r['significant_hlz'] else '':>10s}")
    sig = [r["factor"] for r in rows if r["significant_hlz"] and r["factor"] != "mom12_1"]
    _log(f"\n  → 랭킹 편입 후보(HLZ 기준 통과): {sig or '없음'}")
    payload = {"as_of": dt.date.today().isoformat(), "hold_days": HOLD_DAYS,
               "events": events, "factors": rows,
               "significant_factors": sig,
               "survivorship_caveat": (None if n_pit_ok == events else
                   f"PIT 구성종목 조회 실패 {events - n_pit_ok}회 → 해당 시점은 현재 구성종목 사용"
                   " (생존편향 잔존)"),
               "criteria": "IC>0 & t≥3.0 (Harvey-Liu-Zhu 2016 다중검정 기준)",
               "next_step": "유의 팩터를 kr_stocks.py 랭킹에 '추세+펀더멘털' 형태로 편입(4.7절)"}
    if save:
        os.makedirs("output", exist_ok=True)
        with open(IC_REPORT, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _log(f">>> 저장: {IC_REPORT}")
    return payload


# ------------------------- self-test -------------------------
def _synthetic_kr(n_days=1800, n_syms=60, seed=5):
    """진짜 신호 팩터(good_factor↑ → 미래수익↑)를 심은 합성 데이터."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-02", periods=n_days)
    quality = rng.normal(0, 1, n_syms)                     # 종목 고유 품질
    drift = 0.0002 + 0.0004 * quality                      # 품질이 수익률 결정
    panel = pd.DataFrame(
        {f"{i:06d}": 1000 * np.exp(np.cumsum(rng.normal(drift[i], 0.02, n_days)))
         for i in range(n_syms)}, index=dates)
    funds = {}
    for i in range(n_syms):
        t = f"{i:06d}"
        rev, assets = 1e12, 2e12
        gross = rev * (0.3 + 0.1 * quality[i])             # gross_margin ∝ 품질(예측력 있음)
        ni = rev * 0.05
        funds[t] = {"revenue": [], "gross": [], "ni": [], "assets": [], "equity": [],
                    "ocf": [], "debt": []}
        for y in (2017, 2018, 2019, 2020, 2021, 2022, 2023):
            filed = f"{y+1}-04-01"; end = f"{y}-12-31"
            for k, v in (("revenue", rev), ("gross", gross), ("ni", ni),
                         ("assets", assets), ("equity", assets * 0.5),
                         ("ocf", ni + rng.normal(0, ni * 0.5)), ("debt", assets * 0.5)):
                funds[t][k].append({"end": end, "filed": filed, "val": float(v)})
    return panel, funds


def self_test():
    _log("[self-test] 합성 데이터로 DART 파싱·IC 로직 점검")
    # ① 응답 파싱
    js = {"status": "000", "list": [
        {"fs_div": "CFS", "account_id": "ifrs-full_Revenue", "account_nm": "수익(매출액)",
         "thstrm_amount": "1,000", "rcept_no": "20240315000123", "thstrm_dt": "2023.01.01~2023.12.31"},
        {"fs_div": "CFS", "account_id": "ifrs-full_CostOfSales", "account_nm": "매출원가",
         "thstrm_amount": "600"},
        {"fs_div": "CFS", "account_id": "ifrs-full_Assets", "account_nm": "자산총계",
         "thstrm_amount": "5,000"}]}
    got = _parse_report(js)
    assert got == {"revenue": 1000.0, "assets": 5000.0, "gross": 400.0}, got
    # ② IC: 심어둔 gross_margin 신호가 노이즈 팩터보다 IC 높아야 함
    panel, funds = _synthetic_kr()
    rows, events, _ = run_ic(panel, funds, rebal_days=63, pit_lookup=lambda d: None)
    ic = {r["factor"]: r["ic_mean"] for r in rows}
    assert events > 10 and "gross_margin" in ic, f"이벤트 {events}, 팩터 {list(ic)}"
    others = [v for k, v in ic.items() if k in ("net_margin", "roa", "leverage")]
    assert ic["gross_margin"] > 0.05, f"심은 신호 IC가 낮음: {ic['gross_margin']}"
    assert all(ic["gross_margin"] > o for o in others), f"신호({ic['gross_margin']}) vs 노이즈({others})"
    report(rows, events, 0, save=False)
    _log("[self-test] 통과: DART 파싱 · 심은 신호 IC 검출 OK")


def main():
    ap = argparse.ArgumentParser(description="DART 기반 한국 팩터 IC 검증(4.7절)")
    ap.add_argument("--years", type=int, default=10)
    ap.add_argument("--rebal-days", type=int, default=63)
    ap.add_argument("--collect", action="store_true", help="DART 재무데이터 수집만 실행")
    ap.add_argument("--reports", default="11011", help="쉼표구분 reprt_code(11011=사업보고서)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test(); return
    # 유니버스: 현 시점 코스피200(수집 대상) — IC 단계에서 시점별 멤버십으로 재필터
    today = dt.datetime.now().strftime("%Y%m%d")
    members = None
    for back in range(8):
        d = (dt.datetime.now() - dt.timedelta(days=back)).strftime("%Y%m%d")
        members = kospi200_members(d)
        if members:
            break
    if not members:
        _log("코스피200 구성종목 조회 실패(pykrx/KRX 로그인 문제 — kr_stocks.py 참고)"); sys.exit(1)
    if args.collect:
        collect(members, args.years, tuple(args.reports.split(","))); return
    if not os.path.exists(DART_CACHE):
        _log(f"{DART_CACHE} 없음 — 먼저 --collect 실행"); sys.exit(1)
    with open(DART_CACHE, encoding="utf-8") as f:
        funds = json.load(f)
    panel = build_kr_panel(sorted(set(members) | set(funds)), args.years)
    _log(f"패널 {panel.shape[1]}종목 × {panel.shape[0]}일 · DART {len(funds)}종목")
    rows, events, n_pit_ok = run_ic(panel, funds, args.rebal_days)
    report(rows, events, n_pit_ok)


if __name__ == "__main__":
    main()
