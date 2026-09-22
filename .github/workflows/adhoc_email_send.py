#!/usr/bin/env python3
"""일회성 세션 요약 메일 — 2026-09-22 추천점수(0~10) 검증 결과. adhoc_email.yml 전용.
표준 라이브러리만 사용(의존성 설치 불필요) — sp500_daily_report.send_email()과 동일한
SMTP 방식을 최소 재현."""
import os
import smtplib
import ssl
import sys
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

SUBJECT = "[세션 요약] 추천점수(0~10) 검증 - US/KR 결과 및 이슈"

HTML = """
<div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','Malgun Gothic',sans-serif;
max-width:700px;margin:0 auto;color:#111">
<h2 style="margin:6px 0">\U0001F4CA 세션 요약 — 추천강도(0~10점) 시스템 검증 결과</h2>
<div style="font-size:12px;color:#9ca3af;margin:-4px 0 10px">2026-09-22</div>

<div style="background:#fef2f2;border-left:3px solid #b91c1c;padding:10px 14px;font-size:13px;
line-height:1.7;margin:10px 0"><b>결론부터: 미국·한국 둘 다 아직 통계적으로 검증되지 않음.</b><br>
종합점수(팩터 랭킹) 기반 "밴드 내에서 얼마나 더 좋은가" 추천강도 스코어는 이번 재검증에서
PBO/DSR 게이트를 통과하지 못했습니다. 원래 팩터 랭킹(topn 선정) 자체의 유의성과는 별개 문제입니다.</div>

<h3 style="margin:18px 0 6px">결과 비교</h3>
<table role="presentation" style="border-collapse:collapse;font-size:12px;width:100%;border:1px solid #e5e7eb">
<tr style="background:#f8fafc;color:#6b7280;text-align:left">
<th style="padding:6px 8px">항목</th><th style="padding:6px 8px">US (floor 3.25)</th>
<th style="padding:6px 8px">KR 원래밴드(floor 3.75)</th><th style="padding:6px 8px">KR 확장밴드(floor 3.5~cap 7.0)</th></tr>
<tr><td style="padding:6px 8px;border-top:1px solid #f1f5f9">유효구간 표본(n)</td>
<td style="padding:6px 8px;border-top:1px solid #f1f5f9">계산 불가(None)</td>
<td style="padding:6px 8px;border-top:1px solid #f1f5f9">4</td>
<td style="padding:6px 8px;border-top:1px solid #f1f5f9">27</td></tr>
<tr><td style="padding:6px 8px;border-top:1px solid #f1f5f9">PBO / DSR</td>
<td style="padding:6px 8px;border-top:1px solid #f1f5f9;color:#b91c1c;font-weight:700">0% / 0.63 (FAIL)</td>
<td style="padding:6px 8px;border-top:1px solid #f1f5f9;color:#15803d;font-weight:700">0% / 0.97 (PASS)</td>
<td style="padding:6px 8px;border-top:1px solid #f1f5f9;color:#b91c1c;font-weight:700">0% / 0.77 (FAIL)</td></tr>
<tr><td style="padding:6px 8px;border-top:1px solid #f1f5f9">워크포워드</td>
<td style="padding:6px 8px;border-top:1px solid #f1f5f9">n=10, +0.51%p, 60%양수</td>
<td style="padding:6px 8px;border-top:1px solid #f1f5f9;color:#15803d;font-weight:700">n=20, +7.59%p, 80%양수</td>
<td style="padding:6px 8px;border-top:1px solid #f1f5f9">n=9, +0.37%p, 55.6%양수</td></tr>
<tr><td style="padding:6px 8px;border-top:1px solid #f1f5f9">OOS 홀드아웃</td>
<td style="padding:6px 8px;border-top:1px solid #f1f5f9">n=7(2건), -1.04%p</td>
<td style="padding:6px 8px;border-top:1px solid #f1f5f9;color:#15803d;font-weight:700">+7.61%p</td>
<td style="padding:6px 8px;border-top:1px solid #f1f5f9;color:#b91c1c;font-weight:700">n=8건, -0.85%p(음전환)</td></tr>
</table>

<h3 style="margin:18px 0 6px">핵심 발견</h3>
<div style="font-size:13px;line-height:1.7">
지시하신 대로 한국 쪽 표본부족 문제(A: min_obs 완화, B: 밴드 확장)를 적용했더니 1단계 표본은
늘었지만(n=4→27), 오히려 2·3·4단계 통계적 유의성이 전부 악화되거나 부호가 뒤집혔습니다
(DSR 0.97→0.77, 워크포워드 +7.59%p→+0.37%p, OOS +7.61%p→-0.85%p).
가장 자연스러운 해석은: 원래 좁은 밴드(n=4)의 좋은 결과는 표본이 작아 우연히 좋게 나왔을
가능성이 높고, 표본을 늘려 본 "진짜" 신호는 훨씬 약하다는 것입니다.<br><br>
미국은 정상 데이터(레이트리밋 없는 607/703 커버리지)로 봐도 PBO/DSR 자체가 불통과, OOS
이벤트 2건뿐이라 근거가 약합니다.
</div>

<h3 style="margin:18px 0 6px">제안</h3>
<div style="font-size:13px;line-height:1.7">
현재 데이터로는 추천강도(0~10점) 시스템을 실제 신호에 반영하기보다 "실험적 프로토타입"으로
문서에만 남겨두는 쪽을 권합니다. 기존 팩터 랭킹(1:2:2 가중치, topn 선정) 자체는 이 문제와
무관하게 그대로 유효합니다.
</div>

<div style="font-size:11px;color:#9ca3af;margin-top:16px;line-height:1.5">
규칙 기반 참고용 자료이며 투자 권유가 아닙니다. 판단·책임은 본인에게 있습니다.</div>
</div>
"""


def _parse_recipients(raw: str) -> list:
    return [r.strip() for r in raw.replace(";", ",").split(",") if r.strip()]


def main() -> int:
    user = os.environ.get("SMTP_USER")
    pw = os.environ.get("SMTP_PASS")
    recipients = _parse_recipients(os.environ.get("EMAIL_TO", ""))
    if not (user and pw and recipients):
        print("[오류] SMTP 환경변수/수신자 누락 — 발송 생략", file=sys.stderr)
        return 1

    msg = MIMEMultipart("alternative")
    msg["Subject"], msg["From"], msg["To"] = SUBJECT, user, ", ".join(recipients)
    msg.attach(MIMEText("HTML 미리보기를 지원하는 메일 클라이언트로 확인하세요.", "plain", "utf-8"))
    msg.attach(MIMEText(HTML, "html", "utf-8"))

    try:
        ctx = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ctx) as srv:
            srv.login(user, pw)
            srv.sendmail(user, recipients, msg.as_string())
        print(f"[정보] 메일 발송 완료 -> {len(recipients)}명")
        return 0
    except Exception as e:
        print(f"[오류] 메일 발송 실패: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
