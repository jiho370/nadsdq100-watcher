@AGENTS.md

## Claude Code 전용 메모

지침 본문은 전부 `AGENTS.md` 한 곳에만 둔다. Codex는 `AGENTS.md`를 직접 읽고, Claude Code는
이 파일의 `@AGENTS.md` import를 통해 같은 내용을 읽는다 — 사본이 하나라 갈라질 일이 없다.

**지침을 고칠 때는 이 파일이 아니라 `AGENTS.md`를 고칠 것.**

2026-09-23 이전까지 이 두 파일은 첫 줄 제목만 다른 완전한 중복 사본이었다. 같은 구조를 쓰던
다른 저장소에서 실제로 `AGENTS.md`가 2주 뒤처져 폐지된 절차를 계속 지시하고 있었으므로,
갈라지기 전에 import로 합쳤다.

- 심볼릭 링크(`ln -s AGENTS.md CLAUDE.md`)는 쓰지 말 것. Windows + `core.symlinks=false`라
  링크가 아니라 복사본이 만들어진다(2026-09-22 실측). 공식 문서도 Windows에서는 import 방식을 권한다.
- Claude Code가 `AGENTS.md`를 직접 읽는 기능은 v2.1.277 이상이 필요하다. 요건은 충족돼 있지만,
  텔레메트리가 꺼진 세션이나 Bedrock/Vertex 환경에서는 직접 읽기가 지원되지 않아 지침이 통째로
  안 읽힌다. import는 그런 환경에서도 동작하므로 이 파일을 남겨둔다.
