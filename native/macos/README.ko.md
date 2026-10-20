# Hanglish macOS 입력기

[English](README.md) · [한국어](README.ko.md) · [프로젝트 소개](../../README.ko.md)

한국어를 말할 수 있지만 한글로 적는 데 어려움을 느끼는 사람을 위한 입력기다.
현재는 실험적인 프리뷰 단계다. 전달받은 패키지의 릴리스 안내에서 검증한 환경과 알려진 제한을 확인한다.

Hanglish 0.2.6(빌드 8)은 로마자 문장을 한글 후보로 바꾸는 macOS 입력기다. 입력기 앱,
Python 3.11 런타임, 변환 엔진, 학습 모델을 하나의 설치 이미지에 묶었다. 설치 후에는
개발 서버, 가상 환경, Tailscale 연결이 필요하지 않으며 입력 내용도 네트워크로 보내지
않는다.

현재 패키지는 **Apple Silicon(arm64), 최소 배포 대상 macOS 14 이상**으로 빌드한다.
실제 로컬 검증은 macOS 27에서 진행했다. macOS 14를 포함해 확인하지 않은 버전은
**미검증 프리뷰**로 제공한다. 최소 배포 대상 설정이 호환성을 보장하지는 않는다. 개발 패키지는 ad-hoc 서명 상태이고 비공개용으로
명시된다. 릴리스 모드 패키지는 Developer ID 서명과 Apple 공증을 거친다. 개별 산출물의 상태는 함께 제공되는 manifest와 공증 로그에서 확인한다.

## 다운로드와 설치

공개 설치 파일이 제공되는 버전은 [Hanglish Releases 페이지](https://github.com/LNemo/hanglish-IME/releases)에서 받는다. 설치 파일이 첨부되어 있지 않으면 해당 버전의 공개 다운로드가 없는 상태다. 개발자가 전달한 완성된 설치 파일에도 아래 절차를 적용할 수 있다. Releases 페이지가 열리지 않는다고 Mac에 추가 설정이 필요한 것은 아니다.

`Hanglish-0.2.6-arm64.dmg`처럼 Apple Silicon용 `.dmg` 파일을 선택한다. **Source code** ZIP·tar.gz 파일은 개발용 소스이므로 입력기로 설치할 수 없다. Intel Mac은 지원하지 않는다. **Apple 메뉴 → 이 Mac에 관하여**에서 칩과 macOS 버전을 확인할 수 있다.

파일명에 `development-NONPUBLIC`이 있으면 ad-hoc 서명을 사용하는 별도의 개발용 패키지이며 공개 설치 파일이 아니다. 릴리스 모드 패키지는 Developer ID 서명과 Apple 공증을 거치고, 함께 제공되는 manifest와 공증 로그로 정확한 산출물을 확인한다. macOS가 실행을 차단하면 보안 기능을 끄지 말고 받은 파일이 맞는지 확인한 뒤 정확한 오류 메시지를 알려준다.

1. DMG를 열고 **Hanglish Installer**를 실행한다.
2. **설치 / 업데이트**를 누른다. 관리자 암호는 필요하지 않다.
3. 자동으로 활성화되지 않으면 **시스템 설정 → 키보드 → 텍스트 입력 → 편집 → + →
   한국어 → Hanglish**를 추가한다.
4. 메뉴 막대의 입력기 메뉴에서 `han` 아이콘의 Hanglish를 선택한다.
5. 텍스트 편집기(TextEdit)에서 새 문서를 열고 아래 첫 변환 예시를 따라 한다. 설치가 끝나면 설치 프로그램을 닫고 DMG를 추출해도 된다. 이후에는 설치된 복사본으로 실행된다.

설치기는 영어를 기본으로 표시하고 macOS 언어가 한국어이면 한국어 번역을 사용한다. 현재 macOS 사용자 계정에 설치된다. 엔진과 모델이 포함되어 있으므로 일반 사용에는 터미널 명령, 별도 Python 설치, Apple Developer 계정이나 서버 실행이 필요하지 않다.

입력기 목록에 바로 나타나지 않으면 작업 중인 문서를 저장한 뒤 로그아웃·로그인한다.
개발 Mac에서는 재시동 후 Hanglish를 제거·재추가하여 지속적인 목록 누락을 복구한 적이
있다. 모든 Mac에서 같은 조치로 해결된다는 보장은 아니다.
설치 앱은 입력기 등록과 활성화를 시도하지만, macOS가 시스템 설정에서 사용자의 수동
추가를 요구할 수 있다.

0.2.0에서 업데이트한 뒤 Hanglish를 선택해도 이전 입력기로 남으면, 텍스트 입력 편집
화면에서 Hanglish만 제거한 다음 다시 추가한다. 0.2.1에서 입력 모드 등록 방식이
바뀌었으며, 현재 Mac에서는 이 조치 후 새 모드로 정상 전환되는 것을 확인했다.

설치 위치는 `~/Library/Input Methods/Hanglish.app`이다. 같은 설치 프로그램으로
업데이트하면 기존 설치본을 먼저
`~/Library/Application Support/Hanglish/Backups` 아래에 보관한다. 설치나 등록에
실패하면 이전 설치본 복구를 시도하며, 복구도 실패하면 백업 위치를 안내한다.

## 입력기 선택과 전환 단축키 설정

메뉴 막대에 입력기 메뉴가 보이지 않으면 **시스템 설정 → 키보드 → 텍스트 입력 → 편집**에서 **메뉴 막대에 입력 메뉴 보기**를 켠다. Hanglish가 없으면 같은 화면에서 추가하고 메뉴에서 선택한다.

전환 키는 **시스템 설정 → 키보드 → 키보드 단축키 → 입력 소스**에서 확인하거나 변경한다. 일반적인 단축키는 이전 입력 소스로 전환하는 **Control–Space**, 다음 입력 소스로 전환하는 **Control–Option–Space**다. 사용자가 변경한 설정은 다를 수 있다. 단축키가 ABC와 다른 입력기만 번갈아 선택한다면 우선 메뉴에서 Hanglish를 직접 선택하고 입력 소스에 추가되어 있는지 확인한다. Apple의 [입력 소스 설정](https://support.apple.com/ko-kr/guide/mac-help/mchl84525d76/mac)과 [키보드 단축키 안내](https://support.apple.com/ko-kr/102650)를 참고한다.

Hanglish 전용 설정 앱은 아직 없다. 입력 소스 선택과 전환 단축키는 macOS에서 설정하며, 변환 엔진은 필요할 때 자동 실행된다.

## 사용법

| 조작 | 결과 |
|---|---|
| 로마자 입력 | 조합 중인 원문 표시 |
| Space | 문장 안에 공백 추가 |
| Enter 또는 Tab | 현재 문장 변환 요청 |
| 후보가 나온 뒤 Enter | 선택한 후보 확정 |
| Tab / Shift+Tab, 아래 / 위 | 다음 / 이전 후보 미리보기 |
| 후보 클릭 | 해당 후보 확정 |
| Esc | 원래 로마자를 그대로 확정 |
| Backspace | 원문의 마지막 문자 삭제 후 다시 편집 |
| Shift+Enter | 현재 문자열을 확정하고 앱의 Enter 동작 전달 |
| 방향 이동·단축키·입력 소스 전환 | 현재 조합을 확정하고 앱 동작으로 복귀 |

첫 테스트는 텍스트 편집기에서 Hanglish를 선택하고 `jeoneun haksaengimnida`를 입력한 뒤 Enter를 누른다. `저는 학생입니다` 후보를 확인하고 Enter를 다시 눌러 확정한다. 첫 변환은 엔진 시작 때문에 시간이 더 걸릴 수 있다.

`imnida`를 입력하고 Enter를 누르면 `입니다`가 후보로 표시된다. 결과를
확인한 뒤 Enter를 한 번 더 눌러 확정한다. `testreul sihaenghanda`처럼 문장 안의
공백도 그대로 입력한 뒤 변환한다.

첫 변환 때 입력기 번들 안의 엔진을 자식 프로세스로 자동 실행하므로 평소보다 오래 걸릴
수 있다. 이후 요청은 같은 프로세스를 재사용하며, 120초 동안 요청이 없으면 엔진을
종료한다. 다음 변환 때 자동으로 다시 시작한다. 변환 중 같은 요청을 중복 실행하지 않으며,
오류가 나도 원문을 보존한다. Enter로 재시도하거나 Esc로 원문을 확정할 수 있다.

조합은 최대 2,000자다. 숫자 키는 후보 번호로 소비하지 않으므로 문장에 계속 입력할 수
있다. 조합 내부 커서 이동과 단어 단위 재변환은 아직 지원하지 않는다. 영문을 계속
입력할 때는 ABC 입력 소스로 전환하는 편이 편리하다.

## 업데이트와 제거

업데이트는 DMG의 **Hanglish Installer**를 다시 실행하고 **설치 / 업데이트**를 누른다.
현재 Hanglish가 선택되어 있으면 설치 프로그램이 안전하게 ABC로 전환한 뒤 입력기
프로세스를 종료하고 교체한다. 실행 중이던 이전 입력기가 남거나 목록이 갱신되지 않으면
로그아웃·로그인 후 확인한다.

제거는 설치 프로그램의 **제거 후 보관**을 사용한다. 설치본을 즉시 삭제하지 않고 백업
폴더로 옮기며 입력 소스를 비활성화한다. 시스템 설정 목록에 항목이 남으면 텍스트 입력
편집 화면에서 Hanglish를 제거한다.

기존 [install.sh](install.sh)는 엔진이 없는 경량 빌드의 설치를 차단한다.
엔진과 모델을 포함한 설치·업데이트에는 DMG를 사용한다.

## 개발과 검증

새로 clone한 저장소에는 모델과 일부 필수 데이터가 없어 완전한 설치 패키지를 바로
빌드할 수 없다. 필요한 파일은 [엔진 파일 명세](Packaging/engine-files.json)에 있으며,
모델 없이 실행하는 `test.sh`와 별도로 준비해야 한다.
[개발 환경 준비 안내(영어)](README.md#build-and-verify)에 의존성 설치와 해시로 검증하는 로컬 자산 가져오기 절차를 정리했다. 패키지는 빌드 전에 자산을 검사한다. 가져올 번들은 현재 자산 잠금과 파일·해시가 정확히 일치해야 한다. 출처 메타데이터 정규화 이후에는 이전 0.2.5 설치본이 현재 잠금과 일치하지 않으므로 가져오기 원본으로 사용할 수 없다.

저장소 루트에서 실행한다.

```sh
bash native/macos/test.sh
bash native/macos/test.sh --live "$HOME/Library/Input Methods/Hanglish.app/Contents/Resources"

# 필수 자산이 준비된 뒤 비공개 개발 패키지 생성
bash native/macos/package.sh
```

기본값인 개발 모드에서 `package.sh`는 앱과 설치 프로그램을 만들고 검사한 뒤 기존 파일을
덮어쓰지 않고 다음 산출물을 생성한다.

```text
dist/Hanglish-0.2.6-arm64-development-NONPUBLIC.dmg
dist/Hanglish-0.2.6-arm64-development-NONPUBLIC.dmg.sha256
dist/Hanglish-0.2.6-arm64-development-NONPUBLIC.dmg.json
```

JSON manifest에는 DMG SHA-256, 소스 커밋과 작업 트리의 clean 여부, 입력 파일 해시,
버전 `0.2.6`, 빌드 `8`, 아키텍처, 최소 배포 대상, 패키징 모드가 기록된다. 개발 manifest는
해당 파일이 비공개 개발용이며 배포 준비가 끝나지 않았음을 표시한다. 완성된 DMG에는
Python 런타임이 들어 있으므로 설치 대상 Mac에 Python을 따로 설치할 필요는 없다.

릴리스 모드는 clean 작업 트리와 Developer ID Application 인증서, 인증이 끝난
`notarytool` 키체인 프로필을 이미 가진 Mac에서만 사용한다.

```sh
HANGLISH_PACKAGE_MODE=release \
HANGLISH_SIGNING_IDENTITY='Developer ID Application: Your Name (TEAMID)' \
HANGLISH_TEAM_ID='TEAMID' \
HANGLISH_NOTARY_PROFILE='hanglish-notary' \
bash native/macos/package.sh
```

사전 검사는 인증서·팀 ID·프로필 누락이나 불일치, 변경 사항이 남은 작업 트리를 거부한다.
통과하면 hardened runtime과 timestamp를 적용해 서명하고, DMG 공증과 ticket staple을
수행한 뒤 `hdiutil`, `codesign`, `stapler`, Gatekeeper로 검사한다. 이후
`Hanglish-0.2.6-arm64.dmg`, 체크섬, manifest, 공증 로그를 만든다. 이 기술 검증은 GitHub Release를 발행하거나 모델 라이선스 문제를 해결하지 않는다.
manifest에는 정확한 소스 커밋과 남은 검증 항목이 기록된다.

`test.sh`는 조합 상태, 엔진 프로세스 관리, Swift 소스 타입 검사를 실행한다. `--live`는
지정한 번들 리소스의 실제 내장 엔진을 시작해 대표 문장을 변환하고 선택된 모델 버전을
확인한다. 실제 입력 소스 선택·키 입력·아이콘 표시는 앱에서 별도로 확인해야 한다.

## 현재 범위

- Apple Silicon용으로 빌드하며 최소 배포 대상을 macOS 14로 설정한다. 실제 로컬 검증은 macOS 27에서 진행했다. macOS 14 등 확인하지 않은 버전은 미검증 프리뷰이며, 해당 버전에서의 설치·입력 소스 동작은 확인하지 않았다.
- 암호 입력란이나 일부 특수 텍스트 위젯에서는 운영체제나 앱이 입력기를 제한할 수 있다.
- 현재 개발 산출물은 ad-hoc 서명 상태이고 비공개용으로 표시되므로 공개 릴리스로 배포하면 안 된다. 일반 사용자용 설치 파일에는 Developer ID 서명과 공증이 필요하다.
- 포함된 오픈소스 구성요소의 고지와 라이선스는 패키지에 보존한다. KoELECTRA 기반 모델의 원본 모델에는 공식 프로젝트의 Apache-2.0을 적용하는 기준을 사용한다.
  정확한 출처와 학습 데이터 조건은 외부 구성요소 고지를 참고한다.

자체 코드에는 [CC BY-NC 4.0](../../LICENSE)을 적용한다.
[라이선스 전문(영어)](../../LICENSE)와 [외부 구성요소 고지(영어)](../../THIRD_PARTY_NOTICES.md)를 참고한다.
