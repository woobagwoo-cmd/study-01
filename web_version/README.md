# 손글씨 인식 (한글 + 영문 + 숫자)

EasyOCR 기반 손글씨 인식 웹 앱. 캔버스에 직접 쓰거나 이미지 파일을 업로드해서 텍스트를 인식합니다.

## 실행 방법

```bash
# 가상환경 활성화 (이미 venv 폴더가 있음)
# Windows (PowerShell)
venv\Scripts\Activate.ps1
# Windows (git bash)
source venv/Scripts/activate

# 서버 실행
python app.py
```

브라우저에서 http://127.0.0.1:5000 접속.

## 최초 실행 참고

- 첫 요청 시 EasyOCR이 한글(ko)/영문(en) 인식 모델을 자동 다운로드합니다 (수십~백여 MB, 인터넷 필요).
  다운로드된 모델은 `~/.EasyOCR/model` 에 캐시되어 다음부터는 재다운로드하지 않습니다.
- CPU로 동작하며, 인식에 몇 초 정도 걸릴 수 있습니다. GPU가 있다면 `app.py`의
  `easyocr.Reader(["ko", "en"], gpu=False)` 를 `gpu=True` 로 바꾸면 더 빨라집니다
  (CUDA 지원 torch 필요).

## 구조

- `app.py` — Flask 서버, `/recognize` 엔드포인트에서 EasyOCR로 인식
- `templates/index.html` — 캔버스 그리기 + 파일 업로드 UI (프론트엔드 로직 포함)
- `requirements.txt` — 의존 패키지
