<!-- Created: 2026-09-15 18:54 -->

# MNIST 손글씨 숫자 인식기 (Tkinter)

Tkinter 캔버스에 숫자를 그리면 scikit-learn MLPClassifier가 인식하는 데스크톱 앱입니다.

이 폴더에는 같은 앱의 **두 가지 버전**이 있습니다.

- `digit_recognition.py` — 단일 파일 버전. 모델(`digit_model.joblib`)이 없으면 최초 실행 시 자동 학습합니다.
- `handwriting-digit-recognizer/` — 모듈화 버전. 자동 학습하지 않으므로 `train_model.py`를 먼저 실행해야 합니다.

두 버전 모두 이 폴더의 가상환경(`venv`)을 함께 사용합니다.

## 실행 방법

```bash
# 가상환경이 없다면 새로 생성
python -m venv venv

# 가상환경 활성화
# Windows (PowerShell)
venv\Scripts\Activate.ps1
# Windows (git bash)
source venv/Scripts/activate

# 의존 패키지 설치
pip install -r handwriting-digit-recognizer/requirements.txt
```

### 단일 파일 버전

```bash
python digit_recognition.py
```
또는 Windows에서 `Run_Digit_Recognition.bat` 더블클릭. `digit_model.joblib`이 없으면 최초 실행 시 MNIST 데이터를 내려받아 자동 학습합니다.

### 모듈화 버전

```bash
cd handwriting-digit-recognizer
python train_model.py   # 최초 1회 필수 — 자동 학습 안 함
python digit_app.py
```

## 구조

- `digit_recognition.py` — 단일 파일 버전 (캔버스 + 전처리 + 모델 학습/추론)
- `handwriting-digit-recognizer/` — 모듈화 버전 (`train_model.py` + `preprocess.py` + `digit_app.py`)
- `digit_model.joblib` — 단일 파일 버전이 저장/로드하는 학습된 모델
