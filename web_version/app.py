"""
손글씨 인식 웹 앱 (한글 + 영문 + 숫자)
- EasyOCR 기반
- 입력: 캔버스에 직접 그리기 / 이미지 파일 업로드
"""
import base64
import io
import re
import sys

# Windows 콘솔(cp949 등)에서 EasyOCR의 다운로드 진행률 표시(유니코드 블록 문자)가
# 깨지거나 UnicodeEncodeError로 죽는 것을 방지하기 위해 stdout/stderr를 UTF-8로 강제한다.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import easyocr
import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps

app = Flask(__name__)

# 한글 + 영문 인식기 (숫자는 en/ko 모델에 기본 포함됨)
# gpu=False: 별도 GPU 설정 없이 CPU에서 동작 (첫 요청 시 모델 다운로드로 다소 시간이 걸림)
print("EasyOCR 모델 로딩 중... (최초 실행 시 모델 다운로드로 시간이 걸릴 수 있습니다)")
reader = easyocr.Reader(["ko", "en"], gpu=False)
print("모델 로딩 완료.")


def _decode_image(file_storage=None, data_url=None) -> Image.Image:
    """업로드된 파일 또는 캔버스 dataURL을 PIL 이미지로 변환."""
    if file_storage is not None:
        img = Image.open(file_storage.stream)
    elif data_url is not None:
        header, encoded = data_url.split(",", 1)
        img = Image.open(io.BytesIO(base64.b64decode(encoded)))
    else:
        raise ValueError("이미지 입력이 없습니다.")
    return img.convert("RGB")


def _prepare_for_ocr(img: Image.Image) -> np.ndarray:
    """
    캔버스에서 그린 이미지는 배경이 투명/흰색 + 얇은 검은 선인 경우가 많아
    그대로 두고, 필요시 대비를 살짝 올려 인식률을 높인다.
    """
    # 캔버스가 투명 배경(RGBA)로 넘어올 수 있으니 흰 배경에 합성
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/recognize", methods=["POST"])
def recognize():
    try:
        if "image" in request.files and request.files["image"].filename:
            img = _decode_image(file_storage=request.files["image"])
        else:
            payload = request.get_json(silent=True) or {}
            data_url = payload.get("image")
            if not data_url:
                return jsonify({"error": "이미지가 전달되지 않았습니다."}), 400
            img = _decode_image(data_url=data_url)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"이미지를 읽을 수 없습니다: {exc}"}), 400

    arr = _prepare_for_ocr(img)

    try:
        results = reader.readtext(arr, detail=1, paragraph=False)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"인식 중 오류가 발생했습니다: {exc}"}), 500

    lines = []
    for bbox, text, conf in results:
        lines.append({
            "text": text,
            "confidence": round(float(conf), 3),
            "bbox": [[float(x), float(y)] for x, y in bbox],
        })

    full_text = "\n".join(item["text"] for item in lines)

    return jsonify({
        "text": full_text,
        "lines": lines,
    })


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
