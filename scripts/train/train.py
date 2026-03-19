"""
scripts/train/train.py
YOLOv8 커스텀 모델 학습
"""

import os
import sys
import yaml
from ultralytics import YOLO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import (
    DATA_YAML_PATH, BASE_WEIGHT,
    TRAIN_EPOCHS, TRAIN_IMGSZ, TRAIN_BATCH,
    TRAIN_LR, TRAIN_PATIENCE, TRAIN_DEVICE,
    TRAIN_PROJECT, TRAIN_EXP_NAME,
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def run():
    # data.yaml 확인
    if not os.path.exists(DATA_YAML_PATH):
        print(f"[ERROR] data.yaml 없음: {DATA_YAML_PATH}")
        return False

    with open(DATA_YAML_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    print(f"[train] data.yaml 로드 완료: {data}")

    # 모델 로드
    if not os.path.exists(BASE_WEIGHT):
        print(f"[ERROR] 가중치 파일 없음: {BASE_WEIGHT}")
        return False

    model = YOLO(BASE_WEIGHT)
    print(f"[train] 모델 로드 완료: {BASE_WEIGHT}")
    print(f"[train] 학습 시작 (device={TRAIN_DEVICE}, epochs={TRAIN_EPOCHS})")

    try:
        results = model.train(
            data      = DATA_YAML_PATH,
            epochs    = TRAIN_EPOCHS,
            imgsz     = TRAIN_IMGSZ,
            batch     = TRAIN_BATCH,
            project   = TRAIN_PROJECT,
            name      = TRAIN_EXP_NAME,
            optimizer = "SGD",
            lr0       = TRAIN_LR,
            device    = TRAIN_DEVICE,
            verbose   = True,
            patience  = TRAIN_PATIENCE,
        )
        print("[train] 학습 완료!")
    except Exception as e:
        print(f"[ERROR] 학습 실패: {e}")
        return False

    weights_path = os.path.join(TRAIN_PROJECT, TRAIN_EXP_NAME, "weights", "best.pt")
    if os.path.exists(weights_path):
        print(f"[train] best.pt 저장됨: {weights_path}")
    else:
        print(f"[WARN] best.pt를 찾을 수 없습니다: {weights_path}")

    return True


if __name__ == "__main__":
    run()
