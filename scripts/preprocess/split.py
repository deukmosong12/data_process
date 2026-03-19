"""
scripts/preprocess/split.py
YOLO 포맷 라벨 + 이미지를 train / val 로 분리
"""

import os
import sys
import shutil
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import (
    RAW_IMAGES_DIR, YOLO_LABELS_DIR,
    TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR,
    VAL_IMAGES_DIR,   VAL_LABELS_DIR,
    TRAIN_RATIO,
)


def run(
    images_dir: str  = RAW_IMAGES_DIR,
    labels_dir: str  = YOLO_LABELS_DIR,
    train_ratio: float = TRAIN_RATIO,
):
    for d in [TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR, VAL_IMAGES_DIR, VAL_LABELS_DIR]:
        os.makedirs(d, exist_ok=True)

    image_files = [
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    random.shuffle(image_files)
    split_idx = int(len(image_files) * train_ratio)
    train_files = image_files[:split_idx]
    val_files   = image_files[split_idx:]

    def copy_split(file_list, dest_img, dest_lbl):
        for fname in file_list:
            shutil.copy(os.path.join(images_dir, fname), os.path.join(dest_img, fname))
            lbl = os.path.splitext(fname)[0] + ".txt"
            src_lbl = os.path.join(labels_dir, lbl)
            if os.path.exists(src_lbl):
                shutil.copy(src_lbl, os.path.join(dest_lbl, lbl))

    copy_split(train_files, TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR)
    copy_split(val_files,   VAL_IMAGES_DIR,   VAL_LABELS_DIR)

    print(f"[split] 완료: train {len(train_files)}개 / val {len(val_files)}개")
    return True


if __name__ == "__main__":
    run()
