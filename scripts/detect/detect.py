"""
scripts/detect/detect.py
YOLOv8 모델로 이미지 추론 → 바운딩 박스 저장 + 시각화 이미지 출력
"""

import os
import sys
import time
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import (
    DETECT_WEIGHT, DETECT_IMAGES_DIR,
    LABEL_IMAGE_DIR, INFORMATION_DIR,
    BOX_MIN_WIDTH, BOX_MIN_HEIGHT, BOX_CONF,
)


def run(
    weight_path: str  = DETECT_WEIGHT,
    images_dir:  str  = DETECT_IMAGES_DIR,
    label_img_dir: str = LABEL_IMAGE_DIR,
    info_dir:    str  = INFORMATION_DIR,
    box_min_w:   int  = BOX_MIN_WIDTH,
    box_min_h:   int  = BOX_MIN_HEIGHT,
    conf_thresh: float = BOX_CONF,
):
    from ultralytics import YOLO

    if not os.path.exists(weight_path):
        print(f"[ERROR] 가중치 없음: {weight_path}")
        return False

    os.makedirs(label_img_dir, exist_ok=True)
    os.makedirs(info_dir, exist_ok=True)

    model = YOLO(weight_path)
    print(f"[detect] 모델 로드 완료: {weight_path}")

    image_files = [
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    print(f"[detect] 총 {len(image_files)}개 이미지 처리 시작")

    start = time.time()
    for image_name in image_files:
        image_path = os.path.join(images_dir, image_name)
        image = cv2.imread(image_path)
        if image is None:
            print(f"[WARN] 이미지 로드 실패: {image_name}")
            continue

        results = model(image, conf=conf_thresh)

        info_path  = os.path.join(info_dir, f"{os.path.splitext(image_name)[0]}.txt")
        label_path = os.path.join(label_img_dir, image_name)

        with open(info_path, "w") as info_file:
            for result in results:
                boxes       = result.boxes.xyxy
                confidences = result.boxes.conf

                for box, conf in zip(boxes, confidences):
                    x1, y1, x2, y2 = map(int, box)
                    conf = float(conf)

                    if (x2 - x1 > box_min_w) and (y2 - y1 > box_min_h):
                        info_file.write(f"{x1} {y1} {x2} {y2}\n")
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imwrite(label_path, image)
        print(f"  ✓ {image_name}")

    elapsed = time.time() - start
    print(f"[detect] 완료: {len(image_files)}개 / {elapsed:.1f}초")
    return True


if __name__ == "__main__":
    run()
