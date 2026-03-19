"""
scripts/detect/genderdetect.py
성별 분류 모델(genderbest.pt)로 이미지 추론 → 성별 라벨 저장
"""

import os
import sys
import time
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import DETECT_IMAGES_DIR, LABEL_IMAGE_DIR, INFORMATION_DIR, ROOT

GENDER_WEIGHT = os.path.join(ROOT, "models", "weights", "genderbest.pt")
CLASS_LABELS  = {0: "male", 1: "female"}


def run(
    weight_path:   str = GENDER_WEIGHT,
    images_dir:    str = DETECT_IMAGES_DIR,
    label_img_dir: str = LABEL_IMAGE_DIR,
    info_dir:      str = INFORMATION_DIR,
):
    from ultralytics import YOLO

    if not os.path.exists(weight_path):
        print(f"[ERROR] 성별 모델 가중치 없음: {weight_path}")
        return False

    os.makedirs(label_img_dir, exist_ok=True)
    os.makedirs(info_dir, exist_ok=True)

    model = YOLO(weight_path)
    image_files = [
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    print(f"[genderdetect] {len(image_files)}개 이미지 처리 시작")

    start = time.time()
    for image_name in image_files:
        image = cv2.imread(os.path.join(images_dir, image_name))
        if image is None:
            continue

        results   = model(image)
        info_path = os.path.join(info_dir, f"{os.path.splitext(image_name)[0]}.txt")
        label_path = os.path.join(label_img_dir, image_name)

        with open(info_path, "w") as info_file:
            for result in results:
                for box, conf, cls in zip(
                    result.boxes.xyxy,
                    result.boxes.conf,
                    result.boxes.cls,
                ):
                    x1, y1, x2, y2 = map(int, box)
                    cls = int(cls)
                    if (x2 - x1 > 20) and (y2 - y1 > 20):
                        label    = CLASS_LABELS.get(cls, "unknown")
                        label_id = 0 if label == "male" else 1
                        info_file.write(f"{label_id} {x1} {y1} {x2} {y2}\n")

                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        text = f"{label_id}: {label}"
                        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(image, (x1, y1 - th - bl), (x1 + tw, y1), (0, 255, 0), -1)
                        cv2.putText(image, text, (x1, y1 - bl),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        cv2.imwrite(label_path, image)
        print(f"  ✓ {image_name}")

    print(f"[genderdetect] 완료: {time.time() - start:.1f}초")
    return True


if __name__ == "__main__":
    run()
