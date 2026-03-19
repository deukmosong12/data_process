"""
scripts/preprocess/change_form.py
.txt 라벨 → YOLO 정규화 포맷 변환
출력 형식: <class_id> <x_center> <y_center> <width> <height>
"""

import os
import sys
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import LABELS_DIR, RAW_IMAGES_DIR, YOLO_LABELS_DIR, TARGET_CLASSES


def run(
    labels_dir: str = LABELS_DIR,
    images_dir: str = RAW_IMAGES_DIR,
    output_dir: str = YOLO_LABELS_DIR,
    target_classes: list = TARGET_CLASSES,
):
    os.makedirs(output_dir, exist_ok=True)
    class_mapping = {name: idx for idx, name in enumerate(target_classes)}

    converted, skipped = 0, 0
    for label_file in os.listdir(labels_dir):
        if not label_file.endswith(".txt"):
            continue

        label_path = os.path.join(labels_dir, label_file)
        base_name = os.path.splitext(label_file)[0]

        # 대응 이미지 탐색
        image_path = None
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            candidate = os.path.join(images_dir, base_name + ext)
            if os.path.exists(candidate):
                image_path = candidate
                break

        if image_path is None:
            print(f"[WARN] 이미지 없음: {base_name}")
            skipped += 1
            continue

        with Image.open(image_path) as img:
            img_w, img_h = img.size

        yolo_lines = []
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                class_name, x1, y1, x2, y2 = parts
                if class_name not in class_mapping:
                    continue

                x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                x_center = ((x1 + x2) / 2) / img_w
                y_center = ((y1 + y2) / 2) / img_h
                width    = (x2 - x1) / img_w
                height   = (y2 - y1) / img_h

                class_id = class_mapping[class_name]
                yolo_lines.append(
                    f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                )

        if yolo_lines:
            with open(os.path.join(output_dir, label_file), "w") as f:
                f.writelines(yolo_lines)
            converted += 1
        else:
            skipped += 1

    print(f"[change_form] 완료: 변환 {converted}개 / 스킵 {skipped}개 → '{output_dir}'")
    return True


if __name__ == "__main__":
    run()
