"""
scripts/preprocess/maketxt.py
CVAT annotations.xml → 이미지별 .txt 파일 변환
출력 형식: <label> <x1> <y1> <x2> <y2>
"""

import os
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import RAW_XML_PATH, LABELS_DIR


def run(xml_path: str = RAW_XML_PATH, output_dir: str = LABELS_DIR):
    os.makedirs(output_dir, exist_ok=True)

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"[ERROR] XML 파싱 실패: {e}")
        return False

    count = 0
    for image in root.findall("image"):
        image_name = image.get("name")
        boxes = image.findall("box")
        if not boxes:
            continue

        output_file = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}.txt")
        with open(output_file, "w") as f:
            for box in boxes:
                label = box.get("label")
                xtl = box.get("xtl")
                ytl = box.get("ytl")
                xbr = box.get("xbr")
                ybr = box.get("ybr")
                f.write(f"{label} {xtl} {ytl} {xbr} {ybr}\n")
        count += 1

    print(f"[maketxt] 완료: {count}개 이미지 → '{output_dir}'")
    return True


if __name__ == "__main__":
    run()
