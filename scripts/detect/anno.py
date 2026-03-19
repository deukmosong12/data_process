"""
scripts/detect/anno.py
탐지 결과 .txt → annotations.xml 변환
"""

import os
import sys
import xml.etree.ElementTree as ET
from xml.dom import minidom

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import INFORMATION_DIR, ANNOTATION_OUTPUT, IMAGE_WIDTH, IMAGE_HEIGHT


def parse_txt(txt_path: str):
    boxes = []
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                x1, y1, x2, y2 = parts
                boxes.append(("0", float(x1), float(y1), float(x2), float(y2)))
            elif len(parts) == 5:
                label, x1, y1, x2, y2 = parts
                boxes.append((label, float(x1), float(y1), float(x2), float(y2)))
    return boxes


def run(
    info_dir: str    = INFORMATION_DIR,
    output_path: str = ANNOTATION_OUTPUT,
    width: int       = IMAGE_WIDTH,
    height: int      = IMAGE_HEIGHT,
):
    annotations = ET.Element("annotations")

    txt_files = sorted([f for f in os.listdir(info_dir) if f.endswith(".txt")])
    for txt_file in txt_files:
        image_id   = os.path.splitext(txt_file)[0]
        image_name = f"{image_id}.jpg"
        boxes      = parse_txt(os.path.join(info_dir, txt_file))

        img_el = ET.SubElement(annotations, "image", {
            "id": image_id, "name": image_name,
            "width": str(width), "height": str(height),
        })
        for label, x1, y1, x2, y2 in boxes:
            ET.SubElement(img_el, "box", {
                "label":    "male" if label == "0" else "female",
                "source":   "auto",
                "occluded": "0",
                "xtl": f"{x1:.2f}", "ytl": f"{y1:.2f}",
                "xbr": f"{x2:.2f}", "ybr": f"{y2:.2f}",
                "z_order": "0",
            })

    xml_str = minidom.parseString(ET.tostring(annotations)).toprettyxml(indent="  ")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(xml_str)

    print(f"[anno] 완료: {len(txt_files)}개 → '{output_path}'")
    return True


if __name__ == "__main__":
    run()
