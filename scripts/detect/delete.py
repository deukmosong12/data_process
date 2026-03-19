"""
scripts/detect/delete.py
annotations.xml 내 모든 <box> 태그 제거 유틸리티
"""

import os
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import ANNOTATION_OUTPUT


def run(xml_path: str = ANNOTATION_OUTPUT):
    if not os.path.exists(xml_path):
        print(f"[ERROR] 파일 없음: {xml_path}")
        return False

    tree = ET.parse(xml_path)
    root = tree.getroot()

    removed = 0
    for image in root.findall(".//image"):
        for box in image.findall("box"):
            image.remove(box)
            removed += 1

    output_path = xml_path.replace(".xml", "_cleaned.xml")
    ET.indent(tree, space="  ", level=0)
    tree.write(output_path, encoding="utf-8",
               xml_declaration=True, short_empty_elements=False)

    print(f"[delete] 완료: <box> {removed}개 제거 → '{output_path}'")
    return True


if __name__ == "__main__":
    run()
