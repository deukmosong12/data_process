"""
run_detect_pipeline.py
────────────────────────────────────────────────────
YOLOv8 추론 → 바운딩 박스 저장 → annotations.xml 변환
────────────────────────────────────────────────────
"""

import sys
import argparse
import time


def header(step: int, title: str):
    print(f"\n{'='*55}")
    print(f"  STEP {step} | {title}")
    print(f"{'='*55}")


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 탐지 파이프라인")
    parser.add_argument(
        "--skip", nargs="*", type=int, default=[],
        help="건너뛸 step 번호 (예: --skip 1)"
    )
    parser.add_argument(
        "--gender", action="store_true",
        help="성별 분류 모드 (genderbest.pt 사용)"
    )
    args = parser.parse_args()
    skip = set(args.skip)

    total_start = time.time()
    results = {}

    # ── STEP 1: 이미지 탐지 ────────────────────────
    if 1 not in skip:
        if args.gender:
            header(1, "성별 분류 탐지 (genderdetect)")
            from scripts.detect.genderdetect import run
        else:
            header(1, "객체 탐지 + 바운딩 박스 저장 (detect)")
            from scripts.detect.detect import run
        results[1] = run()
    else:
        print("\n[SKIP] STEP 1")

    # ── STEP 2: annotations.xml 생성 ──────────────
    if 2 not in skip:
        header(2, "Annotation XML 변환 (anno)")
        from scripts.detect.anno import run
        results[2] = run()
    else:
        print("\n[SKIP] STEP 2")

    # ── 결과 요약 ──────────────────────────────────
    elapsed = time.time() - total_start
    print(f"\n{'='*55}")
    print("  파이프라인 완료 요약")
    print(f"{'='*55}")
    step_names = {
        1: "탐지 + 바운딩 박스 저장",
        2: "annotations.xml 변환",
    }
    for step, name in step_names.items():
        if step in skip:
            status = "  SKIP"
        elif results.get(step):
            status = " 성공"
        else:
            status = " 실패"
        
    print(f"\n  총 소요 시간: {elapsed:.1f}초")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
