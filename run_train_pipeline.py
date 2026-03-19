"""
run_train_pipeline.py
────────────────────────────────────────────────────
전처리 → 포맷 변환 → 데이터 분할 → YOLOv8 학습을

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
    parser = argparse.ArgumentParser(description="YOLOv8 학습 파이프라인")
    parser.add_argument(
        "--skip", nargs="*", type=int, default=[],
        help="건너뛸 step 번호 (예: --skip 1 2)"
    )
    args = parser.parse_args()
    skip = set(args.skip)

    total_start = time.time()
    results = {}

    # XML → TXT
    if 1 not in skip:
        header(1, "XML → TXT 변환 (maketxt)")
        from scripts.preprocess.maketxt import run
        results[1] = run()
    else:
        print("\n[SKIP] STEP 1")

    # TXT → YOLO 포맷 
    if 2 not in skip:
        header(2, "TXT → YOLO 포맷 변환 (change_form)")
        from scripts.preprocess.change_form import run
        results[2] = run()
    else:
        print("\n[SKIP] STEP 2")

    # Train / Val 분리 
    if 3 not in skip:
        header(3, "Train / Val 데이터 분리 (split)")
        from scripts.preprocess.split import run
        results[3] = run()
    else:
        print("\n[SKIP] STEP 3")

    #YOLOv8 학습 ────────────────────────
    if 4 not in skip:
        header(4, "YOLOv8 모델 학습 (train)")
        from scripts.train.train import run
        results[4] = run()
    else:
        print("\n[SKIP] STEP 4")

    # ── 결과 요약
    elapsed = time.time() - total_start
    print(f"\n{'='*55}")
    print("  파이프라인 완료 요약")
    print(f"{'='*55}")
    step_names = {
        1: "XML → TXT",
        2: "TXT → YOLO 포맷",
        3: "Train / Val 분리",
        4: "YOLOv8 학습",
    }
    for step, name in step_names.items():
        if step in skip:
            status = " SKIP"
        elif results.get(step):
            status = "성공"
        else:
            status = "실패"
        print(f"  STEP {step} | {name:<20} {status}")
    print(f"\n  총 소요 시간: {elapsed:.1f}초")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
