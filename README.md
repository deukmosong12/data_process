# 🦺 YOLO Vision Pipeline

> 시각 장애인용 온디바이스 길안내 AI 장치를 위한  
> YOLOv8 커스텀 모델 학습 + 자동 라벨링 통합 파이프라인

---

## 📁 폴더 구조

```
yolo-vision-pipeline/
│
├── config.py                      ← ⚙️  모든 경로 · 파라미터 중앙 관리
│
├── run_train_pipeline.py          ← 🚀 학습 파이프라인 한 번에 실행
├── run_detect_pipeline.py         ← 🚀 탐지 파이프라인 한 번에 실행
│
├── scripts/
│   ├── preprocess/
│   │   ├── maketxt.py             XML → TXT 변환
│   │   ├── change_form.py         TXT → YOLO 포맷 변환
│   │   └── split.py               train / val 분리
│   ├── train/
│   │   └── train.py               YOLOv8 Fine-tuning
│   └── detect/
│       ├── detect.py              추론 + 바운딩 박스 저장
│       ├── anno.py                TXT → annotations.xml 변환
│       ├── genderdetect.py        성별 분류 탐지
│       └── delete.py              XML 박스 제거 유틸
│
├── dataset/
│   ├── data.yaml                  ← 클래스 정의 파일
│   ├── annotations.xml            ← CVAT 내보낸 파일 여기에 넣기
│   ├── raw_images/                ← 원본 이미지 여기에 넣기
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   └── val/
│       ├── images/
│       └── labels/
│
├── models/
│   └── weights/                   ← yolov8s.pt · best.pt 여기에 넣기
│
├── outputs/
│   ├── label_image/               탐지 결과 시각화 이미지
│   └── information/               바운딩 박스 좌표 TXT
│
└── results/                       학습 결과 (best.pt, metrics 등)
    └── experiment1/
        └── weights/
            └── best.pt
```

---

## ⚙️ 환경 설정

```bash
pip install ultralytics pillow opencv-python pyyaml
```

---

## 🚀 실행 방법

### 사전 준비

| 파일 | 경로 |
|------|------|
| CVAT 내보낸 XML | `dataset/annotations.xml` |
| 원본 이미지 | `dataset/raw_images/` |
| YOLOv8 기본 가중치 | `models/weights/yolov8s.pt` |
| 학습 완료 가중치 | `models/weights/best.pt` (탐지 단계) |

---

### 📌 학습 파이프라인 (STEP 1~4 순서대로 자동 실행)

```bash
python run_train_pipeline.py
```

| Step | 내용 |
|------|------|
| 1 | `annotations.xml` → 이미지별 `.txt` 변환 |
| 2 | `.txt` → YOLO 정규화 포맷 변환 |
| 3 | 이미지를 `train / val` 폴더로 분리 (80:20) |
| 4 | `yolov8s.pt` 기반 Fine-tuning 학습 |

일부 스텝만 건너뛰고 싶을 때:
```bash
python run_train_pipeline.py --skip 1 2   # STEP 1, 2 건너뛰기
```

---

### 📌 탐지 파이프라인 (학습된 모델로 자동 라벨링)

```bash
python run_detect_pipeline.py
```

| Step | 내용 |
|------|------|
| 1 | `best.pt` 로 이미지 추론 → 바운딩 박스 TXT 저장 |
| 2 | TXT → `annotations.xml` 변환 |

성별 분류 모드:
```bash
python run_detect_pipeline.py --gender
```

---

### 📌 개별 스크립트 실행

각 스크립트는 단독으로도 실행 가능합니다.

```bash
python scripts/preprocess/maketxt.py
python scripts/preprocess/change_form.py
python scripts/preprocess/split.py
python scripts/train/train.py
python scripts/detect/detect.py
python scripts/detect/anno.py
```

---

## ⚙️ 설정 변경

`config.py` 파일 하나에서 모든 설정을 수정합니다.

```python
# 학습 파라미터 예시
TRAIN_EPOCHS  = 500
TRAIN_BATCH   = 16
TRAIN_DEVICE  = "cpu"    # GPU: "0"
TRAIN_RATIO   = 0.8      # train:val = 80:20

# 탐지 필터 예시
BOX_MIN_WIDTH  = 20      # 최소 바운딩 박스 너비 (px)
BOX_CONF       = 0.3     # 신뢰도 임계값
```

---

## 🔄 전체 워크플로우

```
CVAT 수작업 라벨링 (annotations.xml)
            ↓
  python run_train_pipeline.py
  [STEP 1] XML → TXT
  [STEP 2] TXT → YOLO 포맷
  [STEP 3] train / val 분리
  [STEP 4] YOLOv8 Fine-tuning
            ↓
     results/experiment1/weights/best.pt
            ↓
  python run_detect_pipeline.py
  [STEP 1] best.pt 로 대용량 이미지 자동 추론
  [STEP 2] annotations.xml 변환
            ↓
     기업 모델 입력 포맷으로 납품
```

---

## ⏱️ 학습 시간 참고

| 환경 | train 4,000장 / val 1,000장 기준 |
|------|----------------------------------|
| CPU  | 약 2시간 이상 |
| GPU  | 약 20~30분 (사양에 따라 상이) |

> GPU 사용 시 `config.py` 에서 `TRAIN_DEVICE = "0"` 으로 변경
