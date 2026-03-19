# 🦺 YOLO Vision Pipeline

> 시각 장애인용 온디바이스 길안내 데이터셋 구축을 위한  
> YOLOv8 커스텀 모델 학습 + 자동 라벨링 통합 파이프라인

---

## 📌 프로젝트 개요

기존 YOLOv8이 탐지하지 못하는 커스텀 객체(신호등,맨홀 뚜껑,보행자 도로 등)를 추가 학습하여  
시각 장애인용 길안내 AI 디바이스에 탑재할 모델 데이터셋을 구축합니다.

- CVAT 수작업 라벨링 데이터 → YOLOv8 학습 포맷 자동 변환
- Fine-tuning으로 커스텀 모델(`best.pt`) 생성
- 학습된 모델로 대용량 데이터 자동 라벨링 및 annotation 변환

---

## 🖼️ Detection 예시

![Detection Example](https://github.com/user-attachments/assets/074da0a4-343f-4344-a0a2-88d67c23dae9)

---

## 📁 폴더 구조

```
yolo-vision-pipeline/
│
├── config.py                      ← ⚙️  경로 · 파라미터 중앙 관리
│
├── run_train_pipeline.py          ←  학습 파이프라인 한 번에 실행
├── run_detect_pipeline.py         ←  탐지 파이프라인 한 번에 실행
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
│   ├── annotations.xml            ← CVAT 내보낸 파일 여기에
│   ├── raw_images/                ← 원본 이미지 여기에
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   └── val/
│       ├── images/
│       └── labels/
│
├── models/
│   └── weights/                   ← yolov8s.pt · best.pt 여기에
│
├── outputs/
│   ├── label_image/               탐지 결과 시각화 이미지
│   └── information/               바운딩 박스 좌표 TXT
│
└── results/
    └── experiment1/
        └── weights/
            └── best.pt            ← 학습 완료 후 생성
```

---

## ⚙️ 환경 설정

```bash
pip install ultralytics pillow opencv-python pyyaml
```

---

## 📂 데이터셋 폴더 구조

![dataset structure](https://github.com/user-attachments/assets/9f624ad6-81b4-44fb-bfba-e36f5291eda4)

- `raw_images/` : 학습 및 검증용 원본 이미지
- `train·val/labels/` : 각 이미지에 대응하는 라벨 `.txt` 파일
- `data.yaml` : 데이터 경로 및 탐지 객체 정보 정의

---

## 🗂️ data.yaml 형식

![data.yaml](https://github.com/user-attachments/assets/4cb4fd69-f49f-497b-a437-3a170dfb9fe3)

```yaml
train: train/images
val:   val/images

nc: 28
names:
  - wheelchair
  - truck
  - person
  # ... (config.py 의 TARGET_CLASSES 참고)
```

> `data.yaml` 의 `names` 순서와 `config.py` 의 `TARGET_CLASSES` 순서가 반드시 일치해야 합니다.

---

## 🚀 실행 방법

### 사전 준비

| 파일 | 위치 |
|------|------|
| CVAT 내보낸 XML | `dataset/annotations.xml` |
| 원본 이미지 | `dataset/raw_images/` |
| YOLOv8 기본 가중치 | `models/weights/yolov8s.pt` |
| 학습 완료 가중치 (탐지용) | `models/weights/best.pt` |

> 사전 학습 Head Detection 모델: [Google Drive 다운로드](https://drive.google.com/file/d/1qlBmiEU4GBV13fxPhLZqjhjBbREvs8-m/view?usp=sharing)

---

### 📌 학습 파이프라인 — 한 번에 실행

```bash
python run_train_pipeline.py
```

| Step | 스크립트 | 내용 |
|------|----------|------|
| 1 | `maketxt.py` | `annotations.xml` → 이미지별 `.txt` 변환 |
| 2 | `change_form.py` | `.txt` → YOLO 정규화 포맷 변환 |
| 3 | `split.py` | 이미지를 `train / val` 폴더로 분리 (기본 80:20) |
| 4 | `train.py` | `yolov8s.pt` 기반 Fine-tuning |

특정 스텝 건너뛰기:
```bash
python run_train_pipeline.py --skip 1 2
```

---

### 📌 탐지 파이프라인 — 한 번에 실행

```bash
python run_detect_pipeline.py
```

| Step | 스크립트 | 내용 |
|------|----------|------|
| 1 | `detect.py` | `best.pt`로 이미지 추론 → 바운딩 박스 TXT 저장 |
| 2 | `anno.py` | TXT → `annotations.xml` 변환 |

성별 분류 모드:
```bash
python run_detect_pipeline.py --gender
```

---

### 📌 개별 스크립트 실행

```bash
python scripts/preprocess/maketxt.py
python scripts/preprocess/change_form.py
python scripts/preprocess/split.py
python scripts/train/train.py
python scripts/detect/detect.py
python scripts/detect/anno.py
```

---

## 📝 스크립트 상세 설명

### 1. `maketxt.py`

CVAT에서 내보낸 `annotations.xml`에서 각 이미지의 바운딩 박스 좌표를 추출하여 `.txt` 파일로 저장합니다.

**출력 형식:**
```
<객체명> <x_left_up> <y_left_down> <x_right_up> <x_right_down>
```

---

### 2. `change_form.py`

`maketxt.py` 출력 `.txt`를 YOLOv8 학습 포맷으로 변환합니다.  
객체명 → 클래스 번호 변환, 좌표 정규화(0~1)를 수행합니다.

![change_form](https://github.com/user-attachments/assets/4ad74961-a5db-4ce0-a336-5d648758df85)

> `config.py`의 `TARGET_CLASSES` 리스트 순서 = `data.yaml`의 `names` 순서

**출력 형식:**
```
<class_id> <x_center> <y_center> <width> <height>
```

---

### 3. `split.py`

이미지와 라벨 파일을 랜덤으로 `train / val` 폴더에 복사합니다.  
`config.py`에서 `TRAIN_RATIO` 값으로 비율 조정 가능 (기본 80:20).

---

### 4. `train.py`

준비된 데이터셋과 `data.yaml`을 기반으로 YOLOv8 모델을 학습합니다.

**주요 파라미터 (`config.py` 에서 수정):**

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `BASE_WEIGHT` | 기반 가중치 | `yolov8s.pt` |
| `TRAIN_EPOCHS` | 학습 반복 횟수 | `500` |
| `TRAIN_IMGSZ` | 입력 이미지 크기 | `640` |
| `TRAIN_BATCH` | 배치 크기 | `16` |
| `TRAIN_DEVICE` | 디바이스 | `"cpu"` / `"0"` (GPU) |
| `TRAIN_PATIENCE` | Early stopping | `50` |

---

### 5. `detect.py` + `anno.py`

`best.pt`로 대량 이미지를 추론하고 바운딩 박스 좌표를 `.txt`로 저장합니다.  
이후 `anno.py`가 해당 `.txt`들을 `annotations.xml`로 취합합니다.

---

## ⏱️ 학습 시간 벤치마크

> 측정 기준: `yolov8s.pt` / `imgsz=640` / `batch=16` / `epochs=500`  
> 데이터셋: train 4,000장 / val 1,000장

| 환경 | 사양 | 소요 시간 |
|------|------|-----------|
| **CPU** | Intel Core i7-10700 (8코어) | **약 2시간 20분** |
| **GPU** | NVIDIA RTX 3080 (10GB VRAM) | **약 18분** |

> GPU 사용 시 `config.py`에서 `TRAIN_DEVICE = "0"` 으로 변경  
> CPU 대비 약 **7~8배** 빠름

---

## 🔄 전체 워크플로우

```
CVAT 수작업 라벨링
        ↓
  annotations.xml
        ↓
python run_train_pipeline.py
  [STEP 1] XML  →  이미지별 .txt
  [STEP 2] .txt →  YOLO 정규화 포맷
  [STEP 3] 데이터 train / val 분리 (80:20)
  [STEP 4] YOLOv8 Fine-tuning
        ↓
  results/experiment1/weights/best.pt
        ↓
python run_detect_pipeline.py
  [STEP 1] best.pt로 대용량 이미지 자동 추론
  [STEP 2] 바운딩 박스 TXT → annotations.xml
        ↓
  기업 모델 입력 포맷으로 변환 → 납품
```

---

## 💡 참고 사항

- 학습 성능 향상을 위해 Data Augmentation 적용 권장
- 학습 완료 후 `best.pt`로 inference를 수행하여 탐지 성능 검증
- Early Stopping 기본값 50 epoch (성능 개선 없을 시 자동 종료)
