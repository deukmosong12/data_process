"""
config.py - 프로젝트 전체 설정 파일
경로, 파라미터 등 모든 설정을 여기서 관리합니다.
"""

import os

# ─────────────────────────────────────────────
#  프로젝트 루트 (이 파일 기준)
# ─────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────
#  [STEP 1] 전처리 - 원본 데이터 경로
# ─────────────────────────────────────────────
RAW_XML_PATH       = os.path.join(ROOT, "dataset", "annotations.xml")  # CVAT 내보낸 XML
RAW_IMAGES_DIR     = os.path.join(ROOT, "dataset", "raw_images")        # 원본 이미지 폴더
LABELS_DIR         = os.path.join(ROOT, "dataset", "labels_raw")        # maketxt 출력 폴더
YOLO_LABELS_DIR    = os.path.join(ROOT, "dataset", "labels_yolo")       # change_form 출력 폴더

# ─────────────────────────────────────────────
#  [STEP 2] 데이터 분할
# ─────────────────────────────────────────────
TRAIN_RATIO = 0.8   # train : val 비율

DATASET_DIR        = os.path.join(ROOT, "dataset")
TRAIN_IMAGES_DIR   = os.path.join(DATASET_DIR, "train", "images")
TRAIN_LABELS_DIR   = os.path.join(DATASET_DIR, "train", "labels")
VAL_IMAGES_DIR     = os.path.join(DATASET_DIR, "val", "images")
VAL_LABELS_DIR     = os.path.join(DATASET_DIR, "val", "labels")
DATA_YAML_PATH     = os.path.join(DATASET_DIR, "data.yaml")

# ─────────────────────────────────────────────
#  [STEP 3] 학습 파라미터
# ─────────────────────────────────────────────
BASE_WEIGHT        = os.path.join(ROOT, "models", "weights", "yolov8s.pt")
TRAIN_EPOCHS       = 500
TRAIN_IMGSZ        = 640
TRAIN_BATCH        = 16
TRAIN_LR           = 0.01
TRAIN_PATIENCE     = 50
TRAIN_DEVICE       = "cpu"   # GPU 사용 시 "0" 으로 변경
TRAIN_PROJECT      = os.path.join(ROOT, "results")
TRAIN_EXP_NAME     = "experiment1"

# ─────────────────────────────────────────────
#  [STEP 4] 탐지 (Detection) 파라미터
# ─────────────────────────────────────────────
DETECT_WEIGHT      = os.path.join(ROOT, "models", "weights", "best.pt")
DETECT_IMAGES_DIR  = os.path.join(ROOT, "dataset", "raw_images")
LABEL_IMAGE_DIR    = os.path.join(ROOT, "outputs", "label_image")
INFORMATION_DIR    = os.path.join(ROOT, "outputs", "information")
ANNOTATION_OUTPUT  = os.path.join(ROOT, "outputs", "annotations.xml")

# 바운딩 박스 최소 크기 필터 (픽셀)
BOX_MIN_WIDTH  = 20
BOX_MIN_HEIGHT = 20
BOX_CONF       = 0.3   # 신뢰도 임계값 (0.0 ~ 1.0)

# annotation 이미지 해상도
IMAGE_WIDTH  = 1408
IMAGE_HEIGHT = 792

# ─────────────────────────────────────────────
#  탐지 클래스 목록 (data.yaml 의 names 와 동일 순서)
# ─────────────────────────────────────────────
TARGET_CLASSES = [
    'wheelchair', 'truck', 'tree_trunk', 'traffic_sign', 'traffic_light',
    'table', 'stroller', 'stop', 'scooter', 'potted_plant', 'pole',
    'person', 'parking_meter', 'movable_signage', 'motorcycle', 'kiosk',
    'fire_hydrant', 'dog', 'chair', 'cat', 'carrier', 'car', 'bus',
    'bollard', 'bicycle', 'bench', 'barricade', 'manhole'
]
