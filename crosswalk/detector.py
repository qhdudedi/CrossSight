"""[1단계] 객체 검출 + 추적  -  부분 모델 1: YOLO11 (COCO 사전학습) + BoT-SORT

역할
    프레임 1장을 받아 "사람"과 "차량"의 위치를 찾고, 프레임이 바뀌어도 같은 물체에 같은 번호(track_id)를 붙인다.

입력 / 출력
    입력 : 영상 프레임 (H, W, 3) BGR numpy 배열 (OpenCV 가 읽은 그대로)
    출력 : Detection 목록 - 각 항목은 bbox(x1,y1,x2,y2 픽셀), class 이름, confidence, track_id

가중치
    GPU/MPS : yolo11s.pt   CPU : yolo11n.pt
    COCO 80 클래스 중 0(person), 2(car), 3(motorcycle), 5(bus), 7(truck) 만 사용한다.

추적 (BoT-SORT + ReID)
    별도 학습 모델이 아니라 YOLO 가 낸 bbox 들을 프레임 간에 이어 붙여 track_id 를 부여하는 추적기다.
    ultralytics 의 model.track(persist=True) 안에 내장되어 있다. 설정은 crosswalk/tracker.yaml 에 있다
    (같은 사람이 가려졌다 다시 나타나도 같은 id 를 유지하도록 외형 비교 + 긴 유지 시간을 준다).

한계
    위치만 안다. 횡단 의도, 차량의 정지 여부, 위반 여부는 판단하지 못한다. (-> 2, 3단계)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# COCO 클래스 id -> 우리가 쓰는 이름
COCO_PERSON = {0}
COCO_VEHICLE = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


@dataclass
class Detection:
    """검출 1건. 2단계(pose.py)가 keypoints / kpt_conf 를 나중에 채워 넣는다."""

    track_id: int  # 추적기가 부여한 번호. 같은 사람/차량이면 프레임이 바뀌어도 같은 값
    kind: str  # "person" | "vehicle"
    cls_name: str  # "person" | "car" | "bus" ...
    conf: float  # 검출 신뢰도 0~1
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 (픽셀)
    keypoints: np.ndarray | None = None  # (17, 2) 관절 좌표(픽셀) - pose.py 가 채움
    kpt_conf: np.ndarray | None = None  # (17,)   관절 신뢰도    - pose.py 가 채움

    @property
    def foot(self) -> tuple[float, float]:
        """bbox 아래쪽 중앙 = 발 위치. 이동 속도 계산과 '가장 가까운 차량' 거리 계산에 쓴다."""
        x1, _, x2, y2 = self.bbox
        return ((x1 + x2) / 2, y2)

    @property
    def height(self) -> float:
        """bbox 높이(픽셀). 사람 키 기준으로 속도를 정규화할 때 쓴다."""
        return self.bbox[3] - self.bbox[1]


def pick_device(device: str = "auto") -> str:
    """'auto' 면 cuda(NVIDIA) > mps(Apple) > cpu 순으로 고른다."""
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


TRACKER_CFG = Path(__file__).with_name("tracker.yaml")  # 추적기 설정 (BoT-SORT + ReID)

# 가중치 파일이 없을 경우 
def ensure_weights(path: Path) -> Path:
    """models/ 에 가중치 파일이 없으면 ultralytics 공식 릴리스에서 같은 파일명으로 내려받는다."""
    if not path.exists():
        from ultralytics.utils.downloads import attempt_download_asset

        path.parent.mkdir(parents=True, exist_ok=True)
        attempt_download_asset(str(path))
    return path


class Detector:
    def __init__(self, models_dir: Path, device: str = "auto", conf: float = 0.3, imgsz: int = 1280):
        from ultralytics import YOLO

        self.device = pick_device(device)

        # CPU 면 가벼운 n 모델, GPU/MPS 면 s 모델
        name = "yolo11n.pt" if self.device == "cpu" else "yolo11s.pt"

        self.model = YOLO(str(ensure_weights(models_dir / name)))
        self.conf = conf  # 새 트랙을 만드는 최소 검출 신뢰도 (tracker.yaml 의 new_track_thresh 와 같은 역할)

        # 추적기가 약한 박스(0.1~conf)로 기존 트랙을 이어 붙일 수 있도록 검출 자체는 낮은 임계값으로 돌린다
        self.track_low_thresh = 0.1
        self.imgsz = imgsz  # YOLO 에 넣기 전 프레임을 이 크기(긴 변)로 줄인다. 4K 영상이라 1280 이 기본
        self.classes = sorted(COCO_PERSON | set(COCO_VEHICLE))  # 사람 + 차량 클래스만 검출

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        """프레임 1장 -> Detection 목록"""
        # track(): 검출 + BoT-SORT 추적을 한 번에 수행. persist=True 로 이전 프레임의 트랙 상태를 유지
        r = self.model.track(
            frame, persist=True, tracker=str(TRACKER_CFG), conf=self.track_low_thresh, iou=0.5,
            imgsz=self.imgsz, classes=self.classes, device=self.device, verbose=False,
        )[0]
        out: list[Detection] = []

        # 검출이 없거나, 아직 트랙 id 가 확정되지 않은 프레임(id 가 None)은 빈 목록
        if r.boxes is None or r.boxes.id is None:
            return out

        # 결과 텐서를 numpy 로 꺼내 검출 1건씩 Detection 으로 변환
        for b, tid, c, cf in zip(
            r.boxes.xyxy.cpu().numpy(),  # (N,4) bbox
            r.boxes.id.cpu().numpy().astype(int),  # (N,) track_id
            r.boxes.cls.cpu().numpy().astype(int),  # (N,) COCO 클래스 id
            r.boxes.conf.cpu().numpy(),  # (N,) 신뢰도
            strict=True,
        ):
            kind = "person" if c in COCO_PERSON else "vehicle"
            cls_name = "person" if kind == "person" else COCO_VEHICLE[c]
            out.append(Detection(int(tid), kind, cls_name, float(cf), tuple(float(v) for v in b)))
        return out
