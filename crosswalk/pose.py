"""[2단계] 자세 / 제스처  -  부분 모델 2: YOLO11-pose (COCO Keypoints 사전학습)

역할
    1단계에서 찾은 "사람" 영역을 잘라 17개 관절 좌표를 얻고, 그 좌표로 사람이 읽을 수 있는 지표(몸 방향, 고개 돌림,
    손짓)를 만든다. 관절 좌표 자체는 3단계 PCPA 의 pose 입력으로도 쓰인다.

입력 / 출력 (계획서 표 2행)
    입력 : 보행자 영역 (검출 bbox 를 1.5배 넓힌 crop)
    출력 : 17개 관절 좌표(원본 프레임 픽셀 기준) + 관절별 confidence
    지표 : body_frontal, facing_x, head_turn, hand_gesture, kpt_valid  (pose_indicators 함수)

왜 crop 을 쓰나
    4K 프레임 전체를 pose 모델에 넣으면 멀리 있는 사람의 관절이 뭉개진다. bbox 주변만 잘라 넣으면 작은 사람도 잘 잡힌다.

가중치
    GPU/MPS : yolo11s-pose.pt   CPU : yolo11n-pose.pt

COCO 17 관절 번호
    0 nose  1 l_eye  2 r_eye  3 l_ear  4 r_ear  5 l_shoulder  6 r_shoulder  7 l_elbow  8 r_elbow
    9 l_wrist  10 r_wrist  11 l_hip  12 r_hip  13 l_knee  14 r_knee  15 l_ankle  16 r_ankle
    (l/r 은 "사람 본인" 기준 좌우. 화면 기준이 아니다.)

한계
    자세만으로 실제 횡단 의도를 확정할 수 없다. -> 3단계 입력으로 넘긴다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from crosswalk.detector import Detection, ensure_weights, pick_device

# 자주 쓰는 관절 번호에 이름을 붙여 둔다
NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP = range(13)


class PoseEstimator:
    def __init__(self, models_dir: Path, device: str = "auto", conf: float = 0.25, crop_size: int = 384,
                 expand: float = 1.5):
        from ultralytics import YOLO

        self.device = pick_device(device)
        name = "yolo11n-pose.pt" if self.device == "cpu" else "yolo11s-pose.pt"
        self.model = YOLO(str(ensure_weights(models_dir / name)))
        self.conf = conf  # pose 모델의 사람 검출 임계값
        self.crop_size = crop_size  # crop 을 pose 모델에 넣을 때의 크기
        self.expand = expand  # bbox 를 몇 배로 넓혀 자를지 (팔을 뻗은 경우 등을 포함하기 위해)

    def _crop_box(self, det: Detection, w: int, h: int) -> tuple[int, int, int, int]:
        """검출 bbox 를 중심 기준으로 expand 배 넓힌 뒤 프레임 경계로 잘라 crop 좌표를 만든다."""
        x1, y1, x2, y2 = det.bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        bw, bh = (x2 - x1) * self.expand, (y2 - y1) * self.expand
        return (int(max(0, cx - bw / 2)), int(max(0, cy - bh / 2)), int(min(w, cx + bw / 2)), int(min(h, cy + bh / 2)))

    def __call__(self, frame: np.ndarray, persons: list[Detection]) -> None:
        """persons 각각에 대해 crop -> pose 추론 -> keypoints / kpt_conf 를 채운다 (in-place, 반환값 없음)."""
        h, w = frame.shape[:2]
        for det in persons:
            # (1) 사람 주변을 잘라낸다
            X1, Y1, X2, Y2 = self._crop_box(det, w, h)
            crop = frame[Y1:Y2, X1:X2]
            if crop.size == 0:
                continue

            # (2) crop 에 pose 모델 실행 (crop 안에 여러 사람이 있을 수 있다)
            r = self.model.predict(crop, conf=self.conf, imgsz=self.crop_size, device=self.device, verbose=False)[0]
            if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
                continue

            # (3) crop 안에서 "원래 검출 bbox 의 중심"에 가장 가까운 사람을 고른다 (다른 사람의 관절이 붙는 것을 방지)
            boxes = r.boxes.xyxy.cpu().numpy()
            centers = (boxes[:, :2] + boxes[:, 2:]) / 2
            target = np.array([(det.bbox[0] + det.bbox[2]) / 2 - X1, (det.bbox[1] + det.bbox[3]) / 2 - Y1])
            j = int(np.argmin(np.linalg.norm(centers - target, axis=1)))

            # (4) crop 좌표계 -> 원본 프레임 좌표계로 되돌린다 (+X1, +Y1)
            kxy = r.keypoints.xy[j].cpu().numpy() + np.array([X1, Y1])
            kcf = r.keypoints.conf
            det.keypoints = kxy.astype(float)
            det.kpt_conf = (kcf[j].cpu().numpy() if kcf is not None else np.ones(17)).astype(float)


def pose_indicators(kp: np.ndarray | None, kc: np.ndarray | None, thr: float = 0.3) -> dict[str, float]:
    """관절 좌표 -> 제스처/자세 지표 (사람이 읽는 값, heuristic 의도 계산에도 사용)

    body_frontal : 어깨폭 / 몸통높이.  1 에 가까우면 카메라를 정면(또는 등)으로, 0 에 가까우면 옆모습
    facing_x     : 머리가 향하는 좌우 방향. -1 화면 왼쪽, +1 화면 오른쪽, 0 정면이거나 알 수 없음
    head_turn    : 코가 두 귀의 중심에서 벗어난 정도 0~1. 클수록 고개를 옆으로 돌리고 있음(주변을 살피는 동작)
    hand_gesture : 1 이면 손목이 어깨보다 높거나(손 들기) 옆으로 크게 뻗음(수신호)
    kpt_valid    : 신뢰도 thr 이상인 관절 비율. 낮으면(원거리/가림) 위 지표를 덜 믿어야 한다

    관절 신뢰도가 thr 미만이면 그 관절은 '안 보임' 으로 취급한다.
    """
    out = {"body_frontal": 0.0, "facing_x": 0.0, "head_turn": 0.0, "hand_gesture": 0.0, "kpt_valid": 0.0}

    if kp is None or kc is None:  # pose 추론 실패
        return out
    ok = kc >= thr  # 관절별 '보임' 여부
    out["kpt_valid"] = float(ok.mean())

    # --- 몸 방향: 어깨폭이 몸통 높이에 비해 넓으면 정면, 좁으면 측면
    if ok[L_SH] and ok[R_SH] and (ok[L_HIP] or ok[R_HIP]):
        sh_w = abs(kp[L_SH, 0] - kp[R_SH, 0])
        hip_y = np.mean([kp[i, 1] for i in (L_HIP, R_HIP) if ok[i]])
        torso_h = abs(hip_y - (kp[L_SH, 1] + kp[R_SH, 1]) / 2) + 1e-6
        out["body_frontal"] = float(min(1.0, sh_w / torso_h))

    # --- 고개 방향: 코가 두 귀 중심에서 어느 쪽으로 치우쳤는지
    if ok[NOSE]:
        ears = [i for i in (L_EAR, R_EAR) if ok[i]]
        if len(ears) == 2:
            mid = (kp[L_EAR, 0] + kp[R_EAR, 0]) / 2
            ear_w = abs(kp[L_EAR, 0] - kp[R_EAR, 0]) + 1e-6
            yaw = float(np.clip((kp[NOSE, 0] - mid) / ear_w, -1, 1))  # -1~1
            out["head_turn"] = abs(yaw)
            out["facing_x"] = float(np.sign(yaw)) if abs(yaw) > 0.15 else 0.0

        elif len(ears) == 1:
            # 귀가 한쪽만 보이면 완전한 옆모습. 사람의 '오른쪽' 귀가 보이면 화면 오른쪽(+x)을 보고 있다
            out["head_turn"] = 1.0
            out["facing_x"] = 1.0 if ears[0] == R_EAR else -1.0

    # --- 손짓: 손목이 어깨보다 위(이미지 y 는 아래로 갈수록 커진다) 또는 어깨폭의 0.9배 이상 옆으로 뻗음
    if ok[L_SH] and ok[R_SH]:
        sh_w = abs(kp[L_SH, 0] - kp[R_SH, 0]) + 1e-6
        for wr, sh in ((L_WR, L_SH), (R_WR, R_SH)):
            if ok[wr] and (kp[wr, 1] < kp[sh, 1] or abs(kp[wr, 0] - kp[sh, 0]) > 0.9 * sh_w):
                out["hand_gesture"] = 1.0
    return out


def pose_to_pcpa(kp: np.ndarray | None, kc: np.ndarray | None, w: int, h: int, thr: float = 0.3) -> np.ndarray:
    """관절 -> PCPA 의 pose 입력 (34,)

    IDD-PeD 학습 데이터의 pose 포맷과 동일하게 맞춘다:
      - COCO 17 관절 순서 그대로 (x0, y0, x1, y1, ...)
      - 좌표는 프레임 크기로 나눠 0~1 로 정규화 (x/W, y/H)
      - 안 보이는 관절은 (0, 0)
    """
    v = np.zeros(34, np.float32)
    if kp is None or kc is None:
        return v
    for i in range(17):
        if kc[i] >= thr:
            v[2 * i] = kp[i, 0] / w
            v[2 * i + 1] = kp[i, 1] / h
    return v
