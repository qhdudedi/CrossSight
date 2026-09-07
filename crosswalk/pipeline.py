"""파이프라인 - 영상 -> [1] 검출/추적 -> [2] 자세/제스처 -> [3] 횡단 의도 -> 보행자별 확률

프레임 루프 한 바퀴에서 일어나는 일 (Pipeline.run 안의 while 문):

    frame 읽기
      ├─ [1] detector(frame)            -> persons, vehicles (bbox + track_id)   * 차량은 '있다' 만 쓴다 (박스 표시용)
      ├─ [2] pose(frame, persons)       -> persons[i].keypoints 채움
      └─ 보행자마다
           ├─ pose_indicators()          -> 몸방향/고개돌림/손짓 지표
           ├─ TrackState.update_motion() -> 걷는 속도(speed_bh), 이동 시작(motion_onset)
           ├─ [3] PCPAInputs.push(...)   -> 16 프레임 버퍼에 이번 프레임 추가
           │      버퍼가 차면 PCPAIntent.predict() -> intent_prob   (intent_stride 프레임마다)
           │      (PCPA 가 없으면 HeuristicIntent.predict(지표) 로 대체)
           └─ TrackState.last_prob 갱신
      └─ overlay 프레임 그리기 -> overlay.mp4 에 쓰기

루프가 끝나면 트랙별 마지막 확률을 모아 {"pedestrians": [{track_id, intent_prob}, ...]} 로 반환한다.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from crosswalk.detector import Detection, Detector
from crosswalk.intent import HeuristicIntent, PCPAInputs, PCPAIntent, local_context_crop
from crosswalk.pose import PoseEstimator, pose_indicators, pose_to_pcpa


# =============================================================================
# 보행자 / 차량의 시계열 상태
# =============================================================================
@dataclass
class TrackState:
    """보행자 1명(track_id 1개)의 누적 상태. 이동 지표 계산과 PCPA 입력 버퍼를 갖는다."""

    feet: list[tuple[float, float]] = field(default_factory=list)  # 프레임별 발 위치 (bbox 아래 중앙)
    heights: list[float] = field(default_factory=list)  # 프레임별 bbox 높이 (키 기준 정규화용)
    ts: list[float] = field(default_factory=list)  # 프레임별 시각(초)
    speeds: list[float] = field(default_factory=list)  # 프레임별 걷는 속도 (body-heights / s)
    pcpa: PCPAInputs | None = None  # 16 프레임 입력 버퍼 (PCPA 사용 시)
    last_prob: float | None = None  # 마지막으로 계산한 의도 확률 (stride 사이 프레임은 이 값을 유지)
    last_ind: dict[str, float] = field(default_factory=dict)  # 마지막 지표 (화면 표시용)

    def update_motion(self, det: Detection, t: float, fps: float) -> dict[str, float]:
        """이번 프레임의 위치를 누적하고 이동 지표 2개를 돌려준다.

        speed_bh     : 최근 0.5초 동안 발 위치가 움직인 거리 / 0.5초 / 사람 키(bbox 높이).
                       픽셀이 아니라 '초당 몇 키 만큼' 이라서 카메라 거리와 무관하게 비교 가능. 보통 걷기 ≈ 0.4~0.8
        motion_onset : 최근 1초 안에 거의 정지(<0.15) 였다가 걷기(>0.4) 로 바뀌었으면 1 (횡단 시작 신호)
        """
        self.feet.append(det.foot)
        self.heights.append(max(det.height, 1.0))
        self.ts.append(t)

        k = max(2, int(0.5 * fps))  # 0.5초에 해당하는 프레임 수
        if len(self.feet) > 1:
            i0 = max(0, len(self.feet) - k)
            d = np.linalg.norm(np.subtract(self.feet[-1], self.feet[i0]))  # 이동 거리(픽셀)
            dt = self.ts[-1] - self.ts[i0]
            speed = d / dt / float(np.mean(self.heights[i0:])) if dt > 0 else 0.0
        else:
            speed = 0.0
        self.speeds.append(float(speed))

        n = max(2, int(1.0 * fps))  # 1초에 해당하는 프레임 수
        recent = self.speeds[-n:]
        onset = float(min(recent) < 0.15 and recent[-1] > 0.4 and recent.index(min(recent)) < len(recent) - 1)
        return {"speed_bh": float(speed), "motion_onset": onset}


# =============================================================================
# 파이프라인 본체
# =============================================================================
class Pipeline:
    def __init__(self, models_dir: Path = Path("models"), device: str = "auto", intent: str = "auto",
                 pcpa_weights: Path = Path("models/pcpa_iddped.h5"), intent_stride: int = 2, imgsz: int = 1280):
        # [1], [2] 모델 준비
        self.detector = Detector(models_dir, device, imgsz=imgsz)
        self.pose = PoseEstimator(models_dir, device)
        self.intent_stride = max(1, intent_stride)  # PCPA 는 무거우므로 n 프레임마다 1번만 계산

        # [3] 의도 모델 준비: auto -> PCPA 시도 후 실패하면 heuristic
        self.heuristic = HeuristicIntent()
        self.pcpa: PCPAIntent | None = None
        if intent in ("auto", "pcpa"):
            try:
                self.pcpa = PCPAIntent(pcpa_weights)
            except Exception as e:  # 가중치 파일 없음 / 로드 실패
                if intent == "pcpa":
                    raise
                print(f"[intent] PCPA 사용 불가 -> heuristic 으로 대체: {e}")
        self.intent_name = self.pcpa.name if self.pcpa else self.heuristic.name
        self.device = self.detector.device

    # ------------------------------------------------------------------
    def run(self, video: Path, out_dir: Path, max_frames: int | None = None, overlay_width: int = 1280,
            show: bool = False, step: bool = False, verbose: bool = False) -> dict:
        """영상 1개를 처음부터 끝까지 처리하고 요약 dict 를 반환한다. 파일은 overlay.mp4 하나를 만든다.

        show    : 처리 과정을 창으로 실시간 표시 (q 종료, space 일시정지/한 프레임씩, 다시 space 로 재생)
        step    : 시작부터 한 프레임씩 멈춤 (space 로 다음 프레임)
        verbose : 프레임마다 단계별 결과를 터미널에 출력
        """
        # --- 영상 열기 / 출력 준비
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise FileNotFoundError(video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out_dir.mkdir(parents=True, exist_ok=True)
        scale = overlay_width / W  # overlay 영상은 용량을 위해 폭 1280 으로 줄여 저장
        writer = cv2.VideoWriter(str(out_dir / "overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                 (overlay_width, int(H * scale)))

        tracks: dict[int, TrackState] = defaultdict(TrackState)  # 보행자 track_id -> 상태
        fi = -1

        # --- 프레임 루프
        while True:
            ok, frame = cap.read()
            if not ok or (max_frames and fi + 1 >= max_frames):
                break
            fi += 1
            t = fi / fps  # 영상 시각(초)

            # [1] 검출 + 추적
            dets = self.detector(frame)
            persons = [d for d in dets if d.kind == "person"]
            vehicles = [d for d in dets if d.kind == "vehicle"]

            # [2] 자세 (persons 각각의 keypoints 를 채운다)
            self.pose(frame, persons)

            # [3] 보행자마다 지표 + 의도 확률
            for d in persons:
                st = tracks[d.track_id]
                ind = pose_indicators(d.keypoints, d.kpt_conf)  # 자세 지표
                ind.update(st.update_motion(d, t, fps))  # 이동 지표

                if self.pcpa is not None:
                    # PCPA 경로: 입력 4종을 버퍼에 넣고, 16 프레임이 차면 stride 마다 추론
                    if st.pcpa is None:
                        st.pcpa = PCPAInputs(W, H)
                    st.pcpa.push(local_context_crop(frame, d.bbox),  # (112,112,3)
                                 pose_to_pcpa(d.keypoints, d.kpt_conf, W, H),  # (34,)
                                 d.bbox)  # speed 입력은 항상 0 (차량 속도는 쓰지 않는다)
                    if st.pcpa.ready and (fi % self.intent_stride == 0 or st.last_prob is None):
                        st.last_prob = self.pcpa.predict(st.pcpa.tensors())
                    prob = st.last_prob  # 트랙 초반 16 프레임 동안은 None
                else:
                    # heuristic 경로: 지표만으로 계산
                    prob = self.heuristic.predict(ind)
                    st.last_prob = prob

                st.last_ind = ind

            # --- 단계별 로그
            if verbose:
                n_pose = sum(d.keypoints is not None for d in persons)
                print(f"[frame {fi:4d} t={t:6.2f}s] 1.detect persons={len(persons)} vehicles={len(vehicles)} | 2.pose ok={n_pose}/{len(persons)}")
                for d in persons:
                    st = tracks[d.track_id]
                    i = st.last_ind
                    p = st.last_prob
                    print(f"      ped#{d.track_id:<3d} spd={i.get('speed_bh', 0):.2f} onset={int(i.get('motion_onset', 0))} "
                          f"head={i.get('head_turn', 0):.2f} gesture={int(i.get('hand_gesture', 0))} kpt={i.get('kpt_valid', 0):.2f} "
                          f"| 3.intent {'buffering ' + str(len(st.pcpa.ctx)) + '/16' if p is None and st.pcpa else f'p={p:.3f}'}")

            # --- 시각화 프레임 저장 / 표시
            self._draw(frame, persons, vehicles, tracks, fi, t, self.intent_name)
            small = cv2.resize(frame, (overlay_width, int(H * scale)))
            writer.write(small)
            if show:
                cv2.imshow("crosswalk-review  [q: quit, space: pause/step]", small)
                key = cv2.waitKey(0 if step else 1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord(" "):
                    step = not step  # 일시정지 <-> 재생 토글. 정지 상태에서 space 는 한 프레임 진행 후 다시 정지
            if not verbose and fi % 10 == 0:
                print(f"\r  frame {fi + 1}/{n_total}  persons={len(persons)} vehicles={len(vehicles)}", end="", flush=True)
        print()
        cap.release()
        writer.release()
        if show:
            cv2.destroyAllWindows()

        # --- 요약: 보행자(track_id)별 마지막으로 계산된 횡단 의도 확률. 확률이 한 번도 안 나온 짧은 트랙(16 프레임 미만)은 제외
        pedestrians = [{"track_id": tid, "intent_prob": round(st.last_prob, 3)}
                       for tid, st in sorted(tracks.items()) if st.last_prob is not None]
        return {"pedestrians": pedestrians}

    # ------------------------------------------------------------------
    # COCO 17 관절을 잇는 뼈대 (pose 결과가 보이도록)
    SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
                (0, 1), (0, 2), (1, 3), (2, 4)]

    @classmethod
    def _draw(cls, frame, persons, vehicles, tracks, fi, t, intent_name=""):
        """overlay 그리기.

        - 파란 박스 = 차량 (종류와 번호만. 속도는 쓰지 않는다)               ... 1단계
        - 보행자 박스 색 = 확률 (초록 0 -> 빨강 1), 노란 점/선 = 관절과 뼈대   ... 2단계
        - 박스 아래 텍스트 = 지표 (spd 걷는속도, onset 이동시작, head 고개돌림, gest 손짓)
        - 왼쪽 위 배너 = 이번 프레임에서 각 단계가 찾은 것
        - 아래쪽 띠 = 확률이 가장 높은 보행자의 PCPA 입력(최근 16 프레임 local context)   ... 3단계
        """
        s = max(1, frame.shape[1] // 1280)  # 4K 에서도 선이 보이도록 굵기 배율
        Hf, Wf = frame.shape[:2]

        # [1] 차량
        for v in vehicles:
            x1, y1, x2, y2 = map(int, v.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 140, 0), 2 * s)
            cv2.putText(frame, f"{v.cls_name}#{v.track_id}", (x1, max(0, y1 - 8 * s)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, (255, 140, 0), 2 * s)

        # [2] 보행자 + 관절 + 지표
        best = None
        for d in persons:
            st = tracks[d.track_id]
            p = st.last_prob
            if p is not None and (best is None or p > (best[1] or 0)):
                best = (st, p)
            col = (0, 255, 0) if p is None else (0, int(255 * (1 - p)), int(255 * p))  # BGR
            x1, y1, x2, y2 = map(int, d.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2 * s)
            label = f"ped#{d.track_id} p={'--' if p is None else f'{p:.2f}'}"
            cv2.putText(frame, label, (x1, max(0, y1 - 8 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * s, col, 2 * s)
            if d.keypoints is not None:
                kp, kc = d.keypoints, d.kpt_conf
                for a, b in cls.SKELETON:
                    if kc[a] >= 0.3 and kc[b] >= 0.3:
                        cv2.line(frame, (int(kp[a, 0]), int(kp[a, 1])), (int(kp[b, 0]), int(kp[b, 1])), (0, 200, 255), 1 * s)
                for (x, y), c in zip(kp, kc, strict=True):
                    if c >= 0.3:
                        cv2.circle(frame, (int(x), int(y)), 3 * s, (0, 255, 255), -1)
            i = st.last_ind
            if i:
                txt = f"spd {i.get('speed_bh', 0):.2f} onset {int(i.get('motion_onset', 0))} head {i.get('head_turn', 0):.2f} gest {int(i.get('hand_gesture', 0))}"
                cv2.putText(frame, txt, (x1, min(Hf - 4, y2 + 18 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, (255, 255, 255), 1 * s)