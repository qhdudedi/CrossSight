"""[3단계] 횡단 의도  -  부분 모델 3: PCPA (Pedestrian Crossing Prediction with Attention), IDD-PeD 학습 체크포인트

역할
    보행자 1명의 "최근 16 프레임" 을 보고 횡단 의도 확률(0~1)을 낸다.
    1, 2단계 결과(bbox, 관절)와 프레임 영상 조각을 PCPA 가 학습된 입력 규격 그대로 조립하는 것이 핵심이다.

입력 / 출력 (계획서 표 3행, 원본 코드 action_predict.py 의 PCPA 클래스 기준)
    local_context : (16, 112, 112, 3)  bbox 를 1.5배 넓힌 정사각형 영상 조각. 픽셀값 0~255 를 그대로(float) 넣는다
    pose          : (16, 34)           COCO 17 관절 (x/W, y/H). 안 보이는 관절은 0
    box           : (16, 4)            x1,y1,x2,y2 픽셀. 학습 해상도(1920x1080) 기준으로 스케일
    speed         : (16, 1)            차량 속도. IDD-PeD 는 자차(블랙박스 차량) OBD 속도.
                                       고정 CCTV 에는 자차가 없으므로 "가장 가까운 차량의 추정 속도" 로 대체한다
    출력          : 횡단 의도 확률 0~1

모델 구조 (build_pcpa)
    local_context ─ C3D(3D CNN) ─ Dense(256, sigmoid) ──────────────┐
    pose  ─ GRU(256) ─ attention ─ Dropout ─┐                        │
    box   ─ GRU(256) ─ attention ─ Dropout ─┼─ concat ─ attention ─ Dense(1, sigmoid) ─> 확률
    speed ─ GRU(256) ─ attention ─ Dropout ─┘                        │
                                                                     (C3D 임베딩도 concat 에 포함)

체크포인트 로딩 방식
    IDD-PeD 공식 intention.zip 안의 intention/pcpa/13Sep2024-21h06m20s/model.h5 를 models/pcpa_iddped.h5 로 둔다.
    이 파일은 TF 2.2 로 저장돼 최신 Keras 가 그대로 열지 못한다(Lambda 레이어 역직렬화 실패).
    그래서 같은 구조를 tf-keras 로 다시 만들고 "레이어 이름" 으로 가중치만 읽는다 (load_weights by_name).
    -> 레이어 이름이 원본과 한 글자라도 다르면 그 레이어는 학습 안 된 상태가 되므로, 로드 후 21개 레이어를 모두 검증한다.

한계
    차량 탑재 시점 + 자차 속도로 학습됐다. 고정 CCTV 에서는 분포가 달라 국내 데이터 미세조정(pcpa_kr_best.h5) 전에는 참고값이다.
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

OBS_LEN = 16  # PCPA 가 보는 프레임 수 (원본 obs_length, 고정값)
CTX = 112  # local context 한 변 크기 (C3D 입력 규격)
INPUT_TYPES = ["local_context", "pose", "box", "speed"]  # 모델 입력 순서 (원본 obs_input_type 과 동일)


# =============================================================================
# 3-1. 입력 조립
# =============================================================================
def local_context_crop(frame: np.ndarray, bbox, enlarge: float = 1.5) -> np.ndarray:
    """프레임 + bbox -> PCPA local context 1장 (112, 112, 3) uint8

    원본 전처리 3단계를 그대로 재현한다:
      1) jitter_bbox('enlarge', 1.5): bbox 를 폭/높이 변화량 중 작은 쪽 기준으로 사방 확장
      2) squarify(ratio=1): 높이에 맞춰 폭을 늘려 정사각형으로. 화면 밖으로 나가면 안쪽으로 밀어 넣음
      3) img_pad('pad_resize', 112): 긴 변을 112 로 축소하고 남는 부분은 검은색(0) 패딩
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1

    # 1) enlarge
    change = min(bw * enlarge, bh * enlarge)
    x1, y1 = x1 - change // 2, y1 - change // 2
    x2, y2 = x2 + change // 2, y2 + change // 2

    # 2) squarify
    width_change = (y2 - y1) - (x2 - x1)
    x1, x2 = x1 - width_change / 2, x2 + width_change / 2
    if x1 < 0:
        x1 = 0
    if x2 > w:
        x1, x2 = x1 - x2 + w, w
    x1, y1, x2, y2 = int(max(0, x1)), int(max(0, y1)), int(min(w, x2)), int(min(h, y2))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((CTX, CTX, 3), np.uint8)

    # 3) pad_resize
    ch, cw = crop.shape[:2]
    ratio = CTX / max(ch, cw)
    rw, rh = max(1, int(cw * ratio)), max(1, int(ch * ratio))
    crop = cv2.resize(crop, (rw, rh))
    out = np.zeros((CTX, CTX, 3), np.uint8)
    wo, ho = (CTX - rw) // 2, (CTX - rh) // 2
    out[ho : ho + rh, wo : wo + rw] = crop
    return out


class PCPAInputs:
    """보행자 트랙 1개의 '최근 16 프레임' 입력 버퍼 (슬라이딩 윈도우)

    pipeline 이 프레임마다 push() 하고, 16개가 차면 tensors() 로 모델 입력을 꺼낸다.
    """

    def __init__(self, frame_w: int, frame_h: int, ref_size: tuple[int, int] = (1920, 1080)):
        # box 입력은 학습 때 픽셀 좌표 그대로였으므로, 우리 영상 크기 -> 학습 해상도로 비례 변환할 배율
        self.sx, self.sy = ref_size[0] / frame_w, ref_size[1] / frame_h
        self.ctx: list[np.ndarray] = []  # local context 조각들
        self.pose: list[np.ndarray] = []  # (34,) 벡터들
        self.box: list[np.ndarray] = []  # (4,) 벡터들
        self.speed: list[float] = []  # km/h

    def push(self, ctx: np.ndarray, pose34: np.ndarray, bbox, speed_kmh: float) -> None:
        """이번 프레임의 입력 4종을 뒤에 붙이고, 16개를 넘으면 가장 오래된 것을 버린다."""
        x1, y1, x2, y2 = bbox
        self.ctx.append(ctx)
        self.pose.append(pose34)
        self.box.append(np.array([x1 * self.sx, y1 * self.sy, x2 * self.sx, y2 * self.sy], np.float32))
        self.speed.append(float(speed_kmh))
        for lst in (self.ctx, self.pose, self.box, self.speed):
            if len(lst) > OBS_LEN:
                del lst[0]

    @property
    def ready(self) -> bool:
        """16 프레임이 모였는가. 그 전에는 확률을 내지 않는다 (트랙 초반 16 프레임은 None)."""
        return len(self.ctx) >= OBS_LEN

    def tensors(self) -> list[np.ndarray]:
        """모델 입력 4개를 배치 차원(1)을 붙여 반환. 16개 미만이면 첫 프레임을 앞에 복제해 채운다."""
        pad = OBS_LEN - len(self.ctx)
        ctx = [self.ctx[0]] * pad + self.ctx
        pose = [self.pose[0]] * pad + self.pose
        box = [self.box[0]] * pad + self.box
        spd = [self.speed[0]] * pad + self.speed
        return [
            np.asarray(ctx, np.float32)[None],  # (1, 16, 112, 112, 3)  0~255 그대로 (학습 때 /255 안 함)
            np.asarray(pose, np.float32)[None],  # (1, 16, 34)
            np.asarray(box, np.float32)[None],  # (1, 16, 4)
            np.asarray(spd, np.float32).reshape(1, OBS_LEN, 1),  # (1, 16, 1)
        ]


# =============================================================================
# 3-2. 모델 구조 재구성 (원본 PCPA.get_model + C3DNet + attention_3d_block)
# =============================================================================
def build_pcpa(hidden: int = 256):
    """원본과 동일한 구조/레이어 이름으로 tf-keras 모델을 만든다. 가중치는 아직 없음(랜덤)."""
    from tf_keras import Model
    from tf_keras.layers import (GRU, Activation, Concatenate, Conv3D, Dense, Dropout, Flatten, Input, Lambda,
                                 MaxPooling3D, ZeroPadding3D, concatenate, dot)
    import tf_keras.backend as K

    def c3d(x):
        """C3D (Tran et al. 2015) 합성곱 부분. 16x112x112x3 -> 8192 차원 벡터"""
        x = Conv3D(64, 3, activation="relu", padding="same", name="conv1")(x)
        x = MaxPooling3D((1, 2, 2), strides=(1, 2, 2), padding="valid", name="pool1")(x)
        x = Conv3D(128, 3, activation="relu", padding="same", name="conv2")(x)
        x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding="valid", name="pool2")(x)
        x = Conv3D(256, 3, activation="relu", padding="same", name="conv3a")(x)
        x = Conv3D(256, 3, activation="relu", padding="same", name="conv3b")(x)
        x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding="valid", name="pool3")(x)
        x = Conv3D(512, 3, activation="relu", padding="same", name="conv4a")(x)
        x = Conv3D(512, 3, activation="relu", padding="same", name="conv4b")(x)
        x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding="valid", name="pool4")(x)
        x = Conv3D(512, 3, activation="relu", padding="same", name="conv5a")(x)
        x = Conv3D(512, 3, activation="relu", padding="same", name="conv5b")(x)
        x = ZeroPadding3D(padding=(0, 1, 1), name="zeropad5")(x)
        x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding="valid", name="pool5")(x)
        return Flatten(name="flatten")(x)

    def attention(hs, modality):
        """many-to-one attention: 시계열 (T, hidden) -> 마지막 상태를 query 로 가중합 -> (hidden,)"""
        size = int(hs.shape[2])
        score_first = Dense(size, use_bias=False, name="attention_score_vec" + modality)(hs)
        h_t = Lambda(lambda t: t[:, -1, :], output_shape=(size,), name="last_hidden_state" + modality)(hs)
        score = dot([score_first, h_t], [2, 1], name="attention_score" + modality)
        weights = Activation("softmax", name="attention_weight" + modality)(score)
        context = dot([hs, weights], [1, 1], name="context_vector" + modality)
        pre = concatenate([context, h_t], name="attention_output" + modality)
        return Dense(hidden, use_bias=False, activation="tanh", name="attention_vector" + modality)(pre)

    # --- 입력 1: local_context -> C3D -> 256 차원 임베딩
    ctx_in = Input(shape=(OBS_LEN, CTX, CTX, 3), name="input_local_context")
    enc = [Dense(hidden, activation="sigmoid", name="emb_c3d")(c3d(ctx_in))]
    inputs = [ctx_in]

    # --- 입력 2~4: pose / box / speed -> 각각 GRU (시계열 출력 유지)
    sizes = {"pose": 34, "box": 4, "speed": 1}
    for name in INPUT_TYPES[1:]:
        inp = Input(shape=(OBS_LEN, sizes[name]), name="input_" + name)
        inputs.append(inp)
        enc.append(GRU(hidden, return_sequences=True, name="enc_" + name)(inp))

    # --- 각 GRU 출력에 attention -> (1, hidden) 로 만들어 C3D 임베딩과 함께 쌓는다
    att = [Lambda(lambda t: K.expand_dims(t, axis=1))(enc[0])]
    # 주의: 원본 코드는 attention 블록 이름을 data_types[i] (i = 0 부터) 로 붙여서 한 칸 밀려 있다.
    #       pose 브랜치 -> "_local_context", box -> "_pose", speed -> "_box".  체크포인트 이름과 맞추려면 그대로 재현해야 한다.
    for i, e in enumerate(enc[1:]):
        a = attention(e, "_" + INPUT_TYPES[i])
        a = Dropout(0.5)(a)
        att.append(Lambda(lambda t: K.expand_dims(t, axis=1))(a))

    # --- 4개 모달리티(각 1 x hidden) 를 시계열처럼 쌓아 한 번 더 attention -> 최종 확률
    x = Concatenate(name="concat_modalities", axis=1)(att)
    x = attention(x, "_modality")
    out = Dense(1, activation="sigmoid", name="output_dense")(x)
    return Model(inputs=inputs, outputs=out, name="PCPA")


class PCPAIntent:
    """사전학습 PCPA 로 의도 확률을 내는 클래스. pipeline 이 트랙마다 predict() 를 호출한다."""

    name = "pcpa-iddped"

    def __init__(self, weights: Path):
        if not weights.exists():
            raise FileNotFoundError(f"PCPA 가중치 없음: {weights}")
        import h5py

        self.model = build_pcpa()
        # 레이어 이름이 같은 것만 골라 가중치를 복사. 모양이 다르면 오류(skip_mismatch=False)
        self.model.load_weights(str(weights), by_name=True, skip_mismatch=False)

        # 검증: 우리 모델의 '가중치 있는 레이어' 이름이 h5 안에 모두 존재해야 한다
        with h5py.File(weights, "r") as f:
            g = f["model_weights"] if "model_weights" in f else f
            h5_layers = {k for k in g.keys() if len(g[k].attrs.get("weight_names", [])) > 0}
        mine = {layer.name for layer in self.model.layers if layer.weights}
        missing = mine - h5_layers
        if missing:
            raise RuntimeError(f"체크포인트에 없는 레이어: {sorted(missing)}")
        self.loaded_layers = sorted(mine)

    def predict(self, inputs: list[np.ndarray]) -> float:
        """PCPAInputs.tensors() 결과 -> 확률 1개"""
        return float(self.model.predict(inputs, verbose=0).ravel()[0])


# =============================================================================
# 3-3. 대체 경로 (PCPA 가중치가 없을 때)
# =============================================================================
class HeuristicIntent:
    """학습되지 않은 규칙 기반 기준선. 2단계 지표와 이동 지표를 로지스틱(시그모이드)으로 합친다.

    p = sigmoid( 1.5*speed_bh + 1.5*motion_onset + 0.8*head_turn + 1.2*hand_gesture - 0.5*body_frontal - 1.5 )
    걷는 속도가 빠르고, 막 움직이기 시작했고, 고개를 돌리고, 손짓을 하면 확률이 오른다.
    PCPA 결과와 비교하는 용도이며 실제 판단 근거로 쓰지 않는다.
    """

    name = "heuristic"

    def __init__(self):
        self.w = {"speed_bh": 1.5, "motion_onset": 1.5, "head_turn": 0.8, "hand_gesture": 1.2,
                  "body_frontal": -0.5, "bias": -1.5}

    def predict(self, ind: dict[str, float]) -> float:
        z = self.w["bias"] + sum(w * ind.get(k, 0.0) for k, w in self.w.items() if k != "bias")
        return 1.0 / (1.0 + math.exp(-z))
