# 전체 흐름 설명

## 한 줄 요약
`run.py` 가 영상을 열고, 프레임마다 **검출(YOLO11) → 자세(YOLO-pose) → 의도(PCPA)** 를 순서대로 돌려
보행자마다 "횡단 의도 확률" 을 기록한다. 결과는 터미널 표 + `outputs/<영상>/` 의 파일 3개.

## 파일 역할

| 파일 | 계획서 단계 | 하는 일 |
|---|---|---|
| `run.py` | – | 인자 파싱 → `Pipeline` 생성 → `run()` → 결과 출력 |
| `crosswalk/pipeline.py` | 연결 | 프레임 루프. 3단계를 호출하고 트랙 상태·결과를 관리 |
| `crosswalk/detector.py` | 1 객체 검출 | YOLO11 + ByteTrack. 프레임 → 사람/차량 bbox + track_id |
| `crosswalk/pose.py` | 2 자세/제스처 | YOLO11-pose. 보행자 crop → 17 관절 → 몸방향/고개돌림/손짓 지표 |
| `crosswalk/intent.py` | 3 횡단 의도 | PCPA 입력 조립 + 모델 재구성/가중치 로드 → 확률 |

## 프레임 1장이 처리되는 순서 (`Pipeline.run`)

```
① cap.read()                      프레임 (3840x2160)
② detector(frame)                 → persons[], vehicles[]     각 Detection: bbox, conf, track_id
③ veh_speed.update(vehicles)      → {차량id: km/h}            (--meters-per-pixel 없으면 전부 0)
④ pose(frame, persons)            → persons[i].keypoints (17,2), kpt_conf (17,)
⑤ 보행자 d 마다
    pose_indicators(kp, kc)       → body_frontal, facing_x, head_turn, hand_gesture, kpt_valid
    st.update_motion(d)           → speed_bh(초당 키 배수), motion_onset(0/1)
    가장 가까운 차량 속도          → near_kmh
    st.pcpa.push(ctx, pose34, bbox, near_kmh)     ← 16 프레임 슬라이딩 버퍼
    16개가 차면 (2 프레임마다)      pcpa.predict(st.pcpa.tensors()) → intent_prob
    rows.append(...)              프레임·보행자 1행
⑥ _draw() → overlay.mp4 에 프레임 기록
```

## PCPA 입력이 만들어지는 과정 (`intent.py`)

| 입력 | 모양 | 어디서 오나 | 처리 |
|---|---|---|---|
| local_context | (16,112,112,3) | 원본 프레임 + bbox | bbox 1.5배 확장 → 정사각형 → 112 로 축소+패딩. 픽셀값 0~255 그대로 |
| pose | (16,34) | 2단계 관절 | (x/W, y/H) 정규화, 안 보이면 0 |
| box | (16,4) | 1단계 bbox | 학습 해상도(1920x1080) 배율로 스케일 |
| speed | (16,1) | 차량 추정 속도 | km/h. 원래는 자차 OBD 속도 자리 |

16 프레임은 `PCPAInputs` 가 트랙별로 들고 있다가, 새 프레임이 오면 가장 오래된 것을 버린다.
그래서 트랙이 시작된 뒤 16 프레임(15fps 기준 약 1초) 동안은 확률이 `None` 이다.

## 가중치 로딩이 특별한 이유 (`intent.py` build_pcpa / PCPAIntent)

IDD-PeD 가 공개한 `model.h5` 는 TF 2.2 로 저장돼 최신 Keras 가 열지 못한다.
그래서 원본 소스와 **똑같은 레이어 이름** 으로 구조를 다시 만들고 `load_weights(by_name=True)` 로 가중치만 옮긴다.
원본 코드의 attention 레이어 이름이 한 칸 밀려 있는 버그(`_local_context` 가 pose 브랜치에 붙음)까지 그대로 따라야
21개 레이어가 전부 맞는다. 로드 후 h5 의 레이어 목록과 비교해 빠진 것이 있으면 오류를 낸다.

## 결과 파일

- `result.json` : `pedestrians`(트랙별 요약) + `frames_detail`(프레임별 확률·지표 전체) + 실행 정보(`intent_model` 등)
- `intent.csv`  : `frames_detail` 과 같은 내용의 표
- `overlay.mp4` : 보행자 박스 색 = 확률(초록 0 → 빨강 1), 노란 점 = 관절, 파란 박스 = 차량(+추정 속도)

## 자주 보는 값의 의미

- `intent_prob` : PCPA 확률. 0.5 이상이면 "횡단하려는 보행자" 로 본다 (임계값은 아직 국내 검증 전)
- `speed_bh`    : 초당 몇 '키' 만큼 움직였나. 0.4~0.8 이면 보통 걷기, 0.15 미만이면 정지
- `motion_onset`: 정지 → 걷기 전환 순간 1
- `hand_gesture`: 손을 들거나 팔을 옆으로 뻗음
- `head_turn`   : 고개를 옆으로 돌린 정도 (주변 살피기)
