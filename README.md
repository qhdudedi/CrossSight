# crosswalk-review
사전학습 모델 3개를 입력→출력 규격대로 연결해,
**영상을 넣으면 보행자별 횡단 의도 확률이 나오는** 구현


| 단계 | 파일 | 가중치 | 입력 → 출력 |
|---|---|---|---|
| 1 객체 검출 | `crosswalk/detector.py` | `yolo11s.pt`(GPU/MPS) / `yolo11n.pt`(CPU), COCO | 프레임 → 사람/차량 bbox, class, confidence, track_id (BoT-SORT+ReID, 같은 사람은 한 id) |
| 2 자세/제스처 | `crosswalk/pose.py` | `yolo11s-pose.pt` / `yolo11n-pose.pt`, COCO Keypoints | 보행자 영역 → 17 관절 좌표 + confidence → body_frontal, facing_x, head_turn, hand_gesture |
| 3 횡단 의도 | `crosswalk/intent.py` | `pcpa_iddped.h5` (IDD-PeD 공식 PCPA 체크포인트) | 16-frame local context(112×112) + pose(34) + bbox(4) + speed(1, 항상 0) → 확률 0~1 |
| 연결 | `crosswalk/pipeline.py`, `run.py` | – | 영상 → 트랙별 프레임 단위 확률 + overlay.mp4 |

## 설치 (uv)
```bash
uv sync          
```
가상환경 활성화는 필요 없다.  
모든 실행은 `uv run ...` 으로 한다.
- 분석: `uv run run.py 영상파일명`
- 서비스 화면(Streamlit): `uv run streamlit run app.py` - 영상을 고르면 오버레이 영상과 보행자별 횡단 의도 확률을 보여준다
YOLO 가중치는 `models/` 에 없으면 처음 실행 때 자동으로 내려받는다.