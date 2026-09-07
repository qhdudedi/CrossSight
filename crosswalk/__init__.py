"""
    영상 프레임
      │
      ▼
    [1] detector.py   YOLO11 (COCO) + BoT-SORT 추적 (설정 tracker.yaml)
      │               프레임 -> 사람/차량 bbox, class, confidence, track_id
      ▼
    [2] pose.py       YOLO11-pose (COCO Keypoints 17)
      │               보행자 영역 -> 17 관절 좌표 + confidence -> 몸 방향 / 고개 돌림 / 손짓 지표
      ▼
    [3] intent.py     PCPA (IDD-PeD 학습 체크포인트)
      │               최근 16 프레임의 local context + pose + bbox (speed 는 항상 0) -> 횡단 의도 확률 0~1
      ▼
    pipeline.py       위 세 단계를 프레임 루프로 연결해 보행자별 확률을 내고 overlay.mp4 를 저장
    
    run.py            명령행 진입점

"""
