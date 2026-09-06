"""crosswalk 패키지 - 비신호 횡단보도 보행자 횡단 의도 추정.

계획서 "모델 선정 상세 및 개발 범위" 의 부분 사전학습 모델 3개를 입력 -> 출력 규격대로 연결한다.

    영상 프레임
      │
      ▼
    [1] detector.py   YOLO11 (COCO) + ByteTrack
      │               프레임 -> 사람/차량 bbox, class, confidence, track_id
      ▼
    [2] pose.py       YOLO11-pose (COCO Keypoints 17)
      │               보행자 영역 -> 17 관절 좌표 + confidence -> 몸 방향 / 고개 돌림 / 손짓 지표
      ▼
    [3] intent.py     PCPA (IDD-PeD 학습 체크포인트)
      │               최근 16 프레임의 local context + pose + bbox + 차량속도 -> 횡단 의도 확률 0~1
      ▼
    pipeline.py       위 세 단계를 프레임 루프로 연결하고 결과(json / csv / overlay.mp4)를 저장
    run.py            명령행 진입점

전체 흐름은 FLOW.md 참고.
"""
