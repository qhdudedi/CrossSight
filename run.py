"""명령행 진입점.

    uv run run.py video/test.mp4
    uv run run.py video/test.mp4 --intent heuristic         # PCPA 없이 지표만으로
    uv run run.py video/test.mp4 --meters-per-pixel 0.02    # 차량 속도(km/h) 추정을 PCPA 입력으로
    uv run run.py video/test.mp4 --max-frames 60            # 앞부분만 빠르게
    uv run run.py video/test.mp4 --show                     # 처리 과정을 창으로 보면서 (space: 일시정지/한 프레임, q: 종료)
    uv run run.py video/test.mp4 --show --step --verbose    # 한 프레임씩 넘기며 터미널 로그도 함께

흐름: 인자 파싱 -> Pipeline 생성(모델 3개 로드) -> Pipeline.run(영상) -> 요약 출력
결과 파일: outputs/<영상이름>/overlay.mp4, result.json, intent.csv
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # TensorFlow C++ 로그(경고) 숨김

from crosswalk.pipeline import Pipeline  # noqa: E402  (환경변수 설정 뒤에 import)


def main() -> None:
    ap = argparse.ArgumentParser(description="영상 -> 보행자별 횡단 의도 확률")
    ap.add_argument("video", type=Path, help="분석할 영상 파일")
    ap.add_argument("--out", type=Path, default=None, help="결과 폴더 (기본 outputs/<영상이름>/)")
    ap.add_argument("--models", type=Path, default=Path("models"), help="가중치 폴더")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    ap.add_argument("--intent", default="auto", choices=["auto", "pcpa", "heuristic"],
                    help="auto: PCPA 가중치가 있으면 PCPA, 없으면 heuristic")
    ap.add_argument("--pcpa-weights", type=Path, default=Path("models/pcpa_iddped.h5"))
    ap.add_argument("--meters-per-pixel", type=float, default=None, help="1픽셀이 몇 m 인지. 차량 속도 추정용. 없으면 speed=0")
    ap.add_argument("--intent-stride", type=int, default=2, help="PCPA 를 몇 프레임마다 계산할지 (클수록 빠름)")
    ap.add_argument("--imgsz", type=int, default=1280, help="YOLO 입력 크기. 4K 영상은 1280 권장")
    ap.add_argument("--max-frames", type=int, default=None, help="앞에서 N 프레임만 처리")
    ap.add_argument("--show", action="store_true", help="처리 과정을 창으로 실시간 표시 (q 종료, space 일시정지/한 프레임씩)")
    ap.add_argument("--step", action="store_true", help="--show 와 함께: 시작부터 한 프레임씩 (space 로 진행)")
    ap.add_argument("--verbose", action="store_true", help="프레임마다 단계별 결과를 터미널에 출력")
    a = ap.parse_args()

    out = a.out or Path("outputs") / a.video.stem

    # 1) 모델 3개 로드 (YOLO 검출, YOLO-pose, PCPA)
    pipe = Pipeline(a.models, a.device, a.intent, a.pcpa_weights, a.meters_per_pixel, a.intent_stride, a.imgsz)
    print(f"device={pipe.device}  intent_model={pipe.intent_name}  video={a.video}")

    # 2) 영상 처리
    res = pipe.run(a.video, out, a.max_frames, show=a.show, step=a.step or False, verbose=a.verbose)

    # 3) 결과 출력
    print(f"\n처리 {res['frames']} 프레임 / {res['elapsed_sec']}s ({res['fps_processed']} fps)  -> {out}/")
    print(f"{'track':>5} {'frames':>6} {'t_first':>7} {'t_last':>6} {'mean':>6} {'max':>6} {'t@max':>6} {'>=0.5':>5} {'gesture':>7} {'onset':>5}")
    for p in res["pedestrians"]:
        print(f"{p['track_id']:>5} {p['frames']:>6} {p['t_first']:>7.2f} {p['t_last']:>6.2f} {p['intent_mean']:>6.3f} "
              f"{p['intent_max']:>6.3f} {p['t_at_max']:>6.2f} {p['frames_over_0.5']:>5} {p['gesture_frames']:>7} {p['motion_onset_frames']:>5}")
    print()
    for p in res["pedestrians"]:
        print(f"보행자 #{p['track_id']} 횡단 의도 확률 : 평균 {p['intent_mean']:.3f} / 최대 {p['intent_max']:.3f} (t={p['t_at_max']:.2f}s)")
    if not res["vehicle_speed_estimated"]:
        print("* --meters-per-pixel 미지정: PCPA speed 입력은 0 으로 들어갔습니다.")


if __name__ == "__main__":
    main()
