"""
흐름: 인자 파싱 -> Pipeline 생성(모델 3개 로드) -> Pipeline.run(영상) -> 보행자별 횡단 의도 확률 출력
"""
from __future__ import annotations

import argparse
import faulthandler
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # TensorFlow C++ 로그(경고) 숨김
os.environ.setdefault("QT_LOGGING_RULES", "*.warning=false")  # "QFontDatabase: Cannot find font directory ..."  Qt 로그 숨김
faulthandler.enable()  # 네이티브 라이브러리(CUDA/TF/torch) 에서 segfault 가 나도 조용히 죽지 않고 파이썬 스택을 찍는다

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
    ap.add_argument("--intent-stride", type=int, default=2, help="PCPA 를 몇 프레임마다 계산할지 (클수록 빠름)")
    ap.add_argument("--imgsz", type=int, default=1280, help="YOLO 입력 크기. 4K 영상은 1280 권장")
    ap.add_argument("--max-frames", type=int, default=None, help="앞에서 N 프레임만 처리")
    ap.add_argument("--show", action="store_true", help="처리 과정을 창으로 실시간 표시 (q 종료, space 일시정지/한 프레임씩)")
    ap.add_argument("--step", action="store_true", help="--show 와 함께: 시작부터 한 프레임씩 (space 로 진행)")
    ap.add_argument("--verbose", action="store_true", help="프레임마다 단계별 결과를 터미널에 출력")
    a = ap.parse_args()

    out = a.out or Path("outputs") / a.video.stem

    # 1) 모델 3개 로드 (YOLO 검출, YOLO-pose, PCPA)
    pipe = Pipeline(a.models, a.device, a.intent, a.pcpa_weights, a.intent_stride, a.imgsz)
    print(f"device={pipe.device}  intent={pipe.intent_name}")

    # 2) 영상 처리
    res = pipe.run(a.video, out, a.max_frames, show=a.show, step=a.step or False, verbose=a.verbose)

    # 3) 결과 출력: 보행자별 횡단 의도 확률 (마지막으로 계산된 값)
    print(f"결과 -> {out}/")
    for p in res["pedestrians"]:
        print(f"보행자 #{p['track_id']} 횡단 의도 확률 : {p['intent_prob']:.2f}")


if __name__ == "__main__":
    main()
