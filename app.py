from __future__ import annotations

import io
import json
import os
import subprocess
import zipfile
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # TensorFlow C++ 로그 숨김

import imageio_ffmpeg  # noqa: E402  ffmpeg 실행 파일을 함께 제공 (브라우저 재생용 H.264 변환)
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from crosswalk.pipeline import Pipeline  # noqa: E402

VIDEO_DIR = Path("video")
OUT_DIR = Path("outputs")
THRESHOLD = 0.5  # 이 확률 이상이면 "횡단 의도 있음"
CSS = Path(__file__).with_name("app.css")  # 꾸미기: 기본 색은 .streamlit/config.toml 테마, 개별 요소 스타일은 app.css


@st.cache_resource(show_spinner="모델 로딩 중...")
def load_pipeline() -> Pipeline:
    """모델 3개는 무거우므로 한 번만 로드해 재사용한다."""
    return Pipeline()


def to_h264(src: Path, dst: Path) -> None:
    """OpenCV 가 쓴 mp4v 영상은 브라우저가 재생하지 못하므로 H.264 로 변환한다."""
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)], check=True)


def download_bundle(video: Path, overlay: Path, df: pd.DataFrame, with_csv: bool) -> tuple[str, bytes, str, str]:
    """저장 버튼용 파일 -> (버튼 라벨, 내용, 파일명, MIME).  csv 포함이면 mp4 + csv 를 zip 하나로 묶는다."""
    if not with_csv:
        return "영상 저장 (mp4)", overlay.read_bytes(), f"{video.stem}_overlay.mp4", "video/mp4"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:  # mp4 는 이미 압축돼 있어 STORED
        z.write(overlay, f"{video.stem}_overlay.mp4")
        z.writestr(f"{video.stem}_pedestrians.csv", df.to_csv(index=False).encode("utf-8-sig"))  # 엑셀 한글 호환
    return "영상 + CSV 저장 (zip)", buf.getvalue(), f"{video.stem}_result.zip", "application/zip"


def result_files(video: Path) -> tuple[Path, Path]:
    """분석 결과 위치: (브라우저용 오버레이 영상, 보행자별 확률 json)"""
    out_dir = OUT_DIR / video.stem
    return out_dir / "overlay_h264.mp4", out_dir / "pedestrians.json"


def analyze(video: Path) -> None:
    """영상 1개 분석 -> overlay_h264.mp4 + pedestrians.json 저장"""
    out_dir = OUT_DIR / video.stem
    res = load_pipeline().run(video, out_dir)
    to_h264(out_dir / "overlay.mp4", out_dir / "overlay_h264.mp4")
    (out_dir / "pedestrians.json").write_text(json.dumps(res["pedestrians"], ensure_ascii=False), encoding="utf-8")


# =============================================================================
# 화면
# =============================================================================
st.set_page_config(page_title="보행자 횡단 의도 관제", page_icon="🚸", layout="wide")
st.markdown(f"<style>{CSS.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

# --- 사이드바: 영상 선택 / 업로드 / 분석
with st.sidebar:
    st.markdown("<div class='sidebar-title'>영상</div>", unsafe_allow_html=True)
    files = sorted(VIDEO_DIR.glob("*.mp4"))
    choice = st.selectbox("저장된 영상", files, format_func=lambda p: p.name) if files else None
    uploaded = st.file_uploader("영상 업로드 (mp4)", type=["mp4"])

    if uploaded is not None:
        VIDEO_DIR.mkdir(exist_ok=True)
        choice = VIDEO_DIR / uploaded.name
        choice.write_bytes(uploaded.getbuffer())
    if choice is None:
        st.info("video/ 폴더에 영상을 넣거나 업로드하세요")
        st.stop()

    overlay, peds_file = result_files(choice)
    done = overlay.exists() and peds_file.exists()
    rerun = st.button("다시 분석" if done else "분석 시작", type="primary")
    st.caption(f"판정 기준: 횡단 의도 확률 {THRESHOLD:.1f} 이상")

if rerun or not done:
    with st.spinner(f"{choice.name} 분석 중... (영상 길이에 따라 수십 초 걸립니다)"):
        analyze(choice)

# --- 상단: 시스템명 + 파일명
st.markdown(f"<div class='topbar'><div class='sys'>보행자 횡단 의도 관제</div><div class='file'>영상: {choice.name}</div></div>",
            unsafe_allow_html=True)

# --- 가운데 영상(넓게) + 오른쪽 판정 패널
peds = json.loads(peds_file.read_text(encoding="utf-8"))
video_col, panel_col = st.columns([3, 1], gap="medium")

with video_col:
    st.video(str(overlay))

with panel_col:
    crossing = [p for p in peds if p["intent_prob"] >= THRESHOLD]
    a, b = st.columns(2)
    a.metric("탐지 보행자", f"{len(peds)}명")
    b.metric("횡단 의도", f"{len(crossing)}명")
    df = pd.DataFrame({
        "보행자": [f"#{p['track_id']}" for p in peds],
        "확률": [p["intent_prob"] for p in peds],
        "판정": ["횡단 의도 있음" if p["intent_prob"] >= THRESHOLD else "-" for p in peds],
    }).sort_values("확률", ascending=False)
    if df.empty:
        st.write("확률이 계산된 보행자가 없습니다 (16 프레임 이상 보인 보행자만 계산)")
    else:
        st.dataframe(df, hide_index=True, width="stretch",
                     column_config={"확률": st.column_config.ProgressColumn("확률", min_value=0.0, max_value=1.0, format="%.2f")})
        st.caption("영상 속 박스의 ped#번호와 표의 보행자 번호가 같습니다")

    # --- 저장: 분석 영상 다운로드. 체크하면 보행자별 확률 CSV 를 함께 zip 으로
    st.divider()
    with_csv = st.checkbox("CSV(보행자별 확률·판정)도 함께 저장")
    label, data, fname, mime = download_bundle(choice, overlay, df, with_csv)
    st.download_button(label, data, file_name=fname, mime=mime, type="primary")
