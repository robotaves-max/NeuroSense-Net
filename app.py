"""Self-contained Streamlit demo. Run: streamlit run app.py"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import streamlit as st

from src.inference import PretrainedEmotionSuite, fuse_probabilities
from src.utils import MELD_EMOTIONS

st.set_page_config(page_title="NeuroSense-Net", page_icon="🎭", layout="wide")


@st.cache_resource(show_spinner=False)
def suite() -> PretrainedEmotionSuite:
    return PretrainedEmotionSuite()


def write_upload(upload) -> Path:
    suffix = Path(upload.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(upload.getbuffer())
        return Path(handle.name)


def read_audio(path: Path):
    import librosa
    return librosa.load(str(path), sr=16000, mono=True)


def read_video(path: Path) -> List[np.ndarray]:
    import cv2
    capture = cv2.VideoCapture(str(path))
    frames: List[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    return frames


st.title("NeuroSense-Net · Multimodal Emotion Analytics")
st.caption("Choose text, audio, video, or any combination. Each available modality contributes to the final emotion prediction.")
st.info("This research demo recognizes MELD-style expressed emotion. It is not a medical or mental-health diagnosis.")

selected = st.multiselect("Input modalities", ["Text", "Audio", "Video"], default=["Text", "Audio", "Video"])
text = st.text_area("Text / transcript", placeholder="Type the spoken dialogue…", disabled="Text" not in selected)
audio = st.file_uploader("Audio (.wav, .mp3, .m4a)", type=["wav", "mp3", "m4a"], disabled="Audio" not in selected)
video = st.file_uploader("Facial video (.mp4, .avi, .mov)", type=["mp4", "avi", "mov"], disabled="Video" not in selected)

if st.button("Analyze emotion", type="primary"):
    supplied = ("Text" in selected and bool(text.strip())) or ("Audio" in selected and audio is not None) or ("Video" in selected and video is not None)
    if not supplied:
        st.error("Provide text, an audio file, or a facial video for one selected modality.")
    else:
        try:
            with st.spinner("Loading pretrained models (first use downloads them), then analyzing inputs…"):
                models = suite()
                probabilities: Dict[str, np.ndarray] = {}
                if "Text" in selected and text.strip():
                    probabilities["Text"] = models.text_probabilities(text)
                if "Audio" in selected and audio is not None:
                    waveform, rate = read_audio(write_upload(audio))
                    probabilities["Audio"] = models.audio_probabilities(waveform, rate)
                if "Video" in selected and video is not None:
                    probabilities["Video"] = models.video_probabilities(read_video(write_upload(video)))
                final = fuse_probabilities(probabilities)
            emotion = MELD_EMOTIONS[int(final.argmax())]
            st.success(f"Predicted emotion: {emotion.title()} · {final.max():.1%} confidence")
            left, right = st.columns(2)
            with left:
                st.subheader("Final emotion probabilities")
                st.bar_chart({label: float(score) for label, score in zip(MELD_EMOTIONS, final)})
            with right:
                st.subheader("Selected-modality confidence")
                st.bar_chart({name: float(score.max()) for name, score in probabilities.items()})
        except Exception as exc:
            st.error(f"Could not analyze the supplied input: {exc}")
            st.caption("Install the optional packages with: pip install -r requirements-optional.txt")
