import ctypes
import logging
from pathlib import Path

from core.media import extract_audio
from core.stt import transcribe_audio
from utils.media import fetch_videos
from utils.storage import load_transcriptions, save_transcriptions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

venv_lib_path = Path(".venv/lib/python3.12/site-packages/nvidia/cublas/lib")
cublas_path = Path(
    ".venv/lib/python3.12/site-packages/nvidia/cublas/lib/libcublas.so.12"
)
if cublas_path.exists():
    ctypes.CDLL(str(cublas_path.absolute()))


def main():
    video_paths = fetch_videos()
    list_path_audio = extract_audio(video_paths)
    transcriptions = transcribe_audio(list_path_audio, model_size="large-v3")
    if transcriptions:
        save_transcriptions(transcriptions)

    transcriptions = load_transcriptions()


if __name__ == "__main__":
    main()
