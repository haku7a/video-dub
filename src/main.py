import logging
import os
from pathlib import Path

from core.media import extract_audio
from core.stt import transcribe_audio
from utils.media import fetch_videos

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


venv_lib_path = Path(".venv/lib/python3.12/site-packages/nvidia/cublas/lib")
if venv_lib_path.exists():
    os.environ["LD_LIBRARY_PATH"] = (
        f"{os.environ.get('LD_LIBRARY_PATH', '')}:{venv_lib_path.absolute()}"
    )


def main():
    video_paths = fetch_videos()
    list_path_audio = extract_audio(video_paths)
    transcriptions = transcribe_audio(list_path_audio, model_size="large-v3")
    print(transcriptions)


if __name__ == "__main__":
    main()
