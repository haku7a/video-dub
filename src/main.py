import ctypes
import logging
from pathlib import Path
import asyncio

from core.media import extract_audio
from core.stt import transcribe_audio
from core.translate import translate_transcriptions
from utils.media import fetch_videos
from utils.storage import load_transcriptions, save_transcriptions
from core.tts import (
    create_audio_snippets,
    glue_audio_fragments,
    merge_video_with_dubbing,
)

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

    translated_data = translate_transcriptions(transcriptions)
    output_path = "output/json/translated_transcriptions.json"
    save_transcriptions(translated_data, output_path)

    asyncio.run(
        create_audio_snippets(
            Path("output/audio_segments"),
            translated_data,
        )
    )

    glue_audio_fragments(
        Path("output/audio_segments"),
        translated_data,
    )

    merge_video_with_dubbing(
        Path("input/videoplayback (2).mp4"),
        Path("output/audio_segments/videoplayback (2)_final_dub.mp3"),
        Path("output/result.mp4"),
    )


if __name__ == "__main__":
    main()
