import ctypes
import logging
from pathlib import Path
import asyncio

from core.media import extract_audio
from core.stt import transcribe_audio
from core.translate import translate_transcriptions
from utils.media import fetch_videos
from utils.storage import (
    load_transcriptions,
    save_transcriptions,
    delete_unnecessar_files,
)
from utils.folders import prepare_project_structure
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
    paths = prepare_project_structure()
    video_paths = fetch_videos()
    list_path_audio = extract_audio(video_paths)
    for path_audio in list_path_audio:
        transcription = transcribe_audio(path_audio, model_size="large-v3")

        translated = translate_transcriptions(transcription)
        save_transcriptions(translated, paths["json"])

        translated_fix = load_transcriptions(
            paths["json"],
            "Django Crash Course – Python Web Framework_transcribed.json",
        )

        asyncio.run(
            create_audio_snippets(
                paths["final_results"],
                translated_fix,
            )
        )

        glue_audio_fragments(
            paths["final_results"],
            translated_fix,
        )

        merge_video_with_dubbing(
            translated_fix.get("audio_file", ""),
            paths["final_results"],
        )

        delete_unnecessar_files(paths["final_results"])


if __name__ == "__main__":
    main()
