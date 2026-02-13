import ctypes
import logging
from pathlib import Path
import asyncio

from src.core.media import extract_audio
from src.core.stt import transcribe_audio
from src.core.translate import translate_transcriptions
from src.utils.media import fetch_videos
from src.utils.storage import (
    load_transcriptions,
    save_transcriptions,
    delete_unnecessar_files,
)
from src.utils.folders import prepare_project_structure
from src.core.tts import (
    create_audio_snippets,
    glue_audio_fragments,
    merge_video_with_dubbing,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    paths = prepare_project_structure()
    video_paths = fetch_videos()
    list_path_audio = extract_audio(video_paths)
    for path_audio in list_path_audio:
        transcription = transcribe_audio(path_audio, model_size="large-v3")

        translated = translate_transcriptions(transcription)
        save_transcriptions(translated, paths["json"])

        input("!!!")

        translated_fix = load_transcriptions(
            paths["json"],
            Path(translated.get("audio_file", "")).stem + "_transcribed.json",
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
