import logging
from pathlib import Path
import asyncio

from src.core.media import extract_audio
from src.utils.media import fetch_videos
from src.utils.storage import (
    load_transcriptions,
    save_transcriptions,
    delete_unnecessar_files,
)
from src.services.stt.faster_whisper_stt import FasterWhisperSTT
from src.contracts import SpeechToText, Translator
from src.services.translate.ollama_translator import OllamaTranslator

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


def main(stt_service: SpeechToText, translator_service: Translator):
    paths = prepare_project_structure()
    video_paths = fetch_videos()
    list_path_audio = extract_audio(video_paths)
    for path_audio in list_path_audio:
        transcription_obj = stt_service.transcribe(path_audio)

        transcription_dict = {
            "audio_file": transcription_obj.audio_file,
            "language": transcription_obj.language,
            "segments": [
                {"start": s.start, "end": s.end, "text": s.text}
                for s in transcription_obj.segments
            ],
        }

        translated_obj = translator_service.translate(
            transcription_obj, target_lang="ru"
        )

        translated_dict = {
            "audio_file": translated_obj.audio_file,
            "language": translated_obj.language,
            "segments": [
                {
                    "start": s.start,
                    "end": s.end,
                    "text": s.text,
                    "translated_text": s.translated_text,
                }
                for s in translated_obj.segments
            ],
        }
        save_transcriptions(translated_dict, paths["json"])

        input("!!!")

        translated_fix = load_transcriptions(
            paths["json"],
            Path(translated_dict["audio_file"]).stem + "_transcribed.json",
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
    stt = FasterWhisperSTT(
        model_size="large-v3",
        device="cuda",
        compute_type="float16",
    )
    translator = OllamaTranslator(
        model="translategemma:4b",
        temperature=0.0,
    )
    main(stt_service=stt, translator_service=translator)
