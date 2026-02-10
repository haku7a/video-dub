import logging
from pathlib import Path
from typing import Any

from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


def transcribe_audio(audio_path: Path, model_size: str = "large-v3") -> dict[str, Any]:
    try:
        model = WhisperModel(
            model_size,
            device="cuda",
            compute_type="float16",
        )

        logger.info(f"Transcribing: {audio_path.name}")

        segments, info = model.transcribe(
            str(audio_path),
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            repetition_penalty=1.2,
            temperature=0,
            condition_on_previous_text=False,
        )

        file_segments = []

        for segment in segments:
            segment_data = {
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": segment.text.strip(),
            }

            file_segments.append(segment_data)

        return {
            "audio_file": audio_path.name,
            "language": info.language,
            "segments": file_segments,
        }

    except Exception as e:
        logger.error(f"Failed to transcribe audio: {e}")
        return {}
