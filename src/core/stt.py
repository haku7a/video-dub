import logging
from pathlib import Path

from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


def transcribe_audio(
    audio_paths: list[Path], model_size: str = "large-v3"
) -> list[dict]:
    all_transcriptions = []
    try:
        model = WhisperModel(
            model_size,
            device="cuda",
            compute_type="float16",
        )

        for audio_path in audio_paths:
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

            all_transcriptions.append(
                {
                    "audio_file": audio_path.name,
                    "language": info.language,
                    "segments": file_segments,
                }
            )

        return all_transcriptions

    except Exception as e:
        logger.error(f"Failed to transcribe audio: {e}")
        return []
