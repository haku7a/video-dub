import logging
from pathlib import Path
from typing import Optional

from faster_whisper import WhisperModel

from src.contracts import SpeechToText, Segment, Transcription

logger = logging.getLogger(__name__)


class FasterWhisperSTT(SpeechToText):
    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
    ):
        super().__init__()
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model: Optional[WhisperModel] = None

    @property
    def model(self):
        if self._model is None:
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
        return self._model

    def transcribe(self, audio_path: Path) -> Transcription:

        logger.info(f"Transcribing: {audio_path.name}")

        segments, info = self.model.transcribe(
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
            file_segments.append(
                Segment(
                    start=round(segment.start, 2),
                    end=round(segment.end, 2),
                    text=segment.text.strip(),
                    translated_text="",
                )
            )

        return Transcription(
            audio_file=audio_path.name,
            language=info.language,
            segments=file_segments,
        )
