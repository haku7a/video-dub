from dataclasses import dataclass
from abc import ABC, abstractmethod
from pathlib import Path


@dataclass
class Segment:
    start: float
    end: float
    text: str
    translated_text: str = ""


@dataclass
class Transcription:
    audio_file: str
    language: str
    segments: list[Segment]


class SpeechToText(ABC):
    @abstractmethod
    def transcribe(self, audio_path: Path) -> Transcription: ...


class Translator(ABC):
    @abstractmethod
    def translate(
        self, transcription: Transcription, target_lang: str = "ru"
    ) -> Transcription: ...


class TextToSpeech(ABC):
    @abstractmethod
    async def generate_audio(
        self,
        text: str,
        output_path: Path,
    ) -> None: ...
