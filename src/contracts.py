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
