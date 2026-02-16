import edge_tts
from pathlib import Path

from src.contracts import TextToSpeech


class EdgeTTSSegmentGenerator(TextToSpeech):
    def __init__(
        self,
        default_voice: str,
    ):
        self.default_voice = default_voice

    async def generate_audio(
        self,
        translated: dict,
        output_path: Path,
    ):

        segments = translated.get("segments", {})
        base_name = Path(translated.get("audio_file", "")).stem

        for i, segment in enumerate(segments):
            text = segment.get("translated_text", "")

            if not text:
                continue

            file_name = f"{base_name} {i}.mp3"
            target_file = output_path / file_name
            communicate = edge_tts.Communicate(text, self.default_voice)
            await communicate.save(str(target_file))
