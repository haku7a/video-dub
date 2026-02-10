from pathlib import Path
import edge_tts
from typing import Any


async def create_audio_snippets(
    output_path: Path,
    translate_transcriptions: list[dict[str, Any]],
    voice: str = "ru-RU-DmitryNeural",
) -> None:
    for video in translate_transcriptions:
        segments = video.get("segments", [])
        base_name = Path(video.get("audio_file", "")).stem
        for i, segment in enumerate(segments):
            text = segment.get("translated_text", "")
            if not text:
                continue
            file_name = f"{base_name} {i}.mp3"
            target_file = output_path / file_name
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(str(target_file))
