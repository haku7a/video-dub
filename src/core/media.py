import logging
import subprocess
from pathlib import Path

import imageio_ffmpeg as ffmpeg

logger = logging.getLogger(__name__)


def extract_audio(video_paths: list[Path]) -> list[Path]:
    audio_paths = []
    try:
        ffmpeg_exe = ffmpeg.get_ffmpeg_exe()

        for pth in video_paths:
            audio_path = pth.with_suffix(".mp3")

            command = [
                ffmpeg_exe,
                "-i",
                str(pth),
                "-vn",
                "-acodec",
                "libmp3lame",
                "-q:a",
                "0",
                str(audio_path),
                "-y",
            ]

            subprocess.run(
                command,
                check=True,
                capture_output=True,
            )
            logger.info(f"Extracted audio to '{audio_path.name}'")
            audio_paths.append(audio_path)

        return audio_paths

    except Exception as e:
        logger.warning(f"An error occurred during audio extraction: {e}")
        return []
