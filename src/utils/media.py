import logging
from pathlib import Path

from src.config import INPUT_DIR, FINAL_RESULTS_DIR

logger = logging.getLogger(__name__)


def fetch_videos() -> list[Path]:
    """Fetch all video files from the input directory."""
    video_extensions = {".mp4", ".avi", ".mkv", ".mov"}

    try:
        videos = [
            f
            for f in INPUT_DIR.iterdir()
            if f.is_file() and f.suffix.lower() in video_extensions
        ]
        logger.info(f"Found {len(videos)} video files in '{INPUT_DIR}'")
        return videos
    except FileNotFoundError:
        logger.warning(f"Input directory '{INPUT_DIR}' does not exist.")
        return []
    except Exception as e:
        logger.error(f"An error occurred while fetching videos: {e}")
        return []


def get_audio_output_path(pth: Path) -> Path:
    """Generate the output audio file path based on the input video path."""
    audio_path = FINAL_RESULTS_DIR / pth.with_suffix(".mp3").name
    return audio_path
