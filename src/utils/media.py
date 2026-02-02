import logging
from pathlib import Path

logger = logging.getLogger(__name__)

INPUT_DIR = Path("input")


def fetch_videos():
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
