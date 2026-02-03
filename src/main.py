import logging

from core.media import extract_audio
from utils.media import fetch_videos

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    video_paths = fetch_videos()
    extract_audio(video_paths)


if __name__ == "__main__":
    main()
