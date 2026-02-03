import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def save_transcriptions(
    data: list[dict], output_path: str = "output/json/transcriptions.json"
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logger.info(f"Transcriptions saved to {path}")
    except Exception as e:
        logger.error(f"Failed to save transcriptions to {path}: {e}")
