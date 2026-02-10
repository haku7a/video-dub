import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def save_transcriptions(
    data: dict[str, Any],
    output_path: Path,
) -> None:
    path = Path(output_path)
    try:
        save_name = Path(data["audio_file"]).stem
        final_path = output_path / f"{save_name}_transcribed.json"
        with final_path.open("w", encoding="utf-8") as f:
            json.dump(
                data,
                f,
                ensure_ascii=False,
                indent=4,
            )
        logger.info(f"Transcriptions saved to {final_path}")
    except Exception as e:
        logger.error(f"Failed to save transcriptions: {e}")


def load_transcriptions(
    file_path: str = "output/json/transcriptions.json",
) -> List[Dict[str, Any]]:
    path = Path(file_path)

    if not path.exists():
        logger.warning(f"Transcription file not found at path: {path}")
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            logger.info(f"Transcriptions successfully loaded from file: {path}")
            return data
    except Exception as e:
        logger.warning(f"Error fetching or reading JSON: {e}")
        return []
