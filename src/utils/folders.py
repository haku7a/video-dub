import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def prepare_project_structure():
    base_dir = Path("output")
    paths = {
        "json": base_dir / "json",
        "final_results": base_dir / "final_results",
    }

    for name, path in paths.items():
        try:
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Folder created: {path}")
        except Exception as e:
            logger.error(f"Failed to create folder {path} :{e}")

    return paths
