import logging
from typing import Any, Dict, List

import ollama

logger = logging.getLogger(__name__)


def translate_transcriptions(data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not data_list:
        logger.warning("No data to translate.")
        return []

    processed_results = []

    for video_data in data_list:
        source_lang = video_data.get("language", "en")
        target_lang = "ru"

        if "segments" not in video_data:
            processed_results.append(video_data)
            continue

        file_name = video_data.get("audio_file", "N/A")
        logger.info(f"Start file translation {file_name}...")

        for i, segment in enumerate(video_data["segments"]):
            original_text = segment.get("text", "")

            if not original_text:
                continue

            try:
                response = ollama.chat(
                    model="translategemma:4b",
                    messages=[
                        {
                            "role": "user",
                            "content": f"Translate from {source_lang} to {target_lang} while preserving the original meaning and style.\
                                Leave specialized terminology, industry jargon, and proper names in the original language.\
                                      Provide ONLY the translated text, no explanations or alternatives NO EXPLANATIONS. \
                                      Text to translate: {original_text}",
                        },
                    ],
                    options={
                        "temperature": 0,
                    },
                )

                translated_text = response["message"]["content"].strip()
                translated_text = translated_text.strip('"').strip("'")

                segment["translated_text"] = translated_text

                logger.info(f"[{file_name}] Segment {i + 1} translated.")

            except Exception as e:
                logger.error(f"Error translating segment {i + 1}: {e}")
                segment["translated_text"] = "[TRANSLATION ERROR]"
        processed_results.append(video_data)

    return processed_results
