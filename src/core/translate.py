import logging
from typing import Any

import ollama

logger = logging.getLogger(__name__)


def translate_transcriptions(data_dict: dict[str, Any]) -> dict[str, Any]:
    if not data_dict:
        logger.warning("No data to translate.")
        return {}

    source_lang = data_dict.get("language", "en")
    target_lang = "ru"

    file_name = data_dict.get("audio_file", "N/A")
    logger.info(f"Start file translation: {file_name}")

    history = []
    for i, segment in enumerate(data_dict["segments"]):
        original_text = segment.get("text", "")
        history.append(original_text)
        if len(history) > 5:
            history.pop(0)
        if not original_text:
            continue

        try:
            response = ollama.chat(
                model="translategemma:4b",
                messages=[
                    {
                        "role": "user",
                        "content": f"History: {history} Translate from {source_lang} to {target_lang} while preserving the original meaning and style.\
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
    return data_dict
