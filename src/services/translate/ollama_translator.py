import logging
from typing import Any

import ollama

from src.contracts import Translator, Transcription, Segment

logger = logging.getLogger(__name__)


class OllamaTranslator(Translator):
    def __init__(
        self,
        model: str = "translategemma:4b",
        temperature: float = 0.0,
        history_size: int = 5,
    ):
        self.model = model
        self.temperature = temperature
        self.history_size = history_size

    def translate(
        self, transcription: Transcription, target_lang: str = "ru"
    ) -> Transcription:

        source_lang = transcription.language
        file_name = transcription.audio_file

        logger.info(f"Start translation: {file_name}")

        history: list[str] = []
        new_segments: list[Segment] = []

        for i, segment in enumerate(transcription.segments):
            original_text = segment.text

            if not original_text:
                new_segments.append(segment)
                continue

            history.append(original_text)
            if len(history) > self.history_size:
                history.pop(0)

            try:
                prompt = (
                    f"History: {history}\n"
                    f"Translate from {source_lang} to {target_lang} while preserving the original meaning and style.\n"
                    f"Leave specialized terminology, industry jargon, and proper names in the original language.\n"
                    f"Provide ONLY the translated text, no explanations or alternatives.\n"
                    f"Text to translate: {original_text}"
                )

                response = ollama.chat(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    options={
                        "temperature": self.temperature,
                    },
                )

                translated_text = (
                    response["message"]["content"].strip().strip('"').strip("'")
                )

                logger.info(f"[{file_name}] Segment {i + 1} translated.")

            except Exception as e:
                logger.error(f"Error translating segment {i + 1}: {e}")
                translated_text = "[TRANSLATION ERROR]"

            new_segments.append(
                Segment(
                    start=segment.start,
                    end=segment.end,
                    text=segment.text,
                    translated_text=translated_text,
                )
            )
        return Transcription(
            audio_file=transcription.audio_file,
            language=transcription.language,
            segments=new_segments,
        )
