from pathlib import Path
import edge_tts
from typing import Any
import ffmpeg


async def create_audio_snippets(
    output_path: Path,
    translated: dict[str, Any],
    voice: str = "ru-RU-DmitryNeural",
) -> None:
    segments = translated.get("segments", [])
    base_name = Path(translated.get("audio_file", "")).stem
    for i, segment in enumerate(segments):
        text = segment.get("translated_text", "")
        if not text:
            continue
        file_name = f"{base_name} {i}.mp3"
        target_file = output_path / file_name
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(str(target_file))


def glue_audio_fragments(
    audio_segments_path: Path,
    audio_fragments: dict[str, Any],
):
    base_name = Path(audio_fragments.get("audio_file", "")).stem
    segments = audio_fragments.get("segments", [])

    voice_streams = []

    for i, segment in enumerate(segments):
        file_path = audio_segments_path / f"{base_name} {i}.mp3"
        if not file_path.exists():
            continue

        probe = ffmpeg.probe(str(file_path))
        duration = float(probe["format"]["duration"])
        target_dur = segment["end"] - segment["start"]

        s = ffmpeg.input(str(file_path)).audio

        if duration > target_dur:
            s = s.filter("atempo", min(duration / target_dur, 2.0))

        start_ms = int(segment["start"] * 1000)
        s = s.filter("adelay", f"{start_ms}|{start_ms}")
        voice_streams.append(s)

    combined_voice = ffmpeg.filter(
        voice_streams, "amix", inputs=len(voice_streams), normalize=0
    )

    background = ffmpeg.input(
        f"output/{audio_fragments.get('audio_file', '')}"
    ).audio.filter("volume", 0.3)

    final_audio = ffmpeg.filter(
        [combined_voice, background], "amix", inputs=2, duration="longest"
    )

    final_audio = final_audio.filter("loudnorm")

    output_file = audio_segments_path / f"{base_name}_final_dub.mp3"
    out = ffmpeg.output(final_audio, str(output_file))
    out.run(overwrite_output=True)
    (f"Ready: {output_file}")


def merge_video_with_dubbing(
    file_name: str,
    place_conservation: Path,
):
    video_path = "input/" + Path(file_name).stem + ".mp4"
    audio_path = str(place_conservation / Path(file_name).stem) + "_final_dub.mp3"
    final_vido_path = str(place_conservation / Path(file_name).stem) + "_final_dub.mp4"

    video_stream = ffmpeg.input(str(video_path)).video

    audio_stream = ffmpeg.input(audio_path).audio

    (
        ffmpeg.output(
            video_stream,
            audio_stream,
            str(final_vido_path),
            vcodec="copy",
            acodec="aac",
            shortest=None,
        ).run(overwrite_output=True)
    )
