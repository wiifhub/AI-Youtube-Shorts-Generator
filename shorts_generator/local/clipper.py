"""Local clipping: ffmpeg subclip + OpenCV face-aware vertical crop.

Two stages per highlight:
  1. Cut the source video to [start, end] with ffmpeg (re-encoded, audio kept).
  2. Reframe the cut to the target aspect ratio. For 9:16 we slide a vertical
     window horizontally across the frame to keep faces centred (Haar
     cascade — same approach as the original repo, no external models).
"""
import os
import re
import shutil
import subprocess
import time
from functools import lru_cache
from pathlib import Path
import textwrap
from typing import Dict, List, Optional, Tuple

from ..config import LOCAL_OUTPUT_DIR


def _remove_with_retry(path: str, attempts: int = 8, delay: float = 0.5) -> None:
    """os.remove with retries — Windows Defender's real-time scanner can
    briefly hold a lock on a freshly-written file, causing WinError 32."""
    for i in range(attempts):
        try:
            os.remove(path)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(delay)


@lru_cache(maxsize=1)
def _find_ffmpeg() -> str:
    """Resolve ffmpeg even when winget/scoop installed it off PATH."""
    found = shutil.which("ffmpeg")
    if found:
        return found

    candidates: List[Path] = []
    local_app = os.environ.get("LOCALAPPDATA", "")
    if local_app:
        winget_root = Path(local_app) / "Microsoft" / "WinGet" / "Packages"
        if winget_root.is_dir():
            candidates.extend(winget_root.glob("**/ffmpeg.exe"))

    user_profile = os.environ.get("USERPROFILE", "")
    if user_profile:
        candidates.extend(
            (Path(user_profile) / "scoop" / "apps" / "ffmpeg").glob("**/ffmpeg.exe")
        )

    for extra in (
        Path(r"C:\ffmpeg\bin\ffmpeg.exe"),
        Path(r"C:\ProgramData\chocolatey\bin\ffmpeg.exe"),
    ):
        candidates.append(extra)

    for path in candidates:
        if path.is_file():
            return str(path)

    raise RuntimeError(
        "ffmpeg was not found on PATH (or in common install locations). "
        "Install it and ensure `ffmpeg` is available, e.g.:\n"
        "    winget install Gyan.FFmpeg\n"
        "Then restart the terminal so PATH updates take effect."
    )


def _ratio(aspect_ratio: str) -> float:
    """Parse '9:16' → 9/16, '1:1' → 1.0."""
    try:
        w, h = aspect_ratio.split(":")
        return float(w) / float(h)
    except (ValueError, ZeroDivisionError):
        return 9.0 / 16.0


def _ass_timestamp(seconds: float) -> str:
    """Format seconds as an ASS timestamp (H:MM:SS.cc)."""
    total_cs = max(0, int(round(float(seconds) * 100)))
    centiseconds = total_cs % 100
    total_seconds = total_cs // 100
    second = total_seconds % 60
    total_minutes = total_seconds // 60
    minute = total_minutes % 60
    hour = total_minutes // 60
    return f"{hour}:{minute:02d}:{second:02d}.{centiseconds:02d}"


def _escape_ass_text(value: object, remove_filler_words: bool = False) -> str:
    """Escape transcript text so it cannot be interpreted as ASS markup."""
    text = " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split())
    if remove_filler_words:
        text = re.sub(r"\b(?:um+|uh+|erm|you know|basically|literally)\b[,.]?", "", text, flags=re.I)
        text = " ".join(text.split())
    text = text.replace("\\", "\\\\").replace("{", r"\{").replace("}", r"\}")
    wrapped = textwrap.wrap(
        text,
        width=38,
        break_long_words=False,
        break_on_hyphens=False,
    )
    return r"\N".join(wrapped) if wrapped else ""


def _caption_style_values(style: str, position: str = "bottom") -> Tuple[int, str, str, str, int, int, int]:
    """Return font/style values for a named caption preset."""
    presets = {
        "clean": (48, "&H00FFFFFF", "&H00000000", "&H99000000", 3, 1, 2),
        "bold": (56, "&H00FFFFFF", "&H00000000", "&H99000000", 8, 1, 2),
        "boxed": (52, "&H00FFFFFF", "&H00000000", "&HCC000000", 1, 3, 2),
        "karaoke": (54, "&H0000FFFF", "&H0000FFFF", "&H99000000", 7, 1, 2),
    }
    values = presets.get((style or "bold").strip().lower(), presets["bold"])
    alignment = {"top": 8, "center": 5, "bottom": 2}.get((position or "bottom").lower(), 2)
    return (*values[:6], alignment)


def _write_ass_captions(
    ass_path: str,
    clip_start: float,
    clip_end: float,
    segments: List[Dict],
    caption_style: str = "bold",
    remove_filler_words: bool = False,
    caption_position: str = "bottom",
) -> int:
    """Write an ASS subtitle file containing transcript segments in a clip."""
    font_size, primary, secondary, back, outline, border_style, alignment = _caption_style_values(caption_style, caption_position)
    header = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "PlayResX: 1080",
        "PlayResY: 1920",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Default,Arial,{font_size},{primary},{secondary},&H00000000,{back},-1,0,0,0,100,100,0,0,{border_style},{outline},2,{alignment},72,72,170,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    lines = list(header)
    count = 0
    for segment in segments or []:
        if not isinstance(segment, dict):
            continue
        try:
            start = float(segment.get("start", 0.0))
            end = float(segment.get("end", 0.0))
        except (TypeError, ValueError):
            continue
        if end <= start or end <= clip_start or start >= clip_end:
            continue
        text = _escape_ass_text(
            segment.get("text", ""), remove_filler_words=remove_filler_words
        )
        if not text:
            continue
        relative_start = max(0.0, start - clip_start)
        relative_end = min(clip_end, end) - clip_start
        if relative_end <= relative_start:
            continue
        if (caption_style or "").strip().lower() == "karaoke":
            timed_words = []
            for word in segment.get("words") or []:
                try:
                    word_start = max(relative_start, float(word["start"]) - clip_start)
                    word_end = min(relative_end, float(word["end"]) - clip_start)
                    word_text = _escape_ass_text(
                        word.get("word", ""), remove_filler_words=remove_filler_words
                    )
                except (KeyError, TypeError, ValueError):
                    continue
                if word_end > word_start and word_text:
                    timed_words.append((word_start, word_end, word_text))
            if not timed_words:
                plain_words = " ".join(text.split(r"\N")).split()
                if plain_words:
                    word_duration = (relative_end - relative_start) / len(plain_words)
                    timed_words = [
                        (
                            relative_start + index * word_duration,
                            relative_start + (index + 1) * word_duration,
                            word,
                        )
                        for index, word in enumerate(plain_words)
                    ]
            if timed_words:
                plain_words = [word for _, _, word in timed_words]
                for word_start, word_end, word in timed_words:
                    highlighted = " ".join(
                        (r"{\c&H0000FFFF&}" + candidate + r"{\c&H00FFFFFF&}")
                        if candidate == word
                        else candidate
                        for candidate in plain_words
                    )
                    lines.append(
                        "Dialogue: 0,{},{},Default,,0,0,0,,{}".format(
                            _ass_timestamp(word_start),
                            _ass_timestamp(word_end),
                            highlighted,
                        )
                    )
                    count += 1
        else:
            lines.append(
                "Dialogue: 0,{},{},Default,,0,0,0,,{}".format(
                    _ass_timestamp(relative_start),
                    _ass_timestamp(relative_end),
                    text,
                )
            )
            count += 1

    Path(ass_path).write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    return count


def _has_caption_window(
    clip_start: float, clip_end: float, segments: Optional[List[Dict]]
) -> bool:
    """Return whether at least one non-empty transcript segment overlaps a clip."""
    for segment in segments or []:
        if not isinstance(segment, dict):
            continue
        try:
            start = float(segment.get("start", 0.0))
            end = float(segment.get("end", 0.0))
        except (TypeError, ValueError):
            continue
        if end > clip_start and start < clip_end and _escape_ass_text(segment.get("text", "")):
            return True
    return False


def _escape_filter_path(path: str) -> str:
    """Escape a Windows/Unix path for ffmpeg's subtitles filter."""
    # The filter parser wants forward slashes and an escaped drive-letter
    # colon. The surrounding single quotes keep spaces in paths intact.
    return str(Path(path).resolve()).replace("\\", "/").replace(":", r"\:")


def _burn_in_captions(
    video_path: str,
    clip_start: float,
    clip_end: float,
    segments: List[Dict],
    out_path: str,
    caption_style: str = "bold",
    remove_filler_words: bool = False,
    caption_position: str = "bottom",
) -> bool:
    """Render transcript captions onto a reframed video with ffmpeg/libass."""
    ass_path = out_path + ".ass"
    try:
        caption_count = _write_ass_captions(
            ass_path,
            clip_start=clip_start,
            clip_end=clip_end,
            segments=segments,
            caption_style=caption_style,
            remove_filler_words=remove_filler_words,
            caption_position=caption_position,
        )
        if not caption_count:
            shutil.copyfile(video_path, out_path)
            return False

        ffmpeg = _find_ffmpeg()
        subtitle_file = _escape_filter_path(ass_path).replace("'", r"\'")
        cmd = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            video_path,
            "-vf",
            f"subtitles=filename='{subtitle_file}'",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            out_path,
        ]
        subprocess.run(cmd, check=True)
        return True
    finally:
        if os.path.exists(ass_path):
            _remove_with_retry(ass_path)


def _cut_subclip(source_path: str, start: float, end: float, out_path: str) -> str:
    """ffmpeg -ss start -to end -> re-encoded mp4 with audio."""
    ffmpeg = _find_ffmpeg()
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-i", source_path,
        "-ss", f"{start:.3f}",
        "-to", f"{end:.3f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k",
        out_path,
    ]
    subprocess.run(cmd, check=True)
    return out_path


def _reframe_vertical(
    in_path: str,
    out_path: str,
    aspect_ratio: str,
    auto_reframe: bool = True,
    crop_position: float = 0.5,
    fit_mode: str = "crop",
    zoom: float = 1.0,
    intro: Optional[str] = None,
    outro: Optional[str] = None,
) -> str:
    """Crop the cut clip to the target aspect ratio, tracking faces if possible."""
    try:
        import cv2  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "opencv-python is required for --mode local. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e

    target_ratio = _ratio(aspect_ratio)
    if (fit_mode or "crop").strip().lower() == "fit_blur":
        ffmpeg = _find_ffmpeg()
        out_h = 1920
        out_w = max(2, int(round(out_h * target_ratio)) // 2 * 2)
        zoom = max(0.5, min(1.5, float(zoom)))
        filter_complex = (
            f"[0:v]scale={out_w}:{out_h}:force_original_aspect_ratio=increase,crop={out_w}:{out_h},boxblur=20:10[bg];"
            f"[0:v]scale={out_w}:{out_h}:force_original_aspect_ratio=decrease,scale=iw*{zoom:.3f}:ih*{zoom:.3f}[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2[v]"
        )
        subprocess.run([
            ffmpeg, "-y", "-loglevel", "error", "-i", in_path,
            "-filter_complex", filter_complex, "-map", "[v]", "-map", "0:a:0?",
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
            "-shortest", out_path,
        ], check=True)
        return out_path
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {in_path}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Compute the largest crop that fits inside the frame at the target ratio.
    if target_ratio < src_w / src_h:
        crop_h = src_h
        crop_w = int(crop_h * target_ratio)
    else:
        crop_w = src_w
        crop_h = int(crop_w / target_ratio)
    crop_w = max(2, crop_w - (crop_w % 2))
    crop_h = max(2, crop_h - (crop_h % 2))

    face_cascade = None
    if auto_reframe:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    silent_path = out_path + ".silent.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(silent_path, fourcc, fps, (crop_w, crop_h))

    last_center: Optional[Tuple[int, int]] = None
    smoothing = 0.15  # how aggressively to chase a new face position
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = []
        if face_cascade is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
            )
        if len(faces) > 0:
            # Pick the largest face — usually the speaker.
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            cx = x + w // 2
            cy = y + h // 2
            if last_center is None:
                last_center = (cx, cy)
            else:
                lx, ly = last_center
                last_center = (
                    int(lx + (cx - lx) * smoothing),
                    int(ly + (cy - ly) * smoothing),
                )
        if last_center is None and not auto_reframe:
            position = max(0.0, min(1.0, float(crop_position)))
            last_center = (int(crop_w / 2 + position * (src_w - crop_w)), src_h // 2)
        if last_center is None:
            last_center = (src_w // 2, src_h // 2)

        cx, cy = last_center
        x0 = max(0, min(src_w - crop_w, cx - crop_w // 2))
        y0 = max(0, min(src_h - crop_h, cy - crop_h // 2))
        cropped = frame[y0:y0 + crop_h, x0:x0 + crop_w]
        writer.write(cropped)

    cap.release()
    writer.release()

    # Mux audio from the cut clip back onto the silent reframed video.
    ffmpeg = _find_ffmpeg()
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-i", silent_path,
        "-i", in_path,
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "128k",
        "-map", "0:v:0", "-map", "1:a:0?",
        "-shortest",
        out_path,
    ]
    subprocess.run(cmd, check=True)
    _remove_with_retry(silent_path)
    return out_path


def crop_clip_local(
    source_path: str,
    start_time: float,
    end_time: float,
    aspect_ratio: str,
    out_path: str,
    caption_segments: Optional[List[Dict]] = None,
    burn_captions: bool = True,
    caption_style: str = "bold",
    remove_silence: bool = False,
    normalize_audio: bool = False,
    denoise_audio: bool = False,
    remove_filler_words: bool = False,
    caption_position: str = "bottom",
    background_music: Optional[str] = None,
    watermark: Optional[str] = None,
    auto_reframe: bool = True,
    crop_position: float = 0.5,
    fit_mode: str = "crop",
    zoom: float = 1.0,
) -> str:
    """Cut + reframe one highlight, optionally burning Whisper captions."""
    cut_path = out_path + ".cut.mp4"
    base_path = out_path + ".base.mp4"
    try:
        _cut_subclip(source_path, start_time, end_time, cut_path)
        _reframe_vertical(
            cut_path,
            base_path,
            aspect_ratio,
            auto_reframe=auto_reframe,
            crop_position=crop_position,
            fit_mode=fit_mode,
            zoom=zoom,
        )
        if burn_captions and caption_segments:
            _burn_in_captions(
                base_path,
                clip_start=start_time,
                clip_end=end_time,
                segments=caption_segments,
                out_path=out_path,
                caption_style=caption_style,
                remove_filler_words=remove_filler_words,
                caption_position=caption_position,
            )
        else:
            os.replace(base_path, out_path)
    finally:
        if os.path.exists(cut_path):
            _remove_with_retry(cut_path)
        if os.path.exists(base_path):
            _remove_with_retry(base_path)
    audio_filters = []
    if remove_silence:
        audio_filters.append("silenceremove=stop_periods=-1:stop_duration=0.35:stop_threshold=-40dB")
    if normalize_audio:
        audio_filters.append("loudnorm=I=-14:TP=-1.5:LRA=11")
    if denoise_audio:
        audio_filters.append("afftdn=nf=-25")
    if audio_filters:
        processed_path = out_path + ".audio.mp4"
        ffmpeg = _find_ffmpeg()
        subprocess.run(
            [
                ffmpeg, "-y", "-loglevel", "error", "-i", out_path,
                "-af", ",".join(audio_filters), "-c:v", "copy",
                "-c:a", "aac", "-b:a", "128k", processed_path,
            ],
            check=True,
        )
        os.replace(processed_path, out_path)
    if background_music or watermark:
        extra_path = out_path + ".extras.mp4"
        ffmpeg = _find_ffmpeg()
        inputs = [ffmpeg, "-y", "-loglevel", "error", "-i", out_path]
        filters = []
        maps = []
        input_index = 1
        if background_music:
            if not os.path.isfile(background_music):
                raise RuntimeError(f"background music file not found: {background_music}")
            inputs += ["-stream_loop", "-1", "-i", background_music]
            filters.append(f"[0:a][{input_index}:a]amix=inputs=2:duration=first:dropout_transition=2[a]")
            maps.append("[a]")
            input_index += 1
        if watermark:
            if not os.path.isfile(watermark):
                raise RuntimeError(f"watermark image not found: {watermark}")
            inputs += ["-i", watermark]
            filters.append(f"[0:v][{input_index}:v]overlay=W-w-24:H-h-24[v]")
            maps.insert(0, "[v]")
        if background_music and not watermark:
            maps.insert(0, "0:v:0")
        if watermark and not background_music:
            maps.append("0:a:0?")
        cmd = inputs
        if filters:
            cmd += ["-filter_complex", ";".join(filters)]
        if maps:
            cmd += ["-map", maps[0]]
            if len(maps) > 1:
                cmd += ["-map", maps[1]]
        else:
            cmd += ["-map", "0:v:0", "-map", "0:a:0?"]
        cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "20", "-c:a", "aac", "-b:a", "128k", "-shortest", extra_path]
        subprocess.run(cmd, check=True)
        os.replace(extra_path, out_path)
    if intro or outro:
        segments = []
        for label, media in (("intro", intro), ("clip", out_path), ("outro", outro)):
            if not media:
                continue
            if label != "clip" and not os.path.isfile(media):
                raise RuntimeError(f"{label} file not found: {media}")
            segments.append(media)
        if len(segments) > 1:
            branded = out_path + ".branded.mp4"
            ffmpeg = _find_ffmpeg()
            args = [ffmpeg, "-y", "-loglevel", "error"]
            for media in segments:
                args += ["-i", media]
            concat = "".join(f"[{i}:v:0][{i}:a:0]" for i in range(len(segments)))
            args += ["-filter_complex", f"{concat}concat=n={len(segments)}:v=1:a=1[v][a]", "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-c:a", "aac", "-b:a", "128k", branded]
            subprocess.run(args, check=True)
            os.replace(branded, out_path)
    return out_path


def crop_highlights_local(
    source_path: str,
    highlights: List[Dict],
    aspect_ratio: str = "9:16",
    out_dir: Optional[str] = None,
    caption_segments: Optional[List[Dict]] = None,
    burn_captions: bool = True,
    caption_style: str = "bold",
    remove_silence: bool = False,
    normalize_audio: bool = False,
    denoise_audio: bool = False,
    remove_filler_words: bool = False,
    background_music: Optional[str] = None,
    watermark: Optional[str] = None,
    auto_reframe: bool = True,
    crop_position: float = 0.5,
    fit_mode: str = "crop",
    zoom: float = 1.0,
    intro: Optional[str] = None,
    outro: Optional[str] = None,
) -> List[Dict]:
    out_dir = out_dir or LOCAL_OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    results: List[Dict] = []
    for i, h in enumerate(highlights, 1):
        out_path = os.path.join(out_dir, f"short_{i:02d}.mp4")
        print(f"[clip/local] {i}/{len(highlights)}: {h.get('title', '(untitled)')}", flush=True)
        captions_for_clip = burn_captions and _has_caption_window(
            float(h["start_time"]), float(h["end_time"]), caption_segments
        )
        try:
            crop_clip_local(
                source_path,
                float(h["start_time"]),
                float(h["end_time"]),
                aspect_ratio,
                out_path,
                caption_segments=caption_segments,
                burn_captions=burn_captions,
                caption_style=caption_style,
                remove_silence=remove_silence,
                normalize_audio=normalize_audio,
                denoise_audio=denoise_audio,
                remove_filler_words=remove_filler_words,
                background_music=background_music,
                watermark=watermark,
                auto_reframe=auto_reframe,
                crop_position=crop_position,
                fit_mode=fit_mode,
                zoom=zoom,
                intro=intro,
                outro=outro,
            )
            item = {**h, "clip_url": out_path}
            try:
                from .visual import extract_thumbnail

                thumbnail_path = os.path.join(out_dir, f"short_{i:02d}.jpg")
                extract_thumbnail(
                    source_path,
                    (float(h["start_time"]) + float(h["end_time"])) / 2.0,
                    thumbnail_path,
                )
                item["thumbnail_path"] = thumbnail_path
            except Exception as exc:
                print(f"[thumbnail/local] {i} skipped: {exc}", flush=True)
            if captions_for_clip:
                item["captions_burned"] = True
            results.append(item)
        except Exception as e:
            print(f"[clip/local] {i} failed: {e}", flush=True)
            results.append({**h, "clip_url": None, "error": str(e)})
    return results
