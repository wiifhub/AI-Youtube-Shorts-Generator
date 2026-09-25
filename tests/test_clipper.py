"""Pure clipping helpers and subtitle timing tests."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest

import shorts_generator.local.clipper as clipper

from shorts_generator.local.clipper import (
    _normalise_cut_ranges,
    _remap_caption_segments,
    _shift_caption_segments,
    _source_ranges_after_timeline_edit,
    _split_canvas_dimensions,
    _write_ass_captions,
)


def test_cut_ranges_are_clamped_sorted_and_merged() -> None:
    ranges = _normalise_cut_ranges(
        10,
        30,
        [
            {"start_time": 25, "end_time": 40},
            {"start_time": 10, "end_time": 15},
            {"start_time": 14.99, "end_time": 20},
        ],
    )
    assert ranges == [(10.0, 20.0), (25.0, 30.0)]


def test_caption_segments_follow_concatenated_ranges() -> None:
    mapped = _remap_caption_segments(
        [{"start": 10, "end": 14, "text": "First"}, {"start": 25, "end": 29, "text": "Second"}],
        [(10, 14), (25, 30)],
    )
    assert mapped[0]["start"] == 0
    assert mapped[0]["end"] == 4
    assert mapped[1]["start"] == 4
    assert mapped[1]["end"] == 8


def test_jump_cut_ranges_map_back_to_source_time_for_sidecar_captions() -> None:
    translated = _source_ranges_after_timeline_edit(
        [(10, 14), (25, 30)],
        [(0, 2), (5, 8)],
    )
    assert translated == [(10, 12), (26, 29)]


def test_prepended_branding_shifts_caption_and_word_timestamps() -> None:
    shifted = _shift_caption_segments(
        [{"start": 0.5, "end": 1.5, "text": "Hello", "words": [{"start": 0.5, "end": 1.0, "word": "Hello"}]}],
        2.0,
    )
    assert shifted[0]["start"] == 2.5
    assert shifted[0]["words"][0]["end"] == 3.0


def test_ass_writer_escapes_text_and_counts_visible_segments(tmp_path: Path) -> None:
    ass = tmp_path / "captions.ass"
    count = _write_ass_captions(
        str(ass),
        0,
        5,
        [{"start": 1, "end": 3, "text": "Hello {world}"}],
        caption_style="boxed",
    )
    content = ass.read_text(encoding="utf-8-sig")
    assert count == 1
    assert "Dialogue:" in content
    assert r"\{world\}" in content


def test_split_canvas_keeps_even_yuv420_dimensions() -> None:
    height, width, panel = _split_canvas_dimensions(240, 9 / 16)

    assert (height, width, panel) == (240, 136, 68)
    assert width % 2 == 0 and panel % 2 == 0


def test_transition_and_ducking_filters_are_wired(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    output = tmp_path / "cut.mp4"
    commands = []

    monkeypatch.setattr(clipper, "_find_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(clipper, "_has_audio_stream", lambda _path: True)

    def fake_run(args, **kwargs):
        commands.append(args)
        Path(args[-1]).write_bytes(b"rendered")

    monkeypatch.setattr(clipper, "_run_command", fake_run)
    clipper._cut_ranges(str(source), [(0, 3), (4, 7)], str(output), transition="fade", transition_duration=0.5)
    assert any("xfade=transition=fade" in str(item) for item in commands[0])

    rendered = tmp_path / "rendered.mp4"
    rendered.write_bytes(b"video")
    music = tmp_path / "music.mp3"
    music.write_bytes(b"music")
    monkeypatch.setattr(clipper, "_media_duration", lambda _path: 8.0)
    clipper._apply_media_extras(
        str(rendered), str(music), None, music_volume=0.2, music_ducking=True, ducking_strength=0.8
    )
    assert rendered.read_bytes() == b"rendered"
    assert any("sidechaincompress" in str(item) for item in commands[-1])


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is required for media regression")
def test_multicut_media_duration_is_compacted_before_render(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    source = tmp_path / "source.mp4"
    output = tmp_path / "clip.mp4"
    subprocess.run(
        [
            shutil.which("ffmpeg") or "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=320x180:rate=12:duration=4",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=4",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(source),
        ],
        check=True,
    )
    from shorts_generator.local.clipper import crop_clip_local

    crop_clip_local(
        str(source),
        0,
        4,
        "9:16",
        str(output),
        cuts=[{"start_time": 0, "end_time": 1}, {"start_time": 3, "end_time": 4}],
        burn_captions=False,
        auto_reframe=False,
        output_height=240,
    )
    probe = subprocess.run(
        [shutil.which("ffprobe") or "ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(output)],
        capture_output=True,
        text=True,
        check=True,
    )
    duration = float(probe.stdout.strip())
    assert 1.5 <= duration <= 2.5


def _ffmpeg_unescape_once(text: str) -> str:
    """Mirror FFmpeg's filter quoting/escaping for a single parser level."""
    out: list[str] = []
    index = 0
    while index < len(text):
        character = text[index]
        if character == "\\" and index + 1 < len(text):
            out.append(text[index + 1])
            index += 2
        elif character == "'":
            index += 1
        else:
            out.append(character)
            index += 1
    return "".join(out)


@pytest.mark.parametrize(
    "dirname",
    ["plain", "space dir", "Bob's Videos", "a,b;c[d]e'f", "out'[1]hflip[v];[v]'", "x',negate'"],
)
def test_caption_filter_filename_survives_both_ffmpeg_parsing_levels(
    monkeypatch, tmp_path: Path, dirname: str
) -> None:
    """A hostile render path must reach FFmpeg intact, not split into filters.

    FFmpeg unescapes a filter argument twice, so a single level of escaping lets
    an apostrophe in the save folder truncate the subtitle path, drop the file,
    or leak the rest of the path into the filtergraph as filter syntax.
    """
    commands: list[list[str]] = []

    def capture(args, **_kwargs):
        commands.append([str(item) for item in args])
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(clipper, "_run_command", capture)
    output_dir = tmp_path / dirname
    output_dir.mkdir(parents=True)
    source = output_dir / "source.mp4"
    source.write_bytes(b"source")
    out_path = output_dir / "short_01.mp4"

    assert clipper._burn_in_captions(
        str(source),
        0.0,
        1.0,
        [{"start": 0.0, "end": 1.0, "text": "hello world"}],
        str(out_path),
    )

    filters = commands[0]
    value = filters[filters.index("-vf") + 1]
    assert value.startswith("subtitles=filename=")
    escaped = value[len("subtitles=filename=") :]
    expected = str(Path(str(out_path) + ".ass").resolve()).replace("\\", "/")
    # Two unescapes model FFmpeg's filtergraph + filter-option parsers.
    assert _ffmpeg_unescape_once(_ffmpeg_unescape_once(escaped)) == expected


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is required for media regression")
def test_captions_render_through_a_folder_name_with_an_apostrophe(tmp_path: Path) -> None:
    folder = tmp_path / "Bob's Videos"
    folder.mkdir()
    source = folder / "source.mp4"
    output = folder / "short_01.mp4"
    subprocess.run(
        [
            shutil.which("ffmpeg") or "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=320x568:rate=24",
            "-t",
            "1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
        capture_output=True,
    )

    assert clipper._burn_in_captions(
        str(source),
        0.0,
        0.8,
        [{"start": 0.0, "end": 0.8, "text": "hello world"}],
        str(output),
    )
    assert output.is_file() and output.stat().st_size > 0
