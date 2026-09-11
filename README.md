# AI YouTube Shorts Generator

[![Powered by MuAPI](https://img.shields.io/badge/Powered%20by-MuAPI-6366f1?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0xMiAyQzYuNDggMiAyIDYuNDggMiAxMnM0LjQ4IDEwIDEwIDEwIDEwLTQuNDggMTAtMTBTMTcuNTIgMiAxMiAyem0tMSAxNHYtNGgtMnYtMmg0djZoLTJ6bTAtOFY2aDJ2MmgtMnoiLz48L3N2Zz4=)](https://muapi.ai?utm_source=github&utm_medium=badge&utm_campaign=ai-youtube-shorts-generator)


**The open-source alternative to Opus Clip, Vidyo.ai, Klap, SubMagic, 2short.ai, and other AI clipping tools.** Drop in any long-form YouTube video and get ranked, viral-ready 9:16 shorts locally or through an optional API. Local rendering has no built-in per-clip service limit; API and LLM provider charges follow their own terms.

Built for creators, agencies, and developers who don't want to pay $20–$300/month or be capped on minutes processed. Uses GPT-class LLM highlight detection and Whisper transcription to extract the most viral-worthy moments and auto-crop them vertically for TikTok, Reels, and Shorts.

<p align="center"><a href="https://www.youtube.com/watch?v=kT1CO4BYV3A"><img src="https://i.ytimg.com/vi/kT1CO4BYV3A/maxresdefault.jpg" width="720"></a></p>
<p align="center"><a href="https://www.youtube.com/watch?v=kT1CO4BYV3A"><b>▶ Watch: Free Unlimited AI Image Generator (Truly no limits, Open Source, No Watermark) </b></a></p>

> **Building your own Opus Clip–style SaaS?** Skip the infra and ship on the same APIs that power this repo:
> - [AI Clipping API](https://muapi.ai/playground/ai-clipping?utm_source=github&utm_medium=readme&utm_campaign=ai-youtube-shorts-generator) — end-to-end clip selection + render
> - [Auto-Crop API](https://muapi.ai/playground/autocrop?utm_source=github&utm_medium=readme&utm_campaign=ai-youtube-shorts-generator) — vertical reframing only

![longshorts](https://github.com/user-attachments/assets/3f5d1abf-bf3b-475f-8abf-5e253003453a)

<p align="center">
  <a href="https://github.com/Anil-matcha/awesome-generative-ai-apps">
    <img src="https://img.shields.io/badge/Part%20of-Awesome%20Generative%20AI%20Apps-FFD700?style=for-the-badge&logo=github&logoColor=black" alt="Awesome Generative AI Apps">
  </a>
</p>

> 🎨 **[Explore 50+ more open-source AI apps →](https://github.com/Anil-matcha/awesome-generative-ai-apps)**

## Why Use This Instead of Opus Clip / Vidyo.ai / Klap?

| | This repo | Opus Clip / Vidyo.ai / Klap / SubMagic |
|---|---|---|
| **Price** | Free + open source (pay only for API usage) | $20–$300/month subscriptions |
| **Per-clip credits** | None — process unlimited videos | Monthly minute caps, overage fees |
| **Watermarks** | Never | On free tiers |
| **Highlight algorithm** | Fully editable virality framework | Black box |
| **Output format** | Any aspect ratio, any resolution | Locked presets |
| **Batch processing** | `xargs` an entire URL list | Manual upload one-by-one |
| **JSON / API output** | Built-in (`--output-json`) | Limited or paid tier only |
| **Self-hostable** | Yes — runs on your machine or server | SaaS only, your videos sit on their servers |
| **White-label / embeddable** | Yes — MIT licensed, import as Python lib | No |

## Features

- **🎬 YouTube In, Vertical Out**: Hand it any YouTube URL — get back N viral-ready 9:16 mp4s
- **🔀 Two Modes — API (fast) or Local (offline)**: Default `--mode api` uses MuAPI for download/transcription/cropping; `--mode local` keeps download, transcription, and rendering on your machine with `yt-dlp`, `faster-whisper`, and `ffmpeg`/`opencv`, while the selected OpenAI or Gemini provider handles highlight ranking
- **🤖 Virality-Aware Highlight Selection**: Clips ranked on hooks, emotional peaks, opinion bombs, revelation moments, conflict, quotable lines, story peaks, and practical value — not just generic "interesting"
- **📈 Score + Hook + Reason for Every Clip**: Each highlight comes with a viral score, an opening hook line, and a one-sentence explanation of why it works
- **🎤 Whisper Transcription, Your Choice**: Cloud (`/openai-whisper` via MuAPI) or local (`faster-whisper`, CPU or CUDA) — same downstream output shape
- **🧩 Long-Video Aware**: Videos over 30 minutes are auto-chunked with overlap so nothing gets missed
- **♻️ Smart Dedupe**: Overlapping highlights are collapsed by score so you never get two near-duplicate clips
- **🎯 Smart Vertical Crop**: API mode uses MuAPI's auto-crop; local mode runs OpenCV face tracking with motion smoothing
- **📱 Any Aspect Ratio**: 9:16 for TikTok/Reels/Shorts, 1:1 for square, anything else by flag
- **🧰 CLI + Python Library**: Use it from the shell or import `generate_shorts(...)` into your own pipeline
- **📦 JSON Output**: `--output-json` dumps the full result (transcript + every candidate highlight + final clip URLs/paths) for downstream automation

### Shorts Studio web features

The local web application adds the complete creator workflow:

- Drag-and-drop video uploads, local paths, batch URLs, persistent jobs, cancellation, and recent-job history.
- Caption presets (Bold, Clean, Boxed, Karaoke) plus custom font, size, color, safe position, word timing, and filler-word cleanup.
- Auto face framing toggle, manual crop position, crop-to-fill, zoom-out with blurred background, foreground zoom, and two-panel split layout.
- Preview rendering, transcript timeline, sentence-boundary snapping, editable timestamps, per-clip regeneration, one-step Undo, and ZIP export.
- Silence trimming, real silent-video jump cuts, loudness normalization, noise reduction, background music, watermark, intro/outro, and automatic thumbnails.
- Project presets for podcast/interview, educational, reaction/gaming, and story videos; generated titles, descriptions, hashtags, and platform publishing metadata.
- Optional Save folder creates named job folders such as `20260910_214500_shorts_source_a1b2c3d4` containing source cache, transcript, clips, thumbnails, `metadata.json`, and any preview media.
- Whisper model/device controls (`tiny` through `large-v3`; Auto, CPU, or CUDA) with live CUDA status and safe CPU fallback.

### Web UI workspace

The reworked UI is organized as a small editing workspace instead of one long form:

- **Dashboard** shows project counts, completion status, GPU availability, recent projects, and one-click resume.
- **Workspace** keeps source setup, a large 9:16 preview, the inspector, transcript, timeline, rendered clips, and the render queue together.
- **Timeline and transcript** expose sentence-aligned markers and clip boundaries so a clip can be selected, adjusted, previewed, regenerated, or undone without rerunning the whole source.
- **Crop editor** provides per-clip Crop or Fit + blur framing, horizontal position, zoom, and optional face tracking. Changes are applied to the selected clip through the inspector or its clip card.
- **Caption designer** provides preset styles, font, size, color, safe-area position, and live preview overlay controls.
- **Render queue** keeps active and completed jobs visible, supports batch sources, cancellation, and persistent job history after a browser refresh.
- **Export center** exposes metadata, language/model/device settings, ZIP export, and the saved job folder from one place.
- **Responsive and accessible layout** adapts to narrow screens, keeps keyboard focus visible, labels controls for assistive technology, and honors reduced-motion preferences.

## Quick Start (No Setup)

Don't want to self-host? The [AI Clipping API](https://muapi.ai/playground/ai-clipping?utm_source=github&utm_medium=readme&utm_campaign=ai-youtube-shorts-generator) gives you the same Opus Clip–style pipeline as a single HTTP call — no Python, no dependencies, pay-per-clip instead of monthly subscriptions.

---

## Installation (Self-Hosted)

### Prerequisites

- Python 3.10+
- For **API mode (default)**: a MuAPI key — powers download, transcription, highlight ranking, and clipping in a single dependency
- For **Local mode** (`--mode local`): `ffmpeg` on your PATH; OpenAI/Gemini keys are optional (without one, deterministic offline transcript ranking is used). Local files stay on this machine, while remote URLs need yt-dlp network access.

### Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/wiifhub/AI-Youtube-Shorts-Generator.git
   cd AI-Youtube-Shorts-Generator
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # Windows PowerShell: .\\venv\\Scripts\\Activate.ps1
   # macOS/Linux:      source venv/bin/activate
   ```

3. **Install Python dependencies:**
   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements-local.txt
   ```

4. **Set up environment variables:**

   Create a `.env` file in the project root:
   ```bash
   # API mode (default)
   MUAPI_API_KEY=your_muapi_key_here

   # Local mode (--mode local)
   LLM_PROVIDER=openai         # openai or gemini
   OPENAI_API_KEY=your_openai_key_here
   OPENAI_MODEL=gpt-4o-mini          # optional, default gpt-4o-mini
   GEMINI_API_KEY=your_gemini_key_here
   GEMINI_MODEL=gemini-2.5-flash      # optional, default gemini-2.5-flash
   LOCAL_WHISPER_MODEL=base          # tiny / base / small / medium / large-v3
   LOCAL_WHISPER_DEVICE=auto         # auto / cpu / cuda
   LOCAL_OUTPUT_DIR=output           # where local mp4s land
   LOCAL_HEURISTIC_FALLBACK=true     # rank offline when no LLM key is configured
   ```

## Usage

### Web UI

On Windows, from the downloaded/cloned project folder, double-click `install_windows.bat` once. It creates the virtual environment, installs all dependencies, and creates `.env` from `.env.example`. Then double-click `start_studio.bat` (or run `python -m web.app` inside the activated environment). The app uses the current local source tree when started this way.

### Portable Windows executable

Download the current [ShortsStudio-v0.6.3-windows.zip](https://github.com/wiifhub/AI-Youtube-Shorts-Generator/releases/download/v0.6.3/ShortsStudio-v0.6.3-windows.zip) and extract the entire ZIP. Open the `ShortsStudio` folder and double-click `unblock_and_start.bat`; it removes the download block from the extracted files and starts `ShortsStudio.exe`. You can also launch the executable directly after choosing **More info > Run anyway** once. Keep the whole folder together because it includes the runtime, CTranslate2, CUDA 12 runtime libraries, and FFmpeg binaries. Add a `.env` file beside the executable with OpenAI or Gemini credentials for AI ranking; otherwise Local mode automatically uses offline transcript ranking. Use **Quit Shorts Studio** in the page to stop the server cleanly.

### Installed Windows application

For a normal Windows installation, download [ShortsStudio-Setup-v0.6.3.exe](https://github.com/wiifhub/AI-Youtube-Shorts-Generator/releases/download/v0.6.3/ShortsStudio-Setup-v0.6.3.exe). The installer adds a Start Menu entry, offers a Desktop shortcut, installs the bundled runtime and FFmpeg, and registers an uninstaller. The app still runs locally at `http://127.0.0.1:7860`; no cloud account is required for the UI.

For a source checkout, install dependencies, then:

```bash
pip install -r requirements.txt
python -m web.app
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860). Paste a YouTube URL, choose local or API mode, or drag in a video file. The UI supports caption presets (Bold, Clean, Boxed, Karaoke), highlight focus modes, silence/audio cleanup, batch sources, manual per-clip timestamp edits, and creator metadata. Jobs and results are saved under `output/jobs/`, so refreshing or reopening the UI keeps the recent-job history and playable local clips. Completed jobs can be downloaded as a ZIP containing local clips and `metadata.json`.

### GPU acceleration (Windows)

> **Current defaults and costs:** API mode and the OpenAI/Gemini ranking step
> use the credentials and billing terms of those providers. Local mode keeps
> downloading, transcription, and rendering on this machine, while its chosen
> LLM provider handles highlight ranking. MuAPI polling defaults to a 5-second
> interval and a 600-second timeout.

The web UI exposes Whisper model and device controls. `Auto` detects a usable CUDA device through CTranslate2 and otherwise falls back to CPU; `CPU` is the compatible fallback; `CUDA GPU` requires a current NVIDIA driver and a CUDA-capable CTranslate2 build. The portable release bundles the CTranslate2 runtime and reports the detected CUDA device in the status line. For source installs, run `install_gpu_windows.bat` if you also want the CUDA-enabled PyTorch diagnostics package; then restart Shorts Studio and choose **CUDA GPU**. If no LLM credential is configured, Local mode remains usable through its deterministic offline transcript ranker. The `requirements-gpu.txt` file documents that optional dependency.

### Professional Windows installer

The Releases page includes both a portable ZIP and `ShortsStudio-Setup-vX.Y.Z.exe`. The setup program installs the bundled runtime and FFmpeg, creates Start Menu and optional Desktop shortcuts, and registers a normal Windows uninstaller. To build it yourself, run `build_portable.bat`, install Inno Setup 6, then run `build_installer.bat` (or `ISCC.exe installer\\ShortsStudio.iss`). `build_portable.bat` also copies the CUDA 12 runtime DLLs when Ollama or the CUDA toolkit is installed, preventing the `cublas64_12.dll` startup error.

### Windows SmartScreen

The portable executable and installer are not digitally signed in this repository, so a fresh download can show **Windows protected your PC** with **Unknown publisher**. This is a Windows reputation warning, not an application error. For the portable ZIP, extract the complete folder and double-click `unblock_and_start.bat`; it removes the download block from the extracted files and starts Shorts Studio. If Windows still prompts, choose **More info > Run anyway** once. The same one-time prompt may appear for the installer. Removing the warning permanently requires an Authenticode certificate issued to `wiifhub`; a self-signed certificate would not be trusted by Windows SmartScreen.

### Single video (API mode — default)

```bash
python main.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

### Single video (Local mode)

```bash
python main.py "https://www.youtube.com/watch?v=VIDEO_ID" --mode local
```

Local mode writes the rendered shorts to `./output/short_01.mp4`, `short_02.mp4`, … (override with `LOCAL_OUTPUT_DIR`). The web UI uses an isolated `output/jobs/<job-id>/` directory per run, or a timestamped subfolder beneath the optional Save folder. Local files stay on this machine; remote URLs require yt-dlp network access, and highlight ranking uses the selected LLM. When transcript segments are available, local clips include burned-in captions generated through ffmpeg/libass; set `LOCAL_BURN_CAPTIONS=false` to disable them.

### With options

```bash
python main.py "https://www.youtube.com/watch?v=VIDEO_ID" \
    --mode api \
    --num-clips 5 \
    --aspect-ratio 9:16 \
    --output-json result.json
```

### Local file or path

In `--mode local`, you can pass a `file://` URL or a direct filesystem path and skip YouTube entirely:

```bash
python main.py "/Users/you/Videos/input.mp4" --mode local
python main.py "file:///Users/you/Videos/input.mp4" --mode local
```

The Python API works the same way:

```python
from shorts_generator import generate_shorts

result = generate_shorts(
    "/Users/you/Videos/input.mp4",
    num_clips=5,
    aspect_ratio="9:16",
    mode="local",
)
for short in result["shorts"]:
    print(short["score"], short["title"], short["clip_url"])
```

Local transcription is cached as an `.srt` file in `LOCAL_OUTPUT_DIR` using the
video's base name. If the cache already exists and is newer than the source
file, the app reuses it instead of running Whisper again.

Local downloads are also cached in `LOCAL_OUTPUT_DIR` as
`source_<youtube_id>.mp4` when the input is a YouTube URL. If that file already
exists, the app skips `yt-dlp` and reuses the cached video.

### Batch processing

Create a `urls.txt` file with one URL per line, then:

```bash
xargs -a urls.txt -I{} python main.py "{}"
```

### CLI flags

| Flag | Default | Notes |
|------|---------|-------|
| `--mode` | `api` | `api` (MuAPI, fast, no setup) or `local` (remote URL, `file://`, or local path + faster-whisper + LLM provider + ffmpeg) |
| `--num-clips` | `3` | How many shorts to render |
| `--aspect-ratio` | `9:16` | Any ratio; `9:16` for TikTok/Reels, `1:1` for square |
| `--format` | `720` | Source download resolution: `360` / `480` / `720` / `1080` |
| `--language` | auto | Force Whisper language code (e.g. `en`) |
| `--output-json` | — | Dump the full result (transcript + all candidates) to a file |

### API mode vs Local mode

| Step | API mode (`--mode api`) | Local mode (`--mode local`) |
|---|---|---|
| Download | MuAPI `/youtube-download` | `yt-dlp` for remote URLs, direct file path for local inputs |
| Transcription | MuAPI `/openai-whisper` | `faster-whisper` (CPU or CUDA) |
| Highlight LLM | MuAPI `gpt-5-mini` | `LLM_PROVIDER=openai` uses OpenAI (`gpt-4o-mini` by default), `LLM_PROVIDER=gemini` uses Gemini (`gemini-2.5-flash` by default) |
| Vertical crop | MuAPI `/autocrop` | `ffmpeg` + OpenCV face tracking |
| Output | hosted URLs | local mp4 paths |
| Required keys | `MUAPI_API_KEY` | `OPENAI_API_KEY` or `GEMINI_API_KEY` (+ `ffmpeg` on PATH) |

## Shorts Studio HTTP API

When the web app is running, these local routes are available:

| Route | Purpose |
|---|---|
| `GET /api/health` | Confirm the server and output directory |
| `GET /api/system` | FFmpeg, Whisper, CUDA, disk, and concurrency status |
| `GET /api/diagnostics` | Runtime paths, job counts, and diagnostics |
| `POST /api/jobs` | Queue one URL or local video |
| `POST /api/jobs/batch` | Queue multiple sources |
| `GET /api/jobs` | List persisted recent jobs |
| `POST /api/jobs/{id}/preview` | Render a review preview |
| `POST /api/jobs/{id}/clips/{index}` | Regenerate one edited clip |
| `POST /api/jobs/{id}/clips/{index}/undo` | Restore the previous clip version |
| `GET /api/jobs/{id}/export` | Download clips, thumbnails, and publishing metadata as ZIP |
| `POST /api/shutdown` | Stop a local/portable server cleanly |

## How It Works

1. **Download**: Fetches the source video from YouTube
2. **Transcribe**: MuAPI `/openai-whisper` produces a timestamped transcript (verbose_json segments)
3. **Detect content type**: An LLM classifies the video (podcast, interview, tutorial, vlog, etc.) and density, so the prompt can be tuned per content style
4. **Long-video chunking**: Videos > 30 min are split into 20-min overlapping chunks
5. **Highlight ranking**: An LLM scans the transcript through a virality framework — hook moments, emotional peaks, opinion bombs, revelations, conflict, quotables, story peaks, practical value — and emits ranked candidates with scores 0–100
6. **Dedupe**: Overlapping candidates are collapsed by score (>50% overlap → keep the higher score)
7. **Top-N selection**: The top `--num-clips` candidates are selected
8. **Auto-crop**: Each highlight is rendered as a vertical short at the requested aspect ratio

**Output**: a list of mp4 URLs plus, for each clip, its title, viral score, hook sentence, and a one-line reason explaining why it should perform.

## Output

Console output looks like:

```
========================================================================
Highlights:    7 candidates → kept top 3
========================================================================

#1  score=92  124.3s → 187.6s
     title:  The one mistake that cost me $50K
     hook:   "Nobody talks about this, but it killed my first startup..."
     clip:   https://.../short_1.mp4

#2  score=88  ...
```

`--output-json result.json` produces:

```json
{
  "source_video_url": "...",
  "transcript": { "duration": 1873.4, "segments": [...] },
  "highlights": [ {...}, {...}, ... ],
  "shorts": [
    {
      "title": "...",
      "start_time": 124.3,
      "end_time": 187.6,
      "score": 92,
      "hook_sentence": "...",
      "virality_reason": "...",
      "clip_url": "https://.../short_1.mp4"
    }
  ]
}
```

## Configuration

### Highlight selection criteria
Edit `shorts_generator/highlights.py`:
- **Virality framework**: `VIRALITY_CRITERIA` — the ranked list of signals the LLM optimizes for
- **System prompt**: `HIGHLIGHT_SYSTEM_PROMPT` — duration sweet spot, hook rules, JSON schema
- **Chunk size**: `CHUNK_SIZE_SECONDS` (default 1200) — chunk length for long videos
- **Long-video threshold**: `LONG_VIDEO_THRESHOLD` (default 1800) — videos longer than this are chunked
- **Chunk overlap**: `CHUNK_OVERLAP_SECONDS` (default 60) — overlap between chunks so cross-boundary clips aren't missed

### Polling / timeout (current default timeout: 600 seconds)
Edit `shorts_generator/config.py` (or set env vars):
- `MUAPI_POLL_INTERVAL` (default 5s) — seconds between job-status polls
- `MUAPI_POLL_TIMEOUT` (default 600s) — give up after this long

### Whisper transcription
Audio is transcribed by MuAPI's `/openai-whisper` endpoint (server-side `whisper-1`). Pass `--language <code>` to lock the recognition to a specific language; otherwise it auto-detects.

For Local mode, the web UI lets you choose `tiny`, `base`, `small`, `medium`, or `large-v3`, plus `Auto`, `CPU`, or `CUDA GPU`. Auto checks CTranslate2's CUDA runtime and falls back to CPU. The status line shows the detected device; the released Windows executable was verified to detect one CUDA device on the build machine.

## Project Structure

```
AI-Youtube-Shorts-Generator/
├── main.py                       CLI entry point
├── requirements.txt              core deps (api mode)
├── requirements-local.txt        optional deps for --mode local
├── .env.example
└── shorts_generator/
    ├── config.py                 env / settings (MuAPI + local LLM + Whisper)
    ├── muapi.py                  generic submit + poll wrapper
    ├── downloader.py             API mode: YouTube download via MuAPI
    ├── transcriber.py            API mode: MuAPI /openai-whisper client
    ├── highlights.py             shared LLM virality ranking (pluggable backend)
    ├── clipper.py                API mode: MuAPI /autocrop
    ├── pipeline.py               mode dispatcher (api ↔ local)
    └── local/                    --mode local backends (offline)
        ├── downloader.py         yt-dlp download
        ├── transcriber.py        faster-whisper transcription
        ├── llm.py                OpenAI or Gemini client selector
        └── clipper.py            ffmpeg cut + OpenCV vertical crop
```

### Windows packaging files

- `launcher.py` is the PyInstaller entry point for the portable executable.
- `web/app.py` and `web/static/index.html` provide the Shorts Studio dashboard and editor workspace.
- `install_windows.bat` creates a source-install virtual environment.
- `install_gpu_windows.bat` adds optional CUDA-enabled PyTorch diagnostics for source installs.
- `installer/ShortsStudio.iss` and `build_installer.bat` build the Inno Setup installer.
- `requirements-gpu.txt` documents the optional GPU dependency; the released EXE already bundles CUDA-capable CTranslate2.

## Troubleshooting

### YouTube returns HTTP 403

Update the source checkout and restart Shorts Studio so yt-dlp uses the current downloader client. Age-restricted or private videos may still reject downloads; use **Choose video** or drag the local MP4 into the UI instead.

### Whisper produced no segments
The video may have no detectable speech, or it may be in a language Whisper struggles with. Try passing `--language en` (or the correct ISO-639-1 code) to skip auto-detection.

### Looking for better results?
The [AI Clipping API](https://muapi.ai/playground/ai-clipping?utm_source=github&utm_medium=readme&utm_campaign=ai-youtube-shorts-generator) uses an improved algorithm that produces higher-quality clips with better highlight detection.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

## Shorts Studio upgrades

Local mode can add visual scene/speaker signals to highlight ranking, generate a thumbnail per clip, burn ASS captions (including karaoke word timing), trim silence, normalize loudness, remove filler words from captions, mix optional background music, and overlay a watermark. The web UI persists jobs, supports drag-and-drop uploads, batch queues, cancellation, timeline review, per-clip regeneration, and ZIP export. Background music and watermark fields accept paths visible to the running machine and apply to local renders.

Framing can be automatic (face tracking) or manual. Turn off **Auto frame faces** in the web UI and use the **Manual crop position** slider to choose the left, center, or right composition.

Project presets provide starting points for podcast/interview, educational, reaction/gaming, and story videos; every preset remains editable before submission.

Use the web UI's optional **Save folder** field to choose a destination. Each job creates a timestamped subfolder containing the source cache, transcript, rendered shorts, thumbnails, and metadata.

Local clip edits can be previewed, regenerated, and reverted with the per-clip **Undo** action. API-mode clips can be regenerated through MuAPI, but local draft previews and Undo are not available for hosted-only media.

Local renders also accept optional intro and outro MP4 paths, and can apply FFmpeg noise reduction alongside loudness normalization.

The generated ZIP includes `metadata.json` plus platform-ready title, description, hashtag, and thumbnail-hook metadata for YouTube Shorts, TikTok, and Instagram Reels. Local preview and regeneration use the same framing, captions, audio, and layout settings as the final render.

To close the standalone app, click **Quit Shorts Studio** in the web page. It stops the local server and leaves a clear closed confirmation page; closing the browser tab alone does not stop a manually launched server.

This project is licensed under the MIT License.

## Related Projects

- [AI Influencer Generator](https://github.com/SamurAIGPT/AI-Influencer-Generator)
- [Text to Video AI](https://github.com/SamurAIGPT/Text-To-Video-AI)
- [Faceless Video Generator](https://github.com/SamurAIGPT/Faceless-Video-Generator)
- [AI B-roll Generator](https://github.com/Anil-matcha/AI-B-roll)
- [No-code YouTube Shorts Generator](https://www.vadoo.tv/clip-youtube-video)
- [ai-creator-academy](https://github.com/Anil-matcha/ai-creator-academy) — free curriculum teaching creators how to monetize AI-generated shorts and video content
