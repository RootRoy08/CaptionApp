SETUP (Windows)
1. Install Python 3.10+ and ffmpeg (winget install Gyan.FFmpeg)  -> ffmpeg + ffprobe must be on PATH
2. pip install -r requirements.txt
3. Host templates_sample.json (e.g. GitHub repo -> raw URL), then set TEMPLATE_FEED_URL in app.py
4. python app.py

USING THE PREVIEW
- Open video -> it shows in the player. Generate captions -> press Play.
- Drag the caption on the video to move it (it snaps to horizontal centre).
- Mouse wheel over the caption, or the Text size slider, changes size.
- Click a row in the caption table to jump to that line.
- Position and size are stored relative to the frame, so the export matches the preview at any resolution.

EXPORT
- Resolution: Original / 1080p / 1440p / 4K (2160p). Target applies to the long side, so vertical videos work too (4K = 2160x3840).
- Smaller sources are upscaled (Lanczos); this makes them bigger, not sharper.
- H.264 = most compatible. H.265 = roughly half the file size at 4K.
- 4K on CPU is slow (a few minutes per minute of video). The progress bar shows real ffmpeg progress.

BUILD .EXE: pip install pyinstaller && pyinstaller --onefile --windowed --collect-all faster_whisper --collect-all indic_transliteration app.py
(ship ffmpeg.exe and ffprobe.exe next to the .exe, or keep them on PATH)
