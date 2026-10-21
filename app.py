"""Hinglish Captioner - Windows desktop app.
Video -> word-synced Hinglish captions (faster-whisper) -> styled with online templates -> SRT / ASS / burned-in MP4.
"""
import os, re, sys, json, time, bisect, shutil, subprocess, tempfile, urllib.request
from collections import deque
from datetime import datetime
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QUrl, QRectF, QPointF, QSizeF, QSize
from PySide6.QtGui import QFont, QFontMetricsF, QPainter, QPainterPath, QPen, QColor, QPixmap, QIcon, QLinearGradient
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QGraphicsVideoItem
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog,
    QListWidget, QListWidgetItem, QComboBox, QProgressBar, QMessageBox, QGraphicsView, QGraphicsScene, QGraphicsItem,
    QSlider, QFrame, QLineEdit, QTabWidget, QButtonGroup, QCheckBox, QColorDialog, QInputDialog, QFontComboBox,
    QDialog, QAbstractItemView)

# When packaged with PyInstaller, make the bundled ffmpeg/ffprobe visible
if getattr(sys, "frozen", False):
    os.environ["PATH"] = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable)) + os.pathsep + os.environ.get("PATH", "")

# ---- CONFIG: point this at a JSON feed of caption templates (see templates_sample.json) ----
TEMPLATE_FEED_URL = os.environ.get("TEMPLATE_FEED_URL", "https://raw.githubusercontent.com/RootRoy08/CaptionApp/main/templates.json")
CHECK_EVERY_MIN = 30
MAX_TEMPLATES = 10
DATA = os.path.join(os.environ.get("APPDATA", os.path.expanduser("~")), "HinglishCaptioner")
CACHE = os.path.join(DATA, "templates.json")
os.makedirs(DATA, exist_ok=True)

_f = os.path.join(DATA, "feed_url.txt")      # optional: put a different feed URL in this file
if os.path.isfile(_f): TEMPLATE_FEED_URL = open(_f, encoding="utf-8").read().strip()

# ---- speech models: stored in %APPDATA%, downloaded with a live percentage
try:   # use huggingface's standard cache, so a model downloaded by an earlier build is reused, not downloaded again
    from huggingface_hub.constants import HF_HUB_CACHE as MODELS
except Exception:
    MODELS = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
ALLOW = ["config.json", "preprocessor_config.json", "model.bin", "tokenizer.json", "vocabulary.*"]
MODEL_GB = {"large-v3": 3.1, "medium": 1.5, "small": 0.5}

def model_repo(size):
    try:
        from faster_whisper.utils import _MODELS
        return _MODELS[size]
    except Exception:
        return f"Systran/faster-whisper-{size}"

def model_ready(size):
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(model_repo(size), cache_dir=MODELS, allow_patterns=ALLOW, local_files_only=True)
        return True
    except Exception:
        return False

def _tree_size(path):
    n = 0
    for root, _, files in os.walk(path):
        for f in files:
            try: n += os.path.getsize(os.path.join(root, f))
            except OSError: pass
    return n

def download_model(size, progress):
    import fnmatch, threading
    from huggingface_hub import snapshot_download, HfApi
    repo = model_repo(size)
    try:
        info = HfApi().model_info(repo, files_metadata=True)
        total = sum((x.size or 0) for x in info.siblings if any(fnmatch.fnmatch(x.rfilename, p) for p in ALLOW))
    except Exception:
        total = 0
    total = total or int(MODEL_GB.get(size, 3) * 1e9)
    folder = os.path.join(MODELS, "models--" + repo.replace("/", "--")); stop = threading.Event()
    def poll():   # measure bytes on disk, so the percentage is real whatever the downloader does internally
        while not stop.wait(0.4): progress(min(99, int(100 * _tree_size(folder) / total)))
    threading.Thread(target=poll, daemon=True).start()
    try: snapshot_download(repo, cache_dir=MODELS, allow_patterns=ALLOW)
    finally: stop.set()
    progress(100)

# ---- optional NVIDIA GPU support: cuBLAS + cuDNN DLLs are downloaded once from PyPI into %APPDATA%
GPU_DIR = os.path.join(DATA, "gpu")
def add_gpu_path():
    if os.path.isdir(GPU_DIR):
        os.environ["PATH"] = GPU_DIR + os.pathsep + os.environ.get("PATH", "")
        try: os.add_dll_directory(GPU_DIR)
        except Exception: pass
add_gpu_path()

def gpu_ready():
    return os.path.isfile(os.path.join(GPU_DIR, "cublas64_12.dll")) and any(
        f.lower().startswith("cudnn") for f in os.listdir(GPU_DIR))

def has_nvidia():
    if sys.platform != "win32": return False
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0: return True
    except Exception:
        pass
    return shutil.which("nvidia-smi") is not None

def pending_steps(model="large-v3"):
    """What the start-up screen still has to download."""
    steps = []
    if has_nvidia() and not gpu_ready():
        steps.append(("NVIDIA GPU libraries  (about 1.3 GB)", install_gpu_libs))
    if not model_ready(model):
        steps.append((f"Speech model {model}  (about {MODEL_GB.get(model, 3):.1f} GB)", lambda p: download_model(model, p)))
    return steps

def install_gpu_libs(progress):
    import zipfile
    os.makedirs(GPU_DIR, exist_ok=True); pkgs = ["nvidia-cublas-cu12", "nvidia-cudnn-cu12"]
    for n, pkg in enumerate(pkgs):
        meta = json.loads(urllib.request.urlopen(f"https://pypi.org/pypi/{pkg}/json", timeout=30).read())
        url = next(f["url"] for f in meta["urls"] if f["filename"].endswith("win_amd64.whl"))
        tmp = os.path.join(DATA, pkg + ".whl")
        with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as out:
            total, got = int(r.headers.get("Content-Length", 0)), 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk: break
                out.write(chunk); got += len(chunk)
                progress(min(99, int((n + got / max(total, 1)) * 100 / len(pkgs))))
        with zipfile.ZipFile(tmp) as z:
            for name in z.namelist():
                if name.lower().endswith(".dll"):
                    with z.open(name) as src, open(os.path.join(GPU_DIR, os.path.basename(name)), "wb") as dst:
                        shutil.copyfileobj(src, dst)
        os.remove(tmp)
    add_gpu_path(); return True

BUNDLED = [{'id': 'classic',
  'name': 'Classic White',
  'published': '2024-01-12',
  'font': 'Arial',
  'size': 56,
  'text': '#FFFFFF',
  'highlight': '#FFFFFF',
  'outline': '#000000',
  'outline_w': 3,
  'bold': True,
  'upper': False,
  'position': 'bottom'},
 {'id': 'yellow-pop',
  'name': 'Yellow Pop',
  'published': '2024-01-11',
  'font': 'Arial Black',
  'size': 60,
  'text': '#FFFFFF',
  'highlight': '#FFD400',
  'outline': '#000000',
  'outline_w': 4,
  'bold': True,
  'upper': True,
  'position': 'middle'},
 {'id': 'neon-green',
  'name': 'Neon Green',
  'published': '2024-01-10',
  'font': 'Impact',
  'size': 62,
  'text': '#FFFFFF',
  'highlight': '#39FF14',
  'outline': '#000000',
  'outline_w': 4,
  'bold': False,
  'upper': True,
  'position': 'middle'},
 {'id': 'red-hot',
  'name': 'Red Hot',
  'published': '2024-01-09',
  'font': 'Impact',
  'size': 62,
  'text': '#FFFFFF',
  'highlight': '#FF3B30',
  'outline': '#000000',
  'outline_w': 5,
  'bold': False,
  'upper': True,
  'position': 'middle'},
 {'id': 'cyan-glow',
  'name': 'Cyan Glow',
  'published': '2024-01-08',
  'font': 'Segoe UI Black',
  'size': 58,
  'text': '#FFFFFF',
  'highlight': '#00E5FF',
  'outline': '#003B46',
  'outline_w': 4,
  'bold': True,
  'upper': True,
  'position': 'bottom'},
 {'id': 'pink-pop',
  'name': 'Pink Pop',
  'published': '2024-01-07',
  'font': 'Arial Black',
  'size': 58,
  'text': '#FFFFFF',
  'highlight': '#FF4DA6',
  'outline': '#3A0020',
  'outline_w': 4,
  'bold': True,
  'upper': True,
  'position': 'middle'},
 {'id': 'clean-minimal',
  'name': 'Clean Minimal',
  'published': '2024-01-06',
  'font': 'Segoe UI',
  'size': 50,
  'text': '#FFFFFF',
  'highlight': '#FFFFFF',
  'outline': '#222222',
  'outline_w': 2,
  'bold': True,
  'upper': False,
  'position': 'bottom'},
 {'id': 'box-black',
  'name': 'Black Box',
  'published': '2024-01-05',
  'font': 'Arial',
  'size': 50,
  'text': '#FFFFFF',
  'highlight': '#FFD400',
  'outline': '#000000',
  'outline_w': 0,
  'bold': True,
  'upper': False,
  'position': 'bottom',
  'box': '#000000',
  'box_pad': 0.18},
 {'id': 'box-yellow',
  'name': 'Yellow Box',
  'published': '2024-01-04',
  'font': 'Arial',
  'size': 50,
  'text': '#111111',
  'highlight': '#E53935',
  'outline': '#000000',
  'outline_w': 0,
  'bold': True,
  'upper': False,
  'position': 'bottom',
  'box': '#FFD400',
  'box_pad': 0.18},
 {'id': 'box-white',
  'name': 'White Box',
  'published': '2024-01-03',
  'font': 'Segoe UI',
  'size': 50,
  'text': '#1A1A1A',
  'highlight': '#1E88E5',
  'outline': '#000000',
  'outline_w': 0,
  'bold': True,
  'upper': False,
  'position': 'bottom',
  'box': '#FFFFFF',
  'box_pad': 0.18},
 {'id': 'fun-comic',
  'name': 'Fun Comic',
  'published': '2024-01-02',
  'font': 'Comic Sans MS',
  'size': 56,
  'text': '#FFFFFF',
  'highlight': '#FFEB3B',
  'outline': '#1A237E',
  'outline_w': 4,
  'bold': True,
  'upper': False,
  'position': 'middle'},
 {'id': 'elegant',
  'name': 'Elegant Serif',
  'published': '2024-01-01',
  'font': 'Georgia',
  'size': 54,
  'text': '#F5E6C8',
  'highlight': '#FFFFFF',
  'outline': '#000000',
  'outline_w': 3,
  'bold': True,
  'upper': False,
  'position': 'bottom'}]
for _b in BUNDLED:
    if _b["id"] in ("yellow-pop", "neon-green"): _b["badge"] = "Popular"

# Roman-script prompt strongly biases Whisper to output Hinglish (Roman Hindi + English words)
PROMPT = "Haan toh guys, aaj hum baat karenge ki yeh video kaise banate hain. Bilkul simple hai, chalo start karte hain."

# ------------------------------------------------------------------ transcription
def deva_to_roman(word):
    """Devanagari -> natural Hinglish spelling (drops the silent final 'a': hama -> ham, bAta -> baat)."""
    from indic_transliteration import sanscript
    s = sanscript.transliterate(word, sanscript.DEVANAGARI, sanscript.ITRANS)
    for a, b in (("R^i", "ri"), ("GY", "gy"), ("kSh", "ksh"), ("Ch", "chh"), ("Sh", "sh"), ("~n", "n"),
                 ("~N", "n"), (".N", "n"), (".n", "n"), (".h", ""), ("H", "h")):
        s = s.replace(a, b)
    s = re.sub(r"M(?=[pbm])", "m", s).replace("M", "n").replace(".", "").replace("^", "")
    if re.search(r"[^aeiouAIUE]a$", s): s = s[:-1]                      # inherent final schwa is not spoken
    s = re.sub(r"^A", "aa", s)
    if len(re.findall(r"[aeiouAIU]", s)) == 1 and not s.endswith("A"): s = s.replace("A", "aa")   # baat, kaam
    return s.replace("A", "a").replace("I", "i").replace("U", "u").lower()

def hinglish(word):
    """Keep Roman text as is; convert any Devanagari that Whisper returned."""
    if not re.search(r"[\u0900-\u097F]", word): return word
    return deva_to_roman(word)

def _run(path, model_size, device, ctype, progress):
    from faster_whisper import WhisperModel
    model = WhisperModel(model_size, device=device, compute_type=ctype, download_root=MODELS,
                         local_files_only=model_ready(model_size))
    segs, info = model.transcribe(path, language="hi", task="transcribe", word_timestamps=True,
                                  vad_filter=True, beam_size=5, initial_prompt=PROMPT,
                                  condition_on_previous_text=False)
    words = []
    for seg in segs:   # segs is lazy: GPU errors can surface here, not only at load time
        for w in seg.words or []:
            t = hinglish(w.word.strip())
            if t:
                words.append({"w": t, "s": w.start, "e": w.end})
        progress(min(99, int(100 * seg.end / max(info.duration, 1))))
    return group_lines(words)

def transcribe(path, model_size, progress):
    """GPU first (int8_float16 keeps large-v3 inside 4 GB of VRAM); falls back to CPU on any CUDA problem."""
    try:
        import ctranslate2
        has_gpu = ctranslate2.get_cuda_device_count() > 0
    except Exception:
        has_gpu = False
    if has_gpu:
        try:
            return _run(path, model_size, "cuda", "int8_float16", progress), "GPU"
        except Exception as e:
            if not re.search(r"cublas|cudnn|cuda|cudart|memory|\.dll", str(e), re.I): raise
            progress(0)
    return _run(path, model_size, "cpu", "int8", progress), "CPU"

def group_lines(words, max_words=6, max_dur=2.8, gap=0.6):
    lines, cur = [], []
    for w in words:
        if cur and (len(cur) >= max_words or w["e"] - cur[0]["s"] > max_dur or w["s"] - cur[-1]["e"] > gap):
            lines.append(cur); cur = []
        cur.append(w)
    if cur: lines.append(cur)
    return [{"start": l[0]["s"], "end": l[-1]["e"], "words": l, "text": " ".join(x["w"] for x in l)} for l in lines]

# ------------------------------------------------------------------ export
def ts_srt(t):
    h, m, s = int(t // 3600), int(t % 3600 // 60), t % 60
    return f"{h:02}:{m:02}:{int(s):02},{int(round((s % 1) * 1000)):03}"

def ts_ass(t):
    h, m, s = int(t // 3600), int(t % 3600 // 60), t % 60
    return f"{h}:{m:02}:{s:05.2f}"

def ass_col(hexc):
    h = hexc.lstrip("#"); return f"&H00{h[4:6]}{h[2:4]}{h[0:2]}"

def write_srt(lines, path):
    with open(path, "w", encoding="utf-8") as f:
        for i, l in enumerate(lines, 1):
            f.write(f"{i}\n{ts_srt(l['start'])} --> {ts_srt(l['end'])}\n{l['text']}\n\n")

NOWIN = getattr(subprocess, "CREATE_NO_WINDOW", 0)

def probe(path):
    """(width, height, duration) with phone-video rotation applied."""
    try:
        out = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
            "stream=width,height:stream_side_data=rotation:stream_tags=rotate:format=duration",
            "-of", "json", path], text=True, creationflags=NOWIN)
        d = json.loads(out); s = d["streams"][0]; w, h = s["width"], s["height"]
        rot = 0
        for sd in s.get("side_data_list", []): rot = sd.get("rotation", rot)
        rot = int(s.get("tags", {}).get("rotate", rot))
        if abs(rot) % 180 == 90: w, h = h, w
        return w, h, float(d["format"]["duration"])
    except Exception:
        return 1920, 1080, 0.0

def disp(t, tpl): return t.upper() if tpl.get("upper") else t

def make_font(family, fpx, bold):
    """ASS font size = full cell height (ascent+descent); Qt pixel size = em. Convert so preview == export."""
    f = QFont(family); f.setBold(bold); f.setPixelSize(max(1, int(fpx)))
    h = QFontMetricsF(f).height()
    if h > 0: f.setPixelSize(max(1, int(round(fpx * fpx / h))))
    return f

def wrap_words(texts, family, fpx, bold, max_w):
    """Greedy line wrap -> list of rows of word indices. Shared by preview and export."""
    f = make_font(family, fpx, bold); fm = QFontMetricsF(f); sp = fm.horizontalAdvance(" ")
    rows, cur, w = [], [], 0.0
    for i, t in enumerate(texts):
        tw = fm.horizontalAdvance(t); add = tw if not cur else sp + tw
        if cur and w + add > max_w: rows.append(cur); cur, w, add = [], 0.0, tw
        cur.append(i); w += add
    if cur: rows.append(cur)
    return rows, f, fm

def write_ass(lines, tpl, style, path, vw, vh, ow, oh):
    """style = {cx, cy, size}: caption centre as fraction of frame, font size as fraction of short side."""
    fpx = style["size"] * min(vw, vh) * (oh / vh)
    outl = 0 if tpl.get("box") else tpl.get("outline_w", 3) / tpl["size"] * fpx
    x, y = style["cx"] * ow, style["cy"] * oh
    bold = bool(tpl.get("bold"))
    hdr = (f"[Script Info]\nScriptType: v4.00+\nWrapStyle: 2\nPlayResX: {ow}\nPlayResY: {oh}\n\n[V4+ Styles]\n"
           "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,"
           "Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding\n"
           f"Style: Default,{tpl['font']},{fpx:.1f},{ass_col(tpl['highlight'])},{ass_col(tpl['text'])},"
           f"{ass_col(tpl['outline'])},&H80000000,{-1 if bold else 0},0,0,0,100,100,0,0,1,{outl:.1f},0,5,0,0,0,1\n\n"
           "[Events]\nFormat: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write(hdr)
        for l in lines:
            ws = l["words"]; texts = [disp(w["w"], tpl) for w in ws]
            rows, _, fm = wrap_words(texts, tpl["font"], fpx, bold, 0.9 * ow)
            if tpl.get("box"):
                sp = fm.horizontalAdvance(" "); pad = tpl.get("box_pad", 0.18) * fpx
                top = y - len(rows) * fpx / 2; bc = ass_col(tpl["box"])[4:]
                for r, row in enumerate(rows):
                    rw = sum(fm.horizontalAdvance(texts[i]) for i in row) + sp * (len(row) - 1)
                    x0, x1, y0, y1 = x - rw / 2 - pad, x + rw / 2 + pad, top + r * fpx, top + (r + 1) * fpx
                    f.write(f"Dialogue: 0,{ts_ass(l['start'])},{ts_ass(l['end'])},Default,,0,0,0,,"
                            f"{{\\an7\\pos(0,0)\\p1\\bord0\\shad0\\1c&H{bc}&}}m {x0:.0f} {y0:.0f} l {x1:.0f} {y0:.0f} "
                            f"{x1:.0f} {y1:.0f} {x0:.0f} {y1:.0f}{{\\p0}}\n")
            out_rows = []
            for row in rows:
                seg = []
                for i in row:
                    nxt = ws[i + 1]["s"] if i + 1 < len(ws) else l["end"]
                    seg.append(f"{{\\k{max(1, int((nxt - ws[i]['s']) * 100))}}}{texts[i]}")
                out_rows.append(" ".join(seg))
            f.write(f"Dialogue: 1,{ts_ass(l['start'])},{ts_ass(l['end'])},Default,,0,0,0,,"
                    f"{{\\an5\\pos({x:.0f},{y:.0f})}}" + "\\N".join(out_rows) + "\n")

def burn(video, tmp, size, src_size, dur, hevc, out_path, prog):
    ow, oh = size; vf = []
    if (ow, oh) != src_size: vf.append(f"scale={ow}:{oh}:flags=lanczos")
    vf.append("ass=c.ass")
    cmd = ["ffmpeg", "-y", "-i", video, "-vf", ",".join(vf), "-c:v", "libx265" if hevc else "libx264",
           "-crf", "18" if hevc else "16", "-preset", "medium", "-pix_fmt", "yuv420p"]
    if hevc: cmd += ["-tag:v", "hvc1"]
    cmd += ["-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-progress", "pipe:1", "-nostats", out_path]
    pr = subprocess.Popen(cmd, cwd=tmp, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                          encoding="utf-8", errors="replace", creationflags=NOWIN)
    tail = deque(maxlen=12)
    for line in pr.stdout:
        line = line.strip(); m = re.match(r"out_time_(?:us|ms)=(\d+)", line)
        if m and dur: prog(min(99, int(100 * int(m[1]) / 1e6 / dur)))
        elif not re.match(r"(frame|fps|stream_|bitrate|total_size|out_time|dup_|drop_|speed|progress)", line): tail.append(line)
    rc = pr.wait(); shutil.rmtree(tmp, ignore_errors=True)
    if rc: raise RuntimeError("ffmpeg failed:\n" + "\n".join(tail))
    return out_path

# ------------------------------------------------------------------ live preview widgets
class CaptionItem(QGraphicsItem):
    """Draggable caption drawn over the video. Scene units == source video pixels."""
    def __init__(self, vw, vh, style, on_change):
        super().__init__()
        self.vw, self.vh, self.style, self.cb = vw, vh, style, on_change
        self.tpl, self.words, self.t, self.ghost = BUNDLED[0], [], -1.0, False
        self.rect, self._key, self.rows, self.texts = QRectF(), None, [], []
        self.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemSendsGeometryChanges)
        self.setCursor(Qt.SizeAllCursor); self.setZValue(10)

    def boundingRect(self): return self.rect

    def set_content(self, words, t, ghost):
        self.words, self.t, self.ghost = words, t, ghost
        fpx = self.style["size"] * min(self.vw, self.vh)
        texts = [disp(w["w"], self.tpl) for w in words]
        key = (tuple(texts), round(fpx, 1), self.tpl["font"], self.tpl.get("bold"))
        if key != self._key:
            self.prepareGeometryChange(); self._key, self.texts, self.fpx = key, texts, fpx
            self.rows, self.font, self.fm = wrap_words(texts, self.tpl["font"], fpx, bool(self.tpl.get("bold")), 0.9 * self.vw)
            sp = self.fm.horizontalAdvance(" ")
            self.rw = [sum(self.fm.horizontalAdvance(texts[i]) for i in r) + sp * (len(r) - 1) for r in self.rows]
            w, h, pad = max(self.rw or [1]), len(self.rows) * fpx, fpx * 0.3
            self.rect = QRectF(-w / 2 - pad, -h / 2 - pad, w + 2 * pad, h + 2 * pad)
        self.update()

    def paint(self, p, opt, widget=None):
        if not self.rows: return
        p.setRenderHint(QPainter.Antialiasing)
        if self.ghost: p.setOpacity(0.45)
        tpl, fm = self.tpl, self.fm
        outl = tpl.get("outline_w", 3) / tpl["size"] * self.fpx
        sp = fm.horizontalAdvance(" "); y = -len(self.rows) * self.fpx / 2; paths = []
        for r, row in enumerate(self.rows):
            x = -self.rw[r] / 2
            for i in row:
                path = QPainterPath(); path.addText(x, y + fm.ascent(), self.font, self.texts[i])
                paths.append((path, self.words[i]["s"] <= self.t)); x += fm.horizontalAdvance(self.texts[i]) + sp
            y += self.fpx
        if tpl.get("box"):
            pad = tpl.get("box_pad", 0.18) * self.fpx; p.setPen(Qt.NoPen); p.setBrush(QColor(tpl["box"]))
            for r in range(len(self.rows)):
                p.drawRect(QRectF(-self.rw[r] / 2 - pad, -len(self.rows) * self.fpx / 2 + r * self.fpx,
                                  self.rw[r] + 2 * pad, self.fpx))
        else:
            p.setPen(QPen(QColor(tpl["outline"]), 2 * outl, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            for path, _ in paths: p.drawPath(path)
        p.setPen(Qt.NoPen)
        for path, hi in paths:
            p.setBrush(QColor(tpl["highlight"] if hi else tpl["text"])); p.drawPath(path)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange and self.scene():
            x, y = min(max(value.x(), 0), self.vw), min(max(value.y(), 0), self.vh)
            if abs(x - self.vw / 2) < self.vw * 0.012: x = self.vw / 2   # snap to horizontal centre
            return QPointF(x, y)
        if change == QGraphicsItem.ItemPositionHasChanged:
            self.style["cx"], self.style["cy"] = self.pos().x() / self.vw, self.pos().y() / self.vh
        return super().itemChange(change, value)

    def wheelEvent(self, e):
        k = 1.06 if e.delta() > 0 else 1 / 1.06
        self.style["size"] = min(0.15, max(0.015, self.style["size"] * k)); self.cb(); e.accept()

class VideoView(QGraphicsView):
    def resizeEvent(self, e): super().resizeEvent(e); self.fit()
    def fit(self):
        if not self.sceneRect().isEmpty(): self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)

# ------------------------------------------------------------------ templates
def load_cache():
    try:
        d = json.load(open(CACHE, encoding="utf-8"))
        if "online" in d: return d
    except Exception:
        pass
    return {"online": [], "seen": []}

def merge(online):
    """Newest online templates first; built-in ones fill the list up to MAX_TEMPLATES."""
    ids = {t["id"] for t in online}
    return (online + [b for b in BUNDLED if b["id"] not in ids])[:MAX_TEMPLATES]

def fetch_templates():
    req = urllib.request.Request(TEMPLATE_FEED_URL, headers={"User-Agent": "HinglishCaptioner"})
    data = json.loads(urllib.request.urlopen(req, timeout=15).read().decode("utf-8"))["templates"]
    need = ("id", "name", "font", "size", "text", "highlight", "outline")
    data = [t for t in data if all(k in t for k in need)]
    data.sort(key=lambda t: t.get("published", ""), reverse=True)
    return data[:MAX_TEMPLATES]

class Worker(QThread):
    done, fail, prog = Signal(object), Signal(str), Signal(int)
    def __init__(self, fn): super().__init__(); self.fn = fn
    def run(self):
        try: self.done.emit(self.fn(self.prog.emit))
        except Exception as e: self.fail.emit(str(e))

# ------------------------------------------------------------------ UI
PRESETS = os.path.join(DATA, "presets.json")
def load_presets():
    try: return json.load(open(PRESETS, encoding="utf-8"))
    except Exception: return []

QSS = """
QWidget { background:#0e0e10; color:#e8e8ea; font-size:13px; }
QLabel { background:transparent; }
QLabel#muted { color:#8a8a90; }
QLabel#section { color:#8a8a90; font-size:11px; font-weight:600; letter-spacing:1px; }
QLabel#pill { background:#1b1b1f; border:1px solid #2c2c31; border-radius:10px; padding:3px 10px; color:#c9c9cf; }
QWidget#panel { background:#151517; }
QPushButton { background:#1f1f23; border:1px solid #2c2c31; border-radius:8px; padding:7px 12px; }
QPushButton:hover { background:#28282d; }
QPushButton#primary { background:#F59E0B; color:#141414; border:0; font-weight:700; }
QPushButton#primary:hover { background:#FBB03B; }
QPushButton[seg="true"] { border-radius:6px; padding:5px 12px; }
QPushButton[seg="true"]:checked { background:#3a2a08; color:#F59E0B; border:1px solid #F59E0B; }
QLineEdit, QComboBox, QFontComboBox { background:#1b1b1f; border:1px solid #2c2c31; border-radius:8px; padding:6px 8px; }
QComboBox QAbstractItemView { background:#1b1b1f; selection-background-color:#3a2a08; }
QLineEdit#cardedit { background:transparent; border:0; padding:2px 0; font-size:14px; font-weight:600; }
QFrame#card { background:#1b1b1f; border:1px solid #26262a; border-radius:10px; }
QFrame#card[active="true"] { border:1px solid #F59E0B; background:#211b0d; }
QListWidget { background:transparent; border:0; outline:0; }
QListWidget#grid::item { color:#c9c9cf; border:1px solid transparent; border-radius:10px; padding:3px; }
QListWidget#grid::item:selected { background:#241c09; border:2px solid #F59E0B; color:#F59E0B; }
QProgressBar { background:#1f1f23; border:0; border-radius:6px; height:14px; text-align:center; color:#ddd; font-size:11px; }
QProgressBar::chunk { background:#F59E0B; border-radius:6px; }
QSlider::groove:horizontal { height:4px; background:#2c2c31; border-radius:2px; }
QSlider::sub-page:horizontal { background:#F59E0B; border-radius:2px; }
QSlider::handle:horizontal { background:#F59E0B; width:12px; margin:-5px 0; border-radius:6px; }
QTabBar::tab { padding:9px 16px; background:transparent; color:#8a8a90; border-bottom:2px solid transparent; }
QTabBar::tab:selected { color:#ffffff; border-bottom:2px solid #F59E0B; }
QTabWidget::pane { border:0; }
QScrollBar:vertical { width:8px; background:transparent; }
QScrollBar::handle:vertical { background:#2c2c31; border-radius:4px; min-height:30px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0; }
QCheckBox { spacing:8px; }
"""

def grab_frame(video, t):
    out = os.path.join(tempfile.gettempdir(), "hc_frame.png")
    try:
        subprocess.run(["ffmpeg", "-y", "-ss", f"{t:.2f}", "-i", video, "-frames:v", "1", "-vf", "scale=480:-2", out],
                       capture_output=True, timeout=30, creationflags=NOWIN)
        return QPixmap(out)
    except Exception:
        return QPixmap()

def make_thumb(tpl, frame, W, H, badge=""):
    """Template card: a frame of YOUR video with a sample caption in this template's style."""
    pm = QPixmap(W, H); pm.fill(QColor("#26262b")); p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing); p.setRenderHint(QPainter.SmoothPixmapTransform)
    if not frame.isNull():
        sc = frame.scaled(W, H, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
        p.drawPixmap((W - sc.width()) // 2, (H - sc.height()) // 2, sc)
    else:
        g = QLinearGradient(0, 0, 0, H); g.setColorAt(0, QColor("#3a3f4b")); g.setColorAt(1, QColor("#1a1c22")); p.fillRect(0, 0, W, H, g)
    fpx = tpl["size"] / 1080 * W * 2.4
    f = make_font(tpl["font"], fpx, bool(tpl.get("bold"))); fm = QFontMetricsF(f); sp = fm.horizontalAdvance(" ")
    words = [disp("Hey", tpl), disp("friends", tpl)]; wtot = sum(fm.horizontalAdvance(w) for w in words) + sp
    cy = {"bottom": 0.84, "middle": 0.55, "top": 0.14}.get(tpl.get("position", "bottom"), 0.84) * H
    x, y = (W - wtot) / 2, cy - fpx / 2
    if tpl.get("box"):
        pad = tpl.get("box_pad", 0.18) * fpx; p.setPen(Qt.NoPen); p.setBrush(QColor(tpl["box"])); p.drawRect(QRectF(x - pad, y, wtot + 2 * pad, fpx))
    paths, cx = [], x
    for i, w in enumerate(words):
        path = QPainterPath(); path.addText(cx, y + fm.ascent(), f, w)
        paths.append((path, tpl["highlight"] if i else tpl["text"])); cx += fm.horizontalAdvance(w) + sp
    if not tpl.get("box"):
        p.setPen(QPen(QColor(tpl["outline"]), max(1.0, 2 * tpl.get("outline_w", 3) / tpl["size"] * fpx), Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        for path, _ in paths: p.drawPath(path)
    p.setPen(Qt.NoPen)
    for path, col in paths: p.setBrush(QColor(col)); p.drawPath(path)
    if badge:
        bf = p.font(); bf.setPixelSize(10); bf.setBold(True); p.setFont(bf)
        bw = p.fontMetrics().horizontalAdvance(badge) + 14
        p.setBrush(QColor("#F59E0B")); p.drawRoundedRect(QRectF(6, 6, bw, 17), 8, 8)
        p.setPen(QColor("#141414")); p.drawText(QRectF(6, 6, bw, 17), Qt.AlignCenter, badge)
    p.end(); return pm

class FocusEdit(QLineEdit):
    focused = Signal()
    def focusInEvent(self, e): super().focusInEvent(e); self.focused.emit()

class CaptionCard(QFrame):
    picked = Signal(int)
    def __init__(self, idx, line):
        super().__init__(); self.idx = idx; self.setObjectName("card")
        v = QVBoxLayout(self); v.setContentsMargins(12, 8, 12, 8); v.setSpacing(2)
        top = QHBoxLayout(); n = QLabel(str(idx + 1)); n.setObjectName("muted")
        self.tm = QLabel(self.fmt(line)); self.tm.setObjectName("muted"); top.addWidget(n); top.addStretch(); top.addWidget(self.tm)
        self.edit = FocusEdit(line["text"]); self.edit.setCursorPosition(0); self.edit.setObjectName("cardedit"); self.edit.focused.connect(lambda: self.picked.emit(self.idx))
        v.addLayout(top); v.addWidget(self.edit)
    @staticmethod
    def fmt(l):
        f = lambda t: f"{int(t // 60)}:{t % 60:04.1f}"
        return f"{f(l['start'])}  →  {f(l['end'])}"
    def set_active(self, on):
        self.setProperty("active", on); self.style().unpolish(self); self.style().polish(self)
    def mousePressEvent(self, e): self.picked.emit(self.idx); super().mousePressEvent(e)

class Timeline(QWidget):
    seek = Signal(float)
    def __init__(self):
        super().__init__(); self.setFixedHeight(104); self.lines, self.dur, self.t = [], 1.0, 0.0
        self.setCursor(Qt.PointingHandCursor)
    def set_data(self, lines, dur): self.lines, self.dur = lines, max(dur, 0.1); self.update()
    def paintEvent(self, e):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing); W, H = self.width(), self.height()
        p.fillRect(self.rect(), QColor("#111113")); d = max(self.dur, 0.1)
        step = next((s for s in (1, 2, 5, 10, 15, 30, 60, 120, 300) if s * W / d >= 70), 600)
        f = p.font(); f.setPixelSize(10); p.setFont(f); p.setPen(QColor("#6b6b72"))
        for s in range(0, int(d) + 1, step):
            x = int(s / d * W); p.drawLine(x, 14, x, 22); p.drawText(x + 4, 12, f"{s // 60:02}:{s % 60:02}")
        for l in self.lines:
            x0 = l["start"] / d * W; w = max(4.0, (l["end"] - l["start"]) / d * W); r = QRectF(x0, 34, w - 2, 44)
            p.setPen(Qt.NoPen); p.setBrush(QColor("#d8c48c")); p.drawRoundedRect(r, 6, 6)
            p.setPen(QColor("#2a2210"))
            p.drawText(r.adjusted(6, 0, -3, 0), Qt.AlignVCenter | Qt.AlignLeft, p.fontMetrics().elidedText(l["text"], Qt.ElideRight, max(0, int(w - 10))))
        x = int(self.t / d * W); p.setPen(QPen(QColor("#F59E0B"), 2)); p.drawLine(x, 0, x, H); p.end()
    def _pos(self, e): self.seek.emit(min(max(e.position().x() / max(self.width(), 1), 0.0), 1.0) * self.dur)
    def mousePressEvent(self, e): self._pos(e)
    def mouseMoveEvent(self, e): self._pos(e)

# ------------------------------------------------------------------ start-up screen (downloads GPU files + model)
class SetupWorker(QThread):
    prog, fail, done = Signal(int, int), Signal(str), Signal()
    def __init__(self, steps): super().__init__(); self.steps = steps
    def run(self):
        try:
            for i, (_, fn) in enumerate(self.steps):
                fn(lambda p, i=i: self.prog.emit(i, int(p))); self.prog.emit(i, 100)
            self.done.emit()
        except Exception as e:
            self.fail.emit(str(e))

class SetupDialog(QDialog):
    def __init__(self, steps):
        super().__init__(); self.steps = steps; self.setWindowTitle("Hinglish Captioner"); self.setMinimumWidth(600)
        v = QVBoxLayout(self); v.setContentsMargins(28, 24, 28, 24); v.setSpacing(8)
        t = QLabel("Getting things ready..."); t.setStyleSheet("font-size:24px; font-weight:700")
        sub = QLabel("One-time setup. Please keep the app open - this can take several minutes."); sub.setObjectName("muted")
        v.addWidget(t); v.addWidget(sub); v.addSpacing(10); self.bars = []
        for name, _ in steps:
            v.addWidget(QLabel(name)); b = QProgressBar(); b.setRange(0, 100); b.setFormat("%p%"); b.setFixedHeight(20)
            v.addWidget(b); self.bars.append(b); v.addSpacing(6)
        self.msg = QLabel(""); self.msg.setWordWrap(True); v.addWidget(self.msg)
        row = QHBoxLayout(); row.addStretch()
        self.retry = QPushButton("Retry"); self.retry.setObjectName("primary"); self.skip = QPushButton("Continue without it")
        self.retry.clicked.connect(self.start); self.skip.clicked.connect(self.reject)
        row.addWidget(self.skip); row.addWidget(self.retry); v.addLayout(row); self.start()
    def start(self):
        self.msg.setText(""); self.retry.hide(); self.skip.hide()
        self.w = SetupWorker(self.steps); self.w.prog.connect(lambda i, p: self.bars[i].setValue(p))
        self.w.done.connect(self.accept); self.w.fail.connect(self.failed); self.w.start()
    def reject(self):   # a download cannot be interrupted safely, so ignore close/Esc while it runs
        if self.w.isRunning():
            self.msg.setText("Still downloading - please keep this window open until it finishes."); return
        super().reject()
    def failed(self, m):
        self.msg.setText(f"Download stopped: {m}\nCheck your internet connection, then press Retry. Nothing already downloaded is lost.")
        self.retry.show(); self.skip.show()

# ------------------------------------------------------------------ main window
RES = {"Source": 0, "720p": 1280, "1080p": 1920, "4K": 3840}   # target = long side in px
POS_Y = {"bottom": 0.86, "middle": 0.5, "top": 0.12}

def seg(text, group):
    b = QPushButton(text); b.setCheckable(True); b.setProperty("seg", True); group.addButton(b); return b

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hinglish Captioner"); self.resize(1480, 900)
        self.video, self.vinfo, self.lines, self.starts, self.workers, self.cap = None, (1920, 1080, 0.0), [], [], [], None
        self.cstyle = {"cx": 0.5, "cy": 0.86, "size": 0.05}
        self.cache = load_cache(); self.templates = merge(self.cache["online"]); self.presets = load_presets()
        seen = set(self.cache.get("seen", [])); self.new_ids = ({t["id"] for t in self.cache["online"]} - seen) if seen else set()
        self.work = dict(self.templates[0]); self.frame = QPixmap(); self.thumb = (146, 200); self.active = -1; self.mode = "templates"

        # --- player
        self.player = QMediaPlayer(); self.audio = QAudioOutput(); self.player.setAudioOutput(self.audio)
        self.scene = QGraphicsScene(self); self.vitem = QGraphicsVideoItem(); self.scene.addItem(self.vitem)
        self.player.setVideoOutput(self.vitem)
        self.view = VideoView(self.scene); self.view.setStyleSheet("background:#050506; border:0")
        self.view.setFrameShape(QFrame.NoFrame); self.view.setRenderHint(QPainter.Antialiasing)

        # --- top bar
        top = QWidget(); top.setObjectName("panel"); tl = QHBoxLayout(top); tl.setContentsMargins(16, 8, 16, 8); tl.setSpacing(8)
        brand = QLabel("Hinglish Captioner"); brand.setStyleSheet("font-weight:700; font-size:15px")
        self.btn_open = QPushButton("Open video"); self.btn_open.clicked.connect(self.open_video)
        self.lbl_proj = QLabel("No video loaded"); self.lbl_proj.setObjectName("muted")
        self.lbl_dev = QLabel("GPU ready" if (gpu_ready() and has_nvidia()) else "CPU mode"); self.lbl_dev.setObjectName("pill")
        self.qgroup = QButtonGroup(self); self.qgroup.setExclusive(True); self.qbtn = {}
        for k in RES:
            self.qbtn[k] = seg(k, self.qgroup)
        self.qbtn["Source"].setChecked(True)
        self.codec = QComboBox(); self.codec.addItems(["H.264", "H.265"]); self.codec.setToolTip("H.265 = smaller 4K files")
        self.btn_srt = QPushButton("SRT"); self.btn_srt.clicked.connect(self.exp_srt)
        self.btn_export = QPushButton("Export"); self.btn_export.setObjectName("primary"); self.btn_export.setMinimumWidth(110)
        self.btn_export.clicked.connect(self.exp_mp4)
        tl.addWidget(brand); tl.addSpacing(10); tl.addWidget(self.btn_open); tl.addWidget(self.lbl_proj); tl.addStretch()
        tl.addWidget(self.lbl_dev); tl.addSpacing(8); tl.addWidget(QLabel("Quality"))
        for b in self.qbtn.values(): tl.addWidget(b)
        tl.addWidget(self.codec); tl.addWidget(self.btn_srt); tl.addWidget(self.btn_export)

        # --- left: captions
        left = QWidget(); left.setObjectName("panel"); left.setFixedWidth(330); lv = QVBoxLayout(left); lv.setContentsMargins(12, 12, 12, 12)
        sec = QLabel("CAPTIONS"); sec.setObjectName("section"); lv.addWidget(sec)
        row = QHBoxLayout(); self.model = QComboBox(); self.model.addItems(list(MODEL_GB)); self.model.setToolTip("large-v3 = best accuracy")
        self.btn_gen = QPushButton("Generate captions"); self.btn_gen.setObjectName("primary"); self.btn_gen.clicked.connect(self.generate)
        row.addWidget(self.model); row.addWidget(self.btn_gen, 1); lv.addLayout(row)
        self.bar = QProgressBar(); self.bar.setRange(0, 100); self.bar.setValue(0); lv.addWidget(self.bar)
        self.lbl = QLabel("Open a video to begin"); self.lbl.setObjectName("muted"); self.lbl.setWordWrap(True); lv.addWidget(self.lbl)
        self.clist = QListWidget(); self.clist.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel); self.clist.setSpacing(6); lv.addWidget(self.clist, 1)

        # --- center: player
        center = QVBoxLayout(); center.setContentsMargins(10, 10, 10, 6); center.addWidget(self.view, 1)
        self.btn_play = QPushButton("Play"); self.btn_play.setFixedWidth(80); self.btn_play.clicked.connect(self.toggle_play)
        self.seek = QSlider(Qt.Horizontal); self.seek.sliderMoved.connect(self.player.setPosition)
        self.player.durationChanged.connect(lambda d: self.seek.setRange(0, d))
        self.player.playbackStateChanged.connect(lambda s: self.btn_play.setText("Pause" if s == QMediaPlayer.PlayingState else "Play"))
        self.lbl_time = QLabel("0:00 / 0:00")
        tr = QHBoxLayout(); tr.setSpacing(10); tr.addWidget(self.btn_play); tr.addWidget(self.seek, 1); tr.addWidget(self.lbl_time)
        self.size_sl = QSlider(Qt.Horizontal); self.size_sl.setRange(15, 150); self.size_sl.setValue(50); self.size_sl.valueChanged.connect(self.size_changed)
        self.btn_reset = QPushButton("Reset position"); self.btn_reset.clicked.connect(self.reset_style)
        st = QHBoxLayout(); st.setSpacing(10); st.addWidget(QLabel("Text size")); st.addWidget(self.size_sl, 1); st.addWidget(self.btn_reset)
        hint = QLabel("Drag the caption on the video to move it. Mouse wheel over it, or the slider, changes size."); hint.setObjectName("muted")
        center.addLayout(tr); center.addLayout(st); center.addWidget(hint)

        # --- right: templates / customize
        right = QWidget(); right.setObjectName("panel"); right.setFixedWidth(364); rv = QVBoxLayout(right); rv.setContentsMargins(6, 6, 6, 6)
        self.tabs = QTabWidget(); rv.addWidget(self.tabs)
        tw = QWidget(); tv = QVBoxLayout(tw); tv.setContentsMargins(8, 10, 8, 4)
        self.mgroup = QButtonGroup(self); self.mgroup.setExclusive(True)
        self.btn_tpl, self.btn_pre = seg("Templates", self.mgroup), seg("My Presets", self.mgroup); self.btn_tpl.setChecked(True)
        self.mgroup.buttonClicked.connect(lambda b: self.set_mode("templates" if b is self.btn_tpl else "presets"))
        mr = QHBoxLayout(); mr.addWidget(self.btn_tpl, 1); mr.addWidget(self.btn_pre, 1); tv.addLayout(mr)
        sr = QHBoxLayout(); self.search = QLineEdit(); self.search.setPlaceholderText("Find a template...")
        self.search.textChanged.connect(lambda _: self.fill_grid())
        self.btn_save = QPushButton("Save Preset"); self.btn_save.clicked.connect(self.save_preset); sr.addWidget(self.search, 1); sr.addWidget(self.btn_save); tv.addLayout(sr)
        sec2 = QLabel("DYNAMIC CAPTIONS"); sec2.setObjectName("section"); tv.addWidget(sec2)
        self.grid = QListWidget(); self.grid.setObjectName("grid"); self.grid.setViewMode(QListWidget.IconMode)
        self.grid.setResizeMode(QListWidget.Adjust); self.grid.setMovement(QListWidget.Static); self.grid.setWrapping(True)
        self.grid.setUniformItemSizes(True); self.grid.setSpacing(2); self.grid.setWordWrap(True)
        self.grid.currentItemChanged.connect(lambda cur, _p: cur and self.set_work(cur.data(Qt.UserRole)))
        tv.addWidget(self.grid, 1)
        self.lbl_t = QLabel(""); self.lbl_t.setObjectName("muted"); tv.addWidget(self.lbl_t)
        self.btn_ref = QPushButton("Refresh templates"); self.btn_ref.clicked.connect(self.refresh); tv.addWidget(self.btn_ref)
        self.tabs.addTab(tw, "Templates")
        cw = QWidget(); cv = QVBoxLayout(cw); cv.setContentsMargins(12, 14, 12, 12); cv.setSpacing(10)
        self.fontbox = QFontComboBox(); self.fontbox.currentFontChanged.connect(lambda f: self.cust("font", f.family()))
        cv.addWidget(QLabel("Font")); cv.addWidget(self.fontbox)
        self.colbtn = {}
        for key, label in (("text", "Text colour"), ("highlight", "Highlight colour (spoken word)"), ("outline", "Outline / box colour")):
            b = QPushButton(label); b.clicked.connect(lambda _=False, k=key: self.pick_color(k)); self.colbtn[key] = b; cv.addWidget(b)
        cv.addWidget(QLabel("Outline width")); self.outsl = QSlider(Qt.Horizontal); self.outsl.setRange(0, 12)
        self.outsl.valueChanged.connect(lambda v: self.cust("outline_w", v)); cv.addWidget(self.outsl)
        self.chk_up = QCheckBox("UPPERCASE"); self.chk_up.toggled.connect(lambda v: self.cust("upper", v))
        self.chk_bold = QCheckBox("Bold"); self.chk_bold.toggled.connect(lambda v: self.cust("bold", v))
        cv.addWidget(self.chk_up); cv.addWidget(self.chk_bold); cv.addStretch()
        note = QLabel("Changes apply to the preview and the export. Use Save Preset on the Templates tab to keep them."); note.setObjectName("muted"); note.setWordWrap(True); cv.addWidget(note)
        self.tabs.addTab(cw, "Customize")

        # --- timeline + assemble
        self.timeline = Timeline(); self.timeline.seek.connect(lambda t: self.player.setPosition(int(t * 1000)))
        root = QVBoxLayout(self); root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0); root.addWidget(top)
        mid = QHBoxLayout(); mid.setSpacing(0); mid.addWidget(left); mid.addLayout(center, 1); mid.addWidget(right); root.addLayout(mid, 1); root.addWidget(self.timeline)

        self.sync_custom(); self.fill_grid(); self.grid.setCurrentRow(0); self.refresh(silent=True)
        self.timer = QTimer(self); self.timer.timeout.connect(lambda: self.refresh(silent=True)); self.timer.start(CHECK_EVERY_MIN * 60 * 1000)
        self.ticker = QTimer(self); self.ticker.timeout.connect(self.tick); self.ticker.start(33)

    # ---- helpers
    def run(self, fn, ok):
        w = Worker(fn); self.workers.append(w)
        w.prog.connect(self.bar.setValue); w.done.connect(ok)
        w.fail.connect(lambda m: (self.lbl.setText("Error"), QMessageBox.warning(self, "Error", m)))
        w.start()

    def tpl(self): return self.work

    # ---- video + preview
    def open_video(self):
        p, _ = QFileDialog.getOpenFileName(self, "Video", "", "Video (*.mp4 *.mov *.mkv *.avi *.webm)")
        if not p: return
        self.video, self.lines, self.starts = p, [], []; self.clist.clear(); self.active = -1
        vw, vh, dur = self.vinfo = probe(p)
        self.scene.setSceneRect(0, 0, vw, vh); self.vitem.setSize(QSizeF(vw, vh))
        if self.cap: self.scene.removeItem(self.cap)
        self.cap = CaptionItem(vw, vh, self.cstyle, self.style_changed); self.scene.addItem(self.cap)
        self.cap.tpl = self.work; self.reset_style()
        self.player.setSource(QUrl.fromLocalFile(p)); self.player.pause(); self.view.fit()
        self.frame = grab_frame(p, min(1.0, dur / 3)); self.thumb = (146, max(84, min(260, int(146 * vh / vw)))); self.fill_grid()
        self.timeline.set_data([], dur); self.lbl_proj.setText(f"{os.path.basename(p)}  -  {vw}x{vh}")
        self.lbl.setText("Video loaded. Press Generate captions.")

    def toggle_play(self):
        self.player.pause() if self.player.playbackState() == QMediaPlayer.PlayingState else self.player.play()

    def jump(self, row):
        if 0 <= row < len(self.lines): self.player.setPosition(int(self.lines[row]["start"] * 1000))

    def mark(self, i, on):
        if 0 <= i < self.clist.count():
            w = self.clist.itemWidget(self.clist.item(i))
            if w: w.set_active(on)

    def tick(self):
        pos, dur = self.player.position(), self.player.duration()
        if not self.seek.isSliderDown(): self.seek.setValue(pos)
        f = lambda ms: f"{ms // 60000}:{ms // 1000 % 60:02}"
        self.lbl_time.setText(f"{f(pos)} / {f(max(dur, 0))}")
        t = pos / 1000; self.timeline.t = t; self.timeline.update()
        if not self.cap: return
        act = -1
        if not self.lines:
            words = [{"w": w, "s": 99, "e": 99} for w in "Caption yahan dikhega".split()]; ghost = True
        else:
            idx = bisect.bisect_right(self.starts, t) - 1
            if idx >= 0 and t <= self.lines[idx]["end"] + 0.05: words, ghost, act = self.lines[idx]["words"], False, idx
            else:   # between lines: faint preview of the next line so it can still be dragged
                words, ghost = self.lines[min(idx + 1, len(self.lines) - 1)]["words"], True
        if act != self.active:
            self.mark(self.active, False); self.mark(act, True); self.active = act
            if act >= 0 and self.player.playbackState() == QMediaPlayer.PlayingState and not self.clist.hasFocus():
                self.clist.scrollToItem(self.clist.item(act), QAbstractItemView.PositionAtCenter)
        self.cap.set_content(words, -1 if ghost else t, ghost)

    def size_changed(self, v): self.cstyle["size"] = v / 1000

    def style_changed(self):   # wheel-resize on the caption -> keep slider in sync
        self.size_sl.blockSignals(True); self.size_sl.setValue(round(self.cstyle["size"] * 1000)); self.size_sl.blockSignals(False)

    def reset_style(self):
        t = self.work
        self.cstyle["size"] = min(0.15, max(0.015, t["size"] / 1080))
        self.cstyle["cx"], self.cstyle["cy"] = 0.5, POS_Y.get(t.get("position", "bottom"), 0.86)
        if self.cap: self.cap.setPos(self.cstyle["cx"] * self.cap.vw, self.cstyle["cy"] * self.cap.vh)
        self.style_changed()

    # ---- templates / customize / presets
    def set_work(self, tpl):
        self.work = dict(tpl)
        if self.cap: self.cap.tpl = self.work
        self.sync_custom()

    def set_mode(self, m): self.mode = m; self.fill_grid()

    def fill_grid(self):
        W, H = self.thumb; src = self.templates if self.mode == "templates" else self.presets
        q = self.search.text().strip().lower(); cur = self.work.get("id"); sel = None
        self.grid.blockSignals(True); self.grid.clear()
        self.grid.setIconSize(QSize(W, H)); self.grid.setGridSize(QSize(W + 14, H + 42))
        for t in src:
            if q and q not in t["name"].lower(): continue
            badge = t.get("badge") or ("New" if t["id"] in self.new_ids else "")
            it = QListWidgetItem(QIcon(make_thumb(t, self.frame, W, H, badge)), t["name"])
            it.setData(Qt.UserRole, t); it.setTextAlignment(Qt.AlignHCenter); self.grid.addItem(it)
            if t["id"] == cur: sel = it
        self.grid.blockSignals(False)
        if sel: self.grid.setCurrentItem(sel)

    def sync_custom(self):
        w = self.work; ws = (self.fontbox, self.outsl, self.chk_up, self.chk_bold)
        for x in ws: x.blockSignals(True)
        self.fontbox.setCurrentFont(QFont(w["font"])); self.outsl.setValue(int(w.get("outline_w", 3)))
        self.chk_up.setChecked(bool(w.get("upper"))); self.chk_bold.setChecked(bool(w.get("bold")))
        for x in ws: x.blockSignals(False)
        for key, b in self.colbtn.items():
            col = w.get("box") if (key == "outline" and w.get("box")) else w.get(key, "#000000")
            b.setStyleSheet(f"background:{col}; color:{'#000' if QColor(col).lightness() > 140 else '#fff'}; border:1px solid #444;")

    def cust(self, key, val): self.work[key] = val

    def pick_color(self, key):
        real = "box" if (key == "outline" and self.work.get("box")) else key
        c = QColorDialog.getColor(QColor(self.work.get(real, "#000000")), self, "Pick a colour")
        if c.isValid(): self.work[real] = c.name().upper(); self.sync_custom()

    def save_preset(self):
        name, ok = QInputDialog.getText(self, "Save preset", "Preset name:")
        if not (ok and name.strip()): return
        self.presets.insert(0, dict(self.work, id=f"preset-{int(time.time())}", name=name.strip(), published="9999", badge=""))
        json.dump(self.presets, open(PRESETS, "w", encoding="utf-8"))
        self.btn_pre.setChecked(True); self.mode = "presets"; self.fill_grid()

    # ---- captions
    def generate(self):
        if not self.video: return QMessageBox.information(self, "Hinglish Captioner", "Open a video first.")
        size = self.model.currentText(); need = not model_ready(size)
        self.lbl.setText(f"Downloading the {size} model (one-time), then transcribing..." if need else "Transcribing...")
        self.bar.setValue(0)
        def job(prog):
            if need: download_model(size, prog); prog(0)
            return transcribe(self.video, size, prog)
        self.run(job, self.got_lines)

    def got_lines(self, res):
        lines, dev = res; self.lines = lines; self.starts = [l["start"] for l in lines]; self.active = -1
        self.bar.setValue(100); self.lbl.setText(f"{len(lines)} caption lines ready ({dev}). Press Play to preview.")
        self.clist.clear()
        for i, l in enumerate(lines):
            card = CaptionCard(i, l); card.picked.connect(self.jump)
            card.edit.editingFinished.connect(lambda i=i, c=card: self.line_edited(i, c.edit.text()))
            it = QListWidgetItem(); it.setSizeHint(QSize(0, 70)); self.clist.addItem(it); self.clist.setItemWidget(it, card)
        self.timeline.set_data(lines, self.vinfo[2] or (lines[-1]["end"] if lines else 1))

    def line_edited(self, row, text):
        l = self.lines[row]; new = text.split(); old = l["words"]
        if not new or text == l["text"]: return
        if len(new) == len(old):
            for w, t in zip(old, new): w["w"] = t
        else:  # word count changed: redistribute timing evenly across the line
            d = (l["end"] - l["start"]) / len(new)
            l["words"] = [{"w": t, "s": l["start"] + i * d, "e": l["start"] + (i + 1) * d} for i, t in enumerate(new)]
        l["text"] = text; self.timeline.update()

    # ---- online templates
    def refresh(self, silent=False):
        def ok(res):
            old = {t["id"] for t in self.cache["online"]}
            new = [t for t in res if t["id"] not in old]
            self.cache = {"online": res, "seen": list(old)}
            json.dump(self.cache, open(CACHE, "w", encoding="utf-8"))
            if old: self.new_ids = {t["id"] for t in new}
            self.templates = merge(res); self.fill_grid()
            self.lbl_t.setText(f"{len(new)} new online template(s)!" if (new and old) else f"Up to date ({datetime.now():%H:%M})")
        w = Worker(lambda p: fetch_templates()); self.workers.append(w); w.done.connect(ok)
        w.fail.connect(lambda m: self.lbl_t.setText("Template feed unreachable - showing built-in templates")); w.start()

    # ---- export
    def exp_srt(self):
        if not self.lines: return QMessageBox.information(self, "Hinglish Captioner", "Generate captions first.")
        p, _ = QFileDialog.getSaveFileName(self, "Save", "captions.srt", "SubRip (*.srt)")
        if p: write_srt(self.lines, p)

    def exp_mp4(self):
        if not (self.lines and self.video): return QMessageBox.information(self, "Hinglish Captioner", "Generate captions first.")
        if not shutil.which("ffmpeg"): return QMessageBox.warning(self, "ffmpeg missing", "Install ffmpeg and add it to PATH.")
        p, _ = QFileDialog.getSaveFileName(self, "Save", "captioned.mp4", "MP4 (*.mp4)")
        if not p: return
        vw, vh, dur = self.vinfo; target = RES[self.qgroup.checkedButton().text()]
        k = target / max(vw, vh) if target else 1.0
        ow, oh = int(round(vw * k / 2) * 2), int(round(vh * k / 2) * 2)
        tmp = tempfile.mkdtemp()
        write_ass(self.lines, self.work, self.cstyle, os.path.join(tmp, "c.ass"), vw, vh, ow, oh)
        self.lbl.setText(f"Rendering {ow}x{oh}..."); self.bar.setValue(0)
        hevc = self.codec.currentIndex() == 1
        self.run(lambda prog: burn(self.video, tmp, (ow, oh), (vw, vh), dur, hevc, p, prog),
                 lambda _: (self.bar.setValue(100), self.lbl.setText("Saved: " + p)))

if __name__ == "__main__":
    a = QApplication(sys.argv); a.setStyle("Fusion"); a.setStyleSheet(QSS)
    steps = pending_steps()
    if steps: SetupDialog(steps).exec()      # start-up screen: GPU libraries + model, with live percentages
    w = App(); w.show(); sys.exit(a.exec())
