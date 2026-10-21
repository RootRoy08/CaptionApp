"""Hinglish Captioner - Windows desktop app.
Video -> word-synced Hinglish captions (faster-whisper) -> styled with online templates -> SRT / ASS / burned-in MP4.
"""
import os, re, sys, json, bisect, shutil, subprocess, tempfile, urllib.request
from collections import deque
from datetime import datetime
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QUrl, QRectF, QPointF, QSizeF, QSize
from PySide6.QtGui import QFont, QFontMetricsF, QPainter, QPainterPath, QPen, QColor, QPixmap, QIcon
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QGraphicsVideoItem
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QTableWidget, QTableWidgetItem, QListWidget, QListWidgetItem, QComboBox,
    QProgressBar, QMessageBox, QHeaderView, QGraphicsView, QGraphicsScene, QGraphicsItem, QSlider)

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
# Roman-script prompt strongly biases Whisper to output Hinglish (Roman Hindi + English words)
PROMPT = "Haan toh guys, aaj hum baat karenge ki yeh video kaise banate hain. Bilkul simple hai, chalo start karte hain."

# ------------------------------------------------------------------ transcription
def hinglish(word):
    """If Whisper returned Devanagari, convert to natural Roman Hinglish."""
    if not re.search(r"[\u0900-\u097F]", word):
        return word
    from indic_transliteration import sanscrit
    from indic_transliteration.sanscrit import transliterate
    s = transliterate(word, sanscrit.DEVANAGARI, sanscrit.ITRANS)
    for a, b in [("aa", "a"), ("ii", "i"), ("uu", "u"), ("~N", "n"), ("N", "n"), ("M", "n"), (".n", "n"),
                 (".h", ""), ("R^i", "ri"), ("Sh", "sh"), ("sh", "sh"), ("T", "t"), ("D", "d"), ("w", "v")]:
        s = s.replace(a, b)
    return s.lower()

def _run(path, model_size, device, ctype, progress):
    from faster_whisper import WhisperModel
    model = WhisperModel(model_size, device=device, compute_type=ctype)
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
def make_icon(tpl):
    """Small thumbnail: 'A' in the text colour, 'a' in the highlight colour, on a video-grey tile."""
    W, H, px = 88, 52, 28
    pm = QPixmap(W, H); pm.fill(QColor("#3c3c3c")); p = QPainter(pm); p.setRenderHint(QPainter.Antialiasing)
    f = make_font(tpl["font"], px, bool(tpl.get("bold"))); fm = QFontMetricsF(f)
    txt = disp("Aa", tpl); w1 = fm.horizontalAdvance(txt[0]); x = (W - fm.horizontalAdvance(txt)) / 2; y = (H - px) / 2
    if tpl.get("box"):
        p.setPen(Qt.NoPen); p.setBrush(QColor(tpl["box"])); p.drawRect(QRectF(x - 7, y - 2, fm.horizontalAdvance(txt) + 14, px + 4))
    parts = [(txt[0], x, tpl["text"]), (txt[1], x + w1, tpl["highlight"])]
    paths = []
    for ch, cx, col in parts:
        path = QPainterPath(); path.addText(cx, y + fm.ascent(), f, ch); paths.append((path, col))
    if not tpl.get("box"):
        p.setPen(QPen(QColor(tpl["outline"]), max(1.5, tpl.get("outline_w", 3) * 0.7), Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        for path, _ in paths: p.drawPath(path)
    p.setPen(Qt.NoPen)
    for path, col in paths: p.setBrush(QColor(col)); p.drawPath(path)
    p.end(); return QIcon(pm)

RES = {"Original size": 0, "1080p": 1920, "1440p": 2560, "4K (2160p)": 3840}   # target = long side in px
POS_Y = {"bottom": 0.86, "middle": 0.5, "top": 0.12}

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hinglish Captioner"); self.resize(1200, 780)
        self.video, self.vinfo, self.lines, self.starts, self.workers, self.cap = None, (1920, 1080, 0.0), [], [], [], None
        self.style = {"cx": 0.5, "cy": 0.86, "size": 0.05}
        self.cache = load_cache(); self.templates = merge(self.cache["online"])

        self.player = QMediaPlayer(); self.audio = QAudioOutput(); self.player.setAudioOutput(self.audio)
        self.scene = QGraphicsScene(self); self.vitem = QGraphicsVideoItem(); self.scene.addItem(self.vitem)
        self.player.setVideoOutput(self.vitem)
        self.view = VideoView(self.scene); self.view.setMinimumHeight(340)
        self.view.setStyleSheet("background:#000; border:0"); self.view.setRenderHint(QPainter.Antialiasing)

        self.btn_open = QPushButton("1. Open video"); self.btn_open.clicked.connect(self.open_video)
        self.model = QComboBox(); self.model.addItems(["large-v3", "medium", "small"])
        self.model.setToolTip("large-v3 = best accuracy (GPU recommended); small = fastest")
        self.btn_gen = QPushButton("2. Generate Hinglish captions"); self.btn_gen.clicked.connect(self.generate)
        self.btn_play = QPushButton("Play"); self.btn_play.clicked.connect(self.toggle_play)
        self.seek = QSlider(Qt.Horizontal); self.seek.sliderMoved.connect(self.player.setPosition)
        self.player.durationChanged.connect(lambda d: self.seek.setRange(0, d))
        self.player.playbackStateChanged.connect(lambda s: self.btn_play.setText(
            "Pause" if s == QMediaPlayer.PlayingState else "Play"))
        self.lbl_time = QLabel("0:00 / 0:00")
        self.size_sl = QSlider(Qt.Horizontal); self.size_sl.setRange(15, 150); self.size_sl.setValue(50)
        self.size_sl.valueChanged.connect(self.size_changed)
        self.btn_reset = QPushButton("Reset position"); self.btn_reset.clicked.connect(self.reset_style)
        self.lbl = QLabel("No video loaded"); self.bar = QProgressBar()
        self.table = QTableWidget(0, 3); self.table.setHorizontalHeaderLabels(["Start", "End", "Caption (editable)"])
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.itemChanged.connect(self.edited); self.table.cellClicked.connect(self.jump)
        self.tlist = QListWidget(); self.tlist.setIconSize(QSize(88, 52)); self.tlist.currentItemChanged.connect(self.apply_tpl)
        self.btn_ref = QPushButton("Refresh templates"); self.btn_ref.clicked.connect(self.refresh)
        self.lbl_t = QLabel("")
        self.btn_gpu = QPushButton("Set up NVIDIA GPU (one-time ~1.3 GB download)")
        self.btn_gpu.clicked.connect(self.setup_gpu)
        if gpu_ready(): self.btn_gpu.setText("NVIDIA GPU support installed")
        self.res = QComboBox(); self.res.addItems(list(RES))
        self.res.setToolTip("Output resolution (long side). Sources smaller than the target are upscaled.")
        self.codec = QComboBox(); self.codec.addItems(["H.264 (most compatible)", "H.265 (smaller 4K files)"])
        self.btn_srt = QPushButton("Export .srt"); self.btn_srt.clicked.connect(self.exp_srt)
        self.btn_mp4 = QPushButton("3. Export video with styled captions"); self.btn_mp4.clicked.connect(self.exp_mp4)

        left, right = QVBoxLayout(), QVBoxLayout()
        top = QHBoxLayout(); [top.addWidget(x) for x in (self.btn_open, self.model, self.btn_gen)]
        tr = QHBoxLayout(); tr.addWidget(self.btn_play); tr.addWidget(self.seek, 1); tr.addWidget(self.lbl_time)
        st = QHBoxLayout(); st.addWidget(QLabel("Text size")); st.addWidget(self.size_sl, 1); st.addWidget(self.btn_reset)
        hint = QLabel("Drag the caption on the video to move it. Mouse wheel over it, or the slider, changes size.")
        hint.setStyleSheet("color:gray")
        left.addLayout(top); left.addWidget(self.view, 5); left.addLayout(tr); left.addLayout(st); left.addWidget(hint)
        left.addWidget(self.lbl); left.addWidget(self.bar); left.addWidget(self.table, 2)
        right.addWidget(QLabel("<b>Latest templates (top 10)</b>")); right.addWidget(self.tlist, 1)
        right.addWidget(self.lbl_t); right.addWidget(self.btn_ref)
        right.addWidget(self.btn_gpu); right.addWidget(QLabel("<b>Export</b>")); right.addWidget(QLabel("Resolution")); right.addWidget(self.res)
        right.addWidget(QLabel("Codec")); right.addWidget(self.codec)
        right.addWidget(self.btn_srt); right.addWidget(self.btn_mp4)
        root = QHBoxLayout(self); root.addLayout(left, 3); root.addLayout(right, 1)

        self.show_templates(); self.refresh(silent=True)
        self.timer = QTimer(self); self.timer.timeout.connect(lambda: self.refresh(silent=True))
        self.timer.start(CHECK_EVERY_MIN * 60 * 1000)
        self.ticker = QTimer(self); self.ticker.timeout.connect(self.tick); self.ticker.start(33)

    # ---- worker helper
    def run(self, fn, ok):
        w = Worker(fn); self.workers.append(w)
        w.prog.connect(self.bar.setValue); w.done.connect(ok)
        w.fail.connect(lambda m: (self.lbl.setText("Error"), QMessageBox.warning(self, "Error", m)))
        w.start()

    # ---- video + preview
    def open_video(self):
        p, _ = QFileDialog.getOpenFileName(self, "Video", "", "Video (*.mp4 *.mov *.mkv *.avi *.webm)")
        if not p: return
        self.video, self.lines, self.starts = p, [], []; self.table.setRowCount(0)
        vw, vh, dur = self.vinfo = probe(p)
        self.scene.setSceneRect(0, 0, vw, vh); self.vitem.setSize(QSizeF(vw, vh))
        if self.cap: self.scene.removeItem(self.cap)
        self.cap = CaptionItem(vw, vh, self.style, self.style_changed); self.scene.addItem(self.cap)
        self.cap.tpl = self.tpl(); self.reset_style()
        self.player.setSource(QUrl.fromLocalFile(p)); self.player.pause(); self.view.fit()
        self.lbl.setText(f"{os.path.basename(p)}  -  {vw}x{vh}")

    def toggle_play(self):
        self.player.pause() if self.player.playbackState() == QMediaPlayer.PlayingState else self.player.play()

    def jump(self, row, _col):
        if row < len(self.lines): self.player.setPosition(int(self.lines[row]["start"] * 1000))

    def tick(self):
        pos = self.player.position(); dur = self.player.duration()
        if not self.seek.isSliderDown(): self.seek.setValue(pos)
        f = lambda ms: f"{ms // 60000}:{ms // 1000 % 60:02}"
        self.lbl_time.setText(f"{f(pos)} / {f(max(dur, 0))}")
        if not self.cap: return
        t = pos / 1000
        if not self.lines:
            words = [{"w": w, "s": 99, "e": 99} for w in "Caption yahan dikhega".split()]; ghost = True
        else:
            idx = bisect.bisect_right(self.starts, t) - 1
            if idx >= 0 and t <= self.lines[idx]["end"] + 0.05: words, ghost = self.lines[idx]["words"], False
            else:   # between lines: faint preview of the next line so it can still be dragged
                nxt = self.lines[min(idx + 1, len(self.lines) - 1)]; words, ghost = nxt["words"], True
        self.cap.set_content(words, -1 if ghost else t, ghost)

    def size_changed(self, v):
        self.style["size"] = v / 1000

    def style_changed(self):   # wheel-resize on the caption -> keep slider in sync
        self.size_sl.blockSignals(True); self.size_sl.setValue(round(self.style["size"] * 1000)); self.size_sl.blockSignals(False)

    def reset_style(self):
        t = self.tpl()
        self.style["size"] = min(0.15, max(0.015, t["size"] / 1080))
        self.style["cx"], self.style["cy"] = 0.5, POS_Y.get(t.get("position", "bottom"), 0.86)
        if self.cap: self.cap.setPos(self.style["cx"] * self.cap.vw, self.style["cy"] * self.cap.vh)
        self.style_changed()

    def apply_tpl(self, cur, _prev=None):
        if self.cap and cur: self.cap.tpl = cur.data(Qt.UserRole)

    def setup_gpu(self):
        if gpu_ready(): return QMessageBox.information(self, "GPU", "GPU support is already installed.")
        self.lbl.setText("Downloading NVIDIA GPU libraries (~1.3 GB)..."); self.bar.setValue(0)
        self.run(install_gpu_libs, lambda _: (self.bar.setValue(100), self.btn_gpu.setText("NVIDIA GPU support installed"),
                 self.lbl.setText("GPU ready - generate captions again")))

    # ---- captions
    def generate(self):
        if not self.video: return QMessageBox.information(self, "Hinglish Captioner", "Open a video first.")
        self.lbl.setText("Transcribing (first run downloads the model)..."); self.bar.setValue(0)
        self.run(lambda prog: transcribe(self.video, self.model.currentText(), prog), self.got_lines)

    def got_lines(self, res):
        lines, dev = res
        self.lines = lines; self.starts = [l["start"] for l in lines]
        self.bar.setValue(100); self.lbl.setText(f"{len(lines)} caption lines ready ({dev}) - press Play to preview")
        self.table.blockSignals(True); self.table.setRowCount(len(lines))
        for i, l in enumerate(lines):
            for c, v in enumerate((ts_srt(l["start"]), ts_srt(l["end"]), l["text"])):
                it = QTableWidgetItem(v)
                if c < 2: it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(i, c, it)
        self.table.blockSignals(False)

    def edited(self, item):
        if item.column() != 2: return
        l = self.lines[item.row()]; new = item.text().split(); old = l["words"]
        if len(new) == len(old):
            for w, t in zip(old, new): w["w"] = t
        else:  # word count changed: redistribute timing evenly across the line
            d = (l["end"] - l["start"]) / max(len(new), 1)
            l["words"] = [{"w": t, "s": l["start"] + i * d, "e": l["start"] + (i + 1) * d} for i, t in enumerate(new)]
        l["text"] = item.text()

    # ---- templates
    def show_templates(self):
        cur = self.tlist.currentItem(); cur_id = cur.data(Qt.UserRole)["id"] if cur else None
        self.tlist.blockSignals(True); self.tlist.clear()
        seen, online = set(self.cache.get("seen", [])), {t["id"] for t in self.cache["online"]}
        row = 0
        for i, t in enumerate(self.templates):
            new = bool(seen) and t["id"] in online and t["id"] not in seen
            it = QListWidgetItem(make_icon(t), ("NEW  " if new else "") + t["name"])
            it.setData(Qt.UserRole, t); self.tlist.addItem(it)
            if t["id"] == cur_id: row = i
        self.tlist.blockSignals(False); self.tlist.setCurrentRow(row)

    def refresh(self, silent=False):
        def ok(res):
            old = {t["id"] for t in self.cache["online"]}
            new = [t for t in res if t["id"] not in old]
            self.cache = {"online": res, "seen": list(old)}
            json.dump(self.cache, open(CACHE, "w", encoding="utf-8"))
            self.templates = merge(res); self.show_templates()
            self.lbl_t.setText(f"{len(new)} new online template(s)!" if new else f"Up to date ({datetime.now():%H:%M})")
        w = Worker(lambda p: fetch_templates()); self.workers.append(w); w.done.connect(ok)
        w.fail.connect(lambda m: self.lbl_t.setText("Template feed unreachable - showing built-in templates")); w.start()

    def tpl(self):
        it = self.tlist.currentItem(); return it.data(Qt.UserRole) if it else BUNDLED[0]

    # ---- export
    def exp_srt(self):
        if not self.lines: return
        p, _ = QFileDialog.getSaveFileName(self, "Save", "captions.srt", "SubRip (*.srt)")
        if p: write_srt(self.lines, p)

    def exp_mp4(self):
        if not (self.lines and self.video): return QMessageBox.information(self, "Hinglish Captioner", "Generate captions first.")
        if not shutil.which("ffmpeg"): return QMessageBox.warning(self, "ffmpeg missing", "Install ffmpeg and add it to PATH.")
        p, _ = QFileDialog.getSaveFileName(self, "Save", "captioned.mp4", "MP4 (*.mp4)")
        if not p: return
        vw, vh, dur = self.vinfo; target = RES[self.res.currentText()]
        k = target / max(vw, vh) if target else 1.0
        ow, oh = int(round(vw * k / 2) * 2), int(round(vh * k / 2) * 2)
        tmp = tempfile.mkdtemp()
        write_ass(self.lines, self.tpl(), self.style, os.path.join(tmp, "c.ass"), vw, vh, ow, oh)
        self.lbl.setText(f"Rendering {ow}x{oh}..."); self.bar.setValue(0)
        hevc = self.codec.currentIndex() == 1
        self.run(lambda prog: burn(self.video, tmp, (ow, oh), (vw, vh), dur, hevc, p, prog),
                 lambda _: (self.bar.setValue(100), self.lbl.setText("Saved: " + p)))

if __name__ == "__main__":
    a = QApplication(sys.argv); w = App(); w.show(); sys.exit(a.exec())
