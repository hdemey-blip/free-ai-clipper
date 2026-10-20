"""Chemin rapide pour l'agent : l'audio entier, puis seulement les secondes de vidéo des clips retenus.

Une vidéo YouTube de 50 min en 1080p pèse près de 2 Go (plus de 4 min de téléchargement). On télécharge
l'audio léger, on transcrit et on choisit les moments, puis on ne prend sur YouTube que les octets des clips.

Les formats DASH de YouTube commencent par un index (boîte sidx) qui donne la position en octets de chaque
fragment de quelques secondes. On lit cet index, puis curl télécharge les fragments voulus par plages, en
parallèle (YouTube bride chaque connexion). ffmpeg ne sait pas faire ça : pour atteindre la 25e minute, il
enchaîne des centaines de petites requêtes, fragment par fragment.

yt-dlp lit YouTube avec son client « lecteur intégré » (web_embedded), qui donne les formats DASH sans compte.
"""

import dataclasses
import json
import struct
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import demusic, highlights, ingest, reframe, render, transcribe
from .captions import PRESETS
from .util import FFMPEG, atomic_path, write_json_atomic

SECTION_PAD = 0.5   # marge de vidéo découpée avant et après le clip
CHUNK = 2**20       # octets par requête
WORKERS = 12        # requêtes en parallèle sur un même fichier


class _Silent:
    def debug(self, msg): pass
    def warning(self, msg): pass
    def error(self, msg): pass


def ydl_options(**extra) -> dict:
    return {"quiet": True, "no_warnings": True, "noprogress": True, "logger": _Silent(), **ingest.YTDLP_JS,
            "extractor_args": {"youtube": {"player_client": ["web_embedded"]}}, **extra}


class Stream:
    """Un format DASH de YouTube (vidéo seule ou audio seul), lu par plages d'octets."""

    def __init__(self, fmt: dict):
        self.fmt = fmt
        self.url = fmt["url"]
        self.headers = sum((["-H", f"{k}: {v}"] for k, v in (fmt.get("http_headers") or {}).items()), [])
        self._index: tuple[bytes, list[tuple[float, float, int, int]]] | None = None

    def get(self, a: int, b: int) -> bytes:
        """Octets a à b inclus. La première requête est parfois refusée (403) : curl réessaie."""
        out = subprocess.run(["curl", "-sL", "--fail", "--retry", "5", "--retry-all-errors", "--retry-delay", "1",
                              "-r", f"{a}-{b}", *self.headers, self.url], capture_output=True, timeout=300)
        if out.returncode != 0 or len(out.stdout) != b - a + 1:
            raise RuntimeError(f"YouTube a refusé les octets {a}-{b} (curl {out.returncode}, "
                               f"{len(out.stdout)} octets reçus)")
        return out.stdout

    def get_many(self, a: int, b: int, tmp_dir: Path) -> bytes:
        """Octets a à b par morceaux de CHUNK, WORKERS requêtes à la fois, dans un seul curl : chaque requête
        démarre lentement (0,2 à 6 Mo/s selon les cas) et lancer un programme coûte 1,5 s sur cette machine."""
        cuts = [(c, min(b, c + CHUNK - 1)) for c in range(a, b + 1, CHUNK)]
        parts = [tmp_dir / f".part_{id(self)}_{a}_{i}" for i in range(len(cuts))]
        # requêtes décrites dans un fichier de configuration : l'adresse fait 2 Ko et Windows limite la ligne
        # de commande à 32 Ko
        quote = lambda s: '"' + str(s).replace("\\", "/").replace('"', '\\"') + '"'
        blocks = []
        for (x, y), part in zip(cuts, parts):
            lines = ["silent", "location", "fail", "retry = 5", "retry-all-errors", "retry-delay = 1",
                     f"range = {x}-{y}", f"url = {quote(self.url)}", f"output = {quote(part)}"]
            lines += [f"header = {quote(h)}" for h in self.headers[1::2]]
            blocks.append("\n".join(lines))
        config = tmp_dir / f".curl_{id(self)}_{a}.txt"
        config.write_text("\nnext\n".join(blocks) + "\n", encoding="utf-8")
        try:
            out = subprocess.run(["curl", "--parallel", "--parallel-max", str(WORKERS), "-K", str(config)],
                                 capture_output=True, timeout=600)
            sizes = [p.stat().st_size if p.exists() else 0 for p in parts]
            if out.returncode != 0 or sizes != [y - x + 1 for x, y in cuts]:
                raise RuntimeError(f"YouTube a refusé une partie des octets {a}-{b} (curl {out.returncode})")
            return b"".join(p.read_bytes() for p in parts)
        finally:
            for p in [*parts, config]:
                p.unlink(missing_ok=True)

    def index(self) -> tuple[bytes, list[tuple[float, float, int, int]]]:
        """(segment d'initialisation, fragments [(début s, fin s, premier octet, dernier octet)])."""
        if self._index:
            return self._index
        head = self.get(0, 65535)
        pos = 0
        while pos + 8 <= len(head):
            size, kind = struct.unpack(">I4s", head[pos:pos + 8])
            if kind == b"sidx":
                break
            pos += size
        else:
            raise RuntimeError("Pas d'index sidx dans ce format")
        if pos + size > len(head):
            head += self.get(len(head), pos + size - 1)
        box = head[pos + 8:pos + size]
        version, timescale = box[0], struct.unpack(">I", box[8:12])[0]
        if version == 0:
            t, first = struct.unpack(">II", box[12:20])
            p = 20
        else:
            t, first = struct.unpack(">QQ", box[12:28])
            p = 28
        count = struct.unpack(">H", box[p + 2:p + 4])[0]
        p += 4
        byte, frags = pos + size + first, []
        for _ in range(count):
            word, dur = struct.unpack(">II", box[p:p + 8])
            length = word & 0x7FFFFFFF
            frags.append((t / timescale, (t + dur) / timescale, byte, byte + length - 1))
            byte += length
            t += dur
            p += 12
        self._index = (head[:pos], frags)  # ftyp + moov ; le sidx n'est pas utile au fichier local
        return self._index

    def section(self, t0: float, t1: float, dest: Path) -> None:
        """Fichier local avec les fragments qui couvrent [t0, t1]. Les horodatages d'origine sont gardés."""
        init, frags = self.index()
        inside = [f for f in frags if f[1] > t0 and f[0] < t1] or frags[-1:]
        with atomic_path(dest) as tmp:
            tmp.write_bytes(init + self.get_many(inside[0][2], inside[-1][3], dest.parent))

    def whole(self, dest: Path) -> None:
        init, frags = self.index()
        with atomic_path(dest) as tmp:
            tmp.write_bytes(init + self.get_many(frags[0][2], frags[-1][3], dest.parent))


def _start(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=start_time", "-of", "csv=p=0",
                          str(path)], capture_output=True, text=True, check=True)
    return float(out.stdout.strip() or 0)


def fetch(url: str, job_dir: Path, log=print) -> tuple[Stream, Stream, Path]:
    """Une seule visite de la page YouTube. Renvoie les flux vidéo et audio des clips et l'audio de
    transcription en WAV 16 kHz, tiré de la version la plus légère de l'audio."""
    import yt_dlp
    with yt_dlp.YoutubeDL(ydl_options()) as ydl:
        formats = ydl.extract_info(url, download=False)["formats"]
    dash = [f for f in formats if f.get("ext") in ("mp4", "m4a") and f.get("url") and "drc" not in f["format_id"]
            and f.get("protocol", "https").startswith("http")]
    # H.264 d'abord : décodé plus vite que le VP9 ou l'AV1 pour la même image
    videos = [f for f in dash if f.get("acodec") == "none" and (f.get("height") or 0) <= 1080]
    videos.sort(key=lambda f: ((f.get("height") or 0), str(f.get("vcodec")).startswith("avc1"), f.get("tbr") or 0))
    # piste d'origine : YouTube ajoute parfois des doublages automatiques (anglais...) en pistes séparées
    audios = [f for f in dash if f.get("vcodec") == "none" and f.get("acodec") not in (None, "none")]
    audios.sort(key=lambda f: (f.get("language_preference") or 0, f.get("abr") or 0))
    if not videos or not audios:
        raise RuntimeError("Pas de format DASH mp4/m4a pour cette vidéo")
    video, audio = Stream(videos[-1]), Stream(audios[-1])
    wav = job_dir / "audio.wav"
    if not wav.exists():
        t = time.time()
        original = audios[-1].get("language_preference") or 0
        light_fmt = min((f for f in audios if (f.get("language_preference") or 0) == original),
                        key=lambda f: f.get("abr") or 0)
        light = job_dir / "audio_light.m4a"
        Stream(light_fmt).whole(light)
        with atomic_path(wav) as tmp:
            subprocess.run([*FFMPEG, "-i", str(light), "-vn", "-ac", "1", "-ar", "16000", str(tmp)], check=True)
        light.unlink(missing_ok=True)
        log(f"  audio de transcription en {time.time() - t:.0f} s ({light_fmt.get('abr') or '?'} kb/s)")
    return video, audio, wav


def subtitle_band(src: Path, meta: dict) -> tuple[int, int, int, int] | None:
    """Zone (x, y, largeur, hauteur) des sous-titres incrustés dans la vidéo source, ou None.

    Un sous-titre incrusté, c'est du texte clair cerné de sombre, dans le bas de l'image, sur une même bande
    d'une image à l'autre. On cherche les lignes où ce motif revient sur au moins un tiers des images."""
    import cv2
    width = 480
    step = max(1, round(meta["fps"] * 1.5))
    masks = []
    for frame in reframe.read_frames(src, 0, meta["duration"], meta, width=width, step=step):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edge = cv2.dilate((gray < 70).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2)
        masks.append((gray > 210) & (edge > 0))
    if len(masks) < 4:
        return None
    masks = np.array(masks)
    density = masks.mean(axis=2)
    h = density.shape[1]
    lo = int(h * 0.55)
    frequent = (density[:, lo:] > 0.012).mean(axis=0) >= 0.35
    best, run_start = (0, 0), None
    for i, on in enumerate(np.append(frequent, False)):
        if on and run_start is None:
            run_start = i
        elif not on and run_start is not None:
            if i - run_start > best[1] - best[0]:
                best = (run_start, i)
            run_start = None
    band = best[1] - best[0]
    if not h * 0.03 <= band <= h * 0.2:
        return None
    scale = meta["height"] / h
    pad = band * 0.35
    y0 = max(0, int((lo + best[0] - pad) * scale))
    y1 = min(meta["height"], int((lo + best[1] + pad) * scale))
    # largeur : colonnes où le texte apparaît dans la bande, sur l'ensemble des images
    cols = masks[:, lo + best[0]:lo + best[1], :].any(axis=1).mean(axis=0) > 0.05
    xs = np.flatnonzero(cols)
    sx = meta["width"] / width
    x0 = max(0, int((xs[0] - band) * sx)) if len(xs) else 0
    x1 = min(meta["width"], int((xs[-1] + band) * sx)) if len(xs) else meta["width"]
    return x0, y0, x1 - x0, y1 - y0


def cut_section(video: Stream, audio: Stream, start: float, dur: float, dest: Path
                ) -> tuple[Path, tuple[int, int, int, int] | None]:
    """[start, start + dur] de la vidéo YouTube, image et son, découpé à l'image près. Les sous-titres
    incrustés dans la source sont floutés dans le même passage. Renvoie le fichier et la zone floutée.

    Réencodage presque sans perte (CRF 12, ultrafast) : une coupe sans réencodage tomberait sur l'image clé
    précédente et décalerait les sous-titres."""
    v, a = dest.with_name(dest.stem + "_v.mp4"), dest.with_name(dest.stem + "_a.m4a")
    try:
        with ThreadPoolExecutor(2) as pool:
            list(pool.map(lambda job: job[0].section(start, start + dur, job[1]), ((video, v), (audio, a))))
        meta = ingest.probe(v)
        band = subtitle_band(v, meta)
        if band:
            x, y, w, h = band
            vf = ["-filter_complex", f"[0:v]split[a][b];[b]crop={w}:{h}:{x}:{y},gblur=sigma=28[s];"
                                     f"[a][s]overlay={x}:{y}[v]", "-map", "[v]"]
        else:
            vf = ["-map", "0:v:0"]
        # les fragments gardent leurs horodatages d'origine : ffmpeg compte -ss depuis le début du fichier
        with atomic_path(dest) as tmp:
            subprocess.run([
                *FFMPEG, "-ss", f"{start - _start(v):.6f}", "-i", str(v), "-ss", f"{start - _start(a):.6f}",
                "-i", str(a), "-t", f"{dur:.6f}", *vf, "-map", "1:a:0",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "12",
                "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
                "-c:a", "pcm_s16le", "-ar", "48000", str(tmp),
            ], check=True, timeout=600)
    finally:
        v.unlink(missing_ok=True)
        a.unlink(missing_ok=True)
    return dest, band


def run_fast(url: str, opts, job_dir: Path, label: str = "", log=print) -> dict:
    """Même résultat que pipeline.run (clips rendus dans job_dir/clips) sans télécharger la vidéo entière.
    opts : pipeline.Options. Le plan des clips est gardé dans plan.json : une relance ne redemande pas à Claude."""
    job_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    step = lambda name: log(f"  [{time.time() - t0:5.0f} s] {name}")

    video, audio, wav = fetch(url, job_dir, log)
    step(f"vidéo {video.fmt.get('height')}p {video.fmt.get('vcodec', '?').split('.')[0]}, audio prêt")
    transcript = transcribe.transcribe(wav, job_dir, opts.whisper_model, opts.language, batched=True)
    words = transcribe.all_words(transcript)
    step(f"transcription : {len(words)} mots ({transcript['duration'] / 60:.0f} min d'audio)")

    def music_mode() -> str:
        if opts.music != "auto":
            return opts.music
        ratio = demusic.detect_music(wav, windows=2, window_s=12)
        mode = "remove" if ratio >= demusic.MUSIC_THRESHOLD else "keep"
        step(f"musique d'origine : {ratio:.0%} hors voix, {'retirée' if mode == 'remove' else 'gardée'}")
        return mode

    background = ThreadPoolExecutor(1)
    music = background.submit(music_mode)  # pendant que Claude choisit les moments
    plan_file = job_dir / "plan.json"
    if plan_file.exists():
        clips = json.loads(plan_file.read_text(encoding="utf-8"))["clips"]
    else:
        clips = highlights.pick_clips(transcript, words, opts.n_clips, opts.min_len, opts.max_len, opts.brain,
                                      log=lambda m: None)
        for i, clip in enumerate(clips, 1):
            clip["name"] = f"clip_{i:02d}"
        write_json_atomic(plan_file, {"source": label or url, "clips": clips})
    step(f"{len(clips)} moments choisis")

    def section(clip: dict) -> tuple[Path, float, tuple[int, int, int, int] | None]:
        start = max(0.0, clip["start"] - SECTION_PAD)
        dur = clip["end"] + render.END_PAD + SECTION_PAD - start
        src, band = cut_section(video, audio, start, dur, job_dir / f"{clip['name']}_src.mkv")
        return src, start, band

    style = dataclasses.replace(PRESETS[opts.style], **opts.caption_overrides)
    out_dir = job_dir / "clips"

    def produce(clip: dict, pending, music: str) -> dict:
        src, offset, band = pending.result()
        if band:
            step(f"{clip['name']} : sous-titres d'origine floutés ({band[2]}x{band[3]} px)")
        local = {**clip, "start": clip["start"] - offset, "end": clip["end"] - offset}
        local_words = [{**w, "start": w["start"] - offset, "end": w["end"] - offset}
                       for w in words if w["end"] > clip["start"] - 2 and w["start"] < clip["end"] + 2]
        info = render.render_clip(src, ingest.probe(src), local, local_words, out_dir, clip["name"], style,
                                  None, opts.show_hook, music, opts.tighten, opts.quality, opts.progress_bar,
                                  log=lambda m: None)
        src.unlink(missing_ok=True)
        step(f"{clip['name']} rendu ({info['duration']} s)")
        return {**clip, **info, "blurred_subtitles": bool(band)}

    # les extraits se téléchargent tous ensemble ; chaque clip est rendu dès que son extrait est là, en parallèle
    # des autres (un rendu attend surtout ffmpeg et ses lancements)
    n = max(1, len(clips))
    with ThreadPoolExecutor(n) as downloads, ThreadPoolExecutor(n) as renders:
        sections = [downloads.submit(section, clip) for clip in clips]
        mode = music.result()
        background.shutdown()
        results = [f.result() for f in [renders.submit(produce, c, s, mode) for c, s in zip(clips, sections)]]
    return {"source": label or url, "clips": results, "dir": str(out_dir), "seconds": round(time.time() - t0)}
