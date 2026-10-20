"""Récupère la vidéo source (fichier local, envoi web ou lien) et en extrait l'audio."""

import os
import shutil
import subprocess
from pathlib import Path

from .util import FFMPEG, atomic_path, ffprobe

VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v", ".flv", ".ts", ".mpg", ".mpeg", ".wmv"}

# YouTube exige un moteur JavaScript pour déchiffrer ses formats ; yt-dlp n'active que deno par défaut, absent ici
YTDLP_JS = {"js_runtimes": {"node": {}}}


def is_url(source: str) -> bool:
    return source.startswith(("http://", "https://"))


def _finished_source(workdir: Path) -> Path | None:
    for p in workdir.glob("source.*"):
        if p.suffix.lower() in VIDEO_EXTS:
            return p
    return None


def store_upload(upload, filename: str, workdir: Path) -> Path:
    """Enregistre un fichier envoyé par l'interface web directement comme source du travail."""
    workdir.mkdir(parents=True, exist_ok=True)
    dest = workdir / f"source{Path(filename).suffix.lower() or '.mp4'}"
    with atomic_path(dest) as tmp:
        upload.save(tmp)
    return dest


def fetch_source(source: str, workdir: Path) -> Path:
    """Place la source dans workdir/source.<ext> (lien physique, copie ou téléchargement) et renvoie son chemin."""
    workdir.mkdir(parents=True, exist_ok=True)
    done = _finished_source(workdir)
    if done:
        return done

    if is_url(source):
        import yt_dlp

        tmp = workdir / "download"
        shutil.rmtree(tmp, ignore_errors=True)
        opts = {
            "outtmpl": str(tmp / "video.%(ext)s"),
            "format": "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080]/b",
            "merge_output_format": "mp4",
            "quiet": True,
            "noprogress": True,
            **YTDLP_JS,
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([source])
        files = [p for p in tmp.glob("video.*") if p.suffix.lower() in VIDEO_EXTS]
        if not files:
            raise RuntimeError("Téléchargement terminé sans fichier vidéo exploitable")
        dest = workdir / f"source{files[0].suffix.lower()}"
        files[0].replace(dest)
        shutil.rmtree(tmp, ignore_errors=True)
        return dest

    src = Path(source)
    if not src.is_file():
        raise FileNotFoundError(f"Fichier introuvable : {source}")
    dest = workdir / f"source{src.suffix.lower()}"
    with atomic_path(dest) as tmp:
        try:
            os.link(src, tmp)  # lien physique : instantané et sans espace disque en plus
        except OSError:
            shutil.copy2(src, tmp)  # autre disque ou système de fichiers sans liens
    return dest


def probe(video: Path) -> dict:
    """Durée, taille, fps et espace colorimétrique de la vidéo."""
    info = ffprobe(video)
    if not info["has_video"]:
        raise ValueError(f"Pas de piste vidéo dans {video.name}")
    return {k: info[k] for k in ("width", "height", "fps", "duration", "color_space")}


def extract_audio(video: Path, workdir: Path) -> Path:
    """Audio mono 16 kHz, le format attendu par Whisper."""
    wav = workdir / "audio.wav"
    if not wav.exists():
        with atomic_path(wav) as tmp:
            subprocess.run([*FFMPEG, "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", str(tmp)], check=True)
    return wav
