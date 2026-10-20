"""Musique de fond ajoutée sous la voix d'un clip déjà rendu.

La musique tourne en boucle sur toute la durée, démarre à un endroit choisi dans le morceau, s'efface quand
quelqu'un parle (compression déclenchée par la voix) et finit en fondu. L'image n'est pas réencodée.
"""

import random
import subprocess
from pathlib import Path

from .util import FFMPEG, atomic_path, ffprobe

AUDIO_EXTS = {".mp3", ".m4a", ".aac", ".wav", ".flac", ".ogg", ".opus"}


def tracks(folder: Path) -> list[Path]:
    return sorted(p for p in folder.glob("*") if p.suffix.lower() in AUDIO_EXTS) if folder.is_dir() else []


def add_music(clip: Path, track: Path, dest: Path, level_db: float = -20.0, seed: int | None = None) -> Path:
    """level_db : niveau de la musique par rapport à son niveau d'origine quand personne ne parle."""
    clip_dur = ffprobe(clip)["duration"]
    track_dur = ffprobe(track)["duration"]
    # départ au hasard dans le premier tiers du morceau : deux clips n'ont pas la même intro
    rng = random.Random(seed)
    offset = rng.uniform(0, max(0.0, min(track_dur / 3, track_dur - clip_dur))) if track_dur > clip_dur else 0.0
    fade = min(1.5, clip_dur / 4)
    graph = (
        f"[1:a]aresample=48000,volume={level_db}dB,atrim=0:{clip_dur:.3f},"
        f"afade=t=in:d=0.4,afade=t=out:st={clip_dur - fade:.3f}:d={fade:.3f}[bed];"
        "[0:a]aresample=48000,asplit[voice][key];"
        "[bed][key]sidechaincompress=threshold=0.03:ratio=5:attack=15:release=350[ducked];"
        "[voice][ducked]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.84[a]"
    )
    with atomic_path(dest) as tmp:
        subprocess.run([
            *FFMPEG, "-i", str(clip), "-stream_loop", "-1", "-ss", f"{offset:.3f}", "-i", str(track),
            "-filter_complex", graph, "-map", "0:v", "-map", "[a]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-t", f"{clip_dur:.3f}",
            str(tmp),
        ], check=True)
    return dest
