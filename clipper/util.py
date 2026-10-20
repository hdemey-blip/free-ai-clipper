"""Petits outils partagés : appels ffmpeg/ffprobe et écriture atomique des fichiers."""

import json
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path

FFMPEG = ["ffmpeg", "-y", "-v", "error", "-nostdin"]


class Stopped(Exception):
    """Travail mis en pause ou retiré de la file. Ce qui est fini reste sur le disque et sera repris."""


def ffprobe(path: Path) -> dict:
    """Un seul appel ffprobe : durée, présence de vidéo, taille, fps et décalage de départ de l'audio."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(out.stdout)
    video = next((s for s in data["streams"] if s.get("codec_type") == "video"), None)
    audio = next((s for s in data["streams"] if s.get("codec_type") == "audio"), None)
    info = {
        "duration": float(data["format"].get("duration") or 0),
        "has_video": video is not None,
        "audio_start": float(audio.get("start_time") or 0) if audio else 0.0,
    }
    if video:
        info.update(width=int(video["width"]), height=int(video["height"]), color_space=video.get("color_space"),
                    fps=_rate(video.get("r_frame_rate")) or _rate(video.get("avg_frame_rate")) or 30.0)
    return info


def _rate(text: str | None) -> float | None:
    """'30000/1001' → 29.97. None pour une cadence absente ou aberrante (certains fichiers à cadence
    variable annoncent 90000/1)."""
    num, _, den = (text or "0/0").partition("/")
    try:
        fps = float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        return None
    return fps if 1 <= fps <= 120 else None


@contextmanager
def atomic_path(dest: Path):
    """Donne un chemin temporaire à côté de dest (même extension, pour ffmpeg) et le renomme en dest si tout
    s'est bien passé. Un travail interrompu ne laisse jamais un fichier tronqué sous le nom final.
    Supprimer le fichier temporaire dans le bloc annule l'écriture."""
    tmp = dest.with_name(f".{dest.stem}.tmp{dest.suffix}")
    try:
        yield tmp
        if tmp.exists():
            os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)


def ask_claude(prompt: str, schema: dict, system: str, stdin: str = "", effort: str = "high",
               timeout: int = 1200) -> dict:
    """Une question à Claude par la CLI Claude Code (abonnement Claude, aucune clé API), réponse au format schema.

    MCP, réglages et outils sont coupés : l'appel ne contient que la consigne et stdin.
    """
    import shutil
    exe = shutil.which("claude")
    if not exe:
        raise RuntimeError("CLI 'claude' introuvable dans le PATH")
    cmd = [
        exe, "-p", prompt,
        "--output-format", "json",
        "--json-schema", json.dumps(schema),
        "--system-prompt", system,
        "--effort", effort,
        "--tools", "",
        "--strict-mcp-config",
        "--setting-sources", "",
        "--disable-slash-commands",
        "--no-session-persistence",
    ]
    proc = subprocess.run(cmd, input=stdin.encode("utf-8"), capture_output=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", "replace")[-500:] or f"code {proc.returncode}")
    data = json.loads(proc.stdout.decode("utf-8"))
    if data.get("is_error") or not data.get("structured_output"):
        raise RuntimeError(str(data.get("result"))[:500])
    return data["structured_output"]


def write_json_atomic(path: Path, data) -> None:
    # les nombres et booléens numpy (scores, masques) deviennent leurs équivalents Python
    text = json.dumps(data, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    with atomic_path(path) as tmp:
        tmp.write_text(text, encoding="utf-8")
