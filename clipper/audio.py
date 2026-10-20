"""Son des clips : extraction des passages gardés, raccords sans clic, volume calé sur le standard des réseaux."""

import json
import re
import subprocess
from pathlib import Path

import numpy as np

from .util import FFMPEG, atomic_path

SR = 48000
TARGET_LUFS = -14.0   # niveau habituel de TikTok, Reels et Shorts
TRUE_PEAK = -1.5
JOIN_FADE_S = 0.012   # fondu aux raccords des coupes de silences (évite les clics)


def assemble(source: Path, window_start: float, window_dur: float, ranges: list[tuple[float, float]],
             dest: Path) -> None:
    """Écrit dans dest (WAV stéréo 48 kHz) les passages [a, b) de la fenêtre, mis bout à bout.

    Les temps de ranges sont relatifs au début de la fenêtre. Fondu très court à chaque raccord,
    et petits fondus d'entrée et de sortie pour que le clip ne démarre ni ne finisse sur un clic.
    """
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{window_start:.3f}", "-i", str(source),
         "-t", f"{window_dur:.3f}", "-vn", "-f", "f32le", "-ac", "2", "-ar", str(SR), "-"],
        capture_output=True, check=True,
    ).stdout
    audio = np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).T
    n_fade = int(JOIN_FADE_S * SR)
    ramp = np.linspace(0, 1, n_fade, dtype=np.float32)
    parts = []
    for i, (a, b) in enumerate(ranges):
        part = audio[:, int(round(a * SR)):int(round(b * SR))].copy()
        if part.shape[1] > 2 * n_fade:
            fade_in = ramp if i else np.linspace(0, 1, n_fade * 4, dtype=np.float32)[-n_fade:] * ramp
            part[:, :n_fade] *= fade_in
            part[:, -n_fade:] *= ramp[::-1]
        parts.append(part)
    out = np.concatenate(parts, axis=1) if parts else np.zeros((2, 0), dtype=np.float32)
    # le clip se termine en douceur sur 0,25 s
    tail = min(out.shape[1], int(0.25 * SR))
    if tail:
        out[:, -tail:] *= np.linspace(1, 0, tail, dtype=np.float32)
    with atomic_path(dest) as tmp:
        subprocess.run([*FFMPEG, "-f", "f32le", "-ac", "2", "-ar", str(SR), "-i", "-", str(tmp)],
                       input=np.ascontiguousarray(out.T).tobytes(), check=True)


def master(src: Path, dest: Path) -> dict:
    """Voix nettoyée des basses parasites puis volume normalisé à -14 LUFS en deux passes (mesure, puis
    correction linéaire : pas de pompage). Renvoie les mesures avant correction."""
    pre = "highpass=f=70"
    probe = subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-i", str(src), "-af",
         f"{pre},loudnorm=I={TARGET_LUFS}:TP={TRUE_PEAK}:LRA=11:print_format=json", "-f", "null", "-"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    match = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", probe.stderr)
    with atomic_path(dest) as tmp:
        if not match:  # mesure impossible (clip trop court ou muet) : on garde le son tel quel
            subprocess.run([*FFMPEG, "-i", str(src), "-ar", str(SR), str(tmp)], check=True)
            return {}
        m = json.loads(match.group(0))
        if float(m["input_i"]) < -70:  # silence quasi total : rien à normaliser
            subprocess.run([*FFMPEG, "-i", str(src), "-ar", str(SR), str(tmp)], check=True)
            return m
        norm = (f"loudnorm=I={TARGET_LUFS}:TP={TRUE_PEAK}:LRA=11:measured_I={m['input_i']}:"
                f"measured_TP={m['input_tp']}:measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:"
                f"offset={m['target_offset']}:linear=true")
        subprocess.run([*FFMPEG, "-i", str(src), "-af", f"{pre},{norm}", "-ar", str(SR), str(tmp)], check=True)
    return m
