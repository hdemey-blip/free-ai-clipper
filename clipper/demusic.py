"""Retrait de la musique de fond : Demucs (htdemucs) sépare la voix du reste, on ne garde que la voix.

L'audio est lu d'un seul flux et traité par morceaux de 2 minutes : la mémoire reste stable même sur une
vidéo de 3 heures. La piste produite démarre au même instant que l'audio d'origine : on peut la remonter
sur la vidéo sans aucun décalage à appliquer.
"""

import subprocess
import threading
from pathlib import Path

import numpy as np

from .util import FFMPEG, atomic_path, ffprobe

MUSIC_MODES = ("keep", "auto", "remove")  # garder / retirer si détectée / toujours retirer

SR = 44100
CHUNK_S = 120           # durée d'un morceau traité d'un coup
OVERLAP_S = 2           # fondu enchaîné entre deux morceaux
MUSIC_THRESHOLD = 0.12  # part d'énergie hors voix au-delà de laquelle on considère qu'il y a de la musique
PCM = ["-f", "f32le", "-ac", "2", "-ar", str(SR)]

_AAC = ["-c:a", "aac", "-b:a", "192k"]
_OPUS = ["-c:a", "libopus", "-b:a", "160k"]
AUDIO_CODECS = {  # codec audio selon le conteneur de sortie
    ".mp4": _AAC, ".m4a": _AAC, ".mov": _AAC, ".mkv": _AAC, ".webm": _OPUS, ".ogg": _OPUS,
    ".mp3": ["-c:a", "libmp3lame", "-b:a", "192k"], ".wav": ["-c:a", "pcm_s16le"], ".flac": ["-c:a", "flac"],
}
AUDIO_ONLY = {".m4a", ".ogg", ".mp3", ".wav", ".flac"}

_model = None
_model_lock = threading.Lock()


def _get_model():
    global _model
    with _model_lock:
        if _model is None:
            from demucs.pretrained import get_model
            _model = get_model("htdemucs")
            _model.eval()
    return _model


def _separate(mix: np.ndarray) -> np.ndarray:
    import torch
    from demucs.apply import apply_model

    model = _get_model()
    wav = torch.from_numpy(mix)
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    with torch.no_grad():
        sources = apply_model(model, ((wav - mean) / std)[None], split=True, overlap=0.25, progress=False)[0]
    return (sources[model.sources.index("vocals")] * std + mean).numpy()


def _decode(src: Path, start: float = 0.0, length: float | None = None):
    cmd = ["ffmpeg", "-v", "error", "-nostdin"]
    if start:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(src)]
    if length:
        cmd += ["-t", f"{length:.3f}"]
    return cmd + ["-vn", *PCM, "-"]


def detect_music(src: Path, windows: int = 3, window_s: float = 20.0) -> float:
    """Part d'énergie hors voix, mesurée sur quelques extraits répartis dans le fichier.

    Sert au mode "auto" : on décide une fois pour toute la vidéo au lieu de séparer chaque clip pour rien.
    """
    total = ffprobe(src)["duration"]
    starts = [total * (i + 1) / (windows + 1) - window_s / 2 for i in range(windows)] if total > window_s else [0]
    mix_e = rest_e = 0.0
    for s in starts:
        raw = subprocess.run(_decode(src, max(0.0, s), window_s), capture_output=True, check=True).stdout
        mix = np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).T.copy()
        if mix.shape[1]:
            rest = mix - _separate(mix)
            mix_e += float(np.sum(mix ** 2))
            rest_e += float(np.sum(rest ** 2))
    return rest_e / (mix_e + 1e-12)


def clean_audio(src: Path, dest_wav: Path, mode: str = "remove", keep: float = 0.0, log=print,
                on_progress=None) -> bool:
    """Écrit dans dest_wav la piste sans musique. Renvoie False si rien n'a été produit.

    mode : "remove" (toujours) ou "auto" (seulement si la séparation trouve vraiment de la musique).
    keep : niveau de musique conservé (0 = aucune, 0.15 = musique très discrète).
    """
    if mode == "keep":
        return False
    info = ffprobe(src)
    chunk, overlap = CHUNK_S * SR, OVERLAP_S * SR
    fade = np.linspace(0, 1, overlap, dtype=np.float32)
    # un seul flux de décodage lu morceau par morceau : pas de saut (-ss), donc aucun décalage entre morceaux
    reader = subprocess.Popen(_decode(src), stdout=subprocess.PIPE)

    def read(n: int) -> np.ndarray:
        data = reader.stdout.read(n * 8)
        return np.frombuffer(data[:len(data) // 8 * 8], dtype=np.float32).reshape(-1, 2).T.copy()

    mix_energy = rest_energy = 0.0
    done_samples = 0
    with atomic_path(dest_wav) as tmp:
        writer = subprocess.Popen([*FFMPEG, *PCM, "-i", "-", str(tmp)], stdin=subprocess.PIPE)

        def write(a: np.ndarray) -> None:
            writer.stdin.write(np.ascontiguousarray(np.clip(a, -1, 1).T, dtype=np.float32).tobytes())

        try:
            # le décodage en PCM perd le décalage de départ de la piste : on le remet en silence
            write(np.zeros((2, round(info["audio_start"] * SR)), dtype=np.float32))
            pending = None  # fin du morceau précédent, gardée pour le fondu avec le suivant
            mix = read(chunk)
            while mix.shape[1]:
                following = read(chunk - overlap)
                voice = _separate(mix)
                rest = mix - voice
                region = slice(overlap, None) if pending is not None else slice(None)  # sans compter deux fois
                mix_energy += float(np.sum(mix[:, region] ** 2))
                rest_energy += float(np.sum(rest[:, region] ** 2))

                out = voice + keep * rest
                if pending is not None:
                    head = min(overlap, out.shape[1])
                    out[:, :head] = pending[:, :head] * (1 - fade[:head]) + out[:, :head] * fade[:head]
                if following.shape[1] == 0 or out.shape[1] <= overlap:
                    write(out)
                    break
                write(out[:, :-overlap])
                pending = out[:, -overlap:]
                done_samples += out.shape[1] - overlap
                if on_progress and info["duration"]:
                    on_progress(min(1.0, done_samples / SR / info["duration"]))
                mix = np.concatenate([mix[:, -overlap:], following], axis=1)
        finally:
            reader.stdout.close()
            reader.wait()
            writer.stdin.close()
            writer.wait()

        ratio = rest_energy / (mix_energy + 1e-12)
        if mix_energy == 0 or (mode == "auto" and ratio < MUSIC_THRESHOLD):
            log(f"Pas de musique détectée ({ratio:.0%} hors voix), audio d'origine conservé")
            tmp.unlink()  # annule l'écriture
            return False
    log(f"Musique retirée ({ratio:.0%} de l'énergie hors voix)")
    return True


def output_path(src: Path, dest: Path | None) -> Path:
    """Sortie par défaut : <nom>_sans_musique.mp4 pour une vidéo, .m4a pour un fichier audio."""
    if dest:
        return dest
    ext = src.suffix.lower() if src.suffix.lower() in AUDIO_CODECS else ".mp4"
    if not ffprobe(src)["has_video"] and ext not in AUDIO_ONLY:
        ext = ".m4a"
    return src.with_name(f"{src.stem}_sans_musique{ext}")


def demusic_file(src: Path, dest: Path | None = None, mode: str = "remove", keep: float = 0.0,
                 log=print) -> Path:
    """Outil autonome : même fichier, musique de fond retirée (la vidéo est copiée sans ré-encodage)."""
    dest = output_path(src, dest)
    codec = AUDIO_CODECS.get(dest.suffix.lower())
    if codec is None:
        raise ValueError(f"Extension de sortie non gérée : {dest.suffix} (choisis parmi {', '.join(AUDIO_CODECS)})")
    wav = dest.with_name(dest.stem + ".voice.wav")
    if not clean_audio(src, wav, mode, keep, log, on_progress=lambda p: log(f"{p:.0%}")):
        return src
    keep_video = ffprobe(src)["has_video"] and dest.suffix.lower() not in AUDIO_ONLY
    video = ["-map", "0:v", "-c:v", "copy"] if keep_video else []
    try:
        with atomic_path(dest) as tmp:
            subprocess.run([*FFMPEG, "-i", str(src), "-i", str(wav), *video, "-map", "1:a", *codec,
                            "-shortest", str(tmp)], check=True)
    finally:
        wav.unlink(missing_ok=True)
    return dest
