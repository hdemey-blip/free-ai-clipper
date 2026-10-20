"""Transcription mot à mot avec faster-whisper (CPU, multilingue)."""

import json
import os
import threading
from pathlib import Path

from .util import write_json_atomic

SENTENCE_END = (".", "!", "?", "…")

_models: dict[str, object] = {}
_models_lock = threading.Lock()


def _model(size: str):
    """Modèle gardé en mémoire entre deux travaux (le serveur web en enchaîne plusieurs)."""
    from faster_whisper import WhisperModel

    with _models_lock:
        if size not in _models:
            threads = max(4, (os.cpu_count() or 8) // 2)  # ≈ cœurs physiques
            _models[size] = WhisperModel(size, device="cpu", compute_type="int8", cpu_threads=threads)
        return _models[size]


def transcribe(audio: Path, workdir: Path, model_size: str = "small", language: str | None = None,
               on_progress=None, batched: bool = False) -> dict:
    """Renvoie {"language", "segments": [{start, end, text, words: [{start, end, word}]}]}.

    Le résultat est mis en cache par modèle et par langue : relancer avec --model large-v3 retranscrit.
    Si on_progress lève une exception (travail mis en pause), les segments déjà transcrits sont gardés dans
    un fichier .partial et la transcription suivante repart de la fin du dernier segment.
    """
    cache = workdir / f"transcript_{model_size}_{language or 'auto'}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    partial = cache.with_suffix(".partial.json")

    import wave

    import numpy as np

    # lecture directe du WAV 16 kHz produit par ffmpeg (évite le décodeur PyAV de faster-whisper)
    with wave.open(str(audio), "rb") as wf:
        rate = wf.getframerate()
        samples = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32)
    samples *= 1 / 32768
    total = len(samples) / rate

    segments, offset, detected = [], 0.0, language
    if partial.exists():
        done = json.loads(partial.read_text(encoding="utf-8"))
        segments, offset, detected = done["segments"], done["offset"], done["language"]

    if batched:
        # morceaux de voix décodés par lots : 2 fois plus rapide sur CPU (5 min d'audio : 14 s en base au lieu de 28)
        from faster_whisper import BatchedInferencePipeline
        segments_iter, info = BatchedInferencePipeline(_model(model_size)).transcribe(
            samples[int(offset * rate):], language=detected, word_timestamps=True, batch_size=8)
    else:
        segments_iter, info = _model(model_size).transcribe(
            samples[int(offset * rate):],
            language=detected,  # la langue détectée au premier passage, pour ne pas en changer en cours de route
            word_timestamps=True,
            vad_filter=True,
            beam_size=5,
        )
    detected = detected or info.language

    def keep_partial():
        end = segments[-1]["end"] if segments else offset
        write_json_atomic(partial, {"language": detected, "offset": end, "segments": segments})

    finished = False
    try:
        for seg in segments_iter:
            words = [
                {"start": round(offset + w.start, 3), "end": round(offset + w.end, 3), "word": w.word.strip()}
                for w in (seg.words or []) if w.word.strip()
            ]
            segments.append({"start": round(offset + seg.start, 3), "end": round(offset + seg.end, 3),
                             "text": seg.text.strip(), "words": words})
            if len(segments) % 25 == 0:
                keep_partial()  # aussi utile si le serveur est coupé net
            if on_progress and total:
                on_progress(min((offset + seg.end) / total, 1.0))
        finished = True
    finally:
        if not finished:
            keep_partial()

    result = {"language": detected, "duration": total, "segments": segments}
    write_json_atomic(cache, result)
    partial.unlink(missing_ok=True)
    return result


def all_words(transcript: dict) -> list[dict]:
    """Mots à plat, avec les élisions recollées (qu + 'elle → qu'elle, j + 'ai → j'ai, -t-il) et la ponctuation
    isolée de la typographie française rattachée au mot précédent par une espace insécable (« dernière ? »)."""
    merged: list[dict] = []
    for w in (w for seg in transcript["segments"] for w in seg["words"]):
        if merged and w["word"].strip() and not any(c.isalnum() for c in w["word"]):
            merged[-1] = {**merged[-1], "word": merged[-1]["word"] + "\u00a0" + w["word"].strip()}
            continue
        glue = merged and (w["word"][0] in "'’-" or merged[-1]["word"][-1] in "'’")
        if glue and w["start"] - merged[-1]["end"] < 0.5:
            merged[-1] = {**merged[-1], "word": merged[-1]["word"] + w["word"], "end": w["end"]}
        else:
            merged.append(dict(w))
    return merged
