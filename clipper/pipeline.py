"""Enchaîne les étapes : source → transcription → choix des moments → rendu des clips."""

import dataclasses
import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import demusic, highlights, ingest, render, transcribe
from .captions import PRESETS, normalize_color
from .util import Stopped, write_json_atomic

ROOT = Path(__file__).resolve().parent.parent
JOBS_DIR = ROOT / "jobs"
WHISPER_MODELS = ("tiny", "base", "small", "medium", "large-v3")


@dataclass
class Options:
    """Réglages d'un travail. Validés ici une fois pour toutes : la CLI et le web ne font que relayer
    le ValueError à l'utilisateur."""
    n_clips: int | None = None
    min_len: int = 20
    max_len: int = 70
    style: str = "bold"
    whisper_model: str = "small"
    language: str | None = None
    logo: str | None = None
    show_hook: bool = True
    caption_overrides: dict = field(default_factory=dict)
    brain: str = "auto"
    music: str = "keep"
    tighten: bool = True          # raccourcit les blancs de plus de 0,8 s
    quality: str = "high"         # fast, high ou max (vitesse d'encodage contre finesse de l'image)
    progress_bar: bool = False

    def __post_init__(self):
        if self.n_clips is not None and self.n_clips <= 0:
            raise ValueError("Le nombre de clips doit être positif")
        if self.min_len <= 0 or self.max_len <= 0:
            raise ValueError("Les durées doivent être positives")
        if self.min_len > self.max_len:
            raise ValueError("La durée minimale doit être inférieure ou égale à la durée maximale")
        for name, value, allowed in (("style", self.style, PRESETS), ("brain", self.brain, highlights.BRAINS),
                                     ("music", self.music, demusic.MUSIC_MODES),
                                     ("modèle", self.whisper_model, WHISPER_MODELS),
                                     ("qualité", self.quality, render.QUALITIES)):
            if value not in allowed:
                raise ValueError(f"{name} inconnu : {value!r} (choix : {', '.join(allowed)})")
        if self.logo and not Path(self.logo).is_file():
            raise ValueError(f"Logo introuvable : {self.logo}")
        for key in ("color", "highlight", "outline"):
            if key in self.caption_overrides:
                self.caption_overrides[key] = normalize_color(self.caption_overrides[key])


DEFAULTS = Options()


def job_id_for(source: str) -> str:
    """Un lien → toujours le même dossier ; un fichier → chemin absolu + taille + date de modification."""
    key = source
    if not ingest.is_url(source) and Path(source).is_file():
        p = Path(source).resolve()
        st = p.stat()
        key = f"{p}|{st.st_size}|{st.st_mtime_ns}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]


PLAN_LOCK = threading.Lock()


def load_plan(job_dir: Path) -> dict | None:
    """Clips prévus pour un travail : {"key", "clips", "order" (ordre de rendu), "skipped", "music"}."""
    path = job_dir / "plan.json"
    with PLAN_LOCK:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def edit_plan(job_dir: Path, change) -> dict:
    """change(plan) modifie le plan sur place. Lecture et écriture sous verrou : l'interface web peut retirer ou
    déplacer un clip pendant que le rendu tourne."""
    path = job_dir / "plan.json"
    with PLAN_LOCK:
        plan = json.loads(path.read_text(encoding="utf-8"))
        change(plan)
        write_json_atomic(path, plan)
        return plan


def run(source: str, opts: Options, job_dir: Path | None = None, status=None, label: str | None = None,
        stop=None) -> dict:
    """status(stage, progress 0-1, message) est appelé à chaque étape (interface web, CLI).
    label : nom affiché dans l'historique (nom d'origine d'un fichier envoyé), source par défaut.
    stop() vrai → Stopped. Chaque étape finie reste sur le disque (source, audio, transcription partielle ou
    complète, clips choisis, clips rendus) : relancer le même travail reprend là où il s'était arrêté."""
    report = status or (lambda stage, p, msg: print(f"[{stage} {p:4.0%}] {msg}"))
    job_dir = job_dir or JOBS_DIR / job_id_for(source)
    job_dir.mkdir(parents=True, exist_ok=True)
    label = label or source
    t0 = time.time()

    def status(stage: str, progress: float, message: str) -> None:
        if stop and stop():
            raise Stopped
        report(stage, progress, message)

    def save(clips: list[dict], note: str = "", rendering: str | None = None) -> None:
        write_json_atomic(job_dir / "result.json", {"source": label, "meta": meta, "clips": clips, "note": note,
                                                    "rendering": rendering})

    status("source", 0.0, "Récupération de la vidéo")
    video = ingest.fetch_source(source, job_dir)
    meta = ingest.probe(video)
    status("source", 1.0, f"{meta['width']}x{meta['height']}, {meta['duration'] / 60:.1f} min")

    status("audio", 0.0, "Extraction de l'audio")
    audio = ingest.extract_audio(video, job_dir)

    status("transcription", 0.0, f"Whisper {opts.whisper_model}")
    transcript = transcribe.transcribe(
        audio, job_dir, opts.whisper_model, opts.language,
        on_progress=lambda p: status("transcription", p, "Transcription en cours"),
    )
    words = transcribe.all_words(transcript)
    status("transcription", 1.0, f"{len(words)} mots, langue {transcript['language']}")

    # les clips choisis sont gardés dans plan.json : une reprise ne redemande pas à Claude
    plan_key = {"model": opts.whisper_model, "lang": opts.language, "clips": opts.n_clips,
                "min": opts.min_len, "max": opts.max_len, "brain": opts.brain}
    plan = load_plan(job_dir)
    previous = []
    if plan and plan["key"] == plan_key:
        status("analyse", 1.0, f"{len(plan['clips'])} clips déjà choisis, repris tels quels")
        result_file = job_dir / "result.json"
        if result_file.exists():
            previous = json.loads(result_file.read_text(encoding="utf-8"))["clips"]
    else:
        status("analyse", 0.0, "Recherche des meilleurs moments")
        clips = highlights.pick_clips(
            transcript, words, opts.n_clips, opts.min_len, opts.max_len, opts.brain,
            log=lambda m: status("analyse", 0.5, m),
        )
        for i, clip in enumerate(clips, 1):
            clip["name"] = f"clip_{i:02d}"
        plan = {"key": plan_key, "clips": clips, "order": [c["name"] for c in clips], "skipped": [], "music": None}
        with PLAN_LOCK:
            write_json_atomic(job_dir / "plan.json", plan)
    note = "" if plan["clips"] else (f"Aucun passage de {opts.min_len} à {opts.max_len} s trouvé : "
                                     "baisse la durée minimale ou vérifie que la vidéo contient de la parole")
    save(previous, note)
    status("analyse", 1.0, f"{len(plan['clips'])} clips retenus" if plan["clips"] else note)

    music = opts.music
    if music == "auto" and plan["clips"]:
        # décidé une fois pour toute la vidéo, sur quelques extraits, plutôt que de séparer chaque clip pour rien
        music = plan["music"]
        if music is None:
            status("musique", 0.0, "Recherche de musique de fond")
            ratio = demusic.detect_music(video)
            music = "remove" if ratio >= demusic.MUSIC_THRESHOLD else "keep"
            edit_plan(job_dir, lambda p: p.update(music=music))
            status("musique", 1.0, f"{ratio:.0%} d'énergie hors voix : musique "
                                   + ("détectée, elle sera retirée" if music == "remove" else "absente, audio gardé"))

    style = dataclasses.replace(PRESETS[opts.style], **opts.caption_overrides)
    logo = Path(opts.logo) if opts.logo else None
    out_dir = job_dir / "clips"
    # un clip déjà rendu avec les mêmes réglages n'est pas refait
    render_key = {"style": dataclasses.asdict(style), "logo": opts.logo, "hook": opts.show_hook, "music": music,
                  "tighten": opts.tighten, "quality": opts.quality, "bar": opts.progress_bar}
    done = {c["name"]: c for c in previous
            if c.get("render_key") == render_key and (out_dir / c["file"]).exists()}

    def finished() -> list[dict]:
        return [done[c["name"]] for c in plan["clips"] if c["name"] in done]

    while True:
        # relu à chaque clip : l'ordre et les clips retirés peuvent changer pendant le rendu
        plan = load_plan(job_dir)
        todo = [n for n in plan["order"] if n not in plan["skipped"] and n not in done]
        if not todo:
            break
        name = todo[0]
        clip = next(c for c in plan["clips"] if c["name"] == name)
        count = len(done) + len(todo)
        progress = len(done) / count
        save(finished(), rendering=name)
        status("rendu", progress, f"Clip {len(done) + 1}/{count} : {clip['title']}")

        def removed(name=name) -> bool:
            return name in (load_plan(job_dir) or {}).get("skipped", [])

        try:
            info = render.render_clip(video, meta, clip, words, out_dir, name, style, logo,
                                      opts.show_hook, music, opts.tighten, opts.quality, opts.progress_bar,
                                      log=lambda m: status("rendu", progress, f"{clip['title']} : {m}"),
                                      stop=lambda: (stop and stop()) or removed())
        except Stopped:
            if stop and stop():
                save(finished())
                raise
            continue  # clip retiré pendant son rendu : on passe au suivant
        done[name] = {**clip, **info, "render_key": render_key}
        save(finished())
    results = finished()
    save(results, note)
    status("terminé", 1.0, f"{len(results)} clips en {time.time() - t0:.0f} s → {out_dir}" if results else note)
    return {"source": label, "meta": meta, "clips": results, "dir": str(out_dir)}
