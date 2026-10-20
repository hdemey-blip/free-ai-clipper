"""Agent TikTok : cherche des reportages réutilisables, les passe dans Clipper et prépare les meilleurs clips à
publier, en local par défaut.

Tout vit dans agent/ :
- config.json      sujets, filtres, chaînes autorisées, réglages (créé au premier lancement) ;
- sons_tiktok.txt  sons de la bibliothèque TikTok, un par ligne, attribués à tour de rôle (ajoutés dans l'appli) ;
- musique/         morceaux à mixer directement dans la vidéo, si on n'utilise pas les sons TikTok ;
- a_publier/       clip.mp4 + clip.txt (son à ajouter, légende, crédit de l'auteur) ;
- a_publier.md     la même chose, tous les clips à la suite ;
- etat.json        vidéos déjà vues, clips produits, prochain son ;
- tiktok.json      seulement si l'envoi en brouillon par l'API TikTok est activé (envoi_tiktok).
"""

import json
from datetime import datetime
from pathlib import Path

from . import discover, fast, music, transcribe
from .pipeline import JOBS_DIR, ROOT, Options
from .tiktok import Account, TikTokError
from .util import write_json_atomic

AGENT_DIR = ROOT / "agent"
CONFIG = AGENT_DIR / "config.json"
STATE = AGENT_DIR / "etat.json"
OUT_DIR = AGENT_DIR / "a_publier"
INDEX = AGENT_DIR / "a_publier.md"
YT_CACHE = AGENT_DIR / "cache_youtube.json"

LIGNE = ("reportages, documentaires et interviews qui apprennent quelque chose : métiers, sciences, voyages, "
         "histoire, initiatives locales ; parole claire et histoires concrètes")
SUJETS = [
    "reportage métier", "documentaire science", "reportage voyage", "interview artisan", "histoire documentaire",
    "reportage nature", "conférence vulgarisation", "reportage cuisine", "portrait entrepreneur", "reportage village",
]

DEFAULT_CONFIG = {
    "sujets": SUJETS,
    "ligne_editoriale": LIGNE,
    "recence": "annee",
    "interet_min": 50,
    "langue": "fr",
    "abonnes_max": 300000,
    "duree_min_minutes": 8,
    "duree_max_minutes": 35,
    "viralite_min": 65,
    "clips_par_lancement": 3,
    "chaines_autorisees": [],
    "chaines_exclues": [],
    "sons_tiktok": "agent/sons_tiktok.txt",
    "musique": {"dossier": "agent/musique", "niveau_db": -20},
    "envoi_tiktok": False,
    "clips_par_video": 3,
    "clipper": {"style": "bold", "modele": "base", "qualite": "turbo", "musique_originale": "auto",
                "min": 20, "max": 50},
}


def load_config() -> dict:
    AGENT_DIR.mkdir(exist_ok=True)
    if not CONFIG.exists():
        write_json_atomic(CONFIG, DEFAULT_CONFIG)
    return {**DEFAULT_CONFIG, **json.loads(CONFIG.read_text(encoding="utf-8"))}


def search(cfg: dict, state: dict, log=print) -> list[dict]:
    """Candidats de discover.find, sans les vidéos déjà vues ni les chaînes exclues. Cache gardé sur disque."""
    cache = json.loads(YT_CACHE.read_text(encoding="utf-8")) if YT_CACHE.exists() else {}
    try:
        found = discover.find(cfg["sujets"], set(state["videos"]), cfg["chaines_autorisees"], cfg["abonnes_max"],
                              cfg["duree_min_minutes"], cfg["duree_max_minutes"], per_topic=12, cache=cache,
                              angle=cfg["ligne_editoriale"], recent=cfg["recence"], min_interest=cfg["interet_min"],
                              log=log)
    finally:
        write_json_atomic(YT_CACHE, cache)
    excluded = {c.lower() for c in cfg["chaines_exclues"]}
    return [v for v in found if v["channel"].lower() not in excluded]


def load_state() -> dict:
    state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
    state.setdefault("videos", {})     # id YouTube → {titre, chaîne, statut, date}
    state.setdefault("clips", [])      # {fichier, légende, statut: prêt|brouillon|échec, ...}
    state.setdefault("morceau", 0)     # prochain son TikTok ou morceau, à tour de rôle
    return state


def tiktok_sounds(cfg: dict) -> list[str]:
    path = ROOT / cfg["sons_tiktok"]
    if not path.exists():
        return []
    lines = (line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    return [line for line in lines if line and not line.startswith("#")]


def account() -> Account:
    return Account(AGENT_DIR / "tiktok.json")


SPEECHLESS = 0.25  # part du clip occupée par la parole sous laquelle le son TikTok devient la bande-son


def _speech_ratio(clip: dict, words: list[dict]) -> float:
    spoken = sum(min(w["end"], clip["end"]) - max(w["start"], clip["start"])
                 for w in words if w["end"] > clip["start"] and w["start"] < clip["end"])
    return spoken / max(clip["end"] - clip["start"], 0.1)


def _sheet(clip: dict) -> str:
    """Fiche d'un clip : ce qu'il faut faire dans l'appli TikTok pour le publier."""
    sound = ""
    if clip.get("son"):
        how = ("Vidéo presque sans parole : son TikTok à 100 %, son d'origine coupé."
               if clip.get("sans_parole") else "En fond : son TikTok vers 15 %, son d'origine à 100 %.")
        sound = f"SON TIKTOK : {clip['son']}\n{how}\n\n"
    return f"{sound}LÉGENDE :\n{clip['legende']}\n"


def _write_sheet(clip: dict) -> None:
    video = Path(clip["fichier"])
    video.with_suffix(".txt").write_text(_sheet(clip), encoding="utf-8")
    new = not INDEX.exists()
    with INDEX.open("a", encoding="utf-8") as f:
        if new:
            f.write("# Clips à publier\n\nVidéos dans a_publier/. Pour chacune : son à ajouter et légende à coller.\n")
        f.write(f"\n## {clip['date']} · {video.name} · viralité {clip['viralite']}\n\n{_sheet(clip)}")


def _caption(clip: dict, video: dict) -> str:
    tags = " ".join("#" + t.lstrip("#") for t in clip.get("hashtags", [])[:5])
    return f"{clip.get('caption', clip['title'])}\n\n{tags}\n\n{discover.credit(video)}".strip()


def send_ready(state: dict, acc: Account | None, limit: int, log=print) -> int:
    """Envoie en brouillon les clips prêts, du plus viral au moins viral. Renvoie le nombre envoyé."""
    if acc is None:
        return 0
    sent = 0
    for clip in sorted((c for c in state["clips"] if c["statut"] == "prêt"), key=lambda c: -c["viralite"]):
        if sent >= limit:
            break
        log(f"envoi en brouillon : {Path(clip['fichier']).name}")
        try:
            clip["publish_id"] = acc.upload_draft(Path(clip["fichier"]), log)
        except TikTokError as exc:
            if "pending" in str(exc) or "spam" in str(exc) or "rate_limit" in str(exc):
                log(f"TikTok refuse d'autres brouillons pour l'instant ({exc}). Les clips restent prêts.")
                break
            clip["statut"], clip["erreur"] = "échec", str(exc)
            log(f"échec : {exc}")
            write_json_atomic(STATE, state)
            continue
        clip["statut"], clip["envoye"] = "brouillon", datetime.now().isoformat(timespec="seconds")
        write_json_atomic(STATE, state)
        sent += 1
    return sent


def _single(name: str):
    """Verrou de fichier tenu tant que le processus vit : une seule production à la fois, même quand une tâche
    planifiée démarre alors que la précédente tourne encore. None si déjà pris."""
    f = open(AGENT_DIR / f"{name}.lock", "w")
    try:
        try:
            import msvcrt
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        except ImportError:  # Linux, macOS
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        return None
    return f


def _at(hhmm: str, after: datetime) -> datetime:
    """Prochain passage à l'heure hhmm ("08:30") à partir de after."""
    h, m = map(int, hhmm.split(":"))
    t = after.replace(hour=h, minute=m, second=0, microsecond=0)
    return t if t > after else datetime.fromtimestamp(t.timestamp() + 86400)


def stock() -> int:
    """Clips prêts dans a_publier/."""
    return sum(c["statut"] == "prêt" and Path(c["fichier"]).exists() for c in load_state()["clips"])


def produce_until(target: int, end: str | None = None, log=print) -> None:
    """Produit des clips jusqu'à en avoir target prêts, ou jusqu'à l'heure end ("08:30"), ou jusqu'à ce que la
    recherche ne trouve plus rien de neuf."""
    lock = _single("production")
    if lock is None:
        log("une production tourne déjà")
        return
    until = _at(end, datetime.now()) if end else None
    empty = 0
    while (have := stock()) < target:
        if until and datetime.now() >= until:
            log(f"{until:%H:%M} atteint : arrêt de la production ({have} clips en stock)")
            return
        log(f"stock : {have}/{target}")
        try:
            made = run_agent(target - have, until=until, log=log)
        except discover.Blocked as exc:
            log(f"{exc} Pause d'une heure.")
            import time
            time.sleep(3600)
            continue
        empty = 0 if made else empty + 1
        if empty >= 3:
            log("plus de vidéo nouvelle à traiter : ajoute des sujets dans agent/config.json")
            return
    log(f"stock atteint : {stock()} clips prêts")


def produce(video: dict, cfg: dict, state: dict, log=print) -> list[dict]:
    """Clipper sur la vidéo, puis son ou musique sur les clips assez viraux. Renvoie les clips prêts.

    Avec une liste de sons TikTok, chaque clip reçoit le nom du son à ajouter dans l'appli et la vidéo garde
    seulement la voix. Sinon, un morceau de musique/ est mixé dans la vidéo.
    """
    c = cfg["clipper"]
    opts = Options(n_clips=cfg["clips_par_video"], min_len=c["min"], max_len=c["max"], style=c["style"],
                   whisper_model=c["modele"], language=cfg["langue"] or None, brain="cli",
                   music=c["musique_originale"], quality=c["qualite"])
    result = fast.run_fast(video["url"], opts, JOBS_DIR / f"yt_{video['id']}",
                           label=f"{video['channel']} · {video['title']}", log=log)
    log(f"  vidéo traitée en {result['seconds'] // 60} min {result['seconds'] % 60:02d} s")
    sounds = tiktok_sounds(cfg)
    songs = [] if sounds else music.tracks(ROOT / cfg["musique"]["dossier"])
    transcript = Path(result["dir"]).parent / f"transcript_{c['modele']}_{cfg['langue'] or 'auto'}.json"
    words = transcribe.all_words(json.loads(transcript.read_text(encoding="utf-8"))) if transcript.exists() else []
    OUT_DIR.mkdir(exist_ok=True)
    ready = []
    for clip in result["clips"]:
        if clip["virality"] < cfg["viralite_min"]:
            log(f"  {clip['name']} écarté, viralité {clip['virality']}")
            continue
        src = Path(result["dir"]) / clip["file"]
        dest = OUT_DIR / f"{datetime.now():%Y%m%d}_{video['id']}_{clip['name']}.mp4"
        sound = song = None
        if sounds:
            sound = sounds[state["morceau"] % len(sounds)]
            state["morceau"] += 1
        elif songs:
            song = songs[state["morceau"] % len(songs)]
            state["morceau"] += 1
            music.add_music(src, song, dest, cfg["musique"]["niveau_db"], seed=hash(dest.name))
        if not song:
            dest.write_bytes(src.read_bytes())
        entry = {
            "fichier": str(dest), "video": video["id"], "titre": clip["title"], "viralite": clip["virality"],
            "son": sound, "sans_parole": bool(words) and _speech_ratio(clip, words) < SPEECHLESS,
            "musique": song.name if song else None, "legende": _caption(clip, video),
            "date": datetime.now().strftime("%d/%m/%Y %H:%M"), "statut": "prêt",
        }
        _write_sheet(entry)
        log(f"  prêt : {dest.name} (viralité {clip['virality']}{', son ' + sound if sound else ''})")
        ready.append(entry)
    return ready


def run_agent(n_clips: int | None = None, tiktok: bool | None = None, until: datetime | None = None,
              log=print) -> int:
    """Produit n_clips nouveaux clips dans a_publier/. Avec tiktok (ou envoi_tiktok dans la config), les envoie
    aussi en brouillon par l'API TikTok. S'arrête avant une nouvelle vidéo passé until. Renvoie le nombre produit."""
    cfg = load_config()
    state = load_state()
    target = n_clips or cfg["clips_par_lancement"]
    tiktok = cfg["envoi_tiktok"] if tiktok is None else tiktok
    acc = account() if tiktok else None
    if acc is not None and not acc.connected:
        raise TikTokError("Compte TikTok non connecté : lance d'abord 'python -m clipper agent login'")

    sent = send_ready(state, acc, target, log)  # d'abord les clips restés prêts au lancement précédent
    want = target - sent
    if want <= 0:
        log(f"{sent} brouillons envoyés, rien de nouveau à produire")
        return 0

    candidates = search(cfg, state, log)
    log(f"{len(candidates)} vidéos retenues")

    produced = 0
    for video in candidates:
        if produced >= want or (until and datetime.now() >= until):
            break
        log(f"→ {video['channel']} ({video['followers'] or '?'} abonnés) : {video['title']}")
        state["videos"][video["id"]] = {"titre": video["title"], "chaine": video["channel"], "url": video["url"],
                                        "licence": video["license"], "statut": "en cours",
                                        "date": datetime.now().isoformat(timespec="seconds")}
        write_json_atomic(STATE, state)
        try:
            ready = produce(video, cfg, state, log)
        except Exception as exc:  # une vidéo qui casse ne doit pas arrêter l'agent
            if "not a bot" in str(exc):  # blocage de YouTube : la vidéo n'y est pour rien, elle sera reprise
                del state["videos"][video["id"]]
                write_json_atomic(STATE, state)
                raise discover.Blocked("YouTube bloque le téléchargement (vérification anti-robot). "
                                       "Relance dans quelques heures.") from None
            state["videos"][video["id"]].update(statut="échec", erreur=str(exc)[:300])
            write_json_atomic(STATE, state)
            log(f"  échec : {exc}")
            continue
        state["clips"].extend(ready)
        state["videos"][video["id"]]["statut"] = f"{len(ready)} clips"
        write_json_atomic(STATE, state)
        produced += len(ready)
        sent += send_ready(state, acc, target - sent, log)

    log(f"Terminé : {produced} nouveaux clips dans {OUT_DIR}, fiches dans {INDEX}"
        + (f" · {sent} brouillons envoyés sur TikTok" if tiktok else ""))
    return produced


def preview(log=print) -> list[dict]:
    """Recherche seule, sans rien télécharger : pour vérifier ce que l'agent choisirait."""
    found = search(load_config(), load_state(), log)
    for v in found:
        log(f"[{v['interest']:3}] {v['channel']} ({v['followers'] or '?'} abonnés) · {v['title']}\n"
            f"      {v['url']} · {v['duration'] // 60} min · {v.get('vet_reason') or 'chaîne autorisée'}")
    return found


def status(log=print) -> None:
    cfg = load_config()
    state = load_state()
    if cfg["envoi_tiktok"]:
        log(f"Envoi TikTok : {'connecté' if account().connected else 'non connecté'}")
    else:
        log("Mode local : les clips restent dans " + str(OUT_DIR))
    log(f"Vidéos traitées : {len(state['videos'])} · clips produits : {len(state['clips'])}")
    log(f"Sons TikTok dans la liste : {len(tiktok_sounds(cfg))} · "
        f"morceaux à mixer : {len(music.tracks(ROOT / cfg['musique']['dossier']))}")
