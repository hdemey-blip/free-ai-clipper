"""Recherche de vidéos réutilisables sur YouTube pour l'agent.

Deux sources seulement :
- les vidéos publiées sous licence Creative Commons Attribution (CC BY) par leur auteur ;
- les chaînes qui ont donné leur accord (liste `autorisees` de la config de l'agent).

Le filtre Creative Commons de YouTube ne suffit pas : beaucoup de chaînes repostent des reportages d'Arte, TF1
ou M6 en cochant CC BY sans en avoir le droit. Chaque candidat passe donc par un filtre de mots (marques connues,
« compilation »…) puis Claude juge, sur la chaîne, le titre et la description, si c'est l'œuvre de la chaîne.
"""

import random
import re
import time
import urllib.parse

from .fast import ydl_options
from .util import ask_claude

# paramètre sp de la recherche YouTube : « Creative Commons », seul ou avec une date de mise en ligne
CC_FILTERS = {None: "EgIwAQ%253D%253D", "semaine": "EgQIAzAB", "mois": "EgQIBDAB", "annee": "EgQIBTAB"}
CC_LICENSE = "Creative Commons"

# chaîne ou titre qui trahit un repost de diffuseur
SUSPECT = re.compile(
    r"\b(arte|tf1|m6|france ?[2345]|france ?t[ée]l[ée]visions?|france ?24|canal\+?|bfm|lci|rmc|w9|c8|nrj ?12|"
    r"envoy[ée] sp[ée]cial|zone interdite|enqu[êe]te exclusive|capital|complément d'enquête|cash investigation|"
    r"national geographic|nat ?geo|discovery|bbc|netflix|history channel|documentaires? (hd|en fran[çc]ais|complet)|"
    r"doc(umentaire)? complet|full documentary|compilation|top ?doc)\b", re.I)

VET_SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "original": {"type": "boolean",
                                 "description": "Vrai seulement si la chaîne a très probablement tourné ce reportage"},
                    "interesting": {"type": "integer", "description": "0-100 : potentiel de clips courts captivants"},
                    "reason": {"type": "string"},
                },
                "required": ["id", "original", "interesting", "reason"],
            },
        },
    },
    "required": ["verdicts"],
}

VET_SYSTEM = (
    "Tu tries des vidéos YouTube marquées Creative Commons pour un compte TikTok qui en tire des extraits en "
    "créditant l'auteur. Beaucoup de chaînes repostent des documentaires de télévision en cochant à tort la licence "
    "CC : ces vidéos-là sont à exclure. Marque original=true seulement quand tout indique que la chaîne a produit "
    "la vidéo elle-même (créateur indépendant qui filme ses propres reportages, voyages, enquêtes, conférences). "
    "Au moindre doute (chaîne qui publie des documentaires de diffuseurs variés, titre traduit, mention d'une "
    "chaîne TV, nom générique du type « Documentaires HD »), mets false. Note ensuite l'intérêt pour des clips "
    "TikTok : sujet surprenant, parole claire, histoires concrètes."
)
ANGLE_RULE = (" Ligne éditoriale du compte : {angle}. La note d'intérêt mesure d'abord l'adéquation à cette ligne : "
              "en dessous de 40 pour une vidéo qui n'y correspond pas, même si elle est bonne.")


class Blocked(RuntimeError):
    """YouTube demande de se connecter pour prouver qu'on n'est pas un robot."""


def _ydl(extra: dict | None = None):
    import yt_dlp
    return yt_dlp.YoutubeDL(ydl_options(skip_download=True, **(extra or {})))


def search(query: str, limit: int = 20, recent: str | None = None) -> list[dict]:
    """Résultats bruts de la recherche YouTube sous licence CC (id, titre, chaîne, durée), sans ouvrir chaque vidéo.
    recent : None, "semaine", "mois" ou "annee" (date de mise en ligne)."""
    url = "https://www.youtube.com/results?search_query=" + urllib.parse.quote(query) + "&sp=" + CC_FILTERS[recent]
    with _ydl({"extract_flat": "in_playlist", "playlistend": limit}) as ydl:
        info = ydl.extract_info(url, download=False)
    return [e for e in info.get("entries") or [] if e.get("id")]


def channel_videos(channel_url: str, limit: int = 15) -> list[dict]:
    """Dernières vidéos d'une chaîne autorisée."""
    url = channel_url.rstrip("/") + "/videos"
    with _ydl({"extract_flat": "in_playlist", "playlistend": limit}) as ydl:
        info = ydl.extract_info(url, download=False)
    return [e for e in info.get("entries") or [] if e.get("id")]


def details(video_id: str) -> dict:
    """Licence, abonnés de la chaîne, description : il faut ouvrir la page de la vidéo."""
    with _ydl() as ydl:
        i = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
    return {
        "id": i["id"], "url": i.get("webpage_url") or f"https://www.youtube.com/watch?v={video_id}",
        "title": i.get("title") or "", "channel": i.get("channel") or i.get("uploader") or "",
        "channel_url": i.get("channel_url") or "", "handle": i.get("uploader_id") or "",
        "followers": i.get("channel_follower_count"), "duration": i.get("duration") or 0,
        "license": i.get("license") or "", "language": i.get("language") or "",
        "description": (i.get("description") or "")[:1200], "upload_date": i.get("upload_date") or "",
    }


def _safe_details(video_id: str) -> dict | None:
    time.sleep(random.uniform(2.0, 5.0))  # rythme humain : YouTube bloque les rafales de requêtes
    try:
        return details(video_id)
    except Exception as exc:
        if "not a bot" in str(exc):
            raise Blocked("YouTube bloque les requêtes de cette connexion (vérification anti-robot). "
                          "Relance dans quelques heures.") from None
        return None  # vidéo privée, supprimée ou bloquée dans le pays


def _same_channel(a: str, b: str) -> bool:
    norm = lambda u: u.lower().rstrip("/").removesuffix("/videos")
    return bool(a) and norm(a) == norm(b)


def vet(candidates: list[dict], angle: str = "", log=print) -> list[dict]:
    """Garde les vidéos que Claude juge originales, triées par intérêt (pour la ligne éditoriale angle)."""
    if not candidates:
        return []
    listing = "\n\n".join(
        f"id: {c['id']}\nchaîne: {c['channel']} ({c['followers'] or '?'} abonnés)\ntitre: {c['title']}\n"
        f"durée: {c['duration'] // 60} min · mise en ligne: {c.get('upload_date') or '?'}\n"
        f"description: {c['description'][:600]}"
        for c in candidates)
    out = ask_claude("Juge chacune de ces vidéos. Elles sont fournies sur stdin.", VET_SCHEMA,
                     VET_SYSTEM + (ANGLE_RULE.format(angle=angle) if angle else ""), stdin=listing, effort="medium", timeout=600)
    verdicts = {v["id"]: v for v in out["verdicts"]}
    kept = []
    for c in candidates:
        v = verdicts.get(c["id"])
        if not v:
            continue
        c = {**c, "interest": v["interesting"], "vet_reason": v["reason"]}
        if v["original"]:
            kept.append(c)
        else:
            log(f"  écartée (repost probable) : {c['channel']} | {c['title'][:60]} | {v['reason']}")
    return sorted(kept, key=lambda c: -c["interest"])


def find(topics: list[str], seen: set[str], authorized: list[str], max_followers: int | None,
         min_minutes: int, max_minutes: int, per_topic: int = 15, cache: dict | None = None,
         max_open: int = 20, angle: str = "", recent: str | None = None, min_interest: int = 0,
         log=print) -> list[dict]:
    """Candidats réutilisables, jamais traités, du plus intéressant au moins intéressant.

    Les chaînes autorisées passent sans vérification de licence ni avis de Claude : leur auteur a dit oui.
    cache (id → détails et avis de Claude) est complété sur place : une vidéo n'est ouverte et jugée qu'une fois.
    """
    cache = {} if cache is None else cache
    found: dict[str, dict] = {}

    def lookup(vid: str) -> dict | None:
        if vid not in cache:
            cache[vid] = _safe_details(vid) or {"missing": True}
        return None if cache[vid].get("missing") else cache[vid]

    for channel in authorized:
        try:
            entries = channel_videos(channel)
        except Exception as exc:  # chaîne renommée ou supprimée : on continue avec les autres
            log(f"chaîne autorisée illisible {channel} : {exc}")
            continue
        for e in entries:
            if e["id"] in seen or e["id"] in found:
                continue
            if not min_minutes * 60 <= (e.get("duration") or 0) <= max_minutes * 60:
                continue
            d = lookup(e["id"])
            if d:
                found[e["id"]] = {**d, "authorized": True, "interest": 100}

    entries: dict[str, dict] = {}
    for topic in topics:
        log(f"recherche : {topic}")
        for e in search(topic, per_topic, recent):
            vid = e["id"]
            if vid in seen or vid in found or vid in entries:
                continue
            if not min_minutes * 60 <= (e.get("duration") or 0) <= max_minutes * 60:
                continue
            if SUSPECT.search(f"{e.get('channel') or ''} {e.get('title') or ''}"):
                continue
            entries[vid] = e

    # 55 pages ouvertes d'affilée ont suffi à déclencher l'anti-robot de YouTube : les autres attendront
    new = [vid for vid in entries if vid not in cache]
    todo = [vid for vid in entries if vid in cache] + new[:max_open]
    log(f"{len(entries)} résultats, {min(len(new), max_open)} à ouvrir sur YouTube"
        + (f" ({len(new) - max_open} gardés pour le prochain lancement)" if len(new) > max_open else ""))
    # une page à la fois : YouTube laisse passer une requête isolée mais bloque les rafales
    infos, blocked = [], False
    for vid in todo:
        if blocked and vid not in cache:
            continue
        try:
            d = lookup(vid)
        except Blocked:
            blocked = True
            log("YouTube freine les requêtes : on continue avec les vidéos déjà ouvertes")
            continue
        if d:
            infos.append(d)

    pending = []
    for d in infos:
        if CC_LICENSE not in d["license"]:
            continue
        if max_followers and d["followers"] and d["followers"] > max_followers:
            continue
        if any(_same_channel(d["channel_url"], a) for a in authorized):
            found[d["id"]] = {**d, "authorized": True, "interest": 100}
            continue
        if SUSPECT.search(d["description"][:400]):
            continue
        pending.append({**d, "authorized": False})

    # un avis donné pour une autre ligne éditoriale ne vaut plus
    unjudged = [d for d in pending if cache[d["id"]].get("verdict", {}).get("angle", "") != angle
                or "verdict" not in cache[d["id"]]]
    if unjudged:
        log(f"{len(unjudged)} vidéos CC à vérifier par Claude")
        kept = {c["id"]: c for c in vet(unjudged, angle, log)}
        for d in unjudged:
            k = kept.get(d["id"])
            cache[d["id"]]["verdict"] = ({"original": True, "interest": k["interest"], "reason": k["vet_reason"]}
                                         if k else {"original": False, "interest": 0, "reason": ""}) | {"angle": angle}
    for d in pending:
        v = cache[d["id"]]["verdict"]
        if v["original"] and v["interest"] >= min_interest:
            found[d["id"]] = {**d, "interest": v["interest"], "vet_reason": v["reason"]}
    return sorted(found.values(), key=lambda c: -c["interest"])


def credit(video: dict) -> str:
    """Mention exigée par la licence CC BY : auteur, titre, lien, licence."""
    who = video.get("handle") or video["channel"]
    line = f"Extrait de « {video['title']} » par {who} ({video['url']})"
    return line + (", licence CC BY" if not video.get("authorized") else ", avec l'accord de l'auteur")
