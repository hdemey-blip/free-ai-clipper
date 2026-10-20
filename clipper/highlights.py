"""Choix des meilleurs moments : Claude lit la transcription, découpe et note chaque clip.

Sans identifiants Anthropic, un score heuristique prend le relais pour que l'outil reste utilisable.
"""

import json
import os
import re
import shutil
from pathlib import Path

from .transcribe import SENTENCE_END
from .util import ask_claude

MODEL = "claude-opus-5-5"

CLIP_SCHEMA = {
    "type": "object",
    "properties": {
        "clips": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "number", "description": "Début en secondes"},
                    "end": {"type": "number", "description": "Fin en secondes"},
                    "title": {"type": "string"},
                    "hook": {"type": "string", "description": "Phrase d'accroche affichée en haut du clip"},
                    "caption": {"type": "string", "description": "Légende du post"},
                    "hashtags": {"type": "array", "items": {"type": "string"}},
                    "keywords": {"type": "array", "items": {"type": "string"},
                                 "description": "Mots du clip colorés dans les sous-titres"},
                    "scores": {
                        "type": "object",
                        "properties": {
                            "hook": {"type": "integer"},
                            "flow": {"type": "integer"},
                            "value": {"type": "integer"},
                            "trend": {"type": "integer"},
                        },
                        "required": ["hook", "flow", "value", "trend"],
                        "additionalProperties": False,
                    },
                    "reason": {"type": "string"},
                },
                "required": ["start", "end", "title", "hook", "caption", "hashtags", "keywords", "scores", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["clips"],
    "additionalProperties": False,
}

SYSTEM = """You are a short-form video editor who has cut thousands of viral TikToks, Reels and Shorts \
from podcasts, interviews and talks. You receive a timestamped transcript and select the moments \
that will perform best as standalone vertical clips.

What makes a good clip:
- It opens on a hook in the first 2 seconds: a bold claim, a question, a surprising number, tension. \
Never start on filler ("so", "and", "um") or mid-sentence.
- It is self-contained: a viewer with no context understands it and gets a payoff \
(an insight, a punchline, a story with an ending, a strong opinion).
- It ends on a complete thought, ideally a punchline or a line that invites comments.
- Clips do not overlap and cover different topics.

Scores are integers 0-100: hook (strength of the first seconds), flow (pacing, no dead air or \
rambling), value (insight, emotion or entertainment), trend (how shareable or debate-worthy the topic is).

Write title, hook, caption and hashtags in the transcript's language. The hook is at most 8 words. \
The caption is 1-2 sentences written for the post, not a summary of the clip. 3 to 5 hashtags, without '#'. \
keywords: 3 to 6 single words copied exactly as spoken in the clip that carry its punch (names, strong \
verbs, surprising words); they are highlighted in a second color in the burned-in subtitles."""


def _format_transcript(words: list[dict]) -> str:
    return "\n".join(f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}" for s in _sentences(words))


def _has_credentials() -> bool:
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    return (Path.home() / ".config" / "anthropic").exists()


def default_clip_count(duration: float) -> int:
    return int(min(12, max(3, duration / 180)))


def _request(transcript: dict, n_clips: int, min_len: int, max_len: int) -> str:
    return (
        f"Video duration: {transcript['duration']:.0f} s. Language: {transcript['language']}.\n"
        f"Select up to {n_clips} clips, each between {min_len} and {max_len} seconds. "
        "Use the timestamps from the transcript for start and end."
    )


def pick_with_cli(transcript: dict, words: list[dict], n_clips: int, min_len: int, max_len: int) -> list[dict]:
    """Passe par la CLI Claude Code (abonnement Claude, aucune clé API).

    MCP, réglages et outils sont coupés : l'appel ne contient que la consigne et la transcription.
    """
    return ask_claude(_request(transcript, n_clips, min_len, max_len) + " The transcript is provided on stdin.",
                      CLIP_SCHEMA, SYSTEM, stdin=f"<transcript>\n{_format_transcript(words)}\n</transcript>")["clips"]


def pick_with_claude(transcript: dict, words: list[dict], n_clips: int, min_len: int, max_len: int) -> list[dict]:
    import anthropic

    client = anthropic.Anthropic()
    user = (
        _request(transcript, n_clips, min_len, max_len)
        + f"\n\n<transcript>\n{_format_transcript(words)}\n</transcript>"
    )
    with client.beta.messages.stream(
        model=MODEL,
        max_tokens=32000,
        system=SYSTEM,
        messages=[{"role": "user", "content": user}],
        output_config={"effort": "high", "format": {"type": "json_schema", "schema": CLIP_SCHEMA}},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    ) as stream:
        response = stream.get_final_message()

    if response.stop_reason == "refusal":
        raise RuntimeError("Claude a refusé la requête")
    if response.stop_reason == "max_tokens":
        raise RuntimeError("Réponse tronquée (max_tokens)")
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)["clips"]


HOOK_WORDS = re.compile(
    r"\b(pourquoi|comment|secret|erreur|jamais|toujours|personne|incroyable|vérité|problème|"
    r"why|how|secret|mistake|never|always|nobody|crazy|truth|problem)\b", re.I)
WEAK_START = re.compile(r"^(et|mais|donc|alors|euh|bah|ben|and|but|so|um|uh|like)\b", re.I)


def _sentences(words: list[dict]) -> list[dict]:
    """Regroupe les mots en phrases (ponctuation finale ou silence de plus de 1,2 s)."""
    out, cur = [], []
    for i, w in enumerate(words):
        cur.append(w)
        nxt = words[i + 1] if i + 1 < len(words) else None
        if w["word"].endswith(SENTENCE_END) or nxt is None or nxt["start"] - w["end"] > 1.2:
            out.append({"start": cur[0]["start"], "end": cur[-1]["end"],
                        "text": " ".join(x["word"] for x in cur)})
            cur = []
    return out


def pick_heuristic(words: list[dict], n_clips: int, min_len: int, max_len: int) -> list[dict]:
    """Fenêtres de phrases entières notées sur le débit, les questions et les mots d'accroche."""
    segs = _sentences(words)
    candidates = []
    for i, first in enumerate(segs):
        text, j = "", i
        while j < len(segs) and segs[j]["end"] - first["start"] <= max_len:
            text += " " + segs[j]["text"]
            length = segs[j]["end"] - first["start"]
            if length >= min_len:
                n_words = len(text.split())
                hook = 30 if HOOK_WORDS.search(first["text"]) or "?" in first["text"] else 0
                weak = 25 if WEAK_START.match(first["text"]) else 0
                pace = min(40, n_words / length * 13)
                punch = min(20, (text.count("?") + text.count("!")) * 5)
                end_clean = 10 if segs[j]["text"].rstrip().endswith(SENTENCE_END) else 0
                candidates.append({"score": hook + pace + punch + end_clean - weak, "start": first["start"],
                                   "end": segs[j]["end"], "text": text.strip(), "opener": first["text"]})
            j += 1

    chosen = []
    for c in _non_overlapping(sorted(candidates, key=lambda c: -c["score"]), n_clips):
        s = int(max(0, min(99, c["score"])))
        opener_words = c["opener"].split()
        hook_line = " ".join(opener_words[:8]).rstrip(",;:") + ("…" if len(opener_words) > 8 else "")
        chosen.append({
            "start": c["start"], "end": c["end"], "title": hook_line, "hook": hook_line,
            "caption": c["text"].split(".")[0][:150], "hashtags": [], "keywords": [],
            "scores": {"hook": s, "flow": s, "value": s, "trend": s},
            "reason": "score heuristique",
        })
    return chosen


def _non_overlapping(ranked: list[dict], n: int) -> list[dict]:
    """Garde, dans l'ordre donné, les intervalles [start, end] qui ne chevauchent aucun déjà gardé, au plus n."""
    kept = []
    for item in ranked:
        if not any(item["start"] < k["end"] and item["end"] > k["start"] for k in kept):
            kept.append(item)
            if len(kept) >= n:
                break
    return kept


def _ends_sentence(words: list[dict], i: int) -> bool:
    """Le mot i termine-t-il une phrase (ponctuation, ou long silence après lui) ?"""
    return (i < 0 or i >= len(words) - 1 or words[i]["word"].endswith(SENTENCE_END)
            or words[i + 1]["start"] - words[i]["end"] > 1.2)


def snap_to_words(clip: dict, words: list[dict], pad: float = 0.12) -> dict:
    """Recale début et fin sur des frontières de mots pour ne jamais couper une syllabe.

    Un clip qui s'ouvre sur la queue de la phrase précédente (« …idées. Souvent on ») démarre à la phrase
    suivante ; une fin coupée à moins de 1,5 s d'une fin de phrase est prolongée jusqu'à elle."""
    i0 = next((i for i, w in enumerate(words) if w["start"] >= clip["start"] - 0.3), None)
    i1 = next((i for i in range(len(words) - 1, -1, -1) if words[i]["end"] <= clip["end"] + 0.3), None)
    if i0 is None or i1 is None or i0 > i1:
        return clip
    if not _ends_sentence(words, i0 - 1):
        i0 = next((j + 1 for j in range(i0, min(i0 + 3, i1)) if _ends_sentence(words, j)), i0)
    if not _ends_sentence(words, i1):
        i1 = next((j for j in range(i1 + 1, min(i1 + 8, len(words))) if words[j]["end"] - words[i1]["end"] <= 1.5
                   and _ends_sentence(words, j)), i1)
    clip["start"] = max(0.0, words[i0]["start"] - pad)
    clip["end"] = words[i1]["end"] + pad
    return clip


PICKERS = {"api": (pick_with_claude, "API"), "cli": (pick_with_cli, "CLI Claude Code")}
BRAINS = ("auto", *PICKERS, "heuristic")


def pick_clips(transcript: dict, words: list[dict], n_clips: int | None = None,
               min_len: int = 20, max_len: int = 70, brain: str = "auto", log=print) -> list[dict]:
    """brain : "api" (clé Anthropic), "cli" (Claude Code), "heuristic", ou "auto" = le premier disponible."""
    n_clips = n_clips or default_clip_count(transcript["duration"])
    if brain == "auto":
        order = (["api"] if _has_credentials() else []) + (["cli"] if shutil.which("claude") else [])
    else:
        order = [brain] if brain in PICKERS else []

    clips = None
    for name in order:
        picker, label = PICKERS[name]
        log(f"Claude ({label}) lit la transcription")
        try:
            clips = picker(transcript, words, n_clips, min_len, max_len)
            log(f"Claude a proposé {len(clips)} clips")
            break
        except Exception as exc:  # réseau, quota, auth : on essaie la suite
            log(f"Échec {name} ({type(exc).__name__}: {exc})")
    if not clips:
        log("Score heuristique" if brain == "heuristic" else
            "Score heuristique (ni clé API ni CLI Claude Code utilisable)" if not order else
            "Passage au score heuristique")
        clips = pick_heuristic(words, n_clips, min_len, max_len)

    return finalize(clips, words, n_clips, min_len, max_len)


def _trim_to(clip: dict, words: list[dict], max_len: float) -> dict:
    """Raccourcit un clip trop long à la dernière fin de phrase (sinon au dernier mot) avant la limite."""
    limit = clip["start"] + max_len
    inside = [w for w in words if clip["start"] <= w["start"] and w["end"] <= limit]
    sentence_ends = [w for w in inside if w["word"].endswith(SENTENCE_END)]
    last = (sentence_ends or inside or [{"end": limit}])[-1]
    return {**clip, "end": last["end"]}


def finalize(clips: list[dict], words: list[dict], n_clips: int, min_len: int, max_len: int) -> list[dict]:
    """Recale sur les mots, fait respecter durées, chevauchements et nombre de clips, puis calcule le score."""
    scored = []
    for clip in clips:
        clip = snap_to_words(dict(clip), words)
        if clip["end"] - clip["start"] > max_len + 1:
            clip = snap_to_words(_trim_to(clip, words, max_len), words)
        if clip["end"] - clip["start"] < max(5, min_len * 0.6):
            continue
        sc = clip["scores"]
        clip["virality"] = max(0, min(100, round(
            0.35 * sc["hook"] + 0.2 * sc["flow"] + 0.3 * sc["value"] + 0.15 * sc["trend"])))
        scored.append(clip)
    return _non_overlapping(sorted(scored, key=lambda c: -c["virality"]), n_clips)
