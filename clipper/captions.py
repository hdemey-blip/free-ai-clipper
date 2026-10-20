"""Sous-titres animés mot à mot (fichier ASS brûlé par ffmpeg/libass)."""

import re
from dataclasses import dataclass
from pathlib import Path

from .transcribe import SENTENCE_END

FONTS_DIR = Path(__file__).parent / "fonts"   # Montserrat (OFL), chargée par libass via fontsdir

# Zones sûres TikTok / Reels / Shorts en 1080x1920 : la barre d'onglets occupe le haut, la légende et la
# musique le bas, la colonne d'icônes la droite entre y ≈ 800 et 1500.
HOOK_TOP = 300
CAPTION_BOTTOM = 600
SIDE_MARGIN, RIGHT_MARGIN = 90, 150


@dataclass
class CaptionStyle:
    font: str = "Montserrat Black"
    size: int = 82
    color: str = "#FFFFFF"
    highlight: str = "#FFE500"
    keyword: str = "#3CFF7A"      # mots clés choisis par Claude et nombres, colorés en permanence
    outline: str = "#000000"
    outline_width: int = 7
    uppercase: bool = True
    max_words: int = 3
    max_chars: int = 18
    pop: bool = True
    hook_font: str = "Montserrat ExtraBold"


PRESETS = {
    "bold": CaptionStyle(),
    "vert": CaptionStyle(highlight="#3CFF7A", keyword="#FFE500"),
    "rose": CaptionStyle(highlight="#FF4FD8", keyword="#FFE500", font="Impact", size=96),
    "sobre": CaptionStyle(font="Montserrat", size=68, highlight="#FFFFFF", keyword="#FFFFFF", uppercase=False,
                          max_words=5, max_chars=28, pop=False, outline_width=5, hook_font="Montserrat"),
}


NAMED_COLORS = {
    "jaune": "#FFE500", "yellow": "#FFE500", "vert": "#3CFF7A", "green": "#3CFF7A",
    "rose": "#FF4FD8", "pink": "#FF4FD8", "rouge": "#FF3B3B", "red": "#FF3B3B",
    "bleu": "#2EA8FF", "blue": "#2EA8FF", "orange": "#FF9B1A", "violet": "#A45CFF", "purple": "#A45CFF",
    "blanc": "#FFFFFF", "white": "#FFFFFF", "noir": "#000000", "black": "#000000", "cyan": "#00E5FF",
}


def normalize_color(value: str) -> str:
    """'#ffe500', 'FFE500', '#FE0' ou un nom ('jaune', 'green'...) → '#FFE500'. ValueError sinon."""
    v = value.strip().lower()
    if v in NAMED_COLORS:
        return NAMED_COLORS[v]
    h = v.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) != 6 or any(c not in "0123456789abcdef" for c in h):
        raise ValueError(f"Couleur invalide : {value!r} (exemples : #FFE500, #FE0, jaune)")
    return "#" + h.upper()


def _ass_color(hex_color: str) -> str:
    h = normalize_color(hex_color).lstrip("#")
    r, g, b = h[0:2], h[2:4], h[4:6]
    return f"&H00{b}{g}{r}"


def _ts(t: float) -> str:
    t = max(0.0, t)
    hours, rem = divmod(t, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{int(hours)}:{int(minutes):02d}:{secs:05.2f}"


def _clean(word: str, upper: bool) -> str:
    word = word.replace("\\", "").replace("{", "(").replace("}", ")")
    return word.upper() if upper else word


def _key(word: str) -> str:
    return re.sub(r"[^\w]", "", word.lower())


def _group(words: list[dict], style: CaptionStyle) -> list[list[dict]]:
    groups, cur = [], []
    for w in words:
        text_len = sum(len(x["word"]) + 1 for x in cur) + len(w["word"])
        gap = cur and w["start"] - cur[-1]["end"] > 0.6
        if cur and (len(cur) >= style.max_words or text_len > style.max_chars or gap):
            groups.append(cur)
            cur = []
        cur.append(w)
        if w["word"].endswith(SENTENCE_END):
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    return groups


def build_ass(words: list[dict], style: CaptionStyle, layout_at, hook: str | None = None,
              keywords: list[str] = (), head_top: float | None = None) -> str:
    """words : mots du clip, horodatés sur la timeline du clip rendu. layout_at(t) donne la mise en page à
    l'instant t : en écran partagé, les sous-titres passent au milieu, sur la jointure des deux panneaux.
    head_top : haut du visage principal (y en 1920) au début du clip, pour placer l'accroche."""
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{style.font},{style.size},{_ass_color(style.color)},&H000000FF,{_ass_color(style.outline)},&H96000000,-1,0,0,0,100,100,0,0,1,{style.outline_width},3,2,{SIDE_MARGIN},{RIGHT_MARGIN},{CAPTION_BOTTOM},1
Style: Hook,{style.hook_font},60,&H00000000,&H000000FF,&H00FFFFFF,&H00FFFFFF,-1,0,0,0,100,100,0,0,3,20,0,8,{SIDE_MARGIN + 20},{RIGHT_MARGIN + 20},{HOOK_TOP},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = []
    if hook:
        # l'accroche ne doit pas couvrir un visage : en écran partagé elle se pose au-dessus de la jointure,
        # sur un gros plan (tête dans la zone du haut) juste au-dessus des sous-titres
        if layout_at(0) == "split":
            where = "\\an2\\pos(540,880)"
        elif head_top is not None and head_top < HOOK_TOP + 170:
            where = f"\\an2\\pos(540,{1920 - CAPTION_BOTTOM - 150})"
        else:
            where = ""
        lines.append(f"Dialogue: 1,{_ts(0)},{_ts(3.2)},Hook,,0,0,0,,"
                     f"{{{where}\\fad(120,250)\\fscx92\\fscy92\\t(0,140,\\fscx100\\fscy100)}}{_clean(hook, False)}")

    hi, kw = _ass_color(style.highlight), _ass_color(style.keyword)
    key_set = {_key(t) for k in keywords for t in k.split() if len(_key(t)) >= 3}

    def is_key(word: str) -> bool:
        k = _key(word)
        return k in key_set or any(c.isdigit() for c in k)

    groups = _group(words, style)
    for gi, group in enumerate(groups):
        next_start = groups[gi + 1][0]["start"] if gi + 1 < len(groups) else group[-1]["end"] + 0.4
        group_end = min(next_start, group[-1]["end"] + 0.35)
        pos = "{\\an5\\pos(540,960)}" if layout_at(group[0]["start"]) == "split" else ""
        for wi, word in enumerate(group):
            start = word["start"]
            end = group[wi + 1]["start"] if wi + 1 < len(group) else group_end
            if end <= start:
                continue
            parts = []
            for k, other in enumerate(group):
                txt = _clean(other["word"], style.uppercase)
                if k == wi:
                    # le mot prononcé grossit d'un coup puis se pose : l'œil suit la voix
                    pop = "\\fscx100\\fscy100\\t(0,70,\\fscx120\\fscy120)\\t(70,160,\\fscx112\\fscy112)" \
                        if style.pop else ""
                    parts.append(f"{{\\c{kw if is_key(other['word']) else hi}{pop}}}{txt}{{\\r}}")
                elif is_key(other["word"]):
                    parts.append(f"{{\\c{kw}}}{txt}{{\\r}}")
                else:
                    parts.append(txt)
            # fondu d'apparition de la ligne : \fad vaut pour toute la ligne, les {\r} des mots ne l'annulent pas
            intro = "{\\fad(70,0)}" if style.pop and wi == 0 else ""
            lines.append(f"Dialogue: 0,{_ts(start)},{_ts(end)},Cap,,0,0,0,,{pos}{intro}{' '.join(parts)}")
    return header + "\n".join(lines) + "\n"
