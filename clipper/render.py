"""Rendu d'un clip : décodage direct de la source, silences resserrés, recadrage image par image,
sous-titres, son masterisé, encodage soigné et vignette."""

import math
import os
import subprocess
from pathlib import Path

import numpy as np

from . import audio, demusic, reframe
from .captions import FONTS_DIR, CaptionStyle, build_ass, normalize_color
from .util import FFMPEG, Stopped, atomic_path

END_PAD = 0.35          # le clip respire un peu après le dernier mot (le son s'y éteint en fondu)
MAX_GAP = 0.8           # un blanc plus long que ça est raccourci…
KEPT_GAP = 0.3          # … à cette durée
MAX_FPS = 60
# preset x264, CRF. "turbo" sert à l'agent : TikTok réencode de toute façon ce qu'on lui envoie
QUALITIES = {"turbo": ("superfast", 20), "fast": ("veryfast", 20), "high": ("slow", 18), "max": ("slower", 16)}
PROGRESS_H = 12


def keep_mask(words: list[dict], n_frames: int, fps: float, tighten: bool) -> np.ndarray:
    """Frames gardées : toutes, sauf le cœur des blancs de plus de MAX_GAP quand tighten est actif.
    Les coupes tombent sur la grille des images, le son est coupé exactement aux mêmes instants."""
    keep = np.ones(n_frames, dtype=bool)
    if tighten:
        for w, nxt in zip(words, words[1:]):
            if nxt["start"] - w["end"] > MAX_GAP:
                a = math.ceil((w["end"] + KEPT_GAP / 2) * fps)
                b = math.floor((nxt["start"] - KEPT_GAP / 2) * fps)
                keep[max(0, a):max(0, b)] = False
    return keep


def _runs(keep: np.ndarray, fps: float) -> list[tuple[float, float]]:
    """Plages gardées, en secondes, à partir du masque des frames."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], keep.astype(np.int8), [0]])))
    return [(a / fps, b / fps) for a, b in zip(edges[::2], edges[1::2])]


def render_clip(source: Path, meta: dict, clip: dict, words: list[dict], out_dir: Path, name: str,
                style: CaptionStyle, logo: Path | None = None, show_hook: bool = True, music: str = "keep",
                tighten: bool = True, quality: str = "high", progress_bar: bool = False, log=print,
                stop=None) -> dict:
    """music : "keep" (audio d'origine) ou "remove" (voix seule). Le mode "auto" est tranché en amont,
    une fois pour toute la vidéo, par pipeline.run. stop() vrai → Stopped, sans fichier final à moitié écrit."""
    out_dir.mkdir(parents=True, exist_ok=True)
    window_wav = out_dir / f"{name}_window.wav"
    voice = out_dir / f"{name}_voice.wav"
    joined = out_dir / f"{name}_joined.wav"
    mastered = out_dir / f"{name}_audio.wav"
    ass = out_dir / f"{name}.ass"
    final = out_dir / f"{name}.mp4"
    cover = out_dir / f"{name}.jpg"

    info = {**meta, "fps": min(meta["fps"], MAX_FPS)}
    fps = info["fps"]
    start = clip["start"]
    dur = min(clip["end"] + END_PAD, meta["duration"]) - start

    try:
        plan = reframe.analyze(source, start, dur, info, stop)
        clip_words = [{**w, "start": w["start"] - start, "end": w["end"] - start}
                      for w in words if w["end"] > start and w["start"] < start + dur]
        keep = keep_mask(clip_words, plan.n_frames, fps, tighten)
        ranges = _runs(keep, fps)
        removed = (len(keep) - int(keep.sum())) / fps
        if removed:
            log(f"{removed:.1f} s de blancs retirés")

        # son : séparation de la voix sur la fenêtre continue (Demucs a besoin du contexte), puis coupes
        # aux mêmes instants que l'image, puis volume calé sur -14 LUFS
        audio_src, audio_start = source, start
        music_removed = False
        if music == "remove":
            subprocess.run([*FFMPEG, "-ss", f"{start:.3f}", "-i", str(source), "-t", f"{dur:.3f}", "-vn",
                            "-ac", "2", "-ar", str(demusic.SR), str(window_wav)], check=True)
            music_removed = demusic.clean_audio(window_wav, voice, music, log=log)
            if music_removed:
                audio_src, audio_start = voice, 0.0
        audio.assemble(audio_src, audio_start, dur, ranges, joined)
        loudness = audio.master(joined, mastered)

        # sous-titres sur la timeline du clip rendu (les blancs retirés décalent les mots)
        kept_before = np.concatenate([[0], np.cumsum(keep)])

        def remap(t: float) -> float:
            return kept_before[int(np.clip(round(t * fps), 0, len(keep)))] / fps

        local = [{**w, "start": remap(w["start"]), "end": remap(w["end"])} for w in clip_words]
        frame_layout = plan.frame_layout[keep]

        def layout_at(t: float) -> str:
            return reframe.LAYOUTS[int(frame_layout[min(len(frame_layout) - 1, max(0, int(t * fps)))])]

        ass.write_text(build_ass(local, style, layout_at, clip.get("hook") if show_hook else None,
                                 clip.get("keywords", []), plan.head_top), encoding="utf-8")

        # ffmpeg tourne dans out_dir : chemins relatifs dans les filtres, sans souci d'échappement Windows
        fonts = Path(os.path.relpath(FONTS_DIR, out_dir)).as_posix()
        cmd = [
            *FFMPEG,
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{reframe.OUT_W}x{reframe.OUT_H}",
            "-r", f"{fps:.5f}", "-i", "-",
            "-i", mastered.name,
        ]
        # netteté adaptative avant les sous-titres (le recadrage agrandit l'image), conversion BT.709 à la fin
        filt = f"[0:v]cas=strength=0.45,ass={ass.name}:fontsdir={fonts}"
        if logo:
            cmd += ["-i", str(logo.resolve())]
            filt += ("[v];[2:v]scale=180:-1,format=rgba,colorchannelmixer=aa=0.9[lg];"
                     "[v][lg]overlay=W-w-60:190")
        filt += (",scale=out_color_matrix=bt709:out_range=tv,format=yuv420p,"
                 "setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709:range=tv[out]")
        preset, crf = QUALITIES[quality]
        bar_color = np.array([int(normalize_color(style.highlight)[i:i + 2], 16) for i in (5, 3, 1)], np.uint8)

        with atomic_path(final) as tmp:
            proc = subprocess.Popen(
                cmd + ["-filter_complex", filt, "-map", "[out]", "-map", "1:a",
                       "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-profile:v", "high",
                       "-g", str(int(round(fps * 2))),
                       "-c:a", "aac", "-b:a", "192k", "-ar", str(audio.SR),
                       "-movflags", "+faststart", "-shortest", tmp.name],
                stdin=subprocess.PIPE, cwd=out_dir,
            )
            canvas = reframe.Canvas()
            total, written = int(keep.sum()), 0
            try:
                for idx, frame in enumerate(reframe.read_frames(source, start, dur, info)):
                    if idx >= len(keep):
                        break
                    if stop and idx % 60 == 0 and stop():
                        raise Stopped
                    if not keep[idx]:
                        continue
                    out = canvas.render(frame, plan, idx)
                    written += 1
                    if progress_bar:
                        out[-PROGRESS_H:, :int(reframe.OUT_W * written / total)] = bar_color
                    proc.stdin.write(out.data)
            except BrokenPipeError:
                pass  # ffmpeg s'est arrêté : son code de retour dit pourquoi
            except Stopped:
                proc.kill()  # inutile de finir d'encoder un clip qu'on jette
                raise
            finally:
                try:
                    proc.stdin.close()
                except BrokenPipeError:
                    pass  # ffmpeg arrêté ou tué : le reste du tampon n'a plus où aller
                proc.wait()
            if proc.returncode != 0:
                raise RuntimeError(f"ffmpeg a échoué sur {name}")

        # vignette : une image du début, accroche affichée
        with atomic_path(cover) as tmp:
            subprocess.run([*FFMPEG, "-ss", "0.9", "-i", str(final), "-frames:v", "1", "-q:v", "2", str(tmp)],
                           check=True)
    finally:
        for f in (window_wav, voice, joined, mastered):
            f.unlink(missing_ok=True)
    return {"file": final.name, "cover": cover.name, "layout": plan.layout, "reframe_stats": plan.stats,
            "music_removed": music_removed, "duration": round(total / fps, 1), "silence_cut": round(removed, 1),
            "loudness_in": loudness.get("input_i")}
