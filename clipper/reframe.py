"""Recadrage 9:16 automatique, plan par plan : suit le visage principal, passe en écran partagé quand deux
personnes sont à l'image, et retombe sur un fond flouté quand personne n'est à l'écran."""

import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .util import Stopped

OUT_W, OUT_H = 1080, 1920
SAMPLE_FPS = 5
DETECT_W = 640
MIN_SHOT_S = 0.4       # un « plan » plus court est rattaché au précédent (flash, mouvement brusque)
CUT_THRESHOLD = 27
MIN_FACE = 0.035        # visage le plus petit retenu, en part de la hauteur (orateur en plan large)
DEAD_ZONE = 0.04      # la caméra ne bouge pas tant que le visage reste dans ±4 % de la largeur
LAYOUTS = ("track", "split", "fit")
YUNET = Path(__file__).parent / "models" / "face_detection_yunet_2023mar.onnx"

_local = threading.local()  # un détecteur par thread : l'interface web traite plusieurs travaux en parallèle


def color_matrix(info: dict) -> str:
    """Matrice YUV → RGB de la source. Sans étiquette, une vidéo HD est presque toujours en BT.709."""
    tag = info.get("color_space")
    known = {"bt709": "bt709", "bt470bg": "bt601", "smpte170m": "bt601", "bt2020nc": "bt2020", "bt2020c": "bt2020"}
    return known.get(tag, "bt709" if info["height"] >= 720 else "bt601")


def read_frames(source: Path, start: float, dur: float, info: dict, width: int | None = None, step: int = 1):
    """Images BGR décodées par ffmpeg directement depuis la source, à cadence constante info["fps"] et avec
    la bonne matrice de couleurs. width : largeur de sortie (taille d'origine sinon). step : une image sur step.
    Les tableaux renvoyés sont réutilisés d'une image à l'autre : les copier pour les garder."""
    w, h = info["width"], info["height"]
    if width:
        w, h = width, int(round(h * width / w / 2)) * 2
    vf = f"fps={info['fps']:.5f}"
    if step > 1:
        vf += f",select='not(mod(n\\,{step}))'"
    vf += f",scale={w}:{h}:in_color_matrix={color_matrix(info)}:flags=bicubic,format=bgr24"
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{start:.3f}", "-i", str(source), "-t", f"{dur:.3f}",
         "-an", "-sn", "-vf", vf, "-fps_mode", "passthrough", "-f", "rawvideo", "-"],
        stdout=subprocess.PIPE,
    )
    buf = np.empty((h, w, 3), dtype=np.uint8)
    view = memoryview(buf).cast("B")
    try:
        while True:
            got = 0
            while got < len(view):
                n = proc.stdout.readinto(view[got:])
                if not n:
                    return
                got += n
            yield buf
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()


@dataclass
class Plan:
    fps: float
    n_frames: int
    frame_layout: np.ndarray                    # indice dans LAYOUTS, par frame
    cx: np.ndarray                              # centre x du cadre (track), par frame
    split: np.ndarray                           # (gauche x, gauche y, droite x, droite y) par frame (split)
    head_top: float | None = None               # haut du visage principal dans l'image 9:16 au début du clip
    stats: dict = field(default_factory=dict)

    @property
    def layout(self) -> str:
        """Mise en page dominante, pour l'affichage."""
        return LAYOUTS[int(np.bincount(self.frame_layout, minlength=3).argmax())]

    def layout_at(self, t: float) -> str:
        return LAYOUTS[int(self.frame_layout[min(self.n_frames - 1, max(0, int(t * self.fps)))])]


def _detect_faces(frame: np.ndarray) -> list[tuple[float, float, float]]:
    """Renvoie [(cx, cy, taille)] en pixels de l'image reçue."""
    h, w = frame.shape[:2]
    detector = getattr(_local, "detector", None)
    if detector is None:
        detector = _local.detector = cv2.FaceDetectorYN.create(str(YUNET), "", (w, h), 0.75, 0.3, 50)
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)
    if faces is None:
        return []
    return [(x + bw / 2, y + bh / 2, bw) for x, y, bw, bh in faces[:, :4] if bw >= h * MIN_FACE]


def _follow(target: np.ndarray, dead: float, fps: float) -> np.ndarray:
    """Caméra d'opérateur : immobile tant que la cible reste dans la zone morte, puis elle la rattrape en
    douceur. Un dernier lissage court gomme les à-coups au démarrage des mouvements."""
    if not len(target):
        return target
    gain = 1 - 0.9 ** (30 / fps)
    pos = float(np.median(target[:max(1, int(fps))]))
    out = np.empty(len(target))
    for i, t in enumerate(target):
        d = t - pos
        if abs(d) > dead:
            pos += (d - np.sign(d) * dead) * gain
        out[i] = pos
    window = max(3, int(fps * 0.4))
    if len(out) > window:
        padded = np.pad(out, (window // 2, window - 1 - window // 2), mode="edge")
        out = np.convolve(padded, np.ones(window) / window, mode="valid")
    return out


def _interp(frames: np.ndarray, samples: list[int], values: list, default: float) -> np.ndarray:
    """Valeurs des échantillons (None = rien détecté) interpolées sur les frames demandées."""
    pts = [(i, v) for i, v in zip(samples, values) if v is not None]
    if not pts:
        return np.full(len(frames), default)
    xs, ys = zip(*pts)
    return np.interp(frames, xs, ys)


def analyze(source: Path, start: float, dur: float, info: dict, stop=None) -> Plan:
    """Lit la fenêtre [start, start + dur] de la source à SAMPLE_FPS, repère les plans et les visages, puis
    choisit pour chaque plan une mise en page et une trajectoire de caméra. stop() vrai → Stopped."""
    w, h, fps = info["width"], info["height"], info["fps"]
    step = max(1, round(fps / SAMPLE_FPS))
    scale = DETECT_W / w

    samples, faces, cuts = [], [], []
    prev = None
    for k, frame in enumerate(read_frames(source, start, dur, info, width=DETECT_W, step=step)):
        if stop and k % 20 == 0 and stop():
            raise Stopped
        # écart moyen teinte/saturation/luminosité : un changement de plan entre deux plans sombres reste net
        # (≈ 50 contre moins de 5 pour un mouvement), là où la seule luminosité ne bouge presque pas
        tiny = cv2.cvtColor(cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
        tiny = tiny.astype(np.int16)
        if prev is not None and np.abs(tiny - prev).mean() > CUT_THRESHOLD:
            cuts.append(k * step)
        prev = tiny
        samples.append(k * step)
        faces.append(sorted(((x / scale, y / scale, s / scale) for x, y, s in _detect_faces(frame)),
                            key=lambda f: -f[2]))

    n_frames = max(1, int(round(dur * fps)), samples[-1] + 1 if samples else 1)
    bounds = [0]
    for c in cuts:
        if c - bounds[-1] >= MIN_SHOT_S * fps:
            bounds.append(c)
    if len(bounds) > 1 and n_frames - bounds[-1] < MIN_SHOT_S * fps:
        bounds.pop()
    bounds.append(n_frames)

    is_landscape = w / h > OUT_W / OUT_H * 1.2
    frame_layout = np.full(n_frames, LAYOUTS.index("fit"), dtype=np.uint8)
    cx = np.full(n_frames, w / 2)
    split = np.tile([w * 0.3, h * 0.4, w * 0.7, h * 0.4], (n_frames, 1))
    last_cx = w / 2
    for a, b in zip(bounds, bounds[1:]):
        idx = [i for i, s in enumerate(samples) if a <= s < b]
        shot_samples = [samples[i] for i in idx]
        singles = [faces[i][0] if faces[i] else None for i in idx]
        pairs = []
        for i in idx:
            two = sorted(faces[i][:2], key=lambda f: f[0])
            pairs.append(two if len(two) == 2 and two[1][0] - two[0][0] > w * 0.3 else None)
        n = max(1, len(idx))
        face_ratio = sum(s is not None for s in singles) / n
        pair_ratio = sum(p is not None for p in pairs) / n
        frames = np.arange(a, b)

        if is_landscape and pair_ratio >= 0.5:
            frame_layout[a:b] = LAYOUTS.index("split")
            for col, (side, axis) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
                vals = [p[side][axis] if p else None for p in pairs]
                track = _interp(frames, shot_samples, vals, split[a, col])
                # x suit en douceur ; y reste fixe sur le plan (la tête bouge peu verticalement)
                split[a:b, col] = _follow(track, w * DEAD_ZONE, fps) if axis == 0 else np.median(track)
        elif is_landscape and face_ratio >= 0.35:
            frame_layout[a:b] = LAYOUTS.index("track")
            track = _interp(frames, shot_samples, [s[0] if s else None for s in singles], last_cx)
            cx[a:b] = _follow(track, w * DEAD_ZONE, fps)
            last_cx = cx[b - 1]

    plan = Plan(fps, n_frames, frame_layout, cx, split)
    if frame_layout[0] == LAYOUTS.index("track"):
        # le recadrage garde toute la hauteur : y source → y sortie par simple mise à l'échelle
        tops = [faces[i][0][1] - 0.75 * faces[i][0][2] for i, s in enumerate(samples)
                if s < 3.2 * fps and faces[i]]
        if tops:
            plan.head_top = float(np.percentile(tops, 20)) * OUT_H / h
    n_samples = max(1, len(samples))
    plan.stats = {
        "face_ratio": round(sum(bool(f) for f in faces) / n_samples, 2),
        "shots": len(bounds) - 1,
        "layouts": {name: round(float(np.mean(frame_layout == i)), 2) for i, name in enumerate(LAYOUTS)
                    if np.any(frame_layout == i)},
    }
    return plan


def _crop(frame: np.ndarray, cx: float, crop_w: int, crop_h: int, cy: float | None = None) -> np.ndarray:
    h, w = frame.shape[:2]
    x0 = int(np.clip(cx - crop_w / 2, 0, w - crop_w))
    y0 = int(np.clip((cy if cy is not None else h / 2) - crop_h / 2, 0, h - crop_h))
    return frame[y0:y0 + crop_h, x0:x0 + crop_w]


class Canvas:
    """Image de sortie 1080x1920 allouée une fois et réutilisée à chaque frame : les redimensionnements
    écrivent directement dedans, sans tableau intermédiaire."""

    def __init__(self):
        self.buf = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)

    def render(self, frame: np.ndarray, plan: Plan, idx: int) -> np.ndarray:
        h, w = frame.shape[:2]
        idx = min(idx, plan.n_frames - 1)
        layout = LAYOUTS[plan.frame_layout[idx]]
        buf = self.buf

        if layout == "track":
            crop_w = min(w, int(round(h * OUT_W / OUT_H)))
            cv2.resize(_crop(frame, plan.cx[idx], crop_w, h), (OUT_W, OUT_H), dst=buf,
                       interpolation=cv2.INTER_CUBIC)
            return buf

        if layout == "split":
            half_h = OUT_H // 2
            crop_h = int(h * 0.8)
            crop_w = min(w, int(crop_h * OUT_W / half_h))
            lx, ly, rx, ry = plan.split[idx]
            for x, y, rows in ((lx, ly, slice(0, half_h)), (rx, ry, slice(half_h, OUT_H))):
                # visage aux deux cinquièmes du panneau : de l'air au-dessus de la tête, pas trop
                part = _crop(frame, x, crop_w, crop_h, cy=y + crop_h * 0.1)
                cv2.resize(part, (OUT_W, rows.stop - rows.start), dst=buf[rows], interpolation=cv2.INTER_CUBIC)
            buf[half_h - 3:half_h + 3] = 0
            return buf

        # fit : vidéo entière au centre, sur un fond flouté et assombri calculé en tout petit
        bg_w = min(w, int(h * OUT_W / OUT_H))
        bg_h = min(h, int(w * OUT_H / OUT_W))
        small = cv2.resize(_crop(frame, w / 2, bg_w, bg_h), (OUT_W // 8, OUT_H // 8), interpolation=cv2.INTER_AREA)
        small = cv2.convertScaleAbs(cv2.GaussianBlur(small, (0, 0), 4), alpha=0.55)
        cv2.resize(small, (OUT_W, OUT_H), dst=buf, interpolation=cv2.INTER_LINEAR)
        scale = min(OUT_W / w, OUT_H / h)
        fw, fh = int(w * scale), int(h * scale)
        y, x = (OUT_H - fh) // 2, (OUT_W - fw) // 2
        cv2.resize(frame, (fw, fh), dst=buf[y:y + fh, x:x + fw], interpolation=cv2.INTER_CUBIC)
        return buf
