"""
Cat Oracle — Video Compositor Backend
=====================================
Incruste les drapeaux directement dans les pixels de la vidéo (rendu "cuit").

Pipeline :
  1. Téléchargement + cache de la vidéo source (httpx, streaming)
  2. Warp homographique de chaque drapeau sur les 4 coins de sa feuille (OpenCV)
  3. Fusion multiply pleine-trame via FFmpeg (le blanc du canvas = neutre)
  4. Encodage H.264 (libx264, crf 18, yuv420p, +faststart) + cache résultat

Lancement :
  pip install -r requirements.txt   (+ ffmpeg installé sur le système)
  uvicorn main:app --port 8000
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import anyio
import cv2
import httpx
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("compositor")

BASE_DIR = Path(__file__).parent
SOURCE_CACHE = BASE_DIR / "cache_sources"
RENDER_DIR = BASE_DIR / "rendered"
SOURCE_CACHE.mkdir(exist_ok=True)
RENDER_DIR.mkdir(exist_ok=True)

MAX_VIDEO_MB = 200

# ffmpeg système si présent, sinon binaires statiques téléchargés automatiquement
FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
if not (FFMPEG and FFPROBE):
    from static_ffmpeg import run as _sf  # pip install static-ffmpeg
    log_msg = "ffmpeg système introuvable → téléchargement des binaires statiques…"
    print(log_msg)
    FFMPEG, FFPROBE = _sf.get_or_fetch_platform_executables_else_raise()

app = FastAPI(title="Cat Oracle Video Compositor", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # en production : restreindre au domaine du front
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/videos", StaticFiles(directory=RENDER_DIR), name="videos")


# ---------------------------------------------------------------- modèles
class Overlay(BaseModel):
    """Un visuel à imprimer sur une feuille."""
    kind: str = Field("flag", description="flag = image opaque effet tissu | draw = texte DRAW encré sur la feuille")
    image: Optional[str] = Field(None, description="PNG en data URI (base64) — requis pour kind=flag")
    slot: Optional[str] = Field(None, description="left | middle | right — pour la détection auto")
    quad: Optional[List[List[float]]] = Field(
        None, min_length=4, max_length=4,
        description="4 coins [x%, y%] : TL, TR, BR, BL (fallback si détection auto impossible)",
    )


class RenderRequest(BaseModel):
    video_url: str
    overlays: List[Overlay] = Field(..., min_length=1, max_length=3)
    opacity: float = Field(1.0, ge=0.5, le=1.0)
    blur_sigma: float = Field(0.6, ge=0.0, le=3.0, description="Flou pour matcher la netteté vidéo")
    auto_detect: bool = Field(True, description="Détecter automatiquement les feuilles blanches")


class RenderResponse(BaseModel):
    url: str
    cached: bool


# ---------------------------------------------------------------- helpers
def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:24]


async def fetch_source(url: str) -> Path:
    """Télécharge la vidéo source en streaming, avec cache disque."""
    key = _sha(url.encode())
    dest = SOURCE_CACHE / f"{key}.mp4"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_suffix(".part")
    try:
        async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
            async with client.stream("GET", url) as r:
                r.raise_for_status()
                size = 0
                with tmp.open("wb") as f:
                    async for chunk in r.aiter_bytes(1 << 16):
                        size += len(chunk)
                        if size > MAX_VIDEO_MB << 20:
                            raise HTTPException(413, "Vidéo source trop volumineuse")
                        f.write(chunk)
        tmp.rename(dest)
        return dest
    except httpx.HTTPError as e:
        tmp.unlink(missing_ok=True)
        raise HTTPException(502, f"Téléchargement vidéo impossible : {e}") from e


def probe_video(video: Path) -> tuple[int, int, float]:
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-show_entries", "format=duration",
         "-of", "json", str(video)],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(out.stdout)
    s = data["streams"][0]
    duration = float(data["format"]["duration"])
    return int(s["width"]), int(s["height"]), duration


def decode_data_uri(uri: str) -> np.ndarray:
    try:
        b64 = uri.split(",", 1)[1] if "," in uri else uri
        buf = np.frombuffer(base64.b64decode(b64), np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)  # BGR
        if img is None:
            raise ValueError("format non décodable")
        return img
    except Exception as e:
        raise HTTPException(422, f"Image overlay invalide : {e}") from e


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Ordonne 4 points en TL, TR, BR, BL."""
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)


def _expand_quad(quad: np.ndarray, factor: float = 1.05) -> np.ndarray:
    """Dilate légèrement le quad autour de son centre pour couvrir les bords sans liseré blanc."""
    c = quad.mean(axis=0)
    return c + (quad - c) * factor


def detect_papers(video: Path, w: int, h: int) -> Optional[Dict[str, List[List[float]]]]:
    """Détecte les 3 feuilles blanches (frame de début : le chat est loin).
    Retourne {slot: quad %} ou None si la détection échoue."""
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, 400)  # ~0,4 s : scène posée, chat au fond
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # blanc = faible saturation + forte luminosité
    mask = cv2.inRange(hsv, (0, 0, 165), (180, 70, 255))
    mask[: int(h * 0.55), :] = 0  # les feuilles sont dans la moitié basse
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    quads = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 0.004 * w * h:  # ignorer le bruit (bols clairs, reflets…)
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            # fallback : rectangle orienté minimal
            approx = cv2.boxPoints(cv2.minAreaRect(cnt)).astype(np.int32).reshape(-1, 1, 2)
        quads.append((area, _order_corners(approx.reshape(4, 2).astype(np.float32))))

    if len(quads) < 3:
        log.warning("détection feuilles : %d trouvée(s), fallback quads front", len(quads))
        return None
    quads = sorted(quads, key=lambda q: -q[0])[:3]              # 3 plus grandes
    quads = sorted(quads, key=lambda q: q[1][:, 0].mean())      # tri gauche → droite
    result = {}
    for slot, (_, quad) in zip(("left", "middle", "right"), quads):
        quad = _expand_quad(quad)
        result[slot] = [[float(x) / w * 100, float(y) / h * 100] for x, y in quad]
    log.info("feuilles détectées : %s", {k: [[round(a, 1), round(b, 1)] for a, b in v] for k, v in result.items()})
    return result


def apply_fabric(img: np.ndarray) -> np.ndarray:
    """Effet drapeau en tissu posé : plis ondulés + lumière directionnelle."""
    ih, iw = img.shape[:2]
    x = np.linspace(0, 3 * np.pi, iw)
    folds = 1.0 + 0.07 * np.sin(x * 1.7) + 0.04 * np.sin(x * 4.3 + 1.2)   # plis verticaux
    light = np.linspace(1.06, 0.92, ih).reshape(-1, 1)                     # lumière du haut
    shade = (folds.reshape(1, -1) * light)[..., None]
    return np.clip(img.astype(np.float32) * shade, 0, 255).astype(np.uint8)


def make_draw_ink() -> tuple[np.ndarray, np.ndarray]:
    """Texte DRAW en encre anti-aliasée : (couleur BGR, alpha).
    Composité sur la vraie feuille de la vidéo → rendu imprimé réaliste."""
    mask = np.zeros((230, 640), np.uint8)
    cv2.putText(mask, "DRAW", (46, 172), cv2.FONT_HERSHEY_TRIPLEX, 4.6, 255, 13, cv2.LINE_AA)
    mask = cv2.GaussianBlur(mask, (0, 0), 1.0)          # bords doux, effet encre
    ink = np.zeros((230, 640, 3), np.uint8)
    ink[:] = (55, 52, 48)                                # gris encre chaud (BGR)
    alpha = (mask * 0.88).astype(np.uint8)               # encre légèrement absorbée
    return ink, alpha


def _warp_pair(img: np.ndarray, alpha: np.ndarray, quad_pct, w: int, h: int):
    ih, iw = img.shape[:2]
    src = np.float32([[0, 0], [iw, 0], [iw, ih], [0, ih]])
    dst = np.float32([[x / 100 * w, y / 100 * h] for x, y in quad_pct])
    m = cv2.getPerspectiveTransform(src, dst)
    wi = cv2.warpPerspective(img, m, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    wa = cv2.warpPerspective(alpha, m, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return wi, wa


def build_overlay_canvas(w: int, h: int, overlays: list[Overlay], blur_sigma: float) -> np.ndarray:
    """Canvas RGBA transparent ; drapeaux opaques effet tissu couvrant la feuille,
    texte DRAW encré (semi-transparent) centré sur la feuille du milieu."""
    canvas = np.zeros((h, w, 4), np.uint8)  # BGRA
    for ov in overlays:
        if not ov.quad:
            continue
        if ov.kind == "draw":
            quad = _expand_quad(np.float32(ov.quad), 0.82)  # centré, marges de papier autour
            img, alpha = make_draw_ink()
        else:
            if not ov.image:
                continue
            quad = np.float32(ov.quad)
            img = apply_fabric(decode_data_uri(ov.image))
            alpha = np.full(img.shape[:2], 255, np.uint8)
        wi, wa = _warp_pair(img, alpha, quad.tolist(), w, h)
        a = (wa.astype(np.float32) / 255)[..., None]
        canvas[..., :3] = (canvas[..., :3] * (1 - a) + wi * a).astype(np.uint8)
        canvas[..., 3] = np.maximum(canvas[..., 3], wa)
    if blur_sigma > 0:  # adoucit les bords + matche la netteté vidéo
        canvas = cv2.GaussianBlur(canvas, (0, 0), blur_sigma)
    return canvas


def composite(video: Path, canvas_png: Path, opacity: float, duration: float, out: Path) -> None:
    """Fusion multiply pleine trame + réencodage H.264 (bonnes pratiques web)."""
    cmd = [
        FFMPEG, "-y",
        "-i", str(video),
        # l'image est bornée à la durée de la vidéo (sinon flux infini avec -loop 1)
        "-loop", "1", "-t", f"{duration:.3f}", "-i", str(canvas_png),
        # drapeau opaque : incrustation par canal alpha (overlay), pas de transparence
        "-filter_complex",
        f"[1:v]format=rgba,colorchannelmixer=aa={opacity}[ov];"
        f"[0:v][ov]overlay=0:0,format=yuv420p[v]",
        "-map", "[v]", "-map", "0:a?",
        "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
        "-c:a", "copy",
        "-movflags", "+faststart",
        "-shortest",
        str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        log.error("ffmpeg: %s", proc.stderr[-2000:])
        raise HTTPException(500, "Échec du compositing FFmpeg")


def render_sync(req: RenderRequest, source: Path, out: Path) -> None:
    w, h, duration = probe_video(source)
    if req.auto_detect:
        detected = detect_papers(source, w, h)
        if detected:
            for ov in req.overlays:
                if ov.slot in detected:
                    ov.quad = detected[ov.slot]
    canvas = build_overlay_canvas(w, h, req.overlays, req.blur_sigma)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        canvas_path = Path(tmp.name)
    try:
        cv2.imwrite(str(canvas_path), canvas)
        composite(source, canvas_path, req.opacity, duration, out)
    finally:
        canvas_path.unlink(missing_ok=True)


# ---------------------------------------------------------------- routes
@app.get("/api/health")
async def health():
    return {"status": "ok", "ffmpeg": Path(FFMPEG).name}


@app.get("/api/papers")
async def papers(video_url: str):
    """Coins des 3 feuilles blanches détectées — pour caler l'overlay frontend."""
    source = await fetch_source(video_url)
    w, h, _ = probe_video(source)
    detected = await anyio.to_thread.run_sync(detect_papers, source, w, h)
    if not detected:
        raise HTTPException(404, "Feuilles non détectées dans cette vidéo")
    return detected


@app.post("/api/render", response_model=RenderResponse)
async def render(req: RenderRequest):
    # clé de cache : vidéo + overlays + réglages → jamais deux fois le même rendu
    key = _sha(json.dumps(req.model_dump(), sort_keys=True).encode())
    out = RENDER_DIR / f"{key}.mp4"
    if out.exists() and out.stat().st_size > 0:
        return RenderResponse(url=f"/videos/{out.name}", cached=True)

    source = await fetch_source(req.video_url)
    # CPU-bound → thread pour ne pas bloquer l'event loop
    await anyio.to_thread.run_sync(render_sync, req, source, out)
    log.info("rendu %s (%.1f Mo)", out.name, out.stat().st_size / 1e6)
    return RenderResponse(url=f"/videos/{out.name}", cached=False)


# ------------------------------------------------------------ frontend
# Le backend sert aussi l'application (index.html à la racine du projet) :
# un seul service à déployer, même origine, zéro CORS.
app.mount("/", StaticFiles(directory=str(BASE_DIR.parent), html=True), name="front")
