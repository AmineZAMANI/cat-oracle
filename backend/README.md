# Cat Oracle — Backend de compositing vidéo

Incruste les drapeaux **dans les pixels de la vidéo** (rendu cuit, indétectable),
au lieu d'une superposition CSS.

## Stack (choix et raisons)

| Besoin | Librairie | Pourquoi |
|---|---|---|
| API HTTP | **FastAPI + uvicorn** | Async, validation Pydantic, standard actuel |
| Warp perspective du drapeau | **OpenCV** (`getPerspectiveTransform` + `warpPerspective`) | Homographie exacte sur les 4 coins de la feuille |
| Fusion + encodage vidéo | **FFmpeg** (CLI, filtre `blend=multiply`, libx264) | Référence industrie ; multiply = le drapeau hérite de la lumière/ombres réelles du papier ; `crf 18 + yuv420p + faststart` = qualité/compatibilité web |
| Téléchargement source | **httpx** (streaming) | Async, limite de taille, cache disque |

Astuce clé : le canvas d'overlay est **blanc** partout sauf sur les feuilles.
En blend *multiply*, le blanc est neutre → un seul passage FFmpeg pleine trame,
pas de masquage complexe, et le drapeau épouse l'éclairage filmé du papier.

## Lancer

```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -r requirements.txt
# ffmpeg requis : brew install ffmpeg | apt install ffmpeg | winget install ffmpeg
uvicorn main:app --port 8000
```

Puis ouvrir `../index.html` et activer **🎬 Rendu serveur** dans le panneau.

## API

`POST /api/render`
```json
{
  "video_url": "https://…/video.mp4",
  "overlays": [
    {"image": "data:image/png;base64,…", "quad": [[4.5,87.5],[21.5,87.2],[20,100],[0.5,100]]}
  ],
  "opacity": 0.95,
  "blur_sigma": 0.6
}
```
→ `{"url": "/videos/<hash>.mp4", "cached": false}`

Les rendus sont cachés par hash (vidéo + drapeaux + coins) : une combinaison
déjà demandée revient instantanément.

## Limites connues

- Si la caméra de la vidéo générée bouge légèrement, l'incrustation (statique) peut dériver — recalibrer les coins.
- Si le chat passe devant une feuille, le drapeau reste visible par-dessus lui (pas de matting). La solution ultime est de régénérer la vidéo IA avec les drapeaux imprimés dès l'image de départ.
