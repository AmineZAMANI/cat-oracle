#!/bin/bash
# Cat Oracle — démarrage du backend (macOS : double-cliquer ce fichier)
set -e
cd "$(dirname "$0")"

echo "🐱 Cat Oracle — Backend de compositing vidéo"
echo "--------------------------------------------"

# 1. ffmpeg (optionnel : sinon binaires statiques téléchargés automatiquement)
if command -v ffmpeg >/dev/null 2>&1; then
  echo "✅ ffmpeg système : $(ffmpeg -version | head -1 | cut -d' ' -f3)"
else
  echo "ℹ️  ffmpeg absent — les binaires statiques seront téléchargés au 1er lancement"
fi

# 2. environnement virtuel
if [ ! -d .venv ]; then
  echo "📦 Création de l'environnement Python…"
  python3 -m venv .venv
fi
source .venv/bin/activate

# 3. dépendances (installées une seule fois)
if [ ! -f .venv/.deps_ok ]; then
  echo "📦 Installation des dépendances…"
  pip install -q --upgrade pip
  pip install -q -r requirements.txt
  touch .venv/.deps_ok
fi
echo "✅ Dépendances prêtes"

# 4. lancement
echo ""
echo "🚀 Backend démarré sur http://localhost:8000"
echo "   Ouvre index.html et active « 🎬 Rendu serveur : ON »"
echo "   (Ctrl+C pour arrêter)"
echo ""
exec uvicorn main:app --port 8000
