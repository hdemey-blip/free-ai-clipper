# Clipper

Clipper découpe une vidéo longue (fichier ou lien YouTube) en clips verticaux 9:16 pour TikTok, Reels ou Shorts. Chaque clip a des sous-titres mot à mot, un cadrage qui suit les visages, un son mis au niveau et un score de potentiel viral. Tout tourne en local sur un PC Windows, sans GPU.

Un agent optionnel cherche lui-même sur YouTube des reportages réutilisables (licence Creative Commons) et en tire des clips à la chaîne.

## Installation

Il faut Python 3.12, ffmpeg et curl dans le PATH. Node.js est conseillé, car yt-dlp s'en sert pour lire YouTube.

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt     # Linux / macOS : .venv/bin/pip
```

Pour choisir les moments, Clipper utilise soit la CLI [Claude Code](https://docs.anthropic.com/claude-code) connectée (`claude` dans le PATH), soit une clé `ANTHROPIC_API_KEY`. Sans l'une ni l'autre, il se rabat sur une heuristique locale.

## Utilisation

```bash
python -m clipper video.mp4 -n 5 --music auto    # vidéo → 5 clips dans jobs/<id>/clips/
python -m clipper "https://youtu.be/..." -n 3    # pareil depuis un lien
python -m clipper serve                          # interface web sur http://127.0.0.1:5190
python -m clipper demusic video.mp4              # retire la musique de fond d'une vidéo
```

Options utiles : `--style bold|vert|rose|sobre`, `--highlight "#FFE500"`, `--font "Impact"`, `--model base|small|medium|large-v3`, `--lang fr`, `--logo logo.png`, `--min 20 --max 70`, `--no-hook`, `--no-tighten`, `--quality fast|high|max`, `--progress-bar`.

## Comment ça marche

| Étape | Outil | Fichier |
|---|---|---|
| Récupération | copie locale, ou yt-dlp (ou le téléchargement par morceaux de `fast.py`) | `ingest.py`, `fast.py` |
| Transcription | faster-whisper, horodatage au mot | `transcribe.py` |
| Choix des moments | Claude lit la transcription, choisit et note les clips (accroche, rythme, valeur), écrit titre, accroche, légende et hashtags | `highlights.py` |
| Recadrage | visages détectés par YuNet, mise en page choisie plan par plan : suivi du visage, écran partagé, ou fond flouté | `reframe.py` |
| Sous-titres | ASS mot à mot, mot prononcé en couleur avec un petit rebond, mots clés en seconde couleur | `captions.py` |
| Musique de fond | Demucs sépare la voix du reste | `demusic.py` |
| Son | raccords sans clic, -14 LUFS, blancs de plus de 0,8 s ramenés à 0,3 s | `audio.py` |
| Rendu | H.264 étiqueté BT.709, vignette, logo, barre de progression | `render.py` |

Quelques détails ont beaucoup compté.

L'image est décodée par un pipe ffmpeg directement depuis la source. Passer par `cv2.VideoCapture` convertissait en BT.601 et ternissait les couleurs.

Les changements de plan sont repérés sur l'écart de teinte et de saturation (HSV) et pas seulement sur la luminosité, sinon les coupes entre deux plans sombres passaient inaperçues.

Image et son sont coupés au même endroit, à l'image près, et les sous-titres sont recalés après chaque blanc retiré. L'écart mesuré entre sous-titre et voix est de 0 ms en médiane. Sous-titres et accroche restent hors des zones que couvrent les boutons de l'appli TikTok.

Chaque fichier est écrit sous un nom temporaire puis renommé. Un travail mis en pause ou interrompu repart de là où il s'était arrêté, en réutilisant la transcription partielle, les moments déjà choisis et les clips déjà rendus.

### Le mode rapide

L'agent passe par `fast.py`, qui traite une vidéo YouTube de 25 minutes en un peu plus de 4 minutes sur un portable.

1. Seul l'audio le plus léger est téléchargé d'abord, pour la transcription. Whisper en mode batch met environ 14 s pour 5 minutes d'audio.
2. Une fois les moments choisis, seuls les morceaux de vidéo utiles sont téléchargés. Les formats DASH de YouTube contiennent un index (`sidx`) : on le lit, puis curl récupère en parallèle les plages d'octets des fragments qui couvrent chaque clip.
3. Chaque clip est rendu dès que sa section est arrivée, en parallèle des autres.
4. Si la source a déjà des sous-titres dans l'image, leur bande est détectée (texte clair à contour sombre, stable d'une image à l'autre) puis floutée, pour éviter deux sous-titres l'un sur l'autre.

Pièges rencontrés en route :

- Sur un fichier distant fragmenté, `ffmpeg -ss` lit tout jusqu'au point demandé, soit 27 s pour atteindre la 25e minute. Lire l'index `sidx` règle le problème.
- Les URL de googlevideo répondent souvent 403 à la première requête. curl s'en sort avec `-L` et des tentatives répétées.
- Sous Windows, la ligne de commande est limitée à 32 Ko et une URL YouTube fait environ 2 Ko, donc les requêtes curl passent par un fichier de configuration (`-K`).
- Certaines vidéos ont plusieurs pistes audio à cause du doublage automatique de YouTube. Il faut choisir explicitement la piste de la langue d'origine.
- Ouvrir des dizaines de pages YouTube d'affilée déclenche la vérification anti-robot. L'agent ouvre les pages une par une, avec un délai, et garde un cache.

## L'agent

```bash
python -m clipper agent cherche            # ce que l'agent choisirait, sans rien télécharger
python -m clipper agent run -n 5           # cherche, découpe et prépare 5 clips
python -m clipper agent produit --stock 50 # produit jusqu'à avoir 50 clips prêts
python -m clipper agent statut
```

Les réglages sont dans `agent/config.json`, créé au premier lancement : sujets de recherche, ligne éditoriale, langue, abonnés maximum, durée des vidéos, clips par vidéo, viralité minimale, chaînes autorisées ou exclues.

L'agent n'utilise que deux types de vidéos :

- les vidéos YouTube sous licence Creative Commons Attribution (CC BY), de chaînes en dessous d'un nombre d'abonnés donné ;
- les vidéos des chaînes dont l'auteur a donné son accord (`chaines_autorisees`).

Beaucoup de chaînes repostent des documentaires de télévision en cochant CC BY sans en avoir le droit, si bien que le filtre Creative Commons de YouTube ne suffit pas. Les noms de diffuseurs sont filtrés, puis Claude lit la chaîne, le titre et la description de chaque vidéo pour écarter les reposts.

Les clips arrivent dans `agent/a_publier/`, chacun avec sa fiche `.txt` : légende, hashtags et crédit de l'auteur, obligatoire avec la licence CC BY.

En option, `tiktok.py` envoie les clips dans la boîte de réception TikTok par l'API officielle (Content Posting API, scope `video.upload`). Il faut créer une appli sur developers.tiktok.com, puis lancer `agent login --client-key ... --client-secret ...` et `agent run --tiktok`.

## Droits

Clipper ne donne aucun droit sur les vidéos que vous découpez. Retirer la musique évite les signalements sur la musique, pas ceux sur la vidéo elle-même. Utilisez vos propres vidéos, des vidéos sous une licence qui le permet ou des vidéos dont l'auteur a donné son accord, et créditez-le.

## Comment ce projet a été fait

Le projet a été écrit avec Claude Code (Anthropic), par itérations, en partant du comportement visible des services commerciaux de clipping par IA : choix des meilleurs moments, recadrage vertical, sous-titres animés. Leur résultat a servi de modèle, mais aucune ligne de leur code n'a été reprise et tout le moteur a été écrit de zéro.

Chaque étape a été mesurée avant d'être optimisée. Le passage de 8 minutes à environ 4 minutes par vidéo, par exemple, vient de profils de temps sur le téléchargement, la transcription, le choix des moments et le rendu.

## Licence

MIT. La police Montserrat est sous licence SIL OFL (`clipper/fonts/OFL.txt`). Le modèle de détection de visages YuNet vient d'OpenCV Zoo (licence MIT).
