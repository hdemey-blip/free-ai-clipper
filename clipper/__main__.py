"""Ligne de commande.

    python -m clipper <fichier|lien> [options]   vidéo longue → clips
    python -m clipper serve [--port 5190]         interface web
    python -m clipper demusic <vidéo> [-o sortie] retire la musique de fond d'une vidéo
"""

import argparse
import json
import sys
from pathlib import Path

from .captions import PRESETS
from .demusic import MUSIC_MODES
from .highlights import BRAINS
from .pipeline import DEFAULTS, WHISPER_MODELS, Options, run
from .render import QUALITIES


def cmd_serve(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="clipper serve", description="Interface web locale")
    parser.add_argument("--port", type=int, default=5190)
    args = parser.parse_args(argv)
    from .web import serve
    serve(args.port)


def cmd_demusic(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="clipper demusic", description="Retire la musique de fond d'une vidéo")
    parser.add_argument("video", help="Vidéo ou fichier audio d'entrée")
    parser.add_argument("-o", "--output", help="Fichier de sortie (par défaut : <nom>_sans_musique.mp4, ou .m4a pour un audio)")
    parser.add_argument("--keep", type=float, default=0.0,
                        help="Niveau de musique conservé, de 0 (aucune) à 1 (tout), ex. 0.15 pour la garder très bas")
    parser.add_argument("--auto", action="store_true", help="Ne rien changer si aucune musique n'est détectée")
    args = parser.parse_args(argv)

    from .demusic import demusic_file
    src = Path(args.video)
    if not src.is_file():
        parser.error(f"fichier introuvable : {src}")
    try:
        result = demusic_file(src, Path(args.output) if args.output else None,
                              "auto" if args.auto else "remove", args.keep)
    except ValueError as exc:
        parser.error(str(exc))
    print(result if result != src else "Aucune musique détectée, fichier inchangé")


def cmd_agent(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="clipper agent", description="Agent TikTok : reportages → clips à publier")
    sub = parser.add_subparsers(dest="action", required=True)
    go = sub.add_parser("run", help="Cherche, découpe et prépare les clips dans agent/a_publier")
    go.add_argument("-n", type=int, help="Nombre de nouveaux clips (config par défaut)")
    go.add_argument("--tiktok", action="store_true", help="Envoie aussi les clips en brouillon par l'API TikTok")
    sub.add_parser("cherche", help="Montre les vidéos que l'agent choisirait, sans rien télécharger")
    sub.add_parser("statut", help="Clips produits, sons disponibles")
    make = sub.add_parser("produit", help="Produit des clips jusqu'à un stock donné (ou jusqu'à une heure)")
    make.add_argument("--stock", type=int, default=200)
    make.add_argument("--fin", help="Heure d'arrêt, par exemple 08:30")
    login = sub.add_parser("login", help="Connecte un compte TikTok pour l'envoi en brouillon (facultatif)")
    login.add_argument("--client-key", required=True)
    login.add_argument("--client-secret", required=True)
    login.add_argument("--redirect", help="Adresse de retour déclarée dans l'appli TikTok")
    args = parser.parse_args(argv)

    from . import agent
    from .discover import Blocked
    from .tiktok import DEFAULT_REDIRECT, TikTokError
    try:
        if args.action == "login":
            name = agent.account().login(args.client_key, args.client_secret, args.redirect or DEFAULT_REDIRECT)
            print("Connecté en tant que", name)
        elif args.action == "run":
            agent.run_agent(args.n, tiktok=args.tiktok or None)
        elif args.action == "produit":
            agent.produce_until(args.stock, args.fin)
        elif args.action == "cherche":
            agent.preview()
        else:
            agent.status()
    except (Blocked, TikTokError) as exc:
        parser.exit(1, f"{exc}\n")


def cmd_clip(argv: list[str]) -> None:
    d = DEFAULTS
    parser = argparse.ArgumentParser(prog="clipper", description="Vidéo longue → clips verticaux sous-titrés",
                                     epilog="Autres commandes : 'clipper serve' (interface web), "
                                            "'clipper demusic <vidéo>' (retirer la musique)")
    parser.add_argument("source", help="Chemin d'une vidéo ou lien (YouTube, Vimeo…)")
    parser.add_argument("-n", "--clips", type=int, help="Nombre de clips (auto par défaut)")
    parser.add_argument("--min", type=int, default=d.min_len, help="Durée minimale d'un clip (s)")
    parser.add_argument("--max", type=int, default=d.max_len, help="Durée maximale d'un clip (s)")
    parser.add_argument("--brain", choices=BRAINS, default=d.brain,
                        help="Qui choisit les moments : api (clé Anthropic), cli (Claude Code, sans clé), "
                             "heuristic, ou auto (le premier disponible)")
    parser.add_argument("--music", choices=MUSIC_MODES, default=d.music,
                        help="Musique de fond : keep, auto (retirée si détectée), remove (toujours retirée)")
    parser.add_argument("--style", choices=sorted(PRESETS), default=d.style, help="Style des sous-titres")
    parser.add_argument("--font", help="Police des sous-titres")
    parser.add_argument("--highlight", help="Couleur du mot actif : #FFE500, #FE0 ou un nom (jaune, vert…)")
    parser.add_argument("--model", choices=WHISPER_MODELS, default=d.whisper_model, help="Modèle Whisper")
    parser.add_argument("--lang", help="Langue forcée (fr, en...), détection auto sinon")
    parser.add_argument("--logo", help="PNG affiché en haut à droite")
    parser.add_argument("--no-hook", action="store_true", help="Pas de titre d'accroche en début de clip")
    parser.add_argument("--no-tighten", action="store_true", help="Garder les blancs tels quels")
    parser.add_argument("--quality", choices=QUALITIES, default=d.quality,
                        help="Encodage : fast (rapide), high (défaut), max (plus lent, image plus fine)")
    parser.add_argument("--progress-bar", action="store_true", help="Barre de progression en bas du clip")
    args = parser.parse_args(argv)

    try:
        opts = Options(
            n_clips=args.clips, min_len=args.min, max_len=args.max, style=args.style,
            whisper_model=args.model, language=args.lang, logo=args.logo, show_hook=not args.no_hook,
            caption_overrides={k: v for k, v in {"font": args.font, "highlight": args.highlight}.items() if v},
            brain=args.brain, music=args.music, tighten=not args.no_tighten, quality=args.quality,
            progress_bar=args.progress_bar,
        )
    except ValueError as exc:
        parser.error(str(exc))
    result = run(args.source, opts)
    for c in result["clips"]:
        keys = ("file", "virality", "duration", "layout", "music_removed", "title")
        print(json.dumps({k: c[k] for k in keys}, ensure_ascii=False))


def main() -> None:
    argv = sys.argv[1:]
    commands = {"serve": cmd_serve, "demusic": cmd_demusic, "agent": cmd_agent}
    if argv and argv[0] in commands:
        commands[argv[0]](argv[1:])
    else:
        cmd_clip(argv)


if __name__ == "__main__":
    main()
