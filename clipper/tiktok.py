"""Envoi des clips en brouillon sur TikTok par l'API officielle (Content Posting API, envoi vers la boîte de
réception de l'appli).

Le clip arrive dans les notifications TikTok du compte connecté. On y ajoute la légende (préparée par l'agent
dans brouillons.md), on peut choisir un son de la bibliothèque TikTok, puis on publie. L'API d'envoi en brouillon
ne prend pas de légende.

Connexion : Login Kit pour ordinateur, avec PKCE et une adresse de retour locale. Les jetons sont gardés dans
agent/tiktok.json et renouvelés tout seuls (jeton d'accès 24 h, jeton de renouvellement 365 jours).
"""

import hashlib
import json
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from .util import write_json_atomic

AUTH_URL = "https://www.tiktok.com/v2/auth/authorize/"
API = "https://open.tiktokapis.com/v2"
SCOPES = "user.info.basic,video.upload"
DEFAULT_REDIRECT = "http://localhost:5191/callback/"
MIN_CHUNK, MAX_CHUNK, CHUNK = 5 * 2**20, 64 * 2**20, 10 * 2**20


class TikTokError(RuntimeError):
    pass


def _http(method: str, url: str, data: bytes | None = None, headers: dict | None = None) -> dict:
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            body = r.read()
    except urllib.error.HTTPError as exc:
        body = exc.read()
        try:
            raise TikTokError(f"HTTP {exc.code} : {json.loads(body)}") from None
        except json.JSONDecodeError:
            raise TikTokError(f"HTTP {exc.code} : {body[:300]!r}") from None
    return json.loads(body) if body else {}


class Account:
    def __init__(self, path: Path):
        self.path = path
        self.cfg = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def save(self) -> None:
        write_json_atomic(self.path, self.cfg)

    @property
    def connected(self) -> bool:
        return bool(self.cfg.get("refresh_token")) and self.cfg.get("refresh_expires_at", 0) > time.time()

    # --- connexion -------------------------------------------------------------------------------------------

    def login(self, client_key: str, client_secret: str, redirect_uri: str = DEFAULT_REDIRECT) -> str:
        """Ouvre la page d'autorisation TikTok et attend le retour sur redirect_uri (serveur local)."""
        self.cfg.update(client_key=client_key, client_secret=client_secret, redirect_uri=redirect_uri)
        verifier = secrets.token_urlsafe(48)
        state = secrets.token_urlsafe(16)
        url = AUTH_URL + "?" + urllib.parse.urlencode({
            "client_key": client_key, "scope": SCOPES, "response_type": "code", "redirect_uri": redirect_uri,
            "state": state,
            # TikTok attend l'empreinte SHA-256 en hexadécimal pour les applis ordinateur
            "code_challenge": hashlib.sha256(verifier.encode()).hexdigest(), "code_challenge_method": "S256",
        })
        code = _wait_for_code(url, redirect_uri, state)
        tokens = _http("POST", API + "/oauth/token/", urllib.parse.urlencode({
            "client_key": client_key, "client_secret": client_secret, "code": code,
            "grant_type": "authorization_code", "redirect_uri": redirect_uri, "code_verifier": verifier,
        }).encode(), {"Content-Type": "application/x-www-form-urlencoded"})
        self._store(tokens)
        return self.user()

    def _store(self, tokens: dict) -> None:
        if "access_token" not in tokens:
            raise TikTokError(f"Connexion refusée : {tokens}")
        now = time.time()
        self.cfg.update(access_token=tokens["access_token"], refresh_token=tokens["refresh_token"],
                        open_id=tokens.get("open_id"), scope=tokens.get("scope"),
                        expires_at=now + tokens["expires_in"] - 120,
                        refresh_expires_at=now + tokens["refresh_expires_in"] - 120)
        self.save()

    def token(self) -> str:
        if not self.connected:
            raise TikTokError("Compte TikTok non connecté : lance 'python -m clipper agent login'")
        if self.cfg["expires_at"] <= time.time():
            self._store(_http("POST", API + "/oauth/token/", urllib.parse.urlencode({
                "client_key": self.cfg["client_key"], "client_secret": self.cfg["client_secret"],
                "grant_type": "refresh_token", "refresh_token": self.cfg["refresh_token"],
            }).encode(), {"Content-Type": "application/x-www-form-urlencoded"}))
        return self.cfg["access_token"]

    def _api(self, path: str, payload: dict) -> dict:
        out = _http("POST", API + path, json.dumps(payload).encode(),
                    {"Authorization": f"Bearer {self.token()}", "Content-Type": "application/json; charset=UTF-8"})
        err = out.get("error") or {}
        if err.get("code") not in (None, "ok"):
            raise TikTokError(f"{err.get('code')} : {err.get('message')} (log_id {err.get('log_id')})")
        return out.get("data") or {}

    def user(self) -> str:
        out = _http("GET", API + "/user/info/?fields=display_name,open_id", None,
                    {"Authorization": f"Bearer {self.token()}"})
        return (out.get("data") or {}).get("user", {}).get("display_name", "?")

    # --- envoi -----------------------------------------------------------------------------------------------

    def upload_draft(self, video: Path, log=print) -> str:
        """Envoie video dans la boîte de réception TikTok du compte. Renvoie le publish_id."""
        size = video.stat().st_size
        if size <= MAX_CHUNK:
            chunk, count = size, 1
        else:
            chunk, count = CHUNK, size // CHUNK  # le dernier morceau prend le reste (jusqu'à 128 Mo permis)
        data = self._api("/post/publish/inbox/video/init/", {"source_info": {
            "source": "FILE_UPLOAD", "video_size": size, "chunk_size": chunk, "total_chunk_count": count}})
        publish_id, upload_url = data["publish_id"], data["upload_url"]
        with video.open("rb") as f:
            for i in range(count):
                start = i * chunk
                end = size - 1 if i == count - 1 else start + chunk - 1
                f.seek(start)
                part = f.read(end - start + 1)
                _http("PUT", upload_url, part, {"Content-Type": "video/mp4", "Content-Length": str(len(part)),
                                                "Content-Range": f"bytes {start}-{end}/{size}"})
        return self.wait(publish_id, log)

    def wait(self, publish_id: str, log=print, timeout: float = 300) -> str:
        deadline = time.time() + timeout
        while time.time() < deadline:
            status = self._api("/post/publish/status/fetch/", {"publish_id": publish_id})
            s = status.get("status")
            if s == "SEND_TO_USER_INBOX" or s == "PUBLISH_COMPLETE":
                return publish_id
            if s == "FAILED":
                raise TikTokError(f"Envoi refusé par TikTok : {status.get('fail_reason')}")
            time.sleep(4)
        log(f"TikTok traite encore {publish_id}, le brouillon arrivera plus tard")
        return publish_id


def _wait_for_code(url: str, redirect_uri: str, state: str) -> str:
    """Serveur local d'un seul appel sur l'adresse de retour. Si le navigateur ne revient pas (adresse de retour
    d'un autre domaine), on peut coller l'adresse finale dans le terminal."""
    target = urllib.parse.urlparse(redirect_uri)
    result: dict = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            result.update({k: v[0] for k, v in q.items()})
            ok = "code" in q and q.get("state", [""])[0] == state
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(("<h2>Compte TikTok connecté, tu peux fermer cet onglet.</h2>" if ok else
                              f"<h2>Échec : {q.get('error_description', q.get('error', ['?']))[0]}</h2>").encode())

        def log_message(self, *args):
            pass

    server = None
    if target.hostname in ("localhost", "127.0.0.1"):
        server = HTTPServer((target.hostname, target.port or 80), Handler)
        threading.Thread(target=server.handle_request, daemon=True).start()
    print("Autorise l'appli dans le navigateur qui s'ouvre :\n" + url)
    webbrowser.open(url)
    if server:
        deadline = time.time() + 300
        while not result and time.time() < deadline:
            time.sleep(0.3)
        server.server_close()
    if not result:
        pasted = input("Colle ici l'adresse complète de la page sur laquelle TikTok t'a renvoyé : ").strip()
        result = {k: v[0] for k, v in urllib.parse.parse_qs(urllib.parse.urlparse(pasted).query).items()}
    if result.get("state") != state or "code" not in result:
        raise TikTokError(f"Autorisation refusée ou incomplète : {result}")
    return result["code"]
