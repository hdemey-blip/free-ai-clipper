"""Interface web locale : déposer une vidéo ou un lien, gérer la file d'attente, voir les clips prévus et
récupérer les clips rendus."""

import dataclasses
import json
import re
import threading
import time
import uuid

from flask import Flask, abort, jsonify, request, send_from_directory

from .captions import PRESETS
from .ingest import store_upload
from .pipeline import DEFAULTS, JOBS_DIR, Options, Stopped, edit_plan, load_plan, run
from .util import write_json_atomic

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 ** 3
LOCK = threading.Lock()
WAKE = threading.Condition(LOCK)
JOBS: dict[str, dict] = {}   # travaux de ce serveur, et travaux relançables retrouvés sur le disque (job.json)
ORDER: list[str] = []        # file d'attente, sans le travail en cours
CURRENT: list[str] = []      # le travail en cours (zéro ou un identifiant)
PUBLIC = ("id", "source", "state", "stage", "progress", "message", "error", "created")


def _job_dir(job_id: str):
    if not re.fullmatch(r"[\w-]+", job_id):
        abort(404)
    return JOBS_DIR / job_id


def _persist(job: dict) -> None:
    """job.json garde de quoi relancer le travail après un redémarrage du serveur."""
    write_json_atomic(JOBS_DIR / job["id"] / "job.json",
                      {"source": job["source"], "src": job["src"], "opts": dataclasses.asdict(job["opts"]),
                       "created": job["created"], "state": job["state"]})


def _restore() -> None:
    """Au démarrage : un travail resté en attente ou en cours a été coupé par l'arrêt du serveur. Il passe en
    pause, prêt à être relancé, et reprendra là où il en était."""
    for path in JOBS_DIR.glob("*/job.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        try:
            opts = Options(**data["opts"])
        except (TypeError, ValueError):
            continue  # réglages d'une ancienne version : le travail reste visible dans l'historique
        state = data["state"]
        message = ""
        if state not in ("done", "error"):
            state, message = "paused", "Interrompu par l'arrêt du serveur. Le travail fait est gardé."
        JOBS[path.parent.name] = {"id": path.parent.name, "source": data["source"], "src": data["src"],
                                  "opts": opts, "created": data["created"], "state": state,
                                  "stage": "", "progress": 0, "message": message, "error": None,
                                  "stop": False, "requeue_at": None}


def _worker() -> None:
    """Un seul travail à la fois : Whisper, Demucs et x264 prennent déjà tous les cœurs du processeur."""
    while True:
        with WAKE:
            while not ORDER:
                WAKE.wait()
            job_id = ORDER.pop(0)
            job = JOBS[job_id]
            job.update(state="running", stop=False, requeue_at=None, error=None,
                       stage="démarrage", progress=0, message="")
            CURRENT[:] = [job_id]
            _persist(job)

        def status(stage, progress, message, job=job):
            with LOCK:
                job.update(stage=stage, progress=progress, message=message)

        stopped = False
        try:
            run(job["src"], job["opts"], JOBS_DIR / job_id, status, job["source"], stop=lambda: job["stop"])
            final = {"state": "done"}
        except Stopped:
            stopped = True
        except Exception as exc:
            final = {"state": "error", "error": f"{type(exc).__name__}: {exc}"}

        with WAKE:
            CURRENT.clear()
            if stopped and job["requeue_at"] is not None:
                ORDER.insert(min(job["requeue_at"], len(ORDER)), job_id)
                final = {"state": "waiting", "stage": "en pause",
                         "message": "Reprendra là où il s'est arrêté."}
            elif stopped:
                final = {"state": "paused", "stage": "en pause",
                         "message": "Retiré de la file. Le travail fait est gardé."}
            job.update(final, stop=False, requeue_at=None)
            _persist(job)
            WAKE.notify_all()


def _stop_current(requeue_at: int | None, message: str) -> None:
    """Arrête le travail en cours à la prochaine vérification (quelques secondes au plus pendant le rendu ou la
    transcription ; après la réponse de Claude ou la fin du téléchargement sinon)."""
    job = JOBS[CURRENT[0]]
    if job["stop"] and job["requeue_at"] is None:
        return  # déjà retiré de la file : on ne le remet pas dedans
    job.update(stop=True, requeue_at=requeue_at, state="stopping", message=message)


def _int(form, name: str) -> int | None:
    """Un champ numérique vidé arrive comme '' : None, et Options prend sa valeur par défaut."""
    raw = (form.get(name) or "").strip()
    return int(raw) if raw else None


@app.post("/api/jobs")
def create_job():
    form = request.form
    d = DEFAULTS
    try:
        min_len, max_len = _int(form, "min_len"), _int(form, "max_len")
        highlight = form.get("highlight", "").strip()
        opts = Options(
            n_clips=_int(form, "clips"),
            min_len=d.min_len if min_len is None else min_len,
            max_len=d.max_len if max_len is None else max_len,
            style=form.get("style", d.style),
            whisper_model=form.get("model", d.whisper_model),
            language=form.get("lang") or None,
            show_hook=form.get("hook") == "on",  # une case décochée n'est pas envoyée du tout
            caption_overrides={"highlight": highlight} if highlight else {},
            brain=form.get("brain", d.brain),
            music=form.get("music", d.music),
            tighten=form.get("tighten") == "on",
            quality=form.get("quality", d.quality),
            progress_bar=form.get("progress") == "on",
        )
    except ValueError as exc:
        return str(exc), 400

    job_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    if "file" in request.files and request.files["file"].filename:
        f = request.files["file"]
        source, label = str(store_upload(f, f.filename, JOBS_DIR / job_id)), f.filename
    elif form.get("url"):
        source = label = form["url"].strip()
    else:
        return "Ajoute une vidéo ou un lien", 400

    (JOBS_DIR / job_id).mkdir(parents=True, exist_ok=True)
    with WAKE:
        JOBS[job_id] = {"id": job_id, "source": label, "src": source, "opts": opts, "created": time.time(),
                        "state": "waiting", "stage": "file d'attente", "progress": 0, "message": "",
                        "error": None, "stop": False, "requeue_at": None}
        ORDER.append(job_id)
        _persist(JOBS[job_id])
        WAKE.notify_all()
    return jsonify({"id": job_id})


@app.get("/api/queue")
def queue_state():
    """Le travail en cours, puis la file dans l'ordre, puis les travaux en pause ou en erreur à relancer."""
    with LOCK:
        line = [JOBS[i] for i in CURRENT + ORDER]
        parked = sorted((j for j in JOBS.values() if j["state"] in ("paused", "error")), key=lambda j: -j["created"])
        return jsonify({"line": [{k: j[k] for k in PUBLIC} for j in line],
                        "parked": [{k: j[k] for k in PUBLIC} for j in parked]})


@app.post("/api/jobs/<job_id>/move")
def move_job(job_id):
    """Place le travail à la position `to` de la file (0 = tout de suite, en pause pour celui qui tourne).
    Sert aussi à relancer un travail en pause. Rien de ce qui est fini n'est refait."""
    to = int((request.get_json(silent=True) or {}).get("to", 10 ** 6))
    with WAKE:
        job = JOBS.get(job_id)
        if job is None:
            return "Ce travail ne peut pas être relancé (créé avant la file d'attente)", 400
        line = CURRENT + ORDER
        to = max(0, min(to, len(line) - (1 if job_id in line else 0)))
        if CURRENT and job_id == CURRENT[0]:
            if to > 0:
                _stop_current(to, "Mise en pause : il reprendra plus loin dans la file.")
        else:
            if job_id in ORDER:
                ORDER.remove(job_id)
            if CURRENT and to == 0:
                ORDER.insert(0, job_id)
                _stop_current(1, f"En pause pour laisser passer {job['source']}")
            else:
                ORDER.insert(to - len(CURRENT), job_id)
            job.update(state="waiting", stage="file d'attente", message="", error=None)
            _persist(job)
        WAKE.notify_all()
    return jsonify({"ok": True})


@app.post("/api/jobs/<job_id>/remove")
def remove_job(job_id):
    """Sort le travail de la file. Source, transcription et clips déjà rendus restent sur le disque."""
    with WAKE:
        job = JOBS.get(job_id)
        if job is None:
            abort(404)
        if CURRENT and job_id == CURRENT[0]:
            job.update(requeue_at=None)
            _stop_current(None, "Arrêt en cours. Le travail fait est gardé.")
        elif job_id in ORDER:
            ORDER.remove(job_id)
            job.update(state="paused", stage="en pause", message="Retiré de la file. Le travail fait est gardé.")
            _persist(job)
    return jsonify({"ok": True})


@app.post("/api/jobs/<job_id>/plan")
def change_plan(job_id):
    """{"skip": nom} retire un clip prévu, {"restore": nom} le remet, {"order": [noms]} change l'ordre de rendu.
    Pris en compte au clip suivant ; retirer le clip en cours de rendu l'arrête aussitôt."""
    body = request.get_json(silent=True) or {}
    job_dir = _job_dir(job_id)
    if load_plan(job_dir) is None:
        abort(404)

    def change(plan):
        names = [c["name"] for c in plan["clips"]]
        if body.get("skip") in names and body["skip"] not in plan["skipped"]:
            plan["skipped"].append(body["skip"])
        if "restore" in body:
            plan["skipped"] = [n for n in plan["skipped"] if n != body["restore"]]
        if "order" in body:
            given = [n for n in body["order"] if n in names]
            plan["order"] = given + [n for n in plan["order"] if n not in given]

    edit_plan(job_dir, change)
    return jsonify({"ok": True})


@app.get("/api/jobs")
def list_jobs():
    out = {}
    for d in sorted(JOBS_DIR.glob("*/result.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        data = json.loads(d.read_text(encoding="utf-8"))
        out[d.parent.name] = {"id": d.parent.name, "source": data["source"], "state": "done",
                              "clips": len(data["clips"]), "created": d.stat().st_mtime}
    with LOCK:
        for job_id, job in JOBS.items():
            out[job_id] = {**out.get(job_id, {}), **{k: job[k] for k in PUBLIC}}
    return jsonify(sorted(out.values(), key=lambda j: -j["created"]))


@app.get("/api/jobs/<job_id>")
def job_status(job_id):
    job_dir = _job_dir(job_id)
    result_file = job_dir / "result.json"
    result = json.loads(result_file.read_text(encoding="utf-8")) if result_file.exists() else None
    with LOCK:
        job = JOBS.get(job_id)
        job = {k: job[k] for k in PUBLIC} if job else {"id": job_id, "state": "done", "error": None,
                                                      "stage": "terminé", "progress": 1, "message": ""}
    if result is None and job_id not in JOBS:
        abort(404)
    job["result"] = result
    job["plan"] = load_plan(job_dir)
    job["resumable"] = job_id in JOBS
    return jsonify(job)


@app.get("/clips/<job_id>/<path:name>")
def clip_file(job_id, name):
    return send_from_directory(_job_dir(job_id) / "clips", name, as_attachment=request.args.get("dl") == "1")


@app.get("/")
def index():
    options = "".join(f'<option value="{k}">{k}</option>' for k in PRESETS)
    return PAGE.replace("{{STYLES}}", options)


def serve(port: int = 5190) -> None:
    JOBS_DIR.mkdir(exist_ok=True)
    _restore()
    threading.Thread(target=_worker, daemon=True).start()
    print(f"Clipper sur http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, threaded=True)


PAGE = r"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Clipper</title>
<style>
:root{--bg:#0d0d10;--panel:#17171c;--line:#2a2a33;--text:#f2f2f5;--mute:#8d8d99;--acc:#ffe500;--ok:#3cff7a;--bad:#ff5a6a}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1180px;margin:0 auto;padding:32px 16px 80px}
h1{font-size:30px;margin:0 0 4px;letter-spacing:-.02em}
h1 span{color:var(--acc)}
.sub{color:var(--mute);margin:0 0 28px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:20px}
#drop{border:2px dashed var(--line);border-radius:12px;padding:30px;text-align:center;cursor:pointer;transition:.15s}
#drop.over,#drop:hover{border-color:var(--acc);background:#1d1d18}
#drop b{display:block;font-size:17px;margin-bottom:4px}
.row{display:flex;gap:12px;margin-top:14px;flex-wrap:wrap}
input[type=text],input[type=number],select{background:#101014;border:1px solid var(--line);color:var(--text);border-radius:9px;padding:10px 12px;font:inherit}
input[type=text]{flex:1;min-width:220px}
label{display:flex;flex-direction:column;gap:4px;font-size:12px;color:var(--mute)}
label.inline{flex-direction:row;align-items:center;gap:8px;font-size:14px;color:var(--text)}
button{background:var(--acc);color:#111;border:0;border-radius:9px;padding:11px 22px;font:600 15px system-ui;cursor:pointer}
button:disabled{opacity:.4;cursor:default}
button.ghost{background:transparent;color:var(--text);border:1px solid var(--line);padding:7px 12px;font-size:13px}
button.ghost:hover:not(:disabled){border-color:var(--acc)}
button.mini{background:transparent;color:var(--text);border:1px solid var(--line);padding:4px 9px;font-size:12px;font-weight:500;border-radius:7px}
button.mini:hover:not(:disabled){border-color:var(--acc)}
button.mini.warn:hover:not(:disabled){border-color:var(--bad);color:var(--bad)}
.bar{height:8px;background:#101014;border-radius:6px;overflow:hidden;margin:10px 0 6px}
.bar i{display:block;height:100%;width:0;background:var(--acc);transition:width .4s}
.stage{font-weight:600;text-transform:capitalize}
.msg{color:var(--mute);font-size:13px}
.err{color:var(--bad)}
h2{font-size:18px;margin:36px 0 10px;display:flex;align-items:baseline;gap:10px}
h2 small{font-size:13px;font-weight:400;color:var(--mute)}
.list{display:flex;flex-direction:column;gap:6px}
.item{display:flex;align-items:center;gap:12px;padding:10px 12px;border:1px solid var(--line);border-radius:10px;background:var(--panel);font-size:13px}
.item.sel{border-color:var(--acc)}
.item .grow{flex:1;min-width:0;cursor:pointer}
.item .name{display:flex;gap:8px;align-items:center;font-weight:600;font-size:14px}
.item .name span:first-child{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.item .bar{height:5px;margin:6px 0 2px}
.item .btns{display:flex;gap:6px;flex-wrap:wrap;justify-content:flex-end}
.item.muted{opacity:.55}
.pos{width:26px;text-align:center;font-weight:700;color:var(--mute);flex:none}
.badge{font-size:11px;font-weight:600;border-radius:20px;padding:1px 8px;border:1px solid var(--line);color:var(--mute);white-space:nowrap}
.badge.run{background:var(--acc);border-color:var(--acc);color:#111}
.badge.ok{border-color:var(--ok);color:var(--ok)}
.badge.bad{border-color:var(--bad);color:var(--bad)}
.hook{color:#cfcfd8;font-size:12px;margin-top:2px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
#job{display:none}
#grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:18px;margin-top:18px}
.clip{background:var(--panel);border:1px solid var(--line);border-radius:14px;overflow:hidden;display:flex;flex-direction:column}
.clip video{width:100%;aspect-ratio:9/16;background:#000;display:block}
.clip .body{padding:12px 14px 14px;display:flex;flex-direction:column;gap:8px;flex:1}
.clip h3{margin:0;font-size:15px;line-height:1.3}
.meta{display:flex;gap:6px;flex-wrap:wrap;font-size:12px;color:var(--mute)}
.pill{border:1px solid var(--line);border-radius:20px;padding:1px 8px}
.score{font-weight:700;color:#111;border-radius:20px;padding:1px 9px;font-size:12px;flex:none}
.cap{font-size:13px;color:#cfcfd8;background:#101014;border-radius:8px;padding:8px 10px}
.actions{display:flex;gap:8px;margin-top:auto}
.actions a{flex:1;text-align:center;text-decoration:none;color:#111;background:var(--acc);border-radius:9px;padding:8px;font-weight:600;font-size:13px}
#history{display:flex;flex-direction:column;gap:6px}
#history div{display:flex;justify-content:space-between;gap:12px;padding:9px 12px;border:1px solid var(--line);border-radius:9px;cursor:pointer;font-size:13px}
#history div:hover{border-color:var(--acc)}
#history span{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
@media (max-width:640px){.item{flex-wrap:wrap}.item .btns{width:100%;justify-content:flex-start}}
</style>
</head>
<body>
<main>
<h1>Clip<span>per</span></h1>
<p class="sub">Une vidéo longue en entrée, des clips verticaux sous-titrés en sortie.</p>

<form id="f" class="card">
  <div id="drop"><b>Glisse une vidéo ici</b><span class="msg" id="fname">ou clique pour choisir un fichier</span>
    <input type="file" name="file" id="file" accept="video/*" hidden></div>
  <div class="row"><input type="text" name="url" id="url" placeholder="…ou colle un lien (YouTube, Vimeo, Twitch…)"></div>
  <div class="row">
    <label>Clips<input type="number" name="clips" min="1" max="30" placeholder="auto" style="width:90px"></label>
    <label>Durée min (s)<input type="number" name="min_len" value="20" style="width:90px"></label>
    <label>Durée max (s)<input type="number" name="max_len" value="70" style="width:90px"></label>
    <label>Sous-titres<select name="style">{{STYLES}}</select></label>
    <label>Couleur du mot<input type="text" name="highlight" placeholder="#FFE500" style="width:100px;min-width:0"></label>
    <label>Transcription<select name="model"><option value="base">rapide</option><option value="small" selected>équilibrée</option><option value="medium">précise (lent)</option></select></label>
    <label>Langue<select name="lang"><option value="">auto</option><option>fr</option><option>en</option><option>es</option><option>de</option><option>it</option><option>pt</option></select></label>
    <label>Choix des moments<select name="brain"><option value="auto">auto</option><option value="cli">Claude Code (sans clé)</option><option value="api">API Anthropic</option><option value="heuristic">heuristique</option></select></label>
    <label>Musique de fond<select name="music"><option value="keep">garder</option><option value="auto">retirer si détectée</option><option value="remove">toujours retirer</option></select></label>
    <label>Qualité d'image<select name="quality"><option value="fast">rapide</option><option value="high" selected>haute</option><option value="max">maximale (lent)</option></select></label>
  </div>
  <div class="row" style="justify-content:space-between;align-items:center">
    <div class="row" style="gap:18px;margin-top:0">
      <label class="inline"><input type="checkbox" name="hook" checked> Titre d'accroche au début</label>
      <label class="inline"><input type="checkbox" name="tighten" checked> Raccourcir les blancs</label>
      <label class="inline"><input type="checkbox" name="progress"> Barre de progression</label>
    </div>
    <button id="go">Ajouter à la file</button>
  </div>
</form>

<h2>File d'attente <small id="qsum"></small></h2>
<div class="list" id="queue"></div>

<section id="job">
  <h2><span id="jname"></span></h2>
  <div class="card">
    <div style="display:flex;justify-content:space-between;align-items:center;gap:12px">
      <span class="stage" id="stage"></span>
      <button class="ghost" id="resume" style="display:none" onclick="move(current, 1e6)" title="Source, transcription et clips déjà rendus sont réutilisés">Relancer là où il en était</button>
    </div>
    <div class="bar"><i id="prog"></i></div><div class="msg" id="smsg"></div>
  </div>
  <h2>Clips prévus <small id="psum"></small></h2>
  <div class="list" id="plan"></div>
  <div id="grid"></div>
</section>

<h2>Historique</h2>
<div id="history"></div>
</main>
<script>
const $ = id => document.getElementById(id);
const drop = $("drop"), file = $("file");
drop.onclick = () => file.click();
file.onchange = () => $("fname").textContent = file.files[0]?.name || "";
["dragover","dragenter"].forEach(e => drop.addEventListener(e, ev => {ev.preventDefault(); drop.classList.add("over")}));
["dragleave","drop"].forEach(e => drop.addEventListener(e, ev => {ev.preventDefault(); drop.classList.remove("over")}));
drop.addEventListener("drop", ev => { file.files = ev.dataTransfer.files; file.onchange(); });

const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const scoreColor = v => v >= 80 ? "var(--ok)" : v >= 60 ? "var(--acc)" : "#ff9b4a";
const LAYOUTS = {track: "suivi visage", split: "écran partagé", fit: "plan large"};
const short = s => String(s || "").split(/[\\/]/).pop();
const mmss = t => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;

$("f").onsubmit = async ev => {
  ev.preventDefault();
  if (!file.files.length && !$("url").value.trim()) return alert("Ajoute une vidéo ou un lien");
  $("go").disabled = true; $("go").textContent = "Envoi…";
  const r = await fetch("/api/jobs", {method: "POST", body: new FormData($("f"))});
  $("go").disabled = false; $("go").textContent = "Ajouter à la file";
  if (!r.ok) return alert(await r.text());
  const {id} = await r.json();
  file.value = ""; $("fname").textContent = "ou clique pour choisir un fichier"; $("url").value = "";
  follow(id);
};

async function post(url, body) {
  const r = await fetch(url, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body || {})});
  if (!r.ok) alert(await r.text());
  tick();
}
const move = (id, to) => post(`/api/jobs/${id}/move`, {to});
const removeJob = id => post(`/api/jobs/${id}/remove`);
const planEdit = body => post(`/api/jobs/${current}/plan`, body);

// ---- file d'attente ----
let current = null, lastQueue = "";
function renderQueue(q) {
  const key = JSON.stringify(q) + current;
  if (key === lastQueue) return;
  lastQueue = key;
  const line = q.line, n = line.length;
  $("qsum").textContent = n ? `${n} travail${n > 1 ? "s" : ""} · un à la fois` : "";
  const row = (j, i) => {
    const running = j.state === "running", stopping = j.state === "stopping", parked = i < 0;
    const badge = running ? '<span class="badge run">en cours</span>'
      : stopping ? '<span class="badge run">pause…</span>'
      : j.state === "error" ? '<span class="badge bad">erreur</span>'
      : parked ? '<span class="badge">en pause</span>'
      : `<span class="badge">${i === 1 && line[0]?.state !== "waiting" ? "suivant" : "en attente"}</span>`;
    const info = j.error ? `<span class="err">${esc(j.error)}</span>`
      : running || stopping ? `${esc(j.stage)} · ${esc(j.message)}`
      : esc(j.message || (parked ? "" : `${i} travail${i > 1 ? "s" : ""} avant`));
    let btns = "";
    if (parked) {
      btns = `<button class="mini" onclick="move('${j.id}', 0)" title="Passe devant le travail en cours, qui reprendra ensuite">Lancer maintenant</button>
              <button class="mini" onclick="move('${j.id}', ${n})">Relancer</button>`;
    } else {
      if (!running && !stopping) {
        btns += `<button class="mini" onclick="move('${j.id}', ${i - 1})" ${i === 0 ? "disabled" : ""} title="Monter">↑</button>
                 <button class="mini" onclick="move('${j.id}', ${i + 1})" ${i === n - 1 ? "disabled" : ""} title="Descendre">↓</button>`;
        if (i > 0) btns += `<button class="mini" onclick="move('${j.id}', 0)" title="Le travail en cours se met en pause et reprendra juste après">Lancer maintenant</button>`;
      } else if (running && n > 1) {
        btns += `<button class="mini" onclick="move('${j.id}', 1)" title="Laisse passer le suivant puis reprend là où il en était">Laisser passer le suivant</button>`;
      }
      if (!stopping) btns += `<button class="mini warn" onclick="removeJob('${j.id}')" title="Sort de la file, le travail déjà fait est gardé">${running ? "Mettre en pause" : "Retirer"}</button>`;
    }
    return `<div class="item ${j.id === current ? "sel" : ""} ${parked ? "muted" : ""}">
      <span class="pos">${parked ? "–" : i + 1}</span>
      <div class="grow" onclick="follow('${j.id}')">
        <div class="name"><span>${esc(short(j.source))}</span>${badge}</div>
        ${running || stopping ? `<div class="bar"><i style="width:${Math.round((j.progress || 0) * 100)}%"></i></div>` : ""}
        <div class="msg">${info}</div>
      </div>
      <div class="btns">${btns}</div></div>`;
  };
  $("queue").innerHTML = (line.map(row).join("") || '<span class="msg">Rien en attente.</span>')
    + q.parked.map(j => row(j, -1)).join("");
}

// ---- travail suivi : clips prévus et clips rendus ----
function follow(id) {
  current = id; lastPlan = ""; shown = ""; lastQueue = "";
  history.replaceState(null, "", "?job=" + id);
  $("job").style.display = "block";
  $("grid").innerHTML = ""; $("plan").innerHTML = "";
  tick();
}

let lastPlan = "", pending = [];
function renderPlan(j) {
  const plan = j.plan, result = j.result;
  const done = new Set((result?.clips || []).map(c => c.name || c.file.replace(/\.mp4$/, "")));
  const active = ["running", "stopping"].includes(j.state);
  const rendering = active ? result?.rendering : null;
  const key = JSON.stringify([plan, [...done], rendering, j.state]);
  if (key === lastPlan) return;
  lastPlan = key;
  if (!plan) {
    $("psum").textContent = "";
    $("plan").innerHTML = j.state === "done" ? "" : '<span class="msg">La liste apparaît dès que les moments sont choisis (après la transcription).</span>';
    return;
  }
  const byName = Object.fromEntries(plan.clips.map(c => [c.name, c]));
  const skipped = new Set(plan.skipped);
  const order = plan.order.filter(n => byName[n]);
  const doneList = plan.clips.map(c => c.name).filter(n => done.has(n));
  pending = order.filter(n => !done.has(n) && !skipped.has(n) && n !== rendering);
  const rows = [...doneList, ...(rendering && !done.has(rendering) ? [rendering] : []), ...pending, ...plan.skipped.filter(n => byName[n] && !done.has(n))];
  $("psum").textContent = `${doneList.length} fait${doneList.length > 1 ? "s" : ""} · ${pending.length + (rendering ? 1 : 0)} à faire`
    + (skipped.size ? ` · ${skipped.size} retiré${skipped.size > 1 ? "s" : ""}` : "");
  $("plan").innerHTML = rows.map((name, i) => {
    const c = byName[name];
    const p = pending.indexOf(name);
    const state = done.has(name) ? '<span class="badge ok">fait</span>'
      : name === rendering ? '<span class="badge run">en rendu</span>'
      : skipped.has(name) ? '<span class="badge">retiré</span>' : '<span class="badge">à faire</span>';
    let btns = "";
    if (p >= 0) {
      btns = `<button class="mini" onclick="shift('${name}', -1)" ${p === 0 ? "disabled" : ""} title="Rendre plus tôt">↑</button>
              <button class="mini" onclick="shift('${name}', 1)" ${p === pending.length - 1 ? "disabled" : ""} title="Rendre plus tard">↓</button>
              <button class="mini warn" onclick="planEdit({skip: '${name}'})">Retirer</button>`;
    } else if (name === rendering) {
      btns = `<button class="mini warn" onclick="planEdit({skip: '${name}'})" title="Arrête ce rendu et passe au clip suivant">Retirer</button>`;
    } else if (skipped.has(name) && !done.has(name)) {
      btns = `<button class="mini" onclick="planEdit({restore: '${name}'})">Remettre</button>`;
    }
    const len = Math.round(c.end - c.start);
    return `<div class="item ${skipped.has(name) && !done.has(name) ? "muted" : ""}">
      <span class="pos">${skipped.has(name) && !done.has(name) ? "–" : i + 1}</span>
      <span class="score" style="background:${scoreColor(c.virality)}">${c.virality}</span>
      <div class="grow" style="cursor:default">
        <div class="name"><span>${esc(c.title)}</span>${state}</div>
        <div class="msg">${mmss(c.start)} → ${mmss(c.end)} dans la vidéo · ${len} s</div>
        ${c.hook ? `<div class="hook">Accroche : ${esc(c.hook)}</div>` : ""}
      </div>
      <div class="btns">${btns}</div></div>`;
  }).join("");
}

function shift(name, delta) {
  const p = pending.indexOf(name), q = p + delta;
  if (p < 0 || q < 0 || q >= pending.length) return;
  const next = [...pending];
  [next[p], next[q]] = [next[q], next[p]];
  planEdit({order: next});
}

let shown = "";
function showClips(id, result) {
  // reconstruire la grille seulement quand un clip arrive : sinon les vidéos en lecture repartiraient à zéro
  const key = id + ":" + result.clips.map(c => c.file).join();
  if (key === shown) return;
  shown = key;
  $("grid").innerHTML = result.clips.map(c => {
    const tags = (c.hashtags || []).map(t => "#" + t.replace(/^#/, "")).join(" ");
    const post = (c.caption + (tags ? "\n\n" + tags : "")).trim();
    const src = `/clips/${id}/${c.file}`;
    return `<div class="clip">
      <video src="${src}" ${c.cover ? `poster="/clips/${id}/${c.cover}"` : ""} controls preload="metadata" playsinline></video>
      <div class="body">
        <h3>${esc(c.title)}</h3>
        <div class="meta"><span class="score" style="background:${scoreColor(c.virality)}">${c.virality}/100</span>
          <span class="pill">${c.duration}s</span><span class="pill">${LAYOUTS[c.layout] || c.layout}</span>
          ${c.music_removed ? '<span class="pill">sans musique</span>' : ""}
          ${c.silence_cut ? `<span class="pill">−${c.silence_cut}s de blancs</span>` : ""}</div>
        <div class="cap">${esc(post).replace(/\n/g, "<br>")}</div>
        <div class="actions"><a href="${src}?dl=1">Télécharger</a>
          <button type="button" class="ghost" data-post="${esc(post)}" onclick="navigator.clipboard.writeText(this.dataset.post);this.textContent='Copié'">Copier la légende</button></div>
      </div></div>`;
  }).join("");
}

const STATES = {waiting: "en attente", running: "", stopping: "mise en pause", paused: "en pause", done: "terminé", error: "erreur"};
function renderJob(j) {
  $("jname").textContent = short(j.result?.source || j.source || j.id);
  const active = ["running", "stopping"].includes(j.state);
  $("stage").textContent = active ? j.stage : STATES[j.state] || j.stage || "";
  $("prog").style.width = Math.round((j.state === "done" ? 1 : j.progress || 0) * 100) + "%";
  const note = j.result?.note;
  $("smsg").innerHTML = j.error ? `<span class="err">${esc(j.error)}</span>`
    : note && j.state === "done" ? `<span class="err">${esc(note)}</span>` : esc(j.message);
  renderPlan(j);
  // relancer : en pause, en erreur, ou terminé avec des clips remis dans la liste
  const resume = j.resumable && (["paused", "error"].includes(j.state) || (j.state === "done" && pending.length));
  $("resume").style.display = resume ? "inline-block" : "none";
  if (j.result) showClips(j.id, j.result);
}

let lastDone = -1;
async function tick() {
  try {
    const [q, j] = await Promise.all([
      fetch("/api/queue").then(r => r.json()),
      current ? fetch("/api/jobs/" + current).then(r => r.ok ? r.json() : null) : null,
    ]);
    renderQueue(q);
    if (j) renderJob(j);
    const doneCount = q.line.length + q.parked.length;
    if (doneCount !== lastDone) { lastDone = doneCount; loadHistory(); }
  } catch (e) { /* serveur coupé : on réessaie au prochain tour */ }
}

async function loadHistory() {
  const jobs = await (await fetch("/api/jobs")).json();
  $("history").innerHTML = jobs.filter(j => !["waiting", "running", "stopping"].includes(j.state))
    .map(j => `<div onclick="follow('${j.id}')"><span>${esc(short(j.source))}</span>
    <span class="msg">${j.state === "error" ? "erreur" : j.state === "paused" ? "en pause" : (j.clips ?? "") + " clips"}</span></div>`).join("")
    || '<span class="msg">Aucun travail pour l\'instant.</span>';
}
setInterval(tick, 1500);
loadHistory();
const fromUrl = new URLSearchParams(location.search).get("job");
if (fromUrl) follow(fromUrl); else tick();
</script>
</body>
</html>
"""
