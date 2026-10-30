"""
CineMatch - production build
----------------------------------------------------------------------
Why it is lag-free:
  * Python runs ONCE: loads movie.pkl + similarity.pkl and pre-computes the
    top-20 neighbours for every movie.
  * The whole interface (search, rows, modal, My List) runs client-side in
    JavaScript, so clicking a poster never triggers a Streamlit rerun.
  * Posters load lazily in parallel, are resized, and cached in the browser.

Run:  streamlit run app_pro.py
Files needed beside the app: movie.pkl, similarity.pkl
"""

import json
import os
import pickle
import re
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="CineMatch", page_icon="🎬", layout="wide",
                   initial_sidebar_state="collapsed")

# Make the embedded app fill the whole window and hide Streamlit chrome.
st.markdown(
    """
    <style>
    #MainMenu, footer, header, [data-testid="stHeader"], [data-testid="stToolbar"],
    [data-testid="stDecoration"], [data-testid="stStatusWidget"] { display: none !important; }
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background: #0b0b0c !important; overflow: hidden !important; }
    .block-container, [data-testid="stMainBlockContainer"] {
        padding: 0 !important; max-width: 100% !important; }
    [data-testid="stVerticalBlock"] { gap: 0 !important; }
    iframe { height: 100vh !important; width: 100% !important; border: 0 !important;
             display: block; color-scheme: dark; }
    </style>
    """,
    unsafe_allow_html=True,
)

PREFERRED_START = ["Inception", "The Dark Knight", "Interstellar", "Avatar",
                   "The Dark Knight Rises", "Iron Man", "Titanic"]
K = 20  # neighbours stored per movie


# ----------------------------------------------------------------------------
# Data + server-side precompute (runs once, cached)
# ----------------------------------------------------------------------------
def _imdb_info(title):
    empty = {"p": None, "y": None, "c": ""}
    query = re.sub(r"[^a-zA-Z0-9]+", "_", str(title)).strip("_")
    if not query:
        return empty
    req = Request(f"https://v3.sg.media-imdb.com/suggestion/x/{query}.json",
                  headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(req, timeout=6) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        items = [i for i in data.get("d", []) if i.get("i", {}).get("imageUrl")]
        pool = [i for i in items if i.get("qid") == "movie"] or items
        if not pool:
            return empty
        exact = [i for i in pool if str(i.get("l", "")).lower() == str(title).lower()]
        best = (exact or pool)[0]
        return {"p": best["i"]["imageUrl"], "y": best.get("y"), "c": best.get("s", "")}
    except Exception:
        return empty


@st.cache_data(show_spinner="Loading CineMatch...")
def build_payload():
    with open("movie.pkl", "rb") as f:
        movies = pd.DataFrame(pickle.load(f))
    titles = movies["title"].astype(str).tolist()
    n = len(movies)

    if os.path.exists("recs.pkl"):
        with open("recs.pkl", "rb") as f:
            recs = pickle.load(f)
    elif os.path.exists("similarity.pkl"):
        with open("similarity.pkl", "rb") as f:
            similarity = pickle.load(f)
        if n != len(similarity):
            raise ValueError(f"Movie data has {n} rows, but similarity data has {len(similarity)} rows.")
        sim = np.array(similarity, dtype=np.float32)
        np.fill_diagonal(sim, -1.0)
        k = min(K, n - 1)
        part = np.argpartition(-sim, k - 1, axis=1)[:, :k]
        rows = np.arange(n)[:, None]
        order = np.argsort(-sim[rows, part], axis=1)
        top = part[rows, order]
        scores = np.clip(np.rint(sim[rows, top] * 100), 0, 100).astype(int)
        recs = [[int(x) for pair in zip(top[i].tolist(), scores[i].tolist()) for x in pair]
                for i in range(n)]
    else:
        raise FileNotFoundError("recs.pkl or similarity.pkl")

    start = 0
    for t in PREFERRED_START:
        if t in titles:
            start = titles.index(t)
            break

    # Pre-fetch the first screen so the very first paint already has posters.
    first = [titles[start]] + [titles[j] for j in recs[start][0::2]]
    seed = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for t, info in zip(first, ex.map(_imdb_info, first)):
            if info["p"]:
                seed[t] = info

    payload = {"titles": titles, "recs": recs, "seed": seed, "start": start}
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    return PAGE.replace("__DATA__", data)


# ----------------------------------------------------------------------------
# Front-end (single self-contained document)
# ----------------------------------------------------------------------------
PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{
  --bg:#0b0b0c; --surface:#161618; --surface-2:#202023; --line:#2b2b30;
  --text:#f4f4f5; --muted:#a1a1aa; --accent:#e11d2e; --good:#46d369;
  --w:clamp(132px,11.5vw,184px); --pad:4vw;
  color-scheme:dark;
}
*{box-sizing:border-box;margin:0;padding:0}
html{background:var(--bg)}
body{background:var(--bg);color:var(--text);font:400 15px/1.5 Inter,"Segoe UI",system-ui,-apple-system,Roboto,Helvetica,Arial,sans-serif;
     -webkit-font-smoothing:antialiased;min-height:100vh}
body.lock{overflow:hidden}
button{font:inherit;color:inherit;cursor:pointer;border:0;background:none}
a{color:inherit;text-decoration:none}
:focus-visible{outline:2px solid #fff;outline-offset:2px}
::-webkit-scrollbar{width:10px;height:10px}
::-webkit-scrollbar-thumb{background:#2c2c31;border-radius:10px}
::-webkit-scrollbar-track{background:transparent}

/* ---------- navigation ---------- */
.nav{position:fixed;inset:0 0 auto 0;z-index:50;height:64px;display:flex;align-items:center;
     justify-content:space-between;padding:0 var(--pad);
     background:linear-gradient(180deg,rgba(11,11,12,.92),rgba(11,11,12,0));transition:background .25s}
.nav.solid{background:rgba(11,11,12,.94);backdrop-filter:blur(10px);border-bottom:1px solid var(--line)}
.nav-l{display:flex;align-items:center;gap:32px}
.logo{font-weight:800;font-size:22px;letter-spacing:.22em;color:var(--accent);cursor:pointer}
.links{display:flex;gap:4px}
.link{padding:8px 12px;border-radius:6px;color:var(--muted);font-weight:500;font-size:14px;transition:color .15s,background .15s}
.link:hover{color:var(--text)}
.link.active{color:var(--text);font-weight:600}
.count{display:inline-block;margin-left:6px;min-width:18px;padding:0 5px;border-radius:9px;background:var(--accent);
       color:#fff;font-size:11px;font-weight:700;line-height:18px;text-align:center}
.nav-r{display:flex;align-items:center;gap:10px}
.search{position:relative;width:min(360px,38vw)}
.search svg{position:absolute;left:12px;top:50%;transform:translateY(-50%);color:var(--muted);pointer-events:none}
.search input{width:100%;height:38px;padding:0 38px 0 38px;border-radius:8px;border:1px solid var(--line);
     background:rgba(32,32,35,.85);color:var(--text);font:inherit;font-size:14px;outline:none;transition:border-color .15s,background .15s}
.search input:focus{border-color:#55555c;background:var(--surface-2)}
.search input::placeholder{color:#7b7b83}
.search kbd{position:absolute;right:10px;top:50%;transform:translateY(-50%);font:600 11px/1 inherit;color:#8a8a92;
     border:1px solid var(--line);border-radius:4px;padding:3px 6px;pointer-events:none}
.sug{position:absolute;top:44px;left:0;right:0;list-style:none;background:var(--surface);border:1px solid var(--line);
     border-radius:10px;padding:6px;box-shadow:0 18px 40px rgba(0,0,0,.6);max-height:360px;overflow:auto}
.sug[hidden]{display:none}
.sug li{padding:9px 12px;border-radius:6px;cursor:pointer;font-size:14px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.sug li.on,.sug li:hover{background:var(--surface-2)}
.sug li.none{color:var(--muted);cursor:default}
.sug mark{background:none;color:#fff;font-weight:700}
.iconbtn{width:38px;height:38px;display:grid;place-items:center;border-radius:8px;border:1px solid var(--line);
     background:rgba(32,32,35,.85);color:var(--text);transition:background .15s}
.iconbtn:hover{background:var(--surface-2)}

/* ---------- hero ---------- */
.hero{position:relative;overflow:hidden;min-height:min(64vh,560px);display:flex;align-items:flex-end;padding:96px var(--pad) 60px}
.hero-bg{position:absolute;inset:-60px;background-size:cover;background-position:center 25%;
     filter:blur(34px) saturate(1.15) brightness(.55);transform:scale(1.15);opacity:0;transition:opacity .7s}
.hero-bg.in{opacity:1}
.hero::after{content:"";position:absolute;inset:0;
     background:linear-gradient(90deg,rgba(11,11,12,.9) 0%,rgba(11,11,12,.5) 55%,rgba(11,11,12,.15) 100%),
                linear-gradient(0deg,var(--bg) 0%,rgba(11,11,12,0) 45%)}
.hero-inner{position:relative;z-index:2;display:flex;gap:clamp(24px,4vw,56px);align-items:flex-end;width:100%;max-width:1360px;margin:0 auto}
.hero-poster{flex:none;width:clamp(170px,16vw,250px);aspect-ratio:2/3;border-radius:8px;overflow:hidden;
     box-shadow:0 24px 60px rgba(0,0,0,.65),0 0 0 1px rgba(255,255,255,.06);position:relative;background:var(--surface)}
.hero-text{min-width:0;max-width:680px}
.eyebrow{font-size:12px;font-weight:700;letter-spacing:.18em;text-transform:uppercase;color:var(--accent);margin-bottom:10px}
.hero h1{font-size:clamp(30px,4.6vw,64px);line-height:1.02;font-weight:800;letter-spacing:-.025em;text-wrap:balance}
.meta{display:flex;align-items:center;gap:14px;margin-top:16px;color:#d4d4d8;font-size:15px;flex-wrap:wrap}
.meta .good{color:var(--good);font-weight:700}
.tag{border:1px solid #6b6b72;border-radius:4px;padding:0 6px;font-size:11px;font-weight:600;line-height:18px;color:#c4c4cb}
.cast{margin-top:12px;color:var(--muted);font-size:15px;max-width:560px}
.cast[hidden],#h-year[hidden],#m-year[hidden],#m-cast[hidden]{display:none}
.actions{display:flex;gap:10px;margin-top:24px;flex-wrap:wrap}
.btn{display:inline-flex;align-items:center;gap:8px;height:44px;padding:0 20px;border-radius:8px;font-weight:600;font-size:15px;
     transition:background .15s,transform .1s;white-space:nowrap}
.btn:active{transform:scale(.98)}
.btn.primary{background:#fff;color:#000}
.btn.primary:hover{background:#e4e4e7}
.btn.sec{background:rgba(120,120,128,.35);color:#fff;backdrop-filter:blur(6px)}
.btn.sec:hover{background:rgba(120,120,128,.55)}

/* ---------- rows ---------- */
.rows{position:relative;z-index:3;margin-top:-28px;padding-bottom:30px}
.row{padding-top:14px}
.row-h{display:flex;align-items:baseline;gap:14px;padding:0 var(--pad)}
.row-h h2{font-size:20px;font-weight:700;letter-spacing:-.01em}
.row-h .sub{color:var(--muted);font-size:13px}
.sc-wrap{position:relative}
.scroller{display:flex;gap:12px;overflow-x:auto;padding:16px var(--pad) 44px;scroll-snap-type:x proximity;
     scroll-padding:0 var(--pad);scrollbar-width:none}
.scroller::-webkit-scrollbar{display:none}
.arrow{position:absolute;top:16px;bottom:44px;width:var(--pad);min-width:40px;z-index:6;display:grid;place-items:center;
     opacity:0;transition:opacity .2s;color:#fff}
.arrow.l{left:0;background:linear-gradient(90deg,rgba(11,11,12,.9),transparent)}
.arrow.r{right:0;background:linear-gradient(270deg,rgba(11,11,12,.9),transparent)}
.sc-wrap:hover .arrow{opacity:1}
.item{flex:0 0 var(--w);scroll-snap-align:start;position:relative}
.item.ranked{flex-basis:calc(var(--w) + 64px);display:flex;align-items:flex-end}
.num{font-size:calc(var(--w)*.92);font-weight:800;line-height:.76;letter-spacing:-.07em;color:var(--bg);
     -webkit-text-stroke:3px #62626a;margin-right:-22px;position:relative;z-index:0;user-select:none}
.num.n2{font-size:calc(var(--w)*.66);margin-right:-14px;letter-spacing:-.1em}
.ranked .card{flex:none;width:var(--w);z-index:1}

.card{position:relative;aspect-ratio:2/3;border-radius:6px;overflow:hidden;background:var(--surface);cursor:pointer;
     transition:transform .22s cubic-bezier(.2,.7,.2,1),box-shadow .22s;will-change:transform;width:100%}
.card:hover,.card:focus-visible{transform:scale(1.07);box-shadow:0 20px 44px rgba(0,0,0,.65);z-index:5}
.poster{position:absolute;inset:0;background:var(--surface)}
.poster img{width:100%;height:100%;object-fit:cover;display:block;opacity:0;transition:opacity .4s}
.poster img.in{opacity:1}
.sk{position:absolute;inset:0;background:linear-gradient(100deg,#161618 30%,#232327 50%,#161618 70%);
    background-size:200% 100%;animation:sh 1.3s linear infinite}
@keyframes sh{to{background-position:-200% 0}}
.noimg{display:flex;align-items:center;justify-content:center;text-align:center;padding:14px;
     background:linear-gradient(160deg,#222226,#141416);color:#9a9aa3;font-weight:700;font-size:15px;line-height:1.25}
.hover{position:absolute;inset:auto 0 0 0;padding:44px 10px 10px;background:linear-gradient(0deg,rgba(0,0,0,.95) 25%,transparent);
     opacity:0;transform:translateY(6px);transition:opacity .2s,transform .2s}
.card:hover .hover,.card:focus-visible .hover{opacity:1;transform:none}
.h-title{font-weight:700;font-size:13px;line-height:1.2;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.h-meta{font-size:12px;color:var(--good);font-weight:600;margin-top:3px;min-height:16px}
.h-row{display:flex;gap:6px;margin-top:8px}
.ib{width:30px;height:30px;border-radius:50%;display:grid;place-items:center;border:1.5px solid rgba(255,255,255,.55);
    background:rgba(0,0,0,.45);color:#fff;transition:background .15s,color .15s,border-color .15s}
.ib:hover{background:#fff;color:#000;border-color:#fff}
@media (hover:none){.hover{opacity:1;transform:none}}

.list-view{padding:100px var(--pad) 30px;min-height:80vh}
.list-view h1{font-size:30px;font-weight:800;letter-spacing:-.02em}
.list-view .sub{color:var(--muted);margin:4px 0 26px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(var(--w),1fr));gap:22px 14px}
.empty{display:flex;flex-direction:column;align-items:center;text-align:center;gap:10px;padding:90px 0;color:var(--muted)}
.empty h2{color:var(--text);font-size:22px}
.footer{padding:30px var(--pad) 40px;border-top:1px solid var(--line);margin:10px var(--pad) 0;padding-left:0;padding-right:0;
     color:#70707a;font-size:12.5px;display:flex;justify-content:space-between;gap:20px;flex-wrap:wrap}

/* ---------- modal ---------- */
.modal{position:fixed;inset:0;z-index:100;background:rgba(0,0,0,.72);backdrop-filter:blur(5px);overflow:auto;
     display:flex;justify-content:center;padding:5vh 18px;animation:fade .2s}
.modal[hidden]{display:none}
.panel{position:relative;width:min(960px,100%);align-self:flex-start;background:var(--surface);border-radius:14px;
     border:1px solid var(--line);box-shadow:0 40px 100px rgba(0,0,0,.7);animation:pop .25s cubic-bezier(.2,.8,.2,1);outline:none}
.x{position:absolute;top:14px;right:14px;width:36px;height:36px;border-radius:50%;display:grid;place-items:center;
     background:rgba(0,0,0,.55);z-index:3;transition:background .15s}
.x:hover{background:rgba(255,255,255,.2)}
.m-top{display:flex;gap:30px;padding:34px 34px 10px}
.m-poster{flex:none;width:min(240px,34%);aspect-ratio:2/3;border-radius:8px;overflow:hidden;position:relative;background:var(--surface-2);
     box-shadow:0 16px 40px rgba(0,0,0,.5)}
.m-info{min-width:0;padding-top:4px;padding-right:40px}
.m-info h2{font-size:clamp(24px,3.2vw,38px);line-height:1.08;font-weight:800;letter-spacing:-.02em}
.m-sub{font-size:16px;font-weight:700;padding:18px 34px 12px}
.m-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(130px,1fr));gap:18px 12px;padding:0 34px 34px}
.m-t{font-size:12.5px;font-weight:600;margin-top:8px;line-height:1.25;color:#d4d4d8}
.m-item .card:hover{transform:scale(1.04)}
@keyframes pop{from{opacity:0;transform:translateY(14px) scale(.98)}to{opacity:1;transform:none}}
@keyframes fade{from{opacity:0}to{opacity:1}}

.toast{position:fixed;left:50%;bottom:28px;transform:translate(-50%,20px);background:#fff;color:#000;font-weight:600;font-size:14px;
     padding:11px 18px;border-radius:8px;box-shadow:0 14px 34px rgba(0,0,0,.5);opacity:0;pointer-events:none;
     transition:opacity .2s,transform .2s;z-index:200}
.toast.show{opacity:1;transform:translate(-50%,0)}

@media (max-width:760px){
  .links{display:none}.m-top{flex-direction:column;padding:24px 20px 6px}.m-poster{width:46%}.m-sub,.m-grid{padding-left:20px;padding-right:20px}
  .hero-poster{display:none}
}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style>
</head>
<body>

<header class="nav" id="nav">
  <div class="nav-l">
    <div class="logo" data-act="view" data-v="home">CINEMATCH</div>
    <nav class="links">
      <button class="link active" data-act="view" data-v="home">Home</button>
      <button class="link" data-act="view" data-v="list">My List<span class="count" id="cnt" hidden></span></button>
    </nav>
  </div>
  <div class="nav-r">
    <div class="search" id="search">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3"/></svg>
      <input id="q" type="text" autocomplete="off" spellcheck="false" aria-label="Search movies">
      <kbd>/</kbd>
      <ul class="sug" id="sug" hidden></ul>
    </div>
    <button class="iconbtn" data-act="surprise" title="Surprise me" aria-label="Surprise me">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M16 3h5v5M4 20L21 3M21 16v5h-5M15 15l6 6M4 4l5 5"/></svg>
    </button>
  </div>
</header>

<main id="app"></main>
<div class="modal" id="modal" hidden></div>
<div class="toast" id="toast" role="status"></div>

<script>
(() => {
"use strict";
const DATA = __DATA__;
const T = DATA.titles, R = DATA.recs;
const $ = s => document.querySelector(s);
const esc = s => String(s).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const IDX = new Map(T.map((t, i) => [t, i]));

const svg = (d, extra = "") => `<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" ${extra}>${d}</svg>`;
const ICON = {
  play:  '<svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M8 5.5v13a1 1 0 0 0 1.5.86l10.5-6.5a1 1 0 0 0 0-1.72L9.5 4.64A1 1 0 0 0 8 5.5z"/></svg>',
  plus:  svg('<path d="M12 5v14M5 12h14"/>'),
  check: svg('<path d="M5 12.5l4.5 4.5L19 7.5"/>'),
  info:  svg('<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/>'),
  left:  svg('<path d="M15 5l-7 7 7 7"/>', 'width="28" height="28"'),
  right: svg('<path d="M9 5l7 7-7 7"/>', 'width="28" height="28"'),
  close: svg('<path d="M6 6l12 12M18 6L6 18"/>'),
  down:  svg('<path d="M6 9l6 6 6-6"/>')
};

/* ---------- storage (falls back to memory if the sandbox blocks it) ---------- */
const store = (() => {
  const mem = {};
  try { localStorage.setItem("__cm", "1"); localStorage.removeItem("__cm");
        return { get: k => localStorage.getItem(k), set: (k, v) => localStorage.setItem(k, v) }; }
  catch (e) { return { get: k => (k in mem ? mem[k] : null), set: (k, v) => { mem[k] = v; } }; }
})();

/* ---------- state ---------- */
const state = { cur: DATA.start, view: "home", history: [], list: [] };
try { JSON.parse(store.get("cm_list") || "[]").forEach(t => { if (IDX.has(t)) state.list.push(IDX.get(t)); }); } catch (e) {}
const saveList = () => { try { store.set("cm_list", JSON.stringify(state.list.map(i => T[i]))); } catch (e) {} };

/* ---------- poster/meta fetching: cached, parallel, prioritised ---------- */
const info = new Map(Object.entries(DATA.seed || {}));
try { const s = JSON.parse(store.get("cm_info_v1") || "{}"); for (const k in s) if (!info.has(k)) info.set(k, s[k]); } catch (e) {}
let persistTimer = null;
const persist = () => { clearTimeout(persistTimer); persistTimer = setTimeout(() => {
  try { const o = {}; let n = 0; for (const [k, v] of info) { if (v && v.p) { o[k] = v; if (++n > 6000) break; } }
        store.set("cm_info_v1", JSON.stringify(o)); } catch (e) {} }, 800); };

async function fetchInfo(title) {
  const fail = { p: null, y: null, c: "", fail: 1 };
  const q = title.replace(/[^a-zA-Z0-9]+/g, "_").replace(/^_+|_+$/g, "");
  if (!q) return fail;
  const ctl = new AbortController(); const to = setTimeout(() => ctl.abort(), 9000);
  try {
    const r = await fetch(`https://v3.sg.media-imdb.com/suggestion/x/${q}.json`, { signal: ctl.signal });
    const d = await r.json();
    const items = (d.d || []).filter(i => i.i && i.i.imageUrl);
    const mv = items.filter(i => i.qid === "movie");
    const pool = mv.length ? mv : items;
    if (!pool.length) return fail;
    const lt = title.toLowerCase();
    const best = pool.find(i => String(i.l || "").toLowerCase() === lt) || pool[0];
    return { p: best.i.imageUrl, y: best.y || null, c: best.s || "" };
  } catch (e) { return fail; } finally { clearTimeout(to); }
}
const pending = new Map(), queue = []; let active = 0;
function pump() {
  while (active < 6 && queue.length) {
    const job = queue.shift(); active++;
    fetchInfo(job.title).then(v => { info.set(job.title, v); if (v.p) persist(); pending.delete(job.title); job.res(v); })
      .finally(() => { active--; pump(); });
  }
}
function getInfo(title, hi) {
  if (info.has(title)) return Promise.resolve(info.get(title));
  let job = pending.get(title);
  if (!job) { job = { title }; job.p = new Promise(r => { job.res = r; }); pending.set(title, job);
              hi ? queue.unshift(job) : queue.push(job); pump(); }
  else if (hi) { const k = queue.indexOf(job); if (k > 0) { queue.splice(k, 1); queue.unshift(job); } }
  return job.p;
}
const sized = (u, w) => u.replace(/\._V1_.*?(\.(?:jpe?g|png))$/i, `._V1_UX${w}_$1`);

function paint(el, inf) {
  const t = el.dataset.t, w = el.dataset.w || 320;
  const meta = el.closest(".card") && el.closest(".card").querySelector(".h-meta");
  if (meta && inf && inf.y) meta.textContent = (el.dataset.s ? el.dataset.s + "% match · " : "") + inf.y;
  const noimg = () => { el.classList.add("noimg"); el.innerHTML = `<span>${esc(t)}</span>`; };
  if (!inf || !inf.p) return noimg();
  const img = new Image(); img.decoding = "async"; img.alt = t; img.draggable = false;
  img.onload = () => { el.replaceChildren(img); requestAnimationFrame(() => img.classList.add("in")); };
  img.onerror = () => { if (img.dataset.fb !== "1") { img.dataset.fb = "1"; img.src = inf.p; } else noimg(); };
  img.src = sized(inf.p, w);
}
function hydrate(root) {
  root.querySelectorAll(".poster[data-t]").forEach(el => {
    getInfo(el.dataset.t, el.dataset.hi === "1").then(inf => paint(el, inf));
  });
}

/* ---------- templates ---------- */
const pairs = (arr, from, to) => { const o = []; for (let k = from * 2; k < Math.min(to * 2, arr.length); k += 2) o.push([arr[k], arr[k + 1]]); return o; };
const scoreOf = i => { const r = R[state.cur]; for (let k = 0; k < r.length; k += 2) if (r[k] === i) return r[k + 1]; return null; };
const trailer = t => `https://www.youtube.com/results?search_query=${encodeURIComponent(t + " official trailer")}`;

function card(i, score, rank) {
  const t = T[i], on = state.list.includes(i);
  const body = `
    <article class="card" data-act="open" data-i="${i}" tabindex="0" role="button" aria-label="${esc(t)}">
      <div class="poster" data-t="${esc(t)}" data-w="320" ${score != null ? `data-s="${score}"` : ""}><div class="sk"></div></div>
      <div class="hover">
        <div class="h-title">${esc(t)}</div>
        <div class="h-meta">${score != null ? score + "% match" : ""}</div>
        <div class="h-row">
          <button class="ib" data-act="similar" data-i="${i}" title="Explore similar" aria-label="Explore similar">${ICON.play}</button>
          <button class="ib" data-act="list" data-i="${i}" title="${on ? "Remove from My List" : "Add to My List"}" aria-label="My List">${on ? ICON.check : ICON.plus}</button>
          <button class="ib" data-act="open" data-i="${i}" title="Details" aria-label="Details">${ICON.down}</button>
        </div>
      </div>
    </article>`;
  return rank ? `<div class="item ranked"><span class="num ${rank > 9 ? "n2" : ""}">${rank}</span>${body}</div>`
              : `<div class="item">${body}</div>`;
}
function row(id, title, sub, inner) {
  return `<section class="row" id="${id}">
    <div class="row-h"><h2>${title}</h2>${sub ? `<span class="sub">${sub}</span>` : ""}</div>
    <div class="sc-wrap">
      <button class="arrow l" data-act="scroll" data-dir="-1" aria-label="Scroll left">${ICON.left}</button>
      <div class="scroller">${inner}</div>
      <button class="arrow r" data-act="scroll" data-dir="1" aria-label="Scroll right">${ICON.right}</button>
    </div></section>`;
}
const listRow = () => state.list.length ? row("r-list", "My List", "", state.list.map(j => card(j, null)).join("")) : "";

function heroHTML(i) {
  const t = T[i], on = state.list.includes(i);
  return `<section class="hero">
    <div class="hero-bg" id="h-bg"></div>
    <div class="hero-inner">
      <div class="hero-poster"><div class="poster" data-t="${esc(t)}" data-w="500" data-hi="1"><div class="sk"></div></div></div>
      <div class="hero-text">
        <div class="eyebrow">Your pick</div>
        <h1>${esc(t)}</h1>
        <div class="meta"><span class="good">${R[i][1]}% top match</span><span id="h-year" hidden></span><span class="tag">HD</span></div>
        <p class="cast" id="h-cast" hidden></p>
        <div class="actions">
          <a class="btn primary" href="${trailer(t)}" target="_blank" rel="noopener">${ICON.play}<span>Watch trailer</span></a>
          <button class="btn sec" data-act="list" data-i="${i}">${on ? ICON.check : ICON.plus}<span>My List</span></button>
          <button class="btn sec" data-act="open" data-i="${i}">${ICON.info}<span>Details</span></button>
        </div>
      </div>
    </div></section>`;
}
const footer = () => `<footer class="footer"><span>CineMatch &middot; content-based recommendations using cosine similarity</span><span>Posters &amp; metadata via IMDb</span></footer>`;

function homeHTML() {
  const i = state.cur, r = R[i];
  let h = heroHTML(i) + `<div class="rows" id="rows">`;
  h += row("r-top", "Top 10 for you", "Because you picked " + esc(T[i]), pairs(r, 0, 10).map(([j, s], n) => card(j, s, n + 1)).join(""));
  const more = pairs(r, 10, 20);
  if (more.length) h += row("r-more", "More like " + esc(T[i]), "", more.map(([j, s]) => card(j, s)).join(""));
  if (state.history.length) h += row("r-hist", "Recently explored", "", state.history.map(j => card(j, null)).join(""));
  h += `<div id="slot-list">${listRow()}</div></div>` + footer();
  return h;
}
function listHTML() {
  if (!state.list.length) return `<div class="list-view"><div class="empty"><h2>Your list is empty</h2>
    <p>Tap the + on any title to save it here.</p><button class="btn primary" data-act="view" data-v="home">Browse picks</button></div></div>` + footer();
  return `<div class="list-view"><h1>My List</h1><p class="sub">${state.list.length} saved title${state.list.length > 1 ? "s" : ""}</p>
    <div class="grid">${state.list.map(j => card(j, null)).join("")}</div></div>` + footer();
}

/* ---------- render ---------- */
const app = $("#app"), nav = $("#nav");
function render() {
  app.innerHTML = state.view === "home" ? homeHTML() : listHTML();
  hydrate(app);
  if (state.view === "home") hydrateHero(state.cur);
  document.querySelectorAll(".link").forEach(b => b.classList.toggle("active", b.dataset.v === state.view));
  syncList();
}
function hydrateHero(i) {
  getInfo(T[i], true).then(inf => {
    if (state.cur !== i || state.view !== "home") return;
    const y = $("#h-year"), c = $("#h-cast"), bg = $("#h-bg");
    if (inf.y && y) { y.textContent = inf.y; y.hidden = false; }
    if (inf.c && c) { c.textContent = "Starring " + inf.c; c.hidden = false; }
    if (inf.p && bg) { bg.style.backgroundImage = `url("${sized(inf.p, 200)}")`; bg.classList.add("in"); }
  });
}
function syncList() {
  document.querySelectorAll('[data-act="list"]').forEach(b => {
    const on = state.list.includes(+b.dataset.i), ic = on ? ICON.check : ICON.plus;
    b.innerHTML = b.classList.contains("btn") ? ic + "<span>My List</span>" : ic;
    b.title = on ? "Remove from My List" : "Add to My List";
  });
  const c = $("#cnt"); c.textContent = state.list.length || ""; c.hidden = !state.list.length;
}
function refreshSlot() {
  if (state.view === "list") { render(); return; }
  const s = $("#slot-list"); if (s) { s.innerHTML = listRow(); hydrate(s); }
}

/* ---------- modal ---------- */
const modal = $("#modal");
function modalHTML(i) {
  const t = T[i], on = state.list.includes(i), s = scoreOf(i);
  const sim = pairs(R[i], 0, 8);
  return `<div class="panel" tabindex="-1" role="dialog" aria-modal="true" aria-label="${esc(t)}">
    <button class="x" data-act="close" aria-label="Close">${ICON.close}</button>
    <div class="m-top">
      <div class="m-poster"><div class="poster" data-t="${esc(t)}" data-w="500" data-hi="1"><div class="sk"></div></div></div>
      <div class="m-info">
        <h2>${esc(t)}</h2>
        <div class="meta">${s != null ? `<span class="good">${s}% match</span>` : ""}<span id="m-year" hidden></span><span class="tag">HD</span></div>
        <p class="cast" id="m-cast" hidden></p>
        <div class="actions">
          <button class="btn primary" data-act="similar" data-i="${i}">${ICON.play}<span>Explore similar</span></button>
          <button class="btn sec" data-act="list" data-i="${i}">${on ? ICON.check : ICON.plus}<span>My List</span></button>
          <a class="btn sec" href="${trailer(t)}" target="_blank" rel="noopener"><span>Trailer</span></a>
        </div>
      </div>
    </div>
    <h3 class="m-sub">More like this</h3>
    <div class="m-grid">${sim.map(([j, sc]) => `<div class="m-item">${card(j, sc)}<div class="m-t">${esc(T[j])}</div></div>`).join("")}</div>
  </div>`;
}
function openModal(i) {
  modal.innerHTML = modalHTML(i); modal.hidden = false; document.body.classList.add("lock"); modal.scrollTop = 0;
  hydrate(modal);
  getInfo(T[i], true).then(inf => {
    const y = modal.querySelector("#m-year"), c = modal.querySelector("#m-cast");
    if (inf.y && y) { y.textContent = inf.y; y.hidden = false; }
    if (inf.c && c) { c.textContent = "Starring " + inf.c; c.hidden = false; }
  });
  const p = modal.querySelector(".panel"); if (p) p.focus({ preventScroll: true });
}
function closeModal() { modal.hidden = true; modal.innerHTML = ""; document.body.classList.remove("lock"); }

/* ---------- actions ---------- */
const toTop = () => window.scrollTo({ top: 0, behavior: "smooth" });
let toastT = null;
function toast(msg) { const el = $("#toast"); el.textContent = msg; el.classList.add("show");
  clearTimeout(toastT); toastT = setTimeout(() => el.classList.remove("show"), 1800); }
function setCurrent(i) {
  if (state.cur !== i) state.history = [state.cur, ...state.history.filter(x => x !== state.cur && x !== i)].slice(0, 12);
  state.cur = i; state.view = "home"; closeModal(); render(); toTop();
}
function toggleList(i) {
  const k = state.list.indexOf(i);
  if (k >= 0) { state.list.splice(k, 1); toast("Removed from My List"); } else { state.list.unshift(i); toast("Added to My List"); }
  saveList(); refreshSlot(); syncList();
}

document.addEventListener("click", e => {
  if (e.target === modal) { closeModal(); return; }
  if (!e.target.closest("#search")) $("#sug").hidden = true;
  const el = e.target.closest("[data-act]"); if (!el) return;
  const act = el.dataset.act, i = +el.dataset.i;
  switch (act) {
    case "view": state.view = el.dataset.v; closeModal(); render(); toTop(); break;
    case "open": openModal(i); break;
    case "similar": setCurrent(i); break;
    case "list": toggleList(i); break;
    case "close": closeModal(); break;
    case "surprise": setCurrent(Math.floor(Math.random() * T.length)); break;
    case "pick": $("#q").value = ""; $("#sug").hidden = true; setCurrent(i); break;
    case "scroll": { const sc = el.parentElement.querySelector(".scroller");
                     sc.scrollBy({ left: +el.dataset.dir * sc.clientWidth * 0.85, behavior: "smooth" }); break; }
  }
});
window.addEventListener("scroll", () => nav.classList.toggle("solid", window.scrollY > 24), { passive: true });

/* ---------- search ---------- */
const lower = T.map(t => t.toLowerCase()), q = $("#q"), sug = $("#sug");
let res = [], sel = 0;
q.placeholder = `Search ${T.length.toLocaleString()} movies`;
function find(v) {
  v = v.trim().toLowerCase(); if (!v) return [];
  const a = [], b = [];
  for (let i = 0; i < lower.length && a.length < 8; i++) {
    const k = lower[i].indexOf(v);
    if (k === 0) a.push(i); else if (k > 0 && b.length < 8) b.push(i);
  }
  return a.concat(b).slice(0, 8);
}
const hl = (t, v) => { const k = t.toLowerCase().indexOf(v.toLowerCase());
  return k < 0 ? esc(t) : esc(t.slice(0, k)) + "<mark>" + esc(t.slice(k, k + v.length)) + "</mark>" + esc(t.slice(k + v.length)); };
function paintSug() {
  const v = q.value.trim();
  if (!v) { sug.hidden = true; return; }
  res = find(v); sel = 0; sug.hidden = false;
  sug.innerHTML = res.length
    ? res.map((i, n) => `<li data-act="pick" data-i="${i}" class="${n === sel ? "on" : ""}">${hl(T[i], v)}</li>`).join("")
    : '<li class="none">No matches</li>';
}
const markSel = () => [...sug.children].forEach((li, n) => li.classList.toggle("on", n === sel));
q.addEventListener("input", paintSug);
q.addEventListener("focus", paintSug);
q.addEventListener("keydown", e => {
  if (e.key === "ArrowDown" && res.length) { e.preventDefault(); sel = (sel + 1) % res.length; markSel(); }
  else if (e.key === "ArrowUp" && res.length) { e.preventDefault(); sel = (sel - 1 + res.length) % res.length; markSel(); }
  else if (e.key === "Enter" && res[sel] != null) { e.preventDefault(); q.value = ""; sug.hidden = true; q.blur(); setCurrent(res[sel]); }
  else if (e.key === "Escape") { sug.hidden = true; q.blur(); }
});
document.addEventListener("keydown", e => {
  const typing = /INPUT|TEXTAREA/.test((document.activeElement || {}).tagName || "");
  if (e.key === "/" && !typing) { e.preventDefault(); q.focus(); }
  else if (e.key === "Escape" && !modal.hidden) closeModal();
  else if (e.key === "Enter" && document.activeElement && document.activeElement.classList.contains("card")) document.activeElement.click();
});

render();
})();
</script>
</body>
</html>
"""

# ----------------------------------------------------------------------------
# Streamlit entry point
# ----------------------------------------------------------------------------
try:
    html_page = build_payload()
except FileNotFoundError as error:
    st.error(f"Could not find model artifacts ({error}). Ensure movie.pkl and recs.pkl (or similarity.pkl) are present in the app root.")
    st.stop()
except Exception as error:
    st.error(f"Could not load the recommender data: {error}")
    st.stop()

components.html(html_page, height=900, scrolling=True)
