# main.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Optional, List, Dict, Tuple
from fastapi.middleware.cors import CORSMiddleware
import os, re, requests, math, time
import hashlib, random

# ────────────────────────────────────────────────────────────────────────────────
# App
# ────────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Startup Incubation AI Analysis API", version="2.7.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ────────────────────────────────────────────────────────────────────────────────
# Schemas
# ────────────────────────────────────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    step_name: str
    question_label: str
    startup_name: Optional[str] = ""
    sector_name: Optional[str] = ""
    context: Optional[str] = ""
    question_type: Optional[str] = ""
    question_key: Optional[str] = ""
    prompt: Optional[str] = ""

class Item(BaseModel):
    question_id: int
    label: Optional[str] = None
    type: str
    answer: Optional[str] = ""
    points: Optional[int] = 10
    has_file: Optional[bool] = False

class StructuredRequest(BaseModel):
    step_name: str
    items: List[Item]

# ────────────────────────────────────────────────────────────────────────────────
# Time/deadline helpers
# ────────────────────────────────────────────────────────────────────────────────
def _now_ms() -> int:
    return int(time.time() * 1000)

def _seconds_left(deadline_ms: Optional[int]) -> float:
    if not deadline_ms:
        return 999.0
    return max(0.0, (deadline_ms - _now_ms()) / 1000.0)

# ────────────────────────────────────────────────────────────────────────────────
# Text utils
# ────────────────────────────────────────────────────────────────────────────────
PLACEHOLDERS = [
    r"\[[^\]]+\]",     # [problème], [date], etc.
    r"\bSegment\s+X\b",
    r"\bY\s*personnes\b",
    r"\bZ\s*€\b",
    r"\bX%\b",
]

def clean(txt: str) -> str:
    """Remove sentences containing placeholders, normalize spaces."""
    t = txt or ""
    for p in PLACEHOLDERS:
        if re.search(p, t, flags=re.I):
            sentences = re.split(r"(?<=[\.\!\?])\s+", t)
            sentences = [s for s in sentences if not re.search(p, s, flags=re.I)]
            t = " ".join(sentences)
    return re.sub(r"\s{2,}", " ", t).strip()

def smart_clean(txt: str) -> str:
    out = clean(txt)
    return out if out else (txt or "").strip()

def split_list(s: str) -> List[str]:
    if not s:
        return []
    parts = re.split(r"[,\n;\|]+", s)
    return [p.strip() for p in parts if p.strip()]

# ────────────────────────────────────────────────────────────────────────────────
# Parse Laravel context -> facts + corpus
# ────────────────────────────────────────────────────────────────────────────────
def extract_facts(context: str) -> Dict[str, str]:
    """
    Parse 'Key: value' lines. Recognized keys (case/space-insensitive):
      Startup, Sector, Pitch, Description, ValueProp, Advantage, Personas,
      CustomerSegments, Problem, Solution, BusinessModel, RevenueStreams,
      KPIs, Channels, ShortTermGoals, LongTermGoals, Geo, Pricing,
      QuestionLabel, QuestionType, QuestionPrompt
    """
    facts: Dict[str, str] = {}
    if not context:
        return facts

    mapping = {
        "startup": "startup",
        "sector": "sector",
        "pitch": "pitch",
        "description": "description",
        "valueprop": "value_prop",
        "advantage": "advantage",
        "personas": "personas",
        "customersegments": "customer_segments",
        "problem": "problem",
        "solution": "solution",
        "businessmodel": "business_model",
        "revenuestreams": "revenue_streams",
        "kpis": "kpis",
        "channels": "channels",
        "shorttermgoals": "short_term_goals",
        "longtermgoals": "long_term_goals",
        "geo": "geo",
        "pricing": "pricing",
        "questionlabel": "question_label",
        "questiontype": "question_type",
        "questionprompt": "question_prompt",
    }

    for raw in (context or "").splitlines():
        if ":" not in raw:
            continue
        k, v = raw.split(":", 1)
        k = k.strip().lower().replace(" ", "")
        v = v.strip()
        if not v:
            continue
        if k in mapping:
            facts[mapping[k]] = v

    # Derive revenue lines
    if "revenue_streams" in facts:
        parts = [p.strip() for p in re.split(r"[;,/]| et ", facts["revenue_streams"]) if p.strip()]
        if parts:
            facts["revenue_primary"] = parts[0]
            if len(parts) > 1:
                facts["revenue_secondary"] = parts[1]

    return facts

def parse_context_lines(context: str) -> Tuple[Dict[str,str], List[Tuple[str,str]]]:
    """
    Return:
      - facts (dict)
      - corpus: list of (label, text) including PrevAnswer & profile fields
    """
    facts = extract_facts(context)
    corpus: List[Tuple[str,str]] = []

    for raw in (context or "").splitlines():
        raw = raw.strip()
        if not raw:
            continue

        # Prev answers: "PrevAnswer: Question => Value"
        if raw.lower().startswith("prevanswer:"):
            m = re.match(r"^PrevAnswer:\s*(.+?)\s*=>\s*(.+)$", raw, flags=re.I)
            if m:
                corpus.append((m.group(1).strip(), m.group(2).strip()))
            continue

        # Profile "Key: Value"
        if ":" in raw:
            k, v = raw.split(":", 1)
            k = k.strip()
            v = v.strip()
            if v:
                corpus.append((k, v))

    return facts, corpus

# ────────────────────────────────────────────────────────────────────────────────
# Style & prompts
# ────────────────────────────────────────────────────────────────────────────────
def style_from_prompt(prompt: str) -> str:
    p = (prompt or "").lower()
    if "bullet" in p or "puce" in p or "liste" in p:
        return "bullets"
    if "court" in p or "1-2 phrases" in p or "concise" in p:
        return "short"
    return "paragraph"

def coach_system_prompt() -> str:
    return (
        "Tu es un coach d'incubation francophone. Tu produis des réponses "
        "spécifiques à CHAQUE startup en t'appuyant sur son profil et ses réponses passées.\n\n"
        "RÈGLES GÉNÉRALES\n"
        "1) Adapter la réponse au secteur, au pitch, au business model et aux objectifs fournis.\n"
        "2) NE PAS réutiliser mot pour mot des phrases d'autres réponses de cette startup : reformule et change l'angle.\n"
        "3) Si une donnée manque, fais une hypothèse prudente (~, ≈) et propose une action de validation concrète.\n"
        "4) Interdit: placeholders ([xxx], X%, Segment X), copier-coller du contexte.\n"
        "5) Style: clair, orienté chiffres, concret. FR par défaut; si le contexte est majoritairement en EN, réponds en EN.\n"
        "6) Respecter le CONTRAT DE SORTIE selon le type:\n"
        "   - text/textarea → une réponse; pas de liste si non demandée; 4–6 lignes par défaut.\n"
        "   - number → UN entier ou décimal sans texte (ex: 1500). Sans symbole.\n"
        "   - date → format ISO AAAA-MM-JJ.\n"
        "   - email → adresse valide.\n"
        "   - file → nom-de-fichier.ext — courte description (1 ligne).\n"
        "   - choice → une valeur parmi les options proposées si connues.\n"
        "7) Si 'liste'/'puces' est demandé → puces (−) concises; sinon paragraphe.\n"
        "8) Terminer par 'Assumptions & Next step: …' si des hypothèses sont posées.\n"
    )

def format_context_snippets(snippets: List[Tuple[str,str]]) -> str:
    out = []
    for i, (lbl, txt) in enumerate(snippets, 1):
        out.append(f"[{i}] {lbl}: {txt}")
    return "\n".join(out)

def _variation_seed(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % 10000

STYLE_VARIANTS = {
    "paragraph": [
        "paragraphe compact, ton pragmatique",
        "paragraphe orienté résultats, ton direct",
        "paragraphe synthétique, ton business",
    ],
    "short": [
        "1–2 phrases, ton tranchant",
        "1–2 phrases, ton orienté métriques",
        "1–2 phrases, ton produit",
    ],
    "bullets": [
        "puces brèves, verbe d'action en tête",
        "puces avec un chiffre par ligne si pertinent",
        "puces MECE, 4–6 items max",
    ],
}

def _forbidden_fragments(snippets: List[Tuple[str,str]], max_chars=320) -> List[str]:
    """Small fragments to NOT copy verbatim (anti copy/paste)."""
    frags, used = [], 0
    for _, txt in snippets[:3]:
        s = re.sub(r"\s+", " ", (txt or "")).strip()
        if 40 <= len(s) <= 160 and used + len(s) <= max_chars:
            frags.append(s); used += len(s)
    return frags

def build_llm_prompt(step_name: str, qlabel: str, qtype: str, style: str,
                     facts: Dict[str, str], rag_snippets: List[Tuple[str,str]]) -> str:
    seed = _variation_seed((facts.get("startup","") + qlabel + step_name)[:256])
    rnd = random.Random(seed)
    style_note = rnd.choice(STYLE_VARIANTS.get(style, STYLE_VARIANTS["paragraph"]))

    header = [
        f"Étape: {step_name}",
        f"Question: {qlabel}",
        f"Type: {qtype}",
        f"Style attendu: {style} ({style_note})",
        "",
        "=== Contexte pertinent (profil & réponses passées) ===",
        format_context_snippets(rag_snippets) or "(aucun extrait pertinent retrouvé)",
        "",
        "=== Faits bruts additionnels ===",
    ]
    keys = [
        "startup","sector","pitch","description","value_prop","advantage","personas",
        "customer_segments","problem","solution","business_model","revenue_streams",
        "kpis","channels","short_term_goals","long_term_goals","geo","pricing"
    ]
    facts_lines = [f"- {k}: {facts[k]}" for k in keys if facts.get(k)]
    header += facts_lines or ["(aucun fait brut additionnel)"]

    forbidden = _forbidden_fragments(rag_snippets)
    if forbidden:
        header += ["", "=== Fragments à NE PAS reprendre textuellement ==="]
        header += [f"- {f}" for f in forbidden]

    header += [
        "",
        "=== CONTRAT DE SORTIE ===",
        "1) Adapter la réponse à la startup et au secteur.",
        "2) Respecter le type (voir système). Interdits: placeholders, copier-coller du contexte.",
        "3) Si hypothèses → terminer par: 'Assumptions & Next step: …' (1 ligne).",
        "",
        "Consigne finale: rédige la réponse maintenant."
    ]
    return "\n".join(header)

OPENERS = [
    "Concrètement, ",
    "En pratique, ",
    "Pour cette startup, ",
    "Dans ce contexte, ",
    ""
]

def vary_opening(txt: str, facts: Dict[str,str], qlabel: str) -> str:
    seed = (facts.get("startup","") + qlabel)[:256]
    r = random.Random(_variation_seed(seed))
    opener = r.choice(OPENERS)
    if txt and opener and not txt.startswith(("−","-")):
        return opener + txt[0].lower() + txt[1:]
    return txt

# ────────────────────────────────────────────────────────────────────────────────
# Type coercion
# ────────────────────────────────────────────────────────────────────────────────
def coerce_by_type(answer: str, qtype: str) -> str:
    a = (answer or "").strip()
    t = (qtype or "").lower()
    if t == "number":
        m = re.findall(r"-?\d+(?:[.,]\d+)?", a)
        return m[0].replace(",", ".") if m else ""
    if t == "date":
        a2 = a.replace(".", "/").replace(" ", "")
        m = re.search(r"(\d{2})/(\d{2})/(\d{4})", a2)
        if m: return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
        m = re.search(r"(\d{4})-(\d{2})-(\d{2})", a2)
        return m.group(0) if m else ""
    if t == "email":
        m = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", a)
        return m.group(0) if m else ""
    if t == "file":
        # Expected: "filename.ext — short description"
        if "—" in a or " - " in a:
            return a
        slug = re.sub(r"[^a-z0-9]+", "-", (a[:40].lower() or "document")).strip("-") or "document"
        if "." not in slug:
            slug = f"{slug}.pdf"
        return f"{slug} — fichier à compléter"
    return a

# ────────────────────────────────────────────────────────────────────────────────
# Similarity & caching
# ────────────────────────────────────────────────────────────────────────────────
def _shingles(text: str, n=6) -> set:
    toks = re.findall(r"\w+", (text or "").lower())
    return set(tuple(toks[i:i+n]) for i in range(max(0, len(toks)-n+1)))

def _jaccard(a: str, b: str) -> float:
    A, B = _shingles(a), _shingles(b)
    if not A or not B: return 0.0
    return len(A & B) / max(1, len(A | B))

def _dedupe_snippets(snips: List[Tuple[str,str]], threshold=0.8) -> List[Tuple[str,str]]:
    out = []
    for lbl, txt in snips:
        if all(_jaccard(txt, t2) < threshold for _, t2 in out):
            out.append((lbl, txt))
    return out

RECENT_CACHE: Dict[str, List[str]] = {}

def _too_similar(a: str, b: str, t=0.90) -> bool:
    return _jaccard(a, b) >= t

def _cache_key(facts: Dict[str,str], step: str) -> str:
    return f"{facts.get('startup','')}-{step}"

def _store_output(key: str, text: str, keep=6):
    RECENT_CACHE.setdefault(key, []).append(text)
    if len(RECENT_CACHE[key]) > keep:
        RECENT_CACHE[key] = RECENT_CACHE[key][-keep:]

# ────────────────────────────────────────────────────────────────────────────────
# LLM + Embeddings (Ollama)
# ────────────────────────────────────────────────────────────────────────────────
def _stable_jitter(seed_text: str, base=0.45, spread=0.15) -> float:
    r = int(hashlib.md5(seed_text.encode("utf-8")).hexdigest(), 16)
    return max(0.1, min(0.95, base + spread * (((r % 200) - 100) / 100.0)))

def llm_generate(prompt: str, sys: str = "", seed_hint: str = "", max_seconds: float = 8.0) -> str:
    """
    Prefer /api/chat; fallback to /api/generate. Tight, dynamic timeouts.
    Default model is tuned for speed with llama 3B.
    """
    try:
        url = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
        model = os.getenv("LLM_MODEL", "llama3.1:3b-instruct")   # ← default for speed
        temp = _stable_jitter(seed_hint or (prompt[:128] + sys[:128]))

        options = {
            "temperature": temp,
            "top_p": 0.9,
            "top_k": 40,
            "num_ctx": 4096,         # 3B: keep modest context for speed
            "repeat_penalty": 1.25,
            "repeat_last_n": 128,
        }

        # 1) try /api/chat
        try:
            payload_chat = {
                "model": model,
                "messages": (
                    [{"role": "system", "content": sys}] if sys else []
                ) + [
                    {"role": "user", "content": prompt}
                ],
                "options": options,
                "stream": False,
            }
            r = requests.post(f"{url}/api/chat", json=payload_chat, timeout=max(3.0, min(12.0, max_seconds)))
            if r.ok:
                data = r.json() or {}
                msg = (data.get("message") or {}).get("content", "")
                if msg:
                    return msg.strip()
        except Exception:
            pass  # fallback

        # 2) fallback /api/generate
        payload_gen = {
            "model": model,
            "prompt": prompt,
            "system": sys or "",
            "options": options,
            "stream": False,
        }
        r2 = requests.post(f"{url}/api/generate", json=payload_gen, timeout=max(3.0, min(12.0, max_seconds)))
        r2.raise_for_status()
        return (r2.json().get("response") or "").strip()

    except Exception:
        return ""

def embed_ollama(texts: List[str], timeout: float = 6.0) -> List[List[float]]:
    """
    Embeddings via Ollama /api/embeddings (default: nomic-embed-text or env EMBED_MODEL).
    """
    url = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
    model = os.getenv("EMBED_MODEL", "nomic-embed-text")
    try:
        payload = {"model": model, "prompt": texts if isinstance(texts, list) else [texts]}
        r = requests.post(f"{url}/api/embeddings", json=payload, timeout=max(2.0, timeout))
        r.raise_for_status()
        j = r.json()
        if "embeddings" in j:
            return j["embeddings"]
        if "embedding" in j:
            emb = j["embedding"]
            return [emb] if isinstance(emb, list) else []
        return [[] for _ in texts]
    except Exception:
        return [[] for _ in texts]

def _cos(a: List[float], b: List[float]) -> float:
    num = sum(x*y for x, y in zip(a, b))
    da = math.sqrt(sum(x*x for x in a)) or 1e-9
    db = math.sqrt(sum(x*x for x in b)) or 1e-9
    return num / (da * db)

# Query expansion (light)
EXPAND = {
    "proposition de valeur": ["UVP", "value proposition", "promesse", "différenciation"],
    "canaux": ["acquisition", "distribution", "marketing channels"],
    "revenus": ["business model", "pricing", "revenue", "monétisation"],
    "personas": ["clients cibles", "segments", "cible", "ICP"],
}

def expand_query(q: str) -> str:
    ql = q.lower()
    extra = []
    for k, syns in EXPAND.items():
        if k in ql:
            extra += syns
    return q + " " + " ".join(sorted(set(extra)))

def build_rag_context(query: str, corpus: List[Tuple[str,str]], k: int = 8, min_chars: int = 30, max_seconds: float = 6.0) -> List[Tuple[str,str]]:
    """
    Top-k closest corpus entries using embeddings. Time-bounded.
    """
    docs = [(lbl, txt) for (lbl, txt) in corpus if len((txt or "").strip()) >= min_chars]
    if not docs:
        return []
    # One embedding call for query, one for doc batch
    q_vecs = embed_ollama([expand_query(query)], timeout=max(2.0, max_seconds * 0.35))
    if not q_vecs or not q_vecs[0]:
        return []
    query_vec = q_vecs[0]
    doc_texts = [d[1] for d in docs]
    doc_vecs = embed_ollama(doc_texts, timeout=max(2.0, max_seconds * 0.65))
    scored = [(lbl, txt, _cos(query_vec, vec)) for (lbl, txt), vec in zip(docs, doc_vecs) if vec]
    scored.sort(key=lambda t: t[2], reverse=True)
    top = [(lbl, txt) for (lbl, txt, _) in scored[:k*2]]
    return _dedupe_snippets(top)[:k]

# ────────────────────────────────────────────────────────────────────────────────
# Rule-based fallbacks (fast paths)
# ────────────────────────────────────────────────────────────────────────────────
def uvp_fallback(qlabel: str, facts: Dict[str, str]) -> str:
    """Build a compact UVP from available facts."""
    if facts.get("value_prop"):
        return smart_clean(facts["value_prop"])
    bits = []
    if facts.get("problem"):
        bits.append(f"Problème: {facts['problem']}")
    if facts.get("solution"):
        bits.append(f"Solution: {facts['solution']}")
    if facts.get("personas"):
        bits.append(f"Cible: {facts['personas']}")
    if facts.get("advantage"):
        bits.append(f"Différenciation: {facts['advantage']}")
    return " — ".join(bits) if bits else ""

def answer_for_question(qlabel: str, facts: Dict[str, str], prompt: str) -> str:
    q = (qlabel or "").lower()
    st = style_from_prompt(prompt)

    def bullets(lines: List[str]) -> str:
        return "\n- " + "\n- ".join([x for x in lines if x])

    if any(x in q for x in ["clients cibles", "client cible", "cible", "persona"]):
        txt = facts.get("personas") or facts.get("customer_segments")
        return bullets(split_list(txt)) if (txt and st == "bullets") else (smart_clean(txt) or "Décrivez 1–2 personas (profil, taille, zone) et l’usage principal.")

    if any(x in q for x in ["source principale de revenus", "revenu principal", "principal revenue"]):
        return facts.get("revenue_primary") or "Indiquez la ligne de revenus dominante (ex : abonnement SaaS, licence, commissions)."

    if any(x in q for x in ["autres sources", "sources secondaires", "revenus secondaires"]):
        return facts.get("revenue_secondary") or "Listez brièvement 1–3 sources secondaires pertinentes."

    if any(x in q for x in ["proposition de valeur", "value prop", "uvp"]):
        return uvp_fallback(qlabel, facts) or ""

    if any(x in q for x in ["business model", "modèle économique"]):
        return facts.get("business_model") or "Expliquez comment l’entreprise gagne de l’argent (mécanique, prix, récurrence)."

    if any(x in q for x in ["kpi", "indicateur", "mesure du succès"]):
        return facts.get("kpis") or "Citez 2–3 KPIs suivis (ex : MRR, CAC, LTV, churn)."

    if "problème" in q:
        return facts.get("problem") or "Résumez le problème et son impact (€, temps, risque)."

    if any(x in q for x in ["solution", "comment résolvez", "comment réglez"]):
        return facts.get("solution") or "Expliquez la solution et le gain mesuré (temps, coût, qualité)."

    if any(x in q for x in ["avantage concurrentiel", "différenciation"]):
        return facts.get("advantage") or "Indiquez votre différenciation (technologie, data, go-to-market, coûts)."

    if any(x in q for x in ["canal", "distribution", "acquisition"]):
        return facts.get("channels") or "Précisez 1–3 canaux d’acquisition/diffusion."

    bits = []
    if facts.get("value_prop"): bits.append(facts["value_prop"])
    if facts.get("business_model"): bits.append(f"Modèle: {facts['business_model']}")
    if facts.get("revenue_primary"): bits.append(f"Revenu principal: {facts['revenue_primary']}")
    if facts.get("kpis"): bits.append(f"KPIs: {facts['kpis']}")
    return " ".join(bits) if bits and st != "bullets" else (("\n- " + "\n- ".join(bits)) if bits else "")

STEP_HINTS = {
    "Personas & Segmentation": [
        "Indiquer 1 persona prioritaire (rôle, taille orga, zone).",
        "Besoin principal et moment d'achat.",
        "1 chiffre d'opportunité (marché/local).",
    ],
    "TAM / SAM / SOM": [
        "TAM global (~), SAM marché cible (~), SOM 2–3 ans (~).",
        "Citer 1–2 sources probables (sans URL).",
        "Décrire méthode rapide (pénétration x ARPA).",
    ],
    "Stratégie Marketing": [
        "Top 2 canaux adaptés au segment.",
        "Objectif CAC/LTV cible.",
        "Expériences à lancer T+30j.",
    ],
}

def step_hint_block(step_name: str) -> str:
    hints = STEP_HINTS.get(step_name, [])
    return "\n- " + "\n- ".join(hints) if hints else ""

# ────────────────────────────────────────────────────────────────────────────────
# Scoring v2 (unchanged logic; fast)
# ────────────────────────────────────────────────────────────────────────────────
def _score_item(step_name: str, it: Item) -> Dict:
    txt = (it.answer or "").strip()
    lower = txt.lower()
    base = 60
    missing = []

    if it.type in ("text", "textarea", "choice"):
        wc = len(re.findall(r"\w+", txt))
        has_num = bool(re.search(r"\d", txt))
        has_pct = "%" in txt
        has_eur = "€" in txt

        if wc < 18: base = 35
        if wc >= 30: base += 10
        if wc >= 50: base += 5
        if has_num: base += 5
        if has_pct: base += 3
        if has_eur: base += 3

        q = (it.label or "").lower()
        if "qui est" in q or ("impact" in q and "qui" in q):
            if not re.search(r"(client|utilisateur|pme|entreprise|segment)", lower):
                missing.append("préciser le segment (profil/taille/zone)")
            if not has_num:
                missing.append("indiquer un ordre de grandeur (nombre, %)")
        elif "effets" in q or ("impact" in q and "négatif" in q):
            if not (has_num or has_pct or "€" in txt):
                missing.append("ajouter au moins un indicateur chifré (€, %, volume)")

    elif it.type == "file":
        base = 85 if it.has_file else 0
        if not it.has_file:
            missing.append("joindre un document")

    elif it.type == "email":
        base = 85 if re.search(r"[\w\.-]+@[\w\.-]+\.\w+", txt) else 20
        if base < 85: missing.append("fournir une adresse valide")

    elif it.type == "date":
        base = 80 if re.search(r"\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}", txt) else 25
        if base < 80: missing.append("utiliser un format date clair (JJ/MM/AAAA)")

    score = max(0, min(100, base))
    status = "validée" if score >= 70 else "à retravailler"

    if score >= 85:
        short = "Excellent : réponse précise, claire et exploitable."
    elif score >= 70:
        short = "Bon niveau. Rendez 1–2 éléments plus concrets (chiffres, période, source)."
    elif score >= 50:
        short = "Base correcte mais incomplète. Ajoutez des chiffres et un ancrage temporel."
    else:
        short = "Insuffisant : structurez, quantifiez et citez au moins une référence."

    suggestion_bullets = [f"• {m}" for m in missing[:3]]

    suggested = ""
    q = (it.label or "").lower()
    if "qui est" in q:
        suggested = "Ex : « PME B2B (50–200 salariés) en IDF ; décideurs Ops/IT ; ~30 comptes au départ. »"
    elif "effets" in q:
        suggested = "Ex : « +25 % de temps perdu/mois (~12 k€), sat. client 3,1/5, 2 incidents/trim. »"
    elif "depuis" in q:
        suggested = "Ex : « Depuis 2023, aggravation T2 2024 (post-refonte SI). »"

    return {
        "question_id": it.question_id,
        "score": score,
        "status": status,
        "feedback": short,
        "points": int(it.points or 10),
        "issues": [],
        "suggestion_bullets": suggestion_bullets,
        "suggested_answer": suggested,
    }

# ────────────────────────────────────────────────────────────────────────────────
# Endpoints
# ────────────────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "version": "2.7.0"}

@app.post("/api/generate")
def generate(req: GenerateRequest, request: Request):
    """
    Pipeline (time-aware):
      0) Parse context -> facts + corpus
      1) RAG (skip if <~8s remaining)
      2) LLM (3B) with dynamic timeout (≤ ~8s)
      3) Rule fallbacks if LLM too short
      4) Cleanup + type coercion + anti-duplication
    """
    # Read deadline header (epoch ms)
    header_deadline = request.headers.get("x-deadline")
    try:
        deadline_ms = int(header_deadline) if header_deadline else None
    except Exception:
        deadline_ms = None

    def time_left() -> float:
        return _seconds_left(deadline_ms)

    # 0) Parse
    facts, corpus = parse_context_lines(req.context or "")

    # 1) RAG (only if enough time)
    rag_snippets: List[Tuple[str,str]] = []
    if time_left() > 8.0 and corpus:
        user_query = f"{req.question_label} || {req.prompt or ''} || {req.step_name}"
        rag_snippets = build_rag_context(user_query, corpus, k=8, min_chars=30, max_seconds=min(6.0, time_left() - 2.0))

    # 2) LLM
    style = style_from_prompt(req.prompt or "")
    qtype = req.question_type or "text"
    sys = coach_system_prompt()
    llm_prompt = build_llm_prompt(req.step_name, req.question_label, qtype, style, facts, rag_snippets)
    seed_hint = f"{facts.get('startup','')}-{req.step_name}-{req.question_label}"

    answer = ""
    if time_left() > 4.5:
        answer = smart_clean(
            llm_generate(
                llm_prompt, sys, seed_hint=seed_hint,
                max_seconds=min(8.0, time_left() - 1.0)
            )
        )

    # 3) Fallbacks
    if len(answer) < 40:
        draft = smart_clean(answer_for_question(req.question_label, facts, req.prompt or ""))
        if len(draft) > len(answer):
            answer = draft
    if len(answer) < 40 and ("proposition de valeur" in (req.question_label or "").lower() or "uvp" in (req.question_label or "").lower()):
        answer = uvp_fallback(req.question_label, facts) or answer

    # 4) Finalize
    final = smart_clean(answer)
    final = vary_opening(final, facts, req.question_label)
    final = coerce_by_type(final, qtype)
    if not final:
        final = "Réponse de travail : précisez la cible, le problème et 1 chiffre clé."

    if len(final) < 40:
        final += step_hint_block(req.step_name)

    # Anti-duplication (only if time allows)
    ck = _cache_key(facts, req.step_name)
    prevs = RECENT_CACHE.get(ck, [])
    if any(_too_similar(final, p) for p in prevs) and time_left() > 4.0:
        llm_prompt2 = llm_prompt + "\n\nConsigne additionnelle: change d'angle et de lexique; évite les mêmes tournures; introduis un chiffre différent si plausible."
        answer2 = smart_clean(
            llm_generate(
                llm_prompt2, sys, seed_hint=f"{ck}-{req.question_label}-retry",
                max_seconds=min(6.0, time_left() - 1.0)
            )
        )
        final2 = coerce_by_type(vary_opening(smart_clean(answer2), facts, req.question_label), qtype)
        if len(final2) > 0:
            final = final2

    _store_output(ck, final)
    return {"answer": final}

@app.post("/api/score_v2")
def score_structured(payload: StructuredRequest):
    if not payload.items:
        return {
            "items": [], "global_score": 0,
            "status": "à retravailler",
            "feedback": "Aucune réponse reçue."
        }
    details = [_score_item(payload.step_name, it) for it in payload.items]
    total_points = sum(max(1, d["points"]) for d in details)
    weighted = sum(d["score"] * max(1, d["points"]) for d in details) / total_points if total_points else 0
    global_score = round(weighted)
    status = "validée" if global_score >= 70 else "à retravailler"
    low = [d for d in details if d["score"] < 70]
    if not low:
        feedback = f"Très bon niveau global ({global_score}%)."
    elif len(low) <= 2:
        feedback = f"Niveau satisfaisant ({global_score}%). Quelques réponses à préciser (voir plan d’action)."
    else:
        feedback = f"Score global {global_score}%. Plusieurs réponses manquent de précision chiffrée et temporelle."
    return {"items": details, "global_score": global_score, "status": status, "feedback": feedback}

# ────────────────────────────────────────────────────────────────────────────────
# Local dev
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
   
    uvicorn.run(app, host="0.0.0.0", port=5005)
