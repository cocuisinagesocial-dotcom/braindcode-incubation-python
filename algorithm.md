# 🧠 Algorithme détaillé - Startup Incubation API

Ce document explique en détail le fonctionnement des algorithmes utilisés dans l'API.

## Table des matières

- [Vue d'ensemble](#vue-densemble)
- [Pipeline de génération](#pipeline-de-génération)
- [Pipeline de scoring](#pipeline-de-scoring)
- [Composants clés](#composants-clés)
- [Optimisations](#optimisations)

## Vue d'ensemble

L'API utilise une combinaison de techniques d'IA :

1. **RAG** (Retrieval-Augmented Generation) : Pour enrichir le contexte
2. **LLM** (Large Language Model) : Pour générer du texte
3. **Règles métier** : Pour valider et scorer
4. **Cache intelligent** : Pour éviter les duplications

## Pipeline de génération

Le pipeline `/api/generate` suit 5 étapes principales :

### Étape 1 : Parse du contexte

**Objectif :** Extraire les informations structurées du contexte Laravel.

**Input :**
```
Startup: MonStartup
Sector: SaaS B2B
Problem: Les PME perdent du temps avec la gestion manuelle
PrevAnswer: Qui est impacté ? => PME B2B de 50-200 employés
```

**Process :**

```python
def parse_context_lines(context: str):
    """
    1. Extrait les faits (Startup, Sector, Problem, etc.)
    2. Construit le corpus pour le RAG (toutes les lignes + PrevAnswer)
    """
    
    # Parsing des faits
    facts = {}
    for line in context.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            facts[normalize(key)] = value.strip()
    
    # Construction du corpus pour RAG
    corpus = []
    for line in context.splitlines():
        if "PrevAnswer:" in line:
            # Format: PrevAnswer: Question => Réponse
            question, answer = parse_prev_answer(line)
            corpus.append((question, answer))
        elif ":" in line:
            key, value = line.split(":", 1)
            corpus.append((key, value))
    
    return facts, corpus
```

**Output :**
- `facts` : `{"startup": "MonStartup", "sector": "SaaS B2B", ...}`
- `corpus` : `[("Problem", "Les PME perdent..."), ("Qui est impacté ?", "PME B2B...")]`

### Étape 2 : RAG (Retrieval-Augmented Generation)

**Objectif :** Trouver les informations les plus pertinentes dans le corpus pour répondre à la question.

**Process :**

#### 2.1 Expansion de la requête

```python
def expand_query(query: str) -> str:
    """
    Enrichit la requête avec des synonymes
    """
    # Mapping de concepts
    expansions = {
        "proposition de valeur": ["UVP", "value prop", "promesse"],
        "canaux": ["acquisition", "distribution", "marketing"],
        "revenus": ["business model", "pricing", "monétisation"]
    }
    
    expanded = query
    for concept, synonyms in expansions.items():
        if concept in query.lower():
            expanded += " " + " ".join(synonyms)
    
    return expanded
```

#### 2.2 Génération des embeddings

```python
def build_rag_context(query: str, corpus: List[Tuple[str, str]]):
    """
    1. Filtrer les documents trop courts (< 30 chars)
    2. Générer l'embedding de la requête
    3. Générer les embeddings des documents
    4. Calculer la similarité cosinus
    5. Trier et retourner les top-k
    """
    
    # 1. Filtrer
    docs = [(label, text) for label, text in corpus if len(text) >= 30]
    
    # 2. Embedding de la requête
    query_expanded = expand_query(query)
    query_vec = embed_ollama([query_expanded])[0]
    
    # 3. Embeddings des documents
    doc_texts = [text for _, text in docs]
    doc_vecs = embed_ollama(doc_texts)
    
    # 4. Calcul des scores
    scores = []
    for (label, text), vec in zip(docs, doc_vecs):
        similarity = cosine_similarity(query_vec, vec)
        scores.append((label, text, similarity))
    
    # 5. Tri et sélection
    scores.sort(key=lambda x: x[2], reverse=True)
    top_k = scores[:8]  # Top 8
    
    # 6. Déduplication
    deduplicated = remove_duplicates(top_k)
    
    return deduplicated
```

**Similarité cosinus :**

```
cos(A, B) = (A · B) / (||A|| × ||B||)

Où :
- A · B = somme(Ai × Bi)  [produit scalaire]
- ||A|| = sqrt(somme(Ai²))  [norme]
```

**Exemple :**

Requête : "Qui est impacté par le problème ?"

Corpus :
1. "Problem: Les PME perdent du temps" → score: 0.72
2. "Qui est impacté ? => PME B2B de 50-200 employés" → score: 0.91 ✓
3. "Solution: Automatiser la gestion" → score: 0.45
4. "Personas: Managers PME" → score: 0.88 ✓

Résultat : Snippets 2 et 4 sont les plus pertinents.

#### 2.3 Déduplication

```python
def dedupe_snippets(snippets: List[Tuple[str, str]]):
    """
    Supprime les snippets trop similaires entre eux
    """
    result = []
    for label, text in snippets:
        # Vérifier si similaire à un snippet déjà ajouté
        is_duplicate = False
        for _, existing_text in result:
            if jaccard_similarity(text, existing_text) > 0.8:
                is_duplicate = True
                break
        
        if not is_duplicate:
            result.append((label, text))
    
    return result
```

**Similarité de Jaccard (shingles) :**

```
J(A, B) = |A ∩ B| / |A ∪ B|

Où A et B sont des ensembles de n-grams (ex: 6 mots consécutifs)
```

### Étape 3 : Génération LLM

**Objectif :** Utiliser le LLM pour générer une réponse personnalisée.

#### 3.1 Construction du prompt

```python
def build_llm_prompt(step, question, qtype, style, facts, rag_snippets):
    """
    Construit un prompt complet et structuré
    """
    
    prompt = f"""
Étape: {step}
Question: {question}
Type: {qtype}
Style: {style}

=== Contexte pertinent ===
{format_snippets(rag_snippets)}

=== Faits bruts ===
- Startup: {facts['startup']}
- Sector: {facts['sector']}
- Problem: {facts['problem']}
...

=== Fragments à NE PAS copier ===
{list_forbidden_fragments(rag_snippets)}

=== CONTRAT DE SORTIE ===
1) Adapter à la startup et au secteur
2) Respecter le type de réponse
3) Pas de placeholders [xxx]
4) Si hypothèses → terminer par "Assumptions & Next step: ..."

Consigne: rédige la réponse maintenant.
"""
    
    return prompt
```

#### 3.2 Variation de style

Pour éviter la répétition, le style varie selon une seed déterministe :

```python
def _variation_seed(text: str) -> int:
    """Génère une seed stable à partir du texte"""
    return int(hashlib.md5(text.encode()).hexdigest(), 16) % 10000

# Variantes de style
STYLE_VARIANTS = {
    "paragraph": [
        "paragraphe compact, ton pragmatique",
        "paragraphe orienté résultats",
        "paragraphe synthétique, ton business"
    ],
    "bullets": [
        "puces brèves, verbe d'action",
        "puces avec chiffres",
        "puces MECE, 4-6 items"
    ]
}

# Sélection
seed = _variation_seed(startup_name + question)
rnd = random.Random(seed)
style_note = rnd.choice(STYLE_VARIANTS[style])
```

#### 3.3 Température dynamique

```python
def _stable_jitter(seed_text: str):
    """
    Génère une température entre 0.3 et 0.6
    basée sur le hash du texte
    """
    hash_val = int(hashlib.md5(seed_text.encode()).hexdigest(), 16)
    normalized = (hash_val % 200 - 100) / 100.0  # -1.0 à 1.0
    
    base = 0.45
    spread = 0.15
    temp = base + spread * normalized
    
    return max(0.1, min(0.95, temp))
```

Cette approche garantit :
- **Cohérence** : Même input → même température
- **Variation** : Différents inputs → différentes températures
- **Contrôle** : Toujours dans une plage raisonnable

#### 3.4 Appel à Ollama

```python
def llm_generate(prompt: str, system: str, max_seconds: float):
    """
    1. Essayer /api/chat (format conversationnel)
    2. Si échec, fallback sur /api/generate
    3. Timeout dynamique
    """
    
    # Calculer la température
    temp = _stable_jitter(prompt[:128] + system[:128])
    
    # Options Ollama
    options = {
        "temperature": temp,
        "top_p": 0.9,
        "top_k": 40,
        "num_ctx": 4096,
        "repeat_penalty": 1.25,
        "repeat_last_n": 128
    }
    
    # Tentative 1 : /api/chat
    try:
        payload = {
            "model": "llama3.1:3b-instruct",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            "options": options,
            "stream": False
        }
        response = requests.post(
            "http://localhost:11434/api/chat",
            json=payload,
            timeout=max_seconds
        )
        if response.ok:
            return response.json()["message"]["content"]
    except:
        pass
    
    # Tentative 2 : /api/generate
    payload = {
        "model": "llama3.1:3b-instruct",
        "prompt": prompt,
        "system": system,
        "options": options,
        "stream": False
    }
    response = requests.post(
        "http://localhost:11434/api/generate",
        json=payload,
        timeout=max_seconds
    )
    return response.json()["response"]
```

### Étape 4 : Fallbacks basés sur règles

**Objectif :** Si le LLM ne répond pas ou donne une réponse trop courte, utiliser des règles.

```python
def answer_for_question(question: str, facts: Dict):
    """
    Règles métier pour réponses rapides
    """
    q = question.lower()
    
    # Règle : Personas / Cible
    if "qui est impacté" in q or "client cible" in q:
        if facts.get("personas"):
            return facts["personas"]
        if facts.get("customer_segments"):
            return facts["customer_segments"]
        return "Décrivez 1-2 personas (profil, taille, zone)"
    
    # Règle : Proposition de valeur
    if "proposition de valeur" in q or "uvp" in q:
        return build_uvp_from_facts(facts)
    
    # Règle : Business model
    if "business model" in q:
        return facts.get("business_model") or \
               "Expliquez comment l'entreprise gagne de l'argent"
    
    # ... autres règles
    
    return ""
```

**Exemple de construction UVP :**

```python
def build_uvp_from_facts(facts: Dict) -> str:
    """
    Construit une UVP à partir des faits disponibles
    """
    if facts.get("value_prop"):
        return facts["value_prop"]
    
    parts = []
    if facts.get("problem"):
        parts.append(f"Problème: {facts['problem']}")
    if facts.get("solution"):
        parts.append(f"Solution: {facts['solution']}")
    if facts.get("personas"):
        parts.append(f"Cible: {facts['personas']}")
    if facts.get("advantage"):
        parts.append(f"Différenciation: {facts['advantage']}")
    
    return " — ".join(parts)
```

### Étape 5 : Post-traitement

#### 5.1 Nettoyage

```python
def smart_clean(text: str) -> str:
    """
    1. Supprimer les phrases avec placeholders [xxx]
    2. Normaliser les espaces
    3. Retourner le résultat ou l'original
    """
    
    placeholders = [r"\[[^\]]+\]", r"\bX%\b", r"\bSegment X\b"]
    
    # Supprimer les phrases problématiques
    for pattern in placeholders:
        if re.search(pattern, text):
            sentences = re.split(r"(?<=[\.\!\?])\s+", text)
            sentences = [s for s in sentences 
                        if not re.search(pattern, s)]
            text = " ".join(sentences)
    
    # Normaliser les espaces
    text = re.sub(r"\s{2,}", " ", text).strip()
    
    return text if text else original_text
```

#### 5.2 Variation d'ouverture

```python
def vary_opening(text: str, facts: Dict, question: str):
    """
    Ajoute un opener varié au début de la réponse
    """
    
    openers = [
        "Concrètement, ",
        "En pratique, ",
        "Pour cette startup, ",
        "Dans ce contexte, ",
        ""
    ]
    
    # Sélection déterministe
    seed = facts["startup"] + question
    rnd = random.Random(_variation_seed(seed))
    opener = rnd.choice(openers)
    
    # Application
    if opener and not text.startswith(("−", "-")):
        return opener + text[0].lower() + text[1:]
    
    return text
```

#### 5.3 Coercition de type

```python
def coerce_by_type(answer: str, qtype: str) -> str:
    """
    Adapte la réponse au type attendu
    """
    
    if qtype == "number":
        # Extraire le premier nombre
        match = re.findall(r"-?\d+(?:[.,]\d+)?", answer)
        return match[0].replace(",", ".") if match else ""
    
    if qtype == "date":
        # Convertir en format ISO
        match = re.search(r"(\d{2})/(\d{2})/(\d{4})", answer)
        if match:
            return f"{match.group(3)}-{match.group(2)}-{match.group(1)}"
        return ""
    
    if qtype == "email":
        # Extraire l'email
        match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", answer)
        return match.group(0) if match else ""
    
    if qtype == "file":
        # Format: filename.ext — description
        if "—" not in answer:
            slug = re.sub(r"[^a-z0-9]+", "-", answer[:40].lower())
            return f"{slug}.pdf — fichier à compléter"
        return answer
    
    return answer
```

#### 5.4 Anti-duplication

**Problème :** Le LLM peut générer la même réponse pour différentes questions.

**Solution :** Cache + détection de similarité

```python
# Cache global
RECENT_CACHE = {}  # {startup-step: [réponses récentes]}

def check_and_regenerate_if_duplicate(answer, facts, step, question):
    """
    1. Calculer la clé de cache
    2. Comparer avec les réponses récentes
    3. Si trop similaire → régénérer
    """
    
    cache_key = f"{facts['startup']}-{step}"
    previous_answers = RECENT_CACHE.get(cache_key, [])
    
    # Vérifier la similarité
    for prev in previous_answers:
        if jaccard_similarity(answer, prev) > 0.90:
            # Trop similaire ! Régénérer
            new_prompt = original_prompt + """
            
Consigne additionnelle: 
Change d'angle et de lexique. Évite les mêmes tournures.
Introduis un chiffre différent si plausible.
"""
            answer = llm_generate(new_prompt, ...)
            break
    
    # Stocker dans le cache
    RECENT_CACHE[cache_key].append(answer)
    if len(RECENT_CACHE[cache_key]) > 6:  # Garder seulement les 6 dernières
        RECENT_CACHE[cache_key] = RECENT_CACHE[cache_key][-6:]
    
    return answer
```

**Similarité de Jaccard avec shingles :**

```python
def jaccard_similarity(text1: str, text2: str) -> float:
    """
    Calcule la similarité entre deux textes
    en utilisant des shingles (n-grams de mots)
    """
    
    # Créer des shingles (6-grams de mots)
    def shingles(text, n=6):
        words = re.findall(r"\w+", text.lower())
        return set(tuple(words[i:i+n]) 
                  for i in range(len(words) - n + 1))
    
    A = shingles(text1)
    B = shingles(text2)
    
    if not A or not B:
        return 0.0
    
    intersection = len(A & B)
    union = len(A | B)
    
    return intersection / union
```

**Exemple :**

```
Text1: "Les PME du secteur retail rencontrent des difficultés"
Text2: "Les PME du secteur retail rencontrent des problèmes"

Shingles (3-grams) de Text1:
- ("les", "pme", "du")
- ("pme", "du", "secteur")
- ("du", "secteur", "retail")
- ("secteur", "retail", "rencontrent")
- ("retail", "rencontrent", "des")
- ("rencontrent", "des", "difficultés")

Shingles de Text2:
- ("les", "pme", "du")
- ("pme", "du", "secteur")
- ("du", "secteur", "retail")
- ("secteur", "retail", "rencontrent")
- ("retail", "rencontrent", "des")
- ("rencontrent", "des", "problèmes")

Intersection: 5 shingles communs
Union: 7 shingles au total
Similarité: 5/7 = 0.71 (71%)
```

## Pipeline de scoring

Le pipeline `/api/score_v2` évalue la qualité des réponses.

### Étape 1 : Analyse du contenu

```python
def _score_item(step: str, item: Item) -> Dict:
    """
    Score un item selon son type et son contenu
    """
    
    text = item.answer.strip()
    base_score = 60  # Score de départ
    missing = []  # Éléments manquants
    
    # 1. Analyse de base
    word_count = len(re.findall(r"\w+", text))
    has_numbers = bool(re.search(r"\d", text))
    has_percentage = "%" in text
    has_currency = "€" in text
    
    # 2. Ajustements selon le contenu
    if word_count < 18:
        base_score = 35  # Trop court
    if word_count >= 30:
        base_score += 10
    if word_count >= 50:
        base_score += 5
    if has_numbers:
        base_score += 5
    if has_percentage:
        base_score += 3
    if has_currency:
        base_score += 3
    
    # 3. Vérifications spécifiques
    question_lower = item.label.lower()
    
    if "qui est impacté" in question_lower:
        # Doit mentionner un segment
        if not re.search(r"(client|pme|entreprise|segment)", text.lower()):
            missing.append("préciser le segment")
            base_score -= 10
        
        # Doit avoir des chiffres
        if not has_numbers:
            missing.append("indiquer un ordre de grandeur")
            base_score -= 10
    
    # 4. Score final
    score = max(0, min(100, base_score))
    
    return {
        "score": score,
        "missing": missing,
        ...
    }
```

### Étape 2 : Génération du feedback

```python
def generate_feedback(score: int, missing: List[str]) -> str:
    """
    Génère un feedback personnalisé
    """
    
    if score >= 85:
        return "Excellent : réponse précise, claire et exploitable."
    
    if score >= 70:
        feedback = "Bon niveau."
        if missing:
            feedback += " Rendez plus concrets : " + ", ".join(missing[:2])
        return feedback
    
    if score >= 50:
        return "Base correcte mais incomplète. Ajoutez des chiffres et un ancrage temporel."
    
    return "Insuffisant : structurez, quantifiez et citez au moins une référence."
```

### Étape 3 : Score global pondéré

```python
def calculate_global_score(items: List[Dict]) -> int:
    """
    Calcule le score global en pondérant par les points
    """
    
    total_points = sum(max(1, item["points"]) for item in items)
    
    if total_points == 0:
        return 0
    
    weighted_sum = sum(
        item["score"] * max(1, item["points"])
        for item in items
    )
    
    return round(weighted_sum / total_points)
```

**Exemple :**

```
Item 1: score=80, points=10 → contribution = 800
Item 2: score=60, points=5  → contribution = 300
Item 3: score=90, points=15 → contribution = 1350

Total points = 30
Weighted sum = 2450
Global score = 2450 / 30 = 81.67 ≈ 82
```

## Composants clés

### 1. Gestion du temps (deadline-aware)

```python
def _seconds_left(deadline_ms: Optional[int]) -> float:
    """
    Calcule le temps restant avant la deadline
    
    Args:
        deadline_ms: Deadline en millisecondes (epoch)
    
    Returns:
        Secondes restantes (999.0 si pas de deadline)
    """
    if not deadline_ms:
        return 999.0
    
    now_ms = int(time.time() * 1000)
    return max(0.0, (deadline_ms - now_ms) / 1000.0)

# Utilisation dans le pipeline
if time_left() > 8.0:
    # Assez de temps pour le RAG
    rag_snippets = build_rag_context(...)

if time_left() > 4.5:
    # Assez de temps pour le LLM
    answer = llm_generate(..., max_seconds=time_left() - 1.0)
```

### 2. Configuration centralisée

```python
class Config:
    """Configuration via variables d'environnement"""
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
    LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:3b-instruct")
    EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
    MAX_LLM_TIMEOUT = float(os.getenv("MAX_LLM_TIMEOUT", "12.0"))
    RAG_TOP_K = int(os.getenv("RAG_TOP_K", "8"))
    # ...

config = Config()

# Utilisation
model = config.LLM_MODEL
timeout = config.MAX_LLM_TIMEOUT
```

### 3. Logging structuré

```python
import logging

logger = logging.getLogger(__name__)

# Dans le code
logger.info(f"Parsed context: {len(facts)} facts, {len(corpus)} items")
logger.info(f"RAG: {len(rag_snippets)} snippets retrieved")
logger.info(f"LLM answer length: {len(answer)}")
logger.warning("LLM answer too short, using fallback")
logger.error(f"Ollama connection failed: {e}")
```

## Optimisations

### 1. Embeddings en batch

Au lieu de :
```python
# ❌ Lent : 1 appel par document
for doc in docs:
    vec = embed_ollama([doc])
```

Faire :
```python
# ✅ Rapide : 1 seul appel
all_docs = [doc for _, doc in docs]
all_vecs = embed_ollama(all_docs)
```

### 2. Early returns

```python
# Si le corpus est vide, pas besoin de RAG
if not corpus:
    return []

# Si pas assez de temps, skip le RAG
if time_left() < 8.0:
    rag_snippets = []
else:
    rag_snippets = build_rag_context(...)
```

### 3. Cache intelligent

```python
# Évite de régénérer la même réponse
RECENT_CACHE = {}  # startup-step → [réponses]

# Stockage
def _store_output(key: str, text: str, keep=6):
    RECENT_CACHE.setdefault(key, []).append(text)
    if len(RECENT_CACHE[key]) > keep:
        RECENT_CACHE[key] = RECENT_CACHE[key][-keep:]

# Vérification
def _too_similar(a: str, b: str) -> bool:
    return jaccard_similarity(a, b) >= 0.90
```

### 4. Timeout adaptatif

```python
# Ajuste le timeout selon le temps restant
max_seconds = min(8.0, time_left() - 1.0)
answer = llm_generate(..., max_seconds=max_seconds)
```

### 5. Modèle quantifié

Utiliser un modèle quantifié (q4_K_M) pour la vitesse :
```
llama3.1:8b-instruct → 5-8s
llama3.1:8b-instruct-q4_K_M → 2-4s (même qualité ~95%)
```

## Complexité algorithmique

| Opération | Complexité | Notes |
|-----------|-----------|-------|
| Parse contexte | O(n) | n = lignes de contexte |
| Embeddings | O(m × d) | m = docs, d = dimension |
| Similarité cosinus | O(d) | d = dimension vecteur |
| Tri scores | O(m log m) | m = nombre de docs |
| Déduplication | O(k²) | k = top-k (petit) |
| LLM génération | O(tokens) | Dépend du modèle |
| Scoring | O(items) | Linéaire |

**Total pour /generate :**
- Meilleur cas : O(n) (pas de RAG, fallback direct)
- Cas moyen : O(n + m log m + tokens) ≈ 3-8 secondes
- Pire cas : O(n + m log m + tokens + retry) ≈ 8-15 secondes

## Diagrammes de flux

### Flux de génération

```
START
  ↓
Parse contexte
  ↓
Temps > 8s? ─NO→ Skip RAG
  ↓ YES
RAG (embeddings + search)
  ↓
Temps > 4.5s? ─NO→ Skip LLM
  ↓ YES
Générer avec LLM
  ↓
Réponse courte? ─YES→ Fallback règles
  ↓ NO
Nettoyer + Coercer type
  ↓
Similaire au cache? ─YES→ Régénérer
  ↓ NO
Stocker dans cache
  ↓
RETURN réponse
```

### Flux de scoring

```
START
  ↓
Pour chaque item:
  ↓
Analyser contenu (mots, nombres, symboles)
  ↓
Appliquer règles métier
  ↓
Calculer score (0-100)
  ↓
Identifier éléments manquants
  ↓
Générer feedback
  ↓
NEXT item
  ↓
Calculer score global pondéré
  ↓
Générer feedback global
  ↓
RETURN résultats
```

## Conclusion

L'algorithme combine :

1. **Intelligence symbolique** (règles métier)
2. **IA générative** (LLM)
3. **Recherche sémantique** (RAG)
4. **Optimisations** (cache, batch, timeouts)

Cette approche hybride garantit :
- ✅ Qualité (réponses personnalisées)
- ✅ Rapidité (3-8 secondes)
- ✅ Robustesse (fallbacks multiples)
- ✅ Cohérence (anti-duplication)

---

Pour toute question sur l'algorithme, consulter le code source avec les commentaires détaillés dans `app/main.py`.
