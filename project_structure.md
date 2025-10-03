# 📁 Structure du projet

Ce document décrit l'organisation complète du projet Startup Incubation API v3.0.

## 🗂️ Arborescence

```
startup-incubation-api/
│
├── 📁 app/                          # Code de l'application
│   ├── __init__.py                  # Package Python
│   └── main.py                      # ⭐ Application principale (1200 lignes)
│
├── 📁 docs/                         # Documentation (optionnel)
│   ├── images/
│   └── examples/
│
├── 📄 .dockerignore                 # Fichiers à ignorer par Docker
├── 📄 .env                          # Configuration (NE PAS COMMITER)
├── 📄 .env.example                  # Template de configuration
├── 📄 .gitignore                    # Fichiers à ignorer par Git
│
├── 📄 ALGORITHM.md                  # 📖 Explication détaillée des algorithmes
├── 📄 CHANGELOG.md                  # 📝 Historique des versions
├── 📄 docker-compose.yml            # 🐳 Stack Docker complète
├── 📄 Dockerfile                    # 🐳 Image Docker de l'API
├── 📄 Makefile                      # 🛠️ Commandes automatisées
├── 📄 MIGRATION.md                  # 🔄 Guide de migration v2.7 → v3.0
├── 📄 PROJECT_STRUCTURE.md          # 📁 Ce fichier
├── 📄 QUICKSTART.md                 # ⚡ Guide de démarrage rapide
├── 📄 README.md                     # 📘 Documentation principale
├── 📄 requirements.txt              # 📦 Dépendances Python
└── 📄 test_main.py                  # 🧪 Tests unitaires
```

## 📦 Fichiers principaux

### `app/main.py` - Cœur de l'application

**Lignes : ~1200**  
**Rôle : Application FastAPI complète**

#### Structure interne :

```python
# 1. Imports et configuration (lignes 1-50)
from fastapi import FastAPI, Request
from pydantic import BaseModel
import os, re, requests, math, time
# ...

# 2. Configuration (lignes 51-100)
class Config:
    OLLAMA_URL = os.getenv("OLLAMA_URL", ...)
    LLM_MODEL = os.getenv("LLM_MODEL", ...)
    # ...

# 3. Application FastAPI (lignes 101-120)
app = FastAPI(title="...", version="3.0.0")
app.add_middleware(CORSMiddleware, ...)

# 4. Schémas Pydantic (lignes 121-200)
class GenerateRequest(BaseModel): ...
class Item(BaseModel): ...
class StructuredRequest(BaseModel): ...

# 5. Utilitaires (lignes 201-400)
def clean(txt: str) -> str: ...
def extract_facts(context: str) -> Dict: ...
def parse_context_lines(context: str) -> Tuple: ...

# 6. Prompts et styles (lignes 401-550)
def coach_system_prompt() -> str: ...
def build_llm_prompt(...) -> str: ...
def vary_opening(...) -> str: ...

# 7. LLM et embeddings (lignes 551-700)
def llm_generate(...) -> str: ...
def embed_ollama(...) -> List[List[float]]: ...

# 8. RAG (lignes 701-850)
def build_rag_context(...) -> List[Tuple]: ...
def expand_query(q: str) -> str: ...

# 9. Similarité et cache (lignes 851-950)
def _jaccard(a: str, b: str) -> float: ...
def _dedupe_snippets(...) -> List: ...
RECENT_CACHE = {}

# 10. Fallbacks (lignes 951-1050)
def uvp_fallback(...) -> str: ...
def answer_for_question(...) -> str: ...

# 11. Scoring (lignes 1051-1150)
def _score_item(...) -> Dict: ...

# 12. Endpoints (lignes 1151-1200)
@app.get("/health")
@app.post("/api/generate")
@app.post("/api/score_v2")
@app.post("/api/score")  # Legacy
```

### `requirements.txt` - Dépendances

```txt
fastapi==0.109.0           # Framework web
uvicorn[standard]==0.27.0  # Serveur ASGI
pydantic==2.5.3            # Validation
python-dotenv==1.0.0       # Variables d'env
httpx==0.26.0              # Client HTTP async
requests==2.31.0           # Client HTTP sync
```

### `test_main.py` - Tests

**Lignes : ~800**  
**Couverture : ~80%**

```python
# Tests des endpoints
def test_health(): ...
def test_generate_minimal(): ...
def test_score_v2_single_item(): ...

# Tests des utilitaires
def test_clean_removes_placeholders(): ...
def test_extract_facts(): ...
def test_jaccard_similarity(): ...

# Tests d'intégration
@pytest.mark.integration
def test_full_generate_pipeline(): ...
```

## 📘 Documentation

### README.md (principal)

**Sections :**
1. Vue d'ensemble
2. Fonctionnalités
3. Architecture
4. Installation
5. Configuration
6. Utilisation
7. API Reference
8. Déploiement
9. Troubleshooting

### ALGORITHM.md (détaillé)

**Sections :**
1. Vue d'ensemble
2. Pipeline de génération (5 étapes)
3. Pipeline de scoring
4. Composants clés (RAG, LLM, Cache)
5. Optimisations
6. Diagrammes de flux

### QUICKSTART.md (rapide)

**Sections :**
1. Installation en 4 étapes
2. Test rapide
3. Alternative Docker
4. Cas d'usage
5. Troubleshooting

### MIGRATION.md (migration)

**Sections :**
1. Changements majeurs v2.7 → v3.0
2. Étapes de migration
3. Adaptation Laravel
4. Comparaison performances
5. Problèmes courants

## 🐳 Docker

### Dockerfile

**Multi-stage build :**
- Stage 1 (builder) : Compilation dépendances
- Stage 2 (runtime) : Image légère finale

**Features :**
- Utilisateur non-root
- Healthcheck intégré
- Optimisé pour prod

### docker-compose.yml

**Services :**
1. **ollama** : Serveur LLM
   - Port 11434
   - Volume persistant
   - Healthcheck

2. **api** : Application FastAPI
   - Port 5005
   - Variables d'env
   - Dépend d'ollama

**Networks :** `startup-network` (bridge)

## 🛠️ Makefile

**Catégories de commandes :**

### Installation
- `make install` : Installer dépendances
- `make env` : Créer .env
- `make models` : Télécharger modèles

### Exécution
- `make run` : Lancer l'API
- `make dev` : Mode développement

### Tests
- `make test` : Tests unitaires
- `make test-cov` : Avec couverture

### Qualité
- `make lint` : Vérifier code
- `make format` : Formater code
- `make check` : Tout vérifier

### Docker
- `make docker-up` : Lancer stack
- `make docker-down` : Arrêter stack
- `make docker-models` : Télécharger modèles

### Nettoyage
- `make clean` : Nettoyer temporaires
- `make clean-all` : Nettoyer tout

### Tout-en-un
- `make setup` : Installation complète
- `make all` : Vérifier tout

## 📊 Métriques du code

### Complexité

| Fichier | Lignes | Fonctions | Classes | Complexité |
|---------|--------|-----------|---------|------------|
| app/main.py | ~1200 | 50+ | 6 | Moyenne |
| test_main.py | ~800 | 60+ | 0 | Simple |

### Couverture des tests

| Module | Couverture |
|--------|-----------|
| Endpoints | 85% |
| Utilitaires texte | 95% |
| Parser contexte | 90% |
| RAG | 75% |
| Scoring | 90% |
| **Global** | **~80%** |

## 🔄 Workflow de développement

### 1. Setup initial

```bash
make setup        # Installation complète
make test         # Vérifier que tout marche
```

### 2. Développement

```bash
make dev          # Lancer en mode dev (auto-reload)
# Coder...
make format       # Formater le code
make check        # Vérifier qualité
make test         # Lancer les tests
```

### 3. Commit

```bash
git add .
git commit -m "feat: nouvelle fonctionnalité"
git push
```

### 4. Déploiement

```bash
make docker-build # Construire l'image
make docker-up    # Lancer en production
```

## 📐 Conventions de code

### Style Python

- **PEP 8** avec quelques adaptations :
  - Longueur de ligne : 120 caractères
  - Strings : doubles quotes `"` préférées
  - Imports : groupés et triés

### Nommage

- **Fonctions** : `snake_case`
- **Classes** : `PascalCase`
- **Constantes** : `UPPER_SNAKE_CASE`
- **Variables privées** : `_leading_underscore`

### Documentation

- **Docstrings** : Google style
- **Commentaires** : Au-dessus du code, pas en fin de ligne
- **Type hints** : Obligatoires pour les fonctions publiques

### Exemple

```python
def build_rag_context(
    query: str,
    corpus: List[Tuple[str, str]],
    k: int = 8,
    max_seconds: float = 6.0
) -> List[Tuple[str, str]]:
    """
    Construit le contexte RAG en trouvant les k snippets les plus pertinents.
    
    Args:
        query: Requête de recherche
        corpus: Corpus de documents (label, texte)
        k: Nombre de résultats
        max_seconds: Timeout maximum
        
    Returns:
        Liste des k meilleurs snippets (label, texte)
    """
    # Implementation...
```

## 🔐 Sécurité

### Fichiers sensibles

**Ne JAMAIS commiter :**
- `.env` (configuration)
- `*.key`, `*.pem` (clés)
- `secrets/` (secrets)
- Logs avec données sensibles

**Protection :**
- `.gitignore` configuré
- Validation Pydantic stricte
- Pas de logs de données sensibles

### Bonnes pratiques

1. **Variables d'env** pour config sensible
2. **Validation** des entrées (Pydantic)
3. **CORS** configuré proprement
4. **Timeouts** sur toutes les requêtes
5. **Healthcheck** pour monitoring

## 📈 Évolution future

### Roadmap v3.1

- [ ] Cache Redis pour embeddings
- [ ] Métriques Prometheus
- [ ] Support multi-langues
- [ ] API key authentication
- [ ] Rate limiting

### Roadmap v4.0

- [ ] Fine-tuning des modèles
- [ ] Support GPU distribué
- [ ] GraphQL API
- [ ] WebSocket pour streaming
- [ ] Dashboard de monitoring

## 🤝 Contribution

### Pour contribuer :

1. Fork le projet
2. Créer une branche : `git checkout -b feature/ma-fonctionnalite`
3. Coder + tests
4. Vérifier : `make all`
5. Commit : `git commit -m "feat: ma fonctionnalité"`
6. Push : `git push origin feature/ma-fonctionnalite`
7. Pull Request

### Checklist PR

- [ ] Code formaté (`make format`)
- [ ] Tests passent (`make test`)
- [ ] Couverture maintenue (>80%)
- [ ] Documentation mise à jour
- [ ] CHANGELOG.md mis à jour

---

**Maintenu par :** Équipe Startup Incubation  
**Version :** 3.0.0  
**Dernière mise à jour :** 2025-10-03
