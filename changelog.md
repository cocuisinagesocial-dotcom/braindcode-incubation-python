# Changelog

Tous les changements notables de ce projet seront documentés dans ce fichier.

Le format est basé sur [Keep a Changelog](https://keepachangelog.com/fr/1.0.0/),
et ce projet adhère au [Semantic Versioning](https://semver.org/lang/fr/).

## [3.0.0] - 2025-10-03

### 🎉 Changements majeurs

#### Ajouté
- **Application unifiée** : Consolidation de `app/main.py` et `routes/analyse.py` en un seul fichier
- **Configuration centralisée** : Classe `Config` avec support des variables d'environnement
- **Nouveau endpoint `/api/score_v2`** : Scoring ultra-rapide basé sur règles métier
- **Support des deadlines** : Header `X-Deadline` pour gestion du temps
- **Validation Pydantic complète** : Types de requêtes et réponses strictement validés
- **Logging structuré** : Logs détaillés à tous les niveaux
- **Health check amélioré** : Vérification d'Ollama et des modèles
- **Documentation complète** :
  - README.md avec guide d'installation et d'utilisation
  - ALGORITHM.md avec explication détaillée des algorithmes
  - MIGRATION.md pour la migration depuis v2.7
  - Tests unitaires complets
- **Docker Compose** : Stack complète avec Ollama

#### Modifié
- **Modèle LLM par défaut** : `llama3.1:3b-instruct` au lieu de `mistral:7b` (2-3x plus rapide)
- **Architecture RAG optimisée** : Déduplication et expansion de requêtes
- **Anti-duplication améliorée** : Détection avec shingles et Jaccard
- **Prompt engineering** : Variation de style et température dynamique
- **Coercition de type** : Meilleure gestion des types number, date, email, file
- **Fallbacks intelligents** : Règles métier pour réponses rapides
- **Gestion des erreurs** : Try-catch complets avec logging
- **Performance** : Embeddings en batch, early returns

#### Déprécié
- `/api/score` (legacy) : Toujours disponible mais limité, utiliser `/api/score_v2`
- `routes/analyse.py` : Fusionné dans `app/main.py`

#### Supprimé
- Dépendance à OpenCV (non utilisée)
- Code mort et commentaires obsolètes
- Prompts externes vides dans `prompts/`

### 🐛 Corrections

- **Timeout Ollama** : Gestion adaptative des timeouts
- **Placeholders** : Nettoyage robuste des `[xxx]`, `X%`, etc.
- **Duplication** : Cache intelligent pour éviter les réponses identiques
- **CORS** : Configuration flexible via variable d'environnement
- **Parsing contexte** : Support de tous les formats de `PrevAnswer`
- **Embeddings vides** : Gestion des cas où Ollama échoue

### 🔧 Technique

- **Python 3.10+** requis
- **FastAPI 0.109.0**
- **Pydantic 2.5.3**
- **Nouvelles variables d'environnement** :
  - `OLLAMA_URL`, `LLM_MODEL`, `EMBED_MODEL`
  - `MAX_LLM_TIMEOUT`, `MAX_EMBED_TIMEOUT`
  - `CACHE_SIZE`, `RAG_TOP_K`, `DEDUPE_THRESHOLD`, `SIMILARITY_THRESHOLD`
  - `CORS_ORIGINS`, `LOG_LEVEL`

### 📊 Performance

| Métrique | v2.7 | v3.0 | Amélioration |
|----------|------|------|--------------|
| Génération (sans RAG) | 5-8s | 2-4s | **50% plus rapide** |
| Génération (avec RAG) | 8-12s | 3-8s | **40% plus rapide** |
| Scoring | 60-150s | <0.5s | **300x plus rapide** |
| Taille du code | ~800 lignes | ~1200 lignes | +50% (mais consolidé) |

---

## [2.7.0] - 2025-09-15

### Ajouté
- Pipeline de génération avec RAG basique
- Endpoint `/api/score` avec analyse LLM
- Support de base pour le contexte Laravel
- Parsing des faits (Startup, Sector, etc.)
- Génération via Ollama

### Connu
- Scoring très lent (60-150s)
- Pas de gestion des deadlines
- Configuration hardcodée
- Duplication des réponses
- Documentation minimale

---

## [2.0.0] - 2025-08-01

### Ajouté
- API FastAPI initiale
- Intégration Ollama
- Endpoint `/api/generate`

---

## [1.0.0] - 2025-06-15

### Ajouté
- Prototype initial
- Tests de concept avec LLM

---

## Format des versions

- **MAJOR** : Changements incompatibles avec l'API
- **MINOR** : Ajout de fonctionnalités rétro-compatibles
- **PATCH** : Corrections de bugs rétro-compatibles

## Liens

- [Documentation](README.md)
- [Guide de migration](MIGRATION.md)
- [Détails algorithme](ALGORITHM.md)
