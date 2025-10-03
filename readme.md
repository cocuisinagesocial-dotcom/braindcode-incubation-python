# 🚀 Startup Incubation AI Analysis API

API d'analyse et de génération de réponses intelligentes pour accompagner les startups en incubation.

## 📋 Table des matières

- [Vue d'ensemble](#vue-densemble)
- [Fonctionnalités](#fonctionnalités)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Utilisation](#utilisation)
- [API Reference](#api-reference)
- [Algorithme détaillé](#algorithme-détaillé)
- [Déploiement](#déploiement)
- [Troubleshooting](#troubleshooting)

## 🎯 Vue d'ensemble

Cette API utilise l'intelligence artificielle (via Ollama) pour :
- **Générer des réponses personnalisées** pour les startups en formation
- **Scorer et évaluer** la qualité des réponses fournies
- **Fournir du feedback constructif** avec des suggestions d'amélioration

### Technologies utilisées

- **FastAPI** : Framework web moderne et performant
- **Ollama** : LLM local (Llama 3.1, Mistral)
- **Python 3.10+** : Langage principal
- **RAG** (Retrieval-Augmented Generation) : Pour des réponses contextuelles

## ✨ Fonctionnalités

### 1. Génération de réponses (`/api/generate`)

- Parse le contexte fourni (profil startup, réponses précédentes)
- Utilise le RAG pour trouver les informations pertinentes
- Génère une réponse personnalisée via LLM
- Applique des fallbacks basés sur règles si nécessaire
- Évite la duplication avec les réponses précédentes

### 2. Scoring structuré (`/api/score_v2`)

- Évalue la qualité de chaque réponse (0-100)
- Calcule un score global pondéré
- Fournit un feedback détaillé
- Propose des suggestions d'amélioration
- Identifie les éléments manquants

### 3. Health check (`/health`)

- Vérifie l'état de l'API
- Teste la connexion à Ollama
- Retourne les modèles configurés

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Client (Laravel)                        │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      │ HTTP/JSON
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                   FastAPI Application                        │
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Endpoints  │  │   RAG Engine │  │   Scoring    │      │
│  │  /generate   │  │  - Embeddings│  │  - Rules     │      │
│  │  /score_v2   │  │  - Search    │  │  - LLM eval  │      │
│  │  /health     │  │  - Dedupe    │  │  - Feedback  │      │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┘      │
│         │                  │                                  │
│         └──────────────────┴──────────────────┐              │
│                                                │              │
│                                         ┌──────▼───────┐     │
│                                         │   Cache      │     │
│                                         │  (In-memory) │     │
│                                         └──────────────┘     │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            │ HTTP API
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                      Ollama Server                           │
│                                                               │
│  ┌──────────────────┐           ┌──────────────────┐        │
│  │  LLM Model       │           │  Embedding Model │        │
│  │  (llama3.1:3b)   │           │  (nomic-embed)   │        │
│  └──────────────────┘           └──────────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Installation

### Prérequis

- Python 3.10+
- Ollama installé et lancé
- Les modèles Ollama téléchargés

### 1. Installer Ollama

```bash
# Linux/Mac
curl -fsSL https://ollama.com/install.sh | sh

# Windows : télécharger depuis https://ollama.com
```

### 2. Télécharger les modèles

```bash
# Modèle LLM (choisir un)
ollama pull llama3.1:3b-instruct       # Rapide, recommandé
ollama pull mistral:7b-instruct-q4_K_M # Meilleure qualité

# Modèle d'embeddings
ollama pull nomic-embed-text
```

### 3. Cloner et installer l'application

```bash
# Cloner le projet
git clone <votre-repo>
cd startup-incubation-api

# Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou
.venv\Scripts\activate     # Windows

# Installer les dépendances
pip install -r requirements.txt
```

### 4. Configuration

```bash
# Copier le fichier d'exemple
cp .env.example .env

# Éditer .env avec vos paramètres
nano .env
```

## ⚙️ Configuration

Toute la configuration se fait via le fichier `.env`. Voir `.env.example` pour les détails.

### Configuration minimale

```env
OLLAMA_URL=http://localhost:11434
LLM_MODEL=llama3.1:3b-instruct
EMBED_MODEL=nomic-embed-text
PORT=5005
```

### Configuration production

```env
# Serveur
HOST=0.0.0.0
PORT=5005

# Ollama (adapter selon votre déploiement)
OLLAMA_URL=http://ollama-service:11434
LLM_MODEL=llama3.1:3b-instruct
EMBED_MODEL=nomic-embed-text

# Timeouts (augmenter si nécessaire)
MAX_LLM_TIMEOUT=15.0
MAX_EMBED_TIMEOUT=8.0

# CORS (spécifier vos domaines)
CORS_ORIGINS=https://votre-domaine.com,https://app.votre-domaine.com

# Logging
LOG_LEVEL=INFO
```

## 🎮 Utilisation

### Démarrer le serveur

```bash
# Mode développement
uvicorn app.main:app --reload --host 0.0.0.0 --port 5005

# Ou directement
python app/main.py
```

### Vérifier le statut

```bash
curl http://localhost:5005/health
```

### Documentation interactive

Ouvrir dans le navigateur :
- Swagger UI : `http://localhost:5005/docs`
- ReDoc : `http://localhost:5005/redoc`

## 📡 API Reference

### POST /api/generate

Génère une réponse personnalisée pour une question.

**Request:**
```json
{
  "step_name": "Personas & Segmentation",
  "question_label": "Qui est impacté par ce problème ?",
  "startup_name": "MonStartup",
  "sector_name": "SaaS B2B",
  "context": "Startup: MonStartup\nSector: SaaS B2B\nProblem: Gestion complexe des stocks\n...",
  "question_type": "textarea",
  "prompt": "Réponse en puces"
}
```

**Response:**
```json
{
  "answer": "- PME du secteur retail (50-200 employés)\n- Managers de supply chain\n- Équipes opérationnelles",
  "metadata": {
    "elapsed_seconds": 2.34,
    "rag_snippets": 5,
    "style": "bullets",
    "question_type": "textarea"
  }
}
```

### POST /api/score_v2

Évalue un ensemble de réponses structurées.

**Request:**
```json
{
  "step_name": "Étape 1",
  "items": [
    {
      "question_id": 1,
      "label": "Question 1",
      "type": "text",
      "answer": "Ma réponse...",
      "points": 10
    }
  ]
}
```

**Response:**
```json
{
  "items": [
    {
      "question_id": 1,
      "score": 75,
      "status": "validée",
      "feedback": "Bon niveau. Rendez 1–2 éléments plus concrets.",
      "points": 10,
      "issues": [],
      "suggestion_bullets": ["• Ajouter des chiffres"],
      "suggested_answer": "Ex : ..."
    }
  ],
  "global_score": 75,
  "status": "validée",
  "feedback": "Niveau satisfaisant (75%)."
}
```

### GET /health

Vérifie l'état de l'API.

**Response:**
```json
{
  "status": "ok",
  "version": "3.0.0",
  "ollama_status": "ok",
  "model": "llama3.1:3b-instruct",
  "embed_model": "nomic-embed-text"
}
```

## 🧠 Algorithme détaillé

Voir le fichier [ALGORITHM.md](ALGORITHM.md) pour une explication détaillée de l'algorithme.

### Pipeline de génération (résumé)

1. **Parse du contexte** : Extraction des faits et du corpus
2. **RAG** : Recherche sémantique des snippets pertinents
3. **Génération LLM** : Création de la réponse avec prompt enrichi
4. **Fallbacks** : Application de règles si réponse insuffisante
5. **Post-traitement** : Nettoyage, coercition, anti-duplication

### Pipeline de scoring (résumé)

1. **Analyse du contenu** : Mots, chiffres, symboles
2. **Calcul du score** : Basé sur des règles métier
3. **Feedback** : Génération de suggestions
4. **Score global** : Moyenne pondérée

## 🐳 Déploiement

### Docker Compose (recommandé)

Créer `docker-compose.yml` :

```yaml
version: '3.8'

services:
  ollama:
    image: ollama/ollama:latest
    container_name: ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    restart: unless-stopped
    
  api:
    build: .
    container_name: startup-api
    ports:
      - "5005:5005"
    environment:
      - OLLAMA_URL=http://ollama:11434
      - LLM_MODEL=llama3.1:3b-instruct
      - EMBED_MODEL=nomic-embed-text
    depends_on:
      - ollama
    restart: unless-stopped

volumes:
  ollama_data:
```

Lancer :
```bash
docker-compose up -d

# Télécharger les modèles dans le container Ollama
docker exec -it ollama ollama pull llama3.1:3b-instruct
docker exec -it ollama ollama pull nomic-embed-text
```

### Dockerfile amélioré

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copier les fichiers
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY .env .env

# Exposer le port
EXPOSE 5005

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5005/health || exit 1

# Lancer l'application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5005"]
```

## 🔧 Troubleshooting

### Problème : Ollama ne répond pas

**Symptômes :**
```json
{"status": "ok", "ollama_status": "error"}
```

**Solutions :**
1. Vérifier qu'Ollama est lancé : `ollama list`
2. Vérifier l'URL dans `.env`
3. Tester manuellement : `curl http://localhost:11434/api/tags`

### Problème : Timeout lors de la génération

**Symptômes :**
- Réponses vides
- Erreurs de timeout dans les logs

**Solutions :**
1. Augmenter `MAX_LLM_TIMEOUT` dans `.env`
2. Utiliser un modèle plus petit (3b au lieu de 7b)
3. Vérifier les ressources système (CPU/RAM)

### Problème : Réponses de mauvaise qualité

**Solutions :**
1. Utiliser un modèle plus gros : `mistral:7b-instruct`
2. Augmenter `RAG_TOP_K` pour plus de contexte
3. Améliorer le contexte fourni dans les requêtes

### Problème : CORS errors

**Symptômes :**
```
Access to XMLHttpRequest at 'http://localhost:5005/api/generate' 
from origin 'http://localhost:8000' has been blocked by CORS policy
```

**Solutions :**
1. Ajouter l'origine dans `CORS_ORIGINS` dans `.env`
2. Exemple : `CORS_ORIGINS=http://localhost:8000,https://app.example.com`

## 📊 Performance

### Benchmarks (modèle 3b, CPU moderne)

| Opération | Temps moyen | Notes |
|-----------|-------------|-------|
| RAG (8 snippets) | 1-2s | Dépend du corpus |
| Génération LLM | 2-5s | Dépend de la longueur |
| Scoring | <0.5s | Basé sur règles |
| Total /generate | 3-8s | Avec RAG |

### Optimisations possibles

1. **GPU** : Utiliser Ollama avec GPU (x5-10 plus rapide)
2. **Cache** : Déjà implémenté (évite les duplications)
3. **Modèle quantifié** : q4_K_M pour rapidité
4. **Batch processing** : Pour scorer plusieurs items

## 🤝 Contribution

Les contributions sont bienvenues !

1. Fork le projet
2. Créer une branche (`git checkout -b feature/AmazingFeature`)
3. Commit (`git commit -m 'Add AmazingFeature'`)
4. Push (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

## 📝 License

Ce projet est sous licence MIT.

## 📧 Contact

Pour toute question : [votre-email@example.com]

---

**Version :** 3.0.0  
**Dernière mise à jour :** 2025
