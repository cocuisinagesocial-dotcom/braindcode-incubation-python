# ⚡ Démarrage rapide - 5 minutes

Guide ultra-rapide pour démarrer avec l'API en moins de 5 minutes.

## 📋 Prérequis

- Python 3.10+
- 4GB RAM minimum
- 10GB espace disque (pour les modèles)

## 🚀 Installation en 4 étapes

### 1. Installer Ollama

```bash
# Linux/Mac
curl -fsSL https://ollama.com/install.sh | sh

# Windows : télécharger depuis https://ollama.com
# Puis redémarrer le terminal
```

### 2. Cloner et configurer

```bash
# Cloner le projet
git clone <votre-repo>
cd startup-incubation-api

# Créer l'environnement virtuel
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou .venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt

# Créer la configuration
cp .env.example .env
```

### 3. Télécharger les modèles

```bash
# Modèle LLM (rapide, ~2GB)
ollama pull llama3.2:3b
# OU si pas disponible : ollama pull mistral:7b-instruct

# Modèle d'embeddings (~270MB)
ollama pull nomic-embed-text
```

### 4. Lancer !

```bash
# Démarrer l'API
python app/main.py

# ✅ API accessible sur http://localhost:5005
```

## 🧪 Test rapide

Dans un autre terminal :

```bash
# Health check
curl http://localhost:5005/health

# Test de génération
curl -X POST http://localhost:5005/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "step_name": "Test",
    "question_label": "Quel est votre client cible ?",
    "context": "Startup: TestCo\nSector: FinTech\nPersonas: Freelances",
    "question_type": "text"
  }'
```

**Résultat attendu :**
```json
{
  "answer": "Les freelances constituent le client cible...",
  "metadata": {...}
}
```

## 🐳 Alternative : Docker (recommandé pour production)

```bash
# Lancer tout avec Docker Compose
docker-compose up -d

# Télécharger les modèles dans le container
docker exec -it startup-ollama ollama pull llama3.1:3b-instruct
docker exec -it startup-ollama ollama pull nomic-embed-text

# Vérifier
curl http://localhost:5005/health
```

## 📚 Prochaines étapes

1. **Documentation complète** : [README.md](README.md)
2. **Comprendre l'algorithme** : [ALGORITHM.md](ALGORITHM.md)
3. **Intégrer avec Laravel** : [MIGRATION.md](MIGRATION.md#adaptation-de-lintégration-laravel)
4. **Explorer l'API** : http://localhost:5005/docs

## 🎯 Cas d'usage rapides

### Générer une réponse

```python
import requests

response = requests.post('http://localhost:5005/api/generate', json={
    'step_name': 'Personas & Segmentation',
    'question_label': 'Qui est votre client cible ?',
    'context': '''
Startup: MonApp
Sector: EdTech
Problem: Les étudiants ont du mal à organiser leurs révisions
Solution: App mobile avec planning intelligent
Personas: Étudiants universitaires, 18-25 ans
    ''',
    'question_type': 'textarea'
})

print(response.json()['answer'])
```

### Scorer des réponses

```python
import requests

response = requests.post('http://localhost:5005/api/score_v2', json={
    'step_name': 'Business Model',
    'items': [
        {
            'question_id': 1,
            'label': 'Qui est impacté ?',
            'type': 'text',
            'answer': 'Les étudiants universitaires en France, environ 2.7M personnes.',
            'points': 10
        },
        {
            'question_id': 2,
            'label': 'Revenu année 1 ?',
            'type': 'number',
            'answer': '50000',
            'points': 5
        }
    ]
})

result = response.json()
print(f"Score global: {result['global_score']}%")
print(f"Status: {result['status']}")
```

## ⚙️ Configuration rapide

Éditer `.env` pour personnaliser :

```env
# Modèle LLM (choisir selon vos besoins)
LLM_MODEL=llama3.1:3b-instruct  # Rapide ✅
# LLM_MODEL=mistral:7b-instruct  # Meilleure qualité mais plus lent

# Timeouts (ajuster selon votre machine)
MAX_LLM_TIMEOUT=12.0  # Augmenter si lent
MAX_EMBED_TIMEOUT=6.0

# Port
PORT=5005  # Changer si déjà utilisé
```

## 🔧 Troubleshooting rapide

### Problème : "Ollama not found"

```bash
# Vérifier qu'Ollama est installé
ollama --version

# Vérifier qu'Ollama tourne
curl http://localhost:11434/api/tags
```

### Problème : "Model not found"

```bash
# Lister les modèles
ollama list

# Télécharger le modèle manquant
ollama pull llama3.1:3b-instruct
```

### Problème : Réponses lentes (>10s)

```bash
# Option 1 : Utiliser un modèle plus petit (déjà le cas)
# Option 2 : Vérifier les ressources
htop  # Linux
# Regarder CPU et RAM

# Option 3 : Augmenter le timeout
# Dans .env : MAX_LLM_TIMEOUT=20.0
```

### Problème : Port déjà utilisé

```bash
# Trouver ce qui utilise le port 5005
lsof -i :5005  # Linux/Mac
netstat -ano | findstr :5005  # Windows

# Tuer le processus ou changer le port dans .env
PORT=5006
```

## 🎉 C'est tout !

Vous êtes prêt à utiliser l'API. Pour aller plus loin :

- **Documentation API interactive** : http://localhost:5005/docs
- **Guide complet** : [README.md](README.md)
- **Support** : Créer une issue sur GitHub

---

**Temps de setup :** ~5 minutes (hors téléchargement des modèles)  
**Version :** 3.0.0  
**Dernière mise à jour :** 2025-10-03