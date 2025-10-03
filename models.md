# 🤖 Guide des modèles Ollama

Ce guide vous aide à choisir les bons modèles pour votre cas d'usage.

## 📊 Comparaison rapide

| Modèle | Taille | Vitesse | Qualité | RAM | Recommandé pour |
|--------|--------|---------|---------|-----|-----------------|
| **llama3.2:3b** | 2GB | ⚡⚡⚡⚡⚡ | ⭐⭐⭐⭐ | 4GB | Production, API rapide ✅ |
| **gemma2:2b** | 1.6GB | ⚡⚡⚡⚡⚡ | ⭐⭐⭐ | 3GB | Ultra-rapide, contraintes |
| **mistral:7b** | 4GB | ⚡⚡⚡⚡ | ⭐⭐⭐⭐⭐ | 8GB | Qualité optimale |
| **llama3.1:8b** | 4.7GB | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | 8GB | Meilleur équilibre |
| **qwen2.5:7b** | 4.7GB | ⚡⚡⚡⚡ | ⭐⭐⭐⭐⭐ | 8GB | Excellent alternatif |
| **phi3:3.8b** | 2.3GB | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ | 4GB | Bon compromis |

## 🎯 Choix selon votre cas

### Vous avez peu de RAM (4GB)
```bash
# Option 1 : Le plus rapide et léger
ollama pull llama3.2:3b

# Option 2 : Ultra-léger
ollama pull gemma2:2b

# Configuration .env
LLM_MODEL=llama3.2:3b
```

### Vous voulez la meilleure qualité (8GB+)
```bash
# Option 1 : Excellent français
ollama pull mistral:7b-instruct

# Option 2 : Très polyvalent
ollama pull llama3.1:8b

# Option 3 : Alternative excellente
ollama pull qwen2.5:7b

# Configuration .env
LLM_MODEL=mistral:7b-instruct
```

### Vous voulez un bon compromis (6GB)
```bash
# Option 1 : Microsoft Phi
ollama pull phi3:3.8b

# Option 2 : Llama 3.2
ollama pull llama3.2:3b

# Configuration .env
LLM_MODEL=phi3:3.8b
```

### Vous avez un GPU
```bash
# Vous pouvez utiliser des modèles plus gros
ollama pull llama3.1:8b
ollama pull mistral:7b-instruct

# Ou même des quantifications non compressées
ollama pull mistral:7b-instruct-q8_0

# Configuration .env
LLM_MODEL=mistral:7b-instruct
```

## 🔍 Vérifier les modèles disponibles

### Voir ce qui est installé
```bash
ollama list
```

### Chercher des modèles
```bash
# Chercher des modèles Llama
ollama search llama

# Chercher des modèles Mistral
ollama search mistral

# Voir tous les modèles populaires
# Visitez : https://ollama.com/library
```

## 📥 Installation des modèles

### Llama 3.2 3B (recommandé)
```bash
ollama pull llama3.2:3b
```

### Mistral 7B
```bash
# Version de base
ollama pull mistral:7b

# Version instruct (meilleure pour notre usage)
ollama pull mistral:7b-instruct

# Version quantifiée Q4 (plus rapide)
ollama pull mistral:7b-instruct-q4_K_M
```

### Gemma 2 2B (ultra-rapide)
```bash
ollama pull gemma2:2b
```

### Llama 3.1 8B
```bash
ollama pull llama3.1:8b
```

### Qwen 2.5 7B
```bash
ollama pull qwen2.5:7b
```

### Phi 3 3.8B
```bash
ollama pull phi3:3.8b
```

## 🧪 Tester un modèle

```bash
# Test interactif
ollama run llama3.2:3b

# Dans le prompt :
>>> Tu es un coach pour startups. Explique ce qu'est une proposition de valeur.

# Quitter : /bye
```

## ⚙️ Configuration dans l'API

### Fichier .env
```env
# Modèle principal
LLM_MODEL=llama3.2:3b

# Modèle d'embeddings (ne pas changer)
EMBED_MODEL=nomic-embed-text

# Augmenter le timeout si modèle lent
MAX_LLM_TIMEOUT=15.0
```

### Tester avec l'API
```bash
# Démarrer l'API
python app/main.py

# Dans un autre terminal
curl http://localhost:5005/health
```

Vous devriez voir :
```json
{
  "status": "ok",
  "ollama_status": "ok",
  "model": "llama3.2:3b",
  "embed_model": "nomic-embed-text"
}
```

## 🔄 Changer de modèle

### Méthode 1 : Via .env
```bash
# Éditer .env
nano .env

# Changer LLM_MODEL
LLM_MODEL=mistral:7b-instruct

# Redémarrer l'API
```

### Méthode 2 : Variable d'environnement
```bash
# Linux/Mac
export LLM_MODEL=mistral:7b-instruct
python app/main.py

# Windows
set LLM_MODEL=mistral:7b-instruct
python app/main.py
```

### Méthode 3 : Docker
```yaml
# docker-compose.yml
services:
  api:
    environment:
      - LLM_MODEL=mistral:7b-instruct
```

## 📊 Benchmarks sur notre API

Tests sur un CPU moderne (Intel i7) avec différents modèles :

| Modèle | Génération simple | Génération + RAG | Mémoire |
|--------|-------------------|------------------|---------|
| llama3.2:3b | 2-3s | 4-6s | 3.5GB |
| gemma2:2b | 1-2s | 3-5s | 2.8GB |
| mistral:7b | 4-6s | 7-10s | 6GB |
| llama3.1:8b | 5-7s | 8-12s | 7GB |
| phi3:3.8b | 2-4s | 5-8s | 4GB |

**Note :** Avec GPU, divisez ces temps par 5-10x.

## 🐛 Problèmes courants

### "Error: pull model manifest: file does not exist"

**Cause :** Le modèle n'existe pas avec ce nom.

**Solution :**
```bash
# Vérifier le nom exact sur https://ollama.com/library
# Exemples de noms corrects :
ollama pull llama3.2:3b          # ✅
ollama pull llama3.2              # ✅ (version par défaut)
ollama pull llama3.1:3b-instruct # ❌ N'EXISTE PAS
```

### "Error: out of memory"

**Solution 1 :** Utiliser un modèle plus petit
```bash
ollama pull gemma2:2b
```

**Solution 2 :** Augmenter la RAM allouée à Docker
```bash
# Dans Docker Desktop : Settings > Resources > Memory
# Allouer au moins 8GB
```

**Solution 3 :** Utiliser une version quantifiée
```bash
# q4_K_M = quantification 4 bits
ollama pull mistral:7b-instruct-q4_K_M
```

### Le modèle est très lent

**Solution 1 :** Utiliser un modèle plus petit
```bash
ollama pull llama3.2:3b
```

**Solution 2 :** Augmenter le timeout
```env
# Dans .env
MAX_LLM_TIMEOUT=20.0
```

**Solution 3 :** Vérifier que vous n'êtes pas en mode swap
```bash
# Linux
free -h

# Si swap utilisé, ajouter de la RAM ou réduire le modèle
```

### La qualité n'est pas bonne

**Solution :** Utiliser un modèle plus gros
```bash
# Passer de 3B à 7B
ollama pull mistral:7b-instruct

# Ou à 8B
ollama pull llama3.1:8b
```

## 🎓 Recommandations finales

### Pour commencer (QUICKSTART)
```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

### Pour production
```bash
# Si serveur avec 8GB+ RAM
ollama pull mistral:7b-instruct
ollama pull nomic-embed-text

# Si serveur avec 4-6GB RAM
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

### Pour développement
```bash
# Version rapide pour itérer vite
ollama pull gemma2:2b
ollama pull nomic-embed-text
```

### Pour qualité maximale
```bash
# Si GPU disponible
ollama pull llama3.1:8b
ollama pull mistral:7b-instruct
ollama pull nomic-embed-text
```

## 📚 Ressources

- **Bibliothèque Ollama :** https://ollama.com/library
- **Documentation Ollama :** https://github.com/ollama/ollama
- **Comparaisons :** https://ollama.com/blog

---

**Dernière mise à jour :** 2025-10-03  
**Version API :** 3.0.0
