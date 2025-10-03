# ================================================
# Makefile - Startup Incubation API v3.0.0
# ================================================

.PHONY: help install dev run test clean docker docker-up docker-down docker-logs models health lint format check all

# Couleurs pour l'affichage
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
NC := \033[0m # No Color

# Variables
PYTHON := python3
PIP := $(PYTHON) -m pip
VENV := .venv
PORT := 5005

# ──────────────────────────────────────────────
# Aide
# ──────────────────────────────────────────────
help: ## Affiche cette aide
	@echo "$(BLUE)Startup Incubation API - Commandes disponibles:$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2}'
	@echo ""

# ──────────────────────────────────────────────
# Installation
# ──────────────────────────────────────────────
install: ## Installer les dépendances
	@echo "$(BLUE)Installation des dépendances...$(NC)"
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt
	@echo "$(GREEN)✓ Installation terminée$(NC)"

install-dev: install ## Installer avec dépendances de dev
	@echo "$(BLUE)Installation des dépendances de développement...$(NC)"
	$(VENV)/bin/pip install pytest pytest-asyncio pytest-cov black flake8 mypy
	@echo "$(GREEN)✓ Installation dev terminée$(NC)"

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
env: ## Créer le fichier .env depuis .env.example
	@if [ ! -f .env ]; then \
		echo "$(BLUE)Création du fichier .env...$(NC)"; \
		cp .env.example .env; \
		echo "$(GREEN)✓ Fichier .env créé. Pensez à le configurer!$(NC)"; \
	else \
		echo "$(YELLOW)⚠ Le fichier .env existe déjà$(NC)"; \
	fi

# ──────────────────────────────────────────────
# Ollama
# ──────────────────────────────────────────────
models: ## Télécharger les modèles Ollama
	@echo "$(BLUE)Téléchargement des modèles Ollama...$(NC)"
	@echo "$(YELLOW)Cela peut prendre plusieurs minutes...$(NC)"
	ollama pull llama3.2:3b || ollama pull mistral:7b-instruct
	ollama pull nomic-embed-text
	@echo "$(GREEN)✓ Modèles téléchargés$(NC)"

models-list: ## Lister les modèles Ollama disponibles
	@echo "$(BLUE)Modèles Ollama installés:$(NC)"
	@ollama list

# ──────────────────────────────────────────────
# Exécution
# ──────────────────────────────────────────────
run: ## Lancer l'application
	@echo "$(BLUE)Démarrage de l'application sur le port $(PORT)...$(NC)"
	$(VENV)/bin/uvicorn app.main:app --host 0.0.0.0 --port $(PORT)

dev: ## Lancer en mode développement (auto-reload)
	@echo "$(BLUE)Démarrage en mode développement...$(NC)"
	$(VENV)/bin/uvicorn app.main:app --reload --host 0.0.0.0 --port $(PORT)

# ──────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────
test: ## Lancer les tests
	@echo "$(BLUE)Exécution des tests...$(NC)"
	$(VENV)/bin/pytest test_main.py -v

test-cov: ## Tests avec couverture
	@echo "$(BLUE)Tests avec couverture...$(NC)"
	$(VENV)/bin/pytest test_main.py -v --cov=app --cov-report=html --cov-report=term

test-integration: ## Tests d'intégration uniquement
	@echo "$(BLUE)Tests d'intégration...$(NC)"
	$(VENV)/bin/pytest test_main.py -v -m integration

# ──────────────────────────────────────────────
# Qualité du code
# ──────────────────────────────────────────────
lint: ## Vérifier le code avec flake8
	@echo "$(BLUE)Vérification du code...$(NC)"
	$(VENV)/bin/flake8 app/ --max-line-length=120 --exclude=__pycache__

format: ## Formater le code avec black
	@echo "$(BLUE)Formatage du code...$(NC)"
	$(VENV)/bin/black app/ --line-length=120

format-check: ## Vérifier le formatage sans modifier
	@echo "$(BLUE)Vérification du formatage...$(NC)"
	$(VENV)/bin/black app/ --check --line-length=120

type-check: ## Vérifier les types avec mypy
	@echo "$(BLUE)Vérification des types...$(NC)"
	$(VENV)/bin/mypy app/ --ignore-missing-imports

check: lint format-check type-check ## Vérifier tout (lint + format + types)
	@echo "$(GREEN)✓ Toutes les vérifications passées$(NC)"

# ──────────────────────────────────────────────
# Health checks
# ──────────────────────────────────────────────
health: ## Vérifier l'état de l'API
	@echo "$(BLUE)Vérification de l'API...$(NC)"
	@curl -s http://localhost:$(PORT)/health | python3 -m json.tool || echo "$(YELLOW)⚠ API non accessible$(NC)"

health-ollama: ## Vérifier l'état d'Ollama
	@echo "$(BLUE)Vérification d'Ollama...$(NC)"
	@curl -s http://localhost:11434/api/tags | python3 -m json.tool || echo "$(YELLOW)⚠ Ollama non accessible$(NC)"

# ──────────────────────────────────────────────
# Docker
# ──────────────────────────────────────────────
docker-build: ## Construire l'image Docker
	@echo "$(BLUE)Construction de l'image Docker...$(NC)"
	docker build -t startup-api:latest .
	@echo "$(GREEN)✓ Image construite$(NC)"

docker-up: ## Lancer avec Docker Compose
	@echo "$(BLUE)Démarrage de la stack Docker...$(NC)"
	docker-compose up -d
	@echo "$(GREEN)✓ Stack démarrée$(NC)"
	@echo "$(YELLOW)N'oubliez pas de télécharger les modèles: make docker-models$(NC)"

docker-down: ## Arrêter Docker Compose
	@echo "$(BLUE)Arrêt de la stack Docker...$(NC)"
	docker-compose down
	@echo "$(GREEN)✓ Stack arrêtée$(NC)"

docker-logs: ## Voir les logs Docker
	docker-compose logs -f

docker-models: ## Télécharger les modèles dans le container Ollama
	@echo "$(BLUE)Téléchargement des modèles dans Docker...$(NC)"
	docker exec -it startup-ollama ollama pull llama3.1:3b-instruct
	docker exec -it startup-ollama ollama pull nomic-embed-text
	@echo "$(GREEN)✓ Modèles téléchargés dans Docker$(NC)"

docker-restart: docker-down docker-up ## Redémarrer Docker Compose

# ──────────────────────────────────────────────
# Nettoyage
# ──────────────────────────────────────────────
clean: ## Nettoyer les fichiers temporaires
	@echo "$(BLUE)Nettoyage...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.log" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".coverage" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)✓ Nettoyage terminé$(NC)"

clean-all: clean ## Nettoyer tout (y compris venv)
	@echo "$(BLUE)Nettoyage complet...$(NC)"
	rm -rf $(VENV)
	@echo "$(GREEN)✓ Nettoyage complet terminé$(NC)"

# ──────────────────────────────────────────────
# Tout-en-un
# ──────────────────────────────────────────────
setup: install env models ## Installation complète (install + env + models)
	@echo "$(GREEN)✓ Installation complète terminée!$(NC)"
	@echo "$(BLUE)Vous pouvez maintenant lancer: make dev$(NC)"

all: clean check test ## Tout vérifier (clean + check + test)
	@echo "$(GREEN)✓ Toutes les vérifications et tests passés!$(NC)"

# ──────────────────────────────────────────────
# Info
# ──────────────────────────────────────────────
info: ## Afficher les informations système
	@echo "$(BLUE)Informations système:$(NC)"
	@echo "  Python: $(shell $(PYTHON) --version)"
	@echo "  Pip: $(shell $(PIP) --version)"
	@echo "  Ollama: $(shell ollama --version 2>/dev/null || echo 'Non installé')"
	@echo "  Docker: $(shell docker --version 2>/dev/null || echo 'Non installé')"
	@echo ""
	@echo "$(BLUE)Configuration:$(NC)"
	@echo "  Port: $(PORT)"
	@echo "  Venv: $(VENV)"
	@echo ""

# Par défaut, afficher l'aide
.DEFAULT_GOAL := help