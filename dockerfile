# ================================================
# Dockerfile - Startup Incubation API v3.0.0
# ================================================

# ──────────────────────────────────────────────
# Stage 1: Builder
# ──────────────────────────────────────────────
FROM python:3.10-slim as builder

WORKDIR /app

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copier et installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ──────────────────────────────────────────────
# Stage 2: Runtime
# ──────────────────────────────────────────────
FROM python:3.10-slim

# Métadonnées
LABEL maintainer="your-email@example.com"
LABEL version="3.0.0"
LABEL description="Startup Incubation AI Analysis API"

# Variables d'environnement
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/root/.local/bin:$PATH

WORKDIR /app

# Installer curl pour healthcheck
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copier les dépendances depuis le builder
COPY --from=builder /root/.local /root/.local

# Copier le code de l'application
COPY app ./app
COPY .env.example .env

# Créer un utilisateur non-root (sécurité)
RUN useradd -m -u 1000 apiuser && \
    chown -R apiuser:apiuser /app

USER apiuser

# Exposer le port
EXPOSE 5005

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5005/health || exit 1

# Commande de démarrage
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5005", "--workers", "1"]