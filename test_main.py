"""
Tests unitaires pour l'API Startup Incubation
Usage: pytest test_main.py -v
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app, extract_facts, parse_context_lines, clean, smart_clean
from app.main import coerce_by_type, _jaccard, _shingles, _cos
from app.main import split_list, uvp_fallback

# ================================================
# Test Client
# ================================================

client = TestClient(app)

# ================================================
# Tests des endpoints
# ================================================

def test_root():
    """Test du endpoint racine"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "running"
    assert "version" in data


def test_health():
    """Test du health check"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "ollama_status" in data
    assert "model" in data


def test_generate_minimal():
    """Test de génération avec requête minimale"""
    payload = {
        "step_name": "Test Step",
        "question_label": "Test Question",
        "context": "Startup: TestCo\nSector: Tech",
        "question_type": "text"
    }
    response = client.post("/api/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert isinstance(data["answer"], str)


def test_generate_with_full_context():
    """Test de génération avec contexte complet"""
    payload = {
        "step_name": "Personas & Segmentation",
        "question_label": "Qui est impacté par ce problème ?",
        "startup_name": "TestStartup",
        "sector_name": "SaaS B2B",
        "context": """Startup: TestStartup
Sector: SaaS B2B
Problem: Gestion complexe des stocks
Personas: PME B2B, 50-200 employés
PrevAnswer: Problème principal => Coûts élevés de gestion""",
        "question_type": "textarea",
        "prompt": "Réponse en puces"
    }
    response = client.post("/api/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["answer"]) > 0


def test_score_v2_empty():
    """Test du scoring avec liste vide"""
    payload = {
        "step_name": "Test",
        "items": []
    }
    response = client.post("/api/score_v2", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["global_score"] == 0
    assert data["status"] == "à retravailler"


def test_score_v2_single_item():
    """Test du scoring avec un seul item"""
    payload = {
        "step_name": "Test Step",
        "items": [
            {
                "question_id": 1,
                "label": "Question test",
                "type": "text",
                "answer": "Ceci est une réponse de test avec des chiffres: 1500€ et 25% de croissance.",
                "points": 10,
                "has_file": False
            }
        ]
    }
    response = client.post("/api/score_v2", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert "global_score" in data
    assert len(data["items"]) == 1
    assert data["items"][0]["score"] > 0


def test_score_v2_multiple_items():
    """Test du scoring avec plusieurs items"""
    payload = {
        "step_name": "Test Step",
        "items": [
            {
                "question_id": 1,
                "label": "Question 1",
                "type": "text",
                "answer": "Réponse complète avec 50 PME et 120 employés en moyenne, budget de 15000€.",
                "points": 10
            },
            {
                "question_id": 2,
                "label": "Question 2",
                "type": "text",
                "answer": "Court",
                "points": 5
            },
            {
                "question_id": 3,
                "label": "Question 3",
                "type": "number",
                "answer": "1500",
                "points": 8
            }
        ]
    }
    response = client.post("/api/score_v2", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 3
    # Item 1 devrait avoir un bon score
    assert data["items"][0]["score"] > 70
    # Item 2 devrait avoir un score faible (trop court)
    assert data["items"][1]["score"] < 50
    # Item 3 devrait avoir un bon score (nombre valide)
    assert data["items"][2]["score"] > 70


def test_score_legacy():
    """Test de l'endpoint de scoring legacy"""
    payload = {
        "step_name": "Test",
        "responses": "Ceci est une réponse de test avec suffisamment de contenu.",
        "questions": ["Question 1", "Question 2"]
    }
    response = client.post("/api/score", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "score" in data
    assert "status" in data
    assert "feedback" in data


# ================================================
# Tests des utilitaires de texte
# ================================================

def test_clean_removes_placeholders():
    """Test du nettoyage des placeholders"""
    text = "Ceci est valide. Cela contient [placeholder] problématique. Fin valide."
    result = clean(text)
    assert "[placeholder]" not in result
    assert "Ceci est valide" in result
    assert "Fin valide" in result


def test_clean_normalizes_spaces():
    """Test de la normalisation des espaces"""
    text = "Texte    avec     espaces     multiples"
    result = clean(text)
    assert "  " not in result


def test_smart_clean():
    """Test du smart_clean"""
    # Cas normal
    text = "Texte normal"
    assert smart_clean(text) == "Texte normal"
    
    # Cas avec placeholder
    text_placeholder = "Texte avec [xxx]"
    result = smart_clean(text_placeholder)
    assert "[xxx]" not in result or result == "Texte avec [xxx]"


def test_split_list():
    """Test de la séparation de listes"""
    # Virgules
    assert split_list("a, b, c") == ["a", "b", "c"]
    
    # Lignes
    assert split_list("a\nb\nc") == ["a", "b", "c"]
    
    # Points-virgules
    assert split_list("a; b; c") == ["a", "b", "c"]
    
    # Mixte
    assert split_list("a, b\nc; d") == ["a", "b", "c", "d"]
    
    # Vide
    assert split_list("") == []
    assert split_list(None) == []


# ================================================
# Tests du parser de contexte
# ================================================

def test_extract_facts():
    """Test de l'extraction de faits"""
    context = """Startup: TestCo
Sector: Tech
Pitch: Solution innovante
Problem: Coûts élevés
Solution: Automatisation"""
    
    facts = extract_facts(context)
    assert facts["startup"] == "TestCo"
    assert facts["sector"] == "Tech"
    assert facts["pitch"] == "Solution innovante"
    assert facts["problem"] == "Coûts élevés"
    assert facts["solution"] == "Automatisation"


def test_extract_facts_revenue_streams():
    """Test de l'extraction des revenus"""
    context = "RevenueStreams: Abonnement SaaS; Licences; Consulting"
    facts = extract_facts(context)
    assert facts["revenue_streams"] == "Abonnement SaaS; Licences; Consulting"
    assert facts["revenue_primary"] == "Abonnement SaaS"
    assert facts["revenue_secondary"] == "Licences"


def test_parse_context_lines():
    """Test du parsing complet du contexte"""
    context = """Startup: TestCo
Sector: Tech
PrevAnswer: Question 1 => Réponse 1
PrevAnswer: Question 2 => Réponse 2"""
    
    facts, corpus = parse_context_lines(context)
    
    # Vérifier les faits
    assert "startup" in facts
    assert facts["startup"] == "TestCo"
    
    # Vérifier le corpus
    assert len(corpus) >= 2
    # Trouver les PrevAnswer dans le corpus
    prev_answers = [item for item in corpus if "Question" in item[0]]
    assert len(prev_answers) >= 2


# ================================================
# Tests de coercition de type
# ================================================

def test_coerce_by_type_number():
    """Test de coercition pour les nombres"""
    assert coerce_by_type("Le prix est 1500", "number") == "1500"
    assert coerce_by_type("Environ 12.5% de croissance", "number") == "12.5"
    assert coerce_by_type("Prix: 1,500.50€", "number") == "1"  # Premier nombre trouvé


def test_coerce_by_type_date():
    """Test de coercition pour les dates"""
    assert coerce_by_type("31/12/2024", "date") == "2024-12-31"
    assert coerce_by_type("2024-12-31", "date") == "2024-12-31"
    assert coerce_by_type("Pas de date", "date") == ""


def test_coerce_by_type_email():
    """Test de coercition pour les emails"""
    assert coerce_by_type("Contact: test@example.com", "email") == "test@example.com"
    assert coerce_by_type("Pas d'email", "email") == ""


def test_coerce_by_type_file():
    """Test de coercition pour les fichiers"""
    result = coerce_by_type("document.pdf — Description", "file")
    assert result == "document.pdf — Description"
    
    result = coerce_by_type("mon document", "file")
    assert "—" in result
    assert result.endswith(".pdf")


def test_coerce_by_type_text():
    """Test de coercition pour le texte"""
    text = "Texte normal"
    assert coerce_by_type(text, "text") == text
    assert coerce_by_type(text, "textarea") == text


# ================================================
# Tests des utilitaires de similarité
# ================================================

def test_shingles():
    """Test de la génération de shingles"""
    text = "les PME du secteur retail"
    shingles = _shingles(text, n=3)
    assert len(shingles) > 0
    assert isinstance(shingles, set)


def test_jaccard_identical():
    """Test de Jaccard sur textes identiques"""
    text = "Les PME du secteur retail rencontrent des difficultés"
    similarity = _jaccard(text, text)
    assert similarity == 1.0


def test_jaccard_different():
    """Test de Jaccard sur textes différents"""
    text1 = "Les PME du secteur retail"
    text2 = "Les grandes entreprises du luxe"
    similarity = _jaccard(text1, text2)
    assert 0 <= similarity < 1.0


def test_jaccard_similar():
    """Test de Jaccard sur textes similaires"""
    text1 = "Les PME du secteur retail rencontrent des difficultés"
    text2 = "Les PME du secteur retail rencontrent des problèmes"
    similarity = _jaccard(text1, text2)
    assert similarity > 0.5  # Devrait être assez similaire


def test_cos_similarity():
    """Test de la similarité cosinus"""
    vec1 = [1.0, 2.0, 3.0]
    vec2 = [1.0, 2.0, 3.0]
    similarity = _cos(vec1, vec2)
    assert abs(similarity - 1.0) < 0.001  # Identiques
    
    vec3 = [3.0, 2.0, 1.0]
    similarity = _cos(vec1, vec3)
    assert 0 < similarity < 1.0


# ================================================
# Tests des fallbacks
# ================================================

def test_uvp_fallback_with_value_prop():
    """Test UVP avec value_prop existante"""
    facts = {"value_prop": "Solution innovante pour PME"}
    result = uvp_fallback("", facts)
    assert "Solution innovante" in result


def test_uvp_fallback_constructed():
    """Test UVP construite à partir d'autres faits"""
    facts = {
        "problem": "Coûts élevés",
        "solution": "Automatisation",
        "personas": "PME B2B"
    }
    result = uvp_fallback("", facts)
    assert "Coûts élevés" in result
    assert "Automatisation" in result
    assert "PME B2B" in result


# ================================================
# Tests de validation Pydantic
# ================================================

def test_generate_request_validation():
    """Test de la validation GenerateRequest"""
    # Requête valide minimale
    payload = {
        "step_name": "Test",
        "question_label": "Question"
    }
    response = client.post("/api/generate", json=payload)
    assert response.status_code == 200


def test_score_request_validation():
    """Test de la validation StructuredRequest"""
    # Requête valide
    payload = {
        "step_name": "Test",
        "items": [
            {
                "question_id": 1,
                "type": "text",
                "answer": "Test"
            }
        ]
    }
    response = client.post("/api/score_v2", json=payload)
    assert response.status_code == 200


# ================================================
# Tests d'erreurs
# ================================================

def test_generate_missing_required_fields():
    """Test avec champs requis manquants"""
    payload = {
        "step_name": "Test"
        # question_label manquant
    }
    response = client.post("/api/generate", json=payload)
    assert response.status_code == 422  # Validation error


def test_score_missing_required_fields():
    """Test du scoring avec champs manquants"""
    payload = {
        # step_name manquant
        "items": []
    }
    response = client.post("/api/score_v2", json=payload)
    assert response.status_code == 422


# ================================================
# Tests d'intégration
# ================================================

@pytest.mark.integration
def test_full_generate_pipeline():
    """Test complet du pipeline de génération"""
    payload = {
        "step_name": "Personas & Segmentation",
        "question_label": "Qui est votre client cible ?",
        "startup_name": "InnovCo",
        "sector_name": "FinTech",
        "context": """Startup: InnovCo
Sector: FinTech
Problem: Gestion complexe des finances pour freelances
Solution: Application mobile simplifiée
Personas: Freelances, 25-45 ans, revenus irréguliers
BusinessModel: Freemium + abonnement premium
PrevAnswer: Problème principal => Manque de visibilité sur cash-flow""",
        "question_type": "textarea",
        "prompt": "Réponse structurée en 4-6 lignes"
    }
    
    response = client.post("/api/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    # Vérifications
    assert "answer" in data
    assert len(data["answer"]) > 50  # Réponse substantielle
    assert "metadata" in data
    assert "elapsed_seconds" in data["metadata"]


@pytest.mark.integration
def test_full_scoring_pipeline():
    """Test complet du pipeline de scoring"""
    payload = {
        "step_name": "Validation Business Model",
        "items": [
            {
                "question_id": 1,
                "label": "Qui est impacté par ce problème ?",
                "type": "textarea",
                "answer": "Les freelances et travailleurs indépendants en France, environ 3 millions de personnes, particulièrement ceux ayant des revenus irréguliers supérieurs à 30k€/an.",
                "points": 15
            },
            {
                "question_id": 2,
                "label": "Quel est votre CA prévisionnel année 1 ?",
                "type": "number",
                "answer": "150000",
                "points": 10
            },
            {
                "question_id": 3,
                "label": "Date de lancement prévue ?",
                "type": "date",
                "answer": "15/06/2025",
                "points": 5
            }
        ]
    }
    
    response = client.post("/api/score_v2", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    # Vérifications
    assert len(data["items"]) == 3
    assert data["global_score"] > 0
    assert data["status"] in ["validée", "à retravailler"]
    
    # Item 1 devrait avoir un bon score (réponse complète)
    assert data["items"][0]["score"] >= 70
    
    # Item 2 devrait avoir un bon score (nombre valide)
    assert data["items"][1]["score"] >= 70
    
    # Item 3 devrait avoir un bon score (date valide)
    assert data["items"][2]["score"] >= 70


# ================================================
# Configuration pytest
# ================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])