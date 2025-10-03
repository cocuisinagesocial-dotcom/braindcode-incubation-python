from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import httpx
import json
import asyncio
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Startup Incubation AI Analysis API",
    description="API for analyzing startup responses using Ollama",
    version="1.0.0"
)

# Pydantic models
class AnalysisRequest(BaseModel):
    step_name: str
    responses: str
    questions: list[str] = []

def generate_prompt(step, response, questions):
    """Generate the prompt for AI analysis"""
    return f"""
Tu es un expert en incubation de startups.

Voici une réponse à évaluer :
{response}

Question :
{" / ".join(questions)}

Étape :
{step}

Analyse la réponse et retourne exactement :
Score : (sur 100)
Feedback : (clair, concis)
Statut : validée / à retravailler

⚠️ Ta réponse doit obligatoirement suivre ce format exact :
Score : ...
Feedback : ...
Statut : ...
""".strip()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Startup Incubation AI Analysis API",
        "version": "1.0.0",
        "status": "running"
    }

# Add health check endpoint without /api prefix
@app.get("/health")
async def health_check_simple():
    """Simple health check endpoint"""
    try:
        # Test Ollama connection
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:11434/api/tags", timeout=5)
            ollama_status = "ok" if response.status_code == 200 else "error"
    except Exception as e:
        ollama_status = f"error: {str(e)}"
    
    return {
        "status": "ok",
        "ollama": ollama_status,
        "message": "FastAPI service is running"
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test Ollama connection
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:11434/api/tags", timeout=5)
            ollama_status = "ok" if response.status_code == 200 else "error"
    except Exception as e:
        ollama_status = f"error: {str(e)}"
    
    return {
        "status": "ok",
        "ollama": ollama_status,
        "message": "FastAPI service is running"
    }

@app.post("/api/score")
async def score_reponses(payload: AnalysisRequest):
    """Score startup responses using AI analysis"""
    logger.info(f"Received scoring request for step: {payload.step_name}")
    
    # Validate input
    if not payload.responses or not payload.responses.strip():
        return {
            "score": 0,
            "feedback": "Réponse vide.",
            "status": "refusée",
            "status_color": "danger"
        }
    
    prompt = generate_prompt(payload.step_name, payload.responses, payload.questions)
    
    try:
        async with httpx.AsyncClient() as client:
            logger.info("Sending request to Ollama...")
            
            # Increase timeout to match Laravel's expectation
            response = await client.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "mistral:7b-instruct-q4_K_M",
                    "prompt": prompt,
                    "stream": True,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": 500
                    }
                },
                timeout=150  # 2.5 minutes timeout
            )

            if response.status_code != 200:
                logger.error(f"Ollama returned status {response.status_code}: {response.text}")
                return {
                    "error": "Erreur Ollama",
                    "details": f"Status: {response.status_code}",
                    "score": 0,
                    "feedback": "Service d'analyse temporairement indisponible",
                    "status": "refusée"
                }

            output = ""
            async for chunk in response.aiter_lines():
                if chunk.strip():
                    try:
                        data = json.loads(chunk)
                        output += data.get("response", "")
                        
                        # Check if generation is done
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue

            logger.info(f"Ollama response received: {len(output)} characters")

            if not output:
                logger.warning("Empty response from Ollama")
                return {
                    "error": "Réponse vide d'Ollama",
                    "details": "No content generated",
                    "score": 0,
                    "feedback": "Impossible d'analyser la réponse",
                    "status": "refusée"
                }

            # Extract values from output
            score = None
            feedback = ""
            status = "refusée"
            status_color = "danger"

            lines = output.splitlines()
            for line in lines:
                line = line.strip()
                if "score" in line.lower() and score is None:
                    try:
                        # Extract numbers from the line
                        numbers = ''.join(filter(str.isdigit, line))
                        if numbers:
                            score = min(100, max(0, int(numbers)))  # Clamp between 0-100
                    except ValueError:
                        logger.warning(f"Could not parse score from line: {line}")
                        continue
                        
                elif "feedback" in line.lower() and not feedback:
                    parts = line.split(":", 1)
                    if len(parts) > 1:
                        feedback = parts[1].strip()
                        
                elif "statut" in line.lower():
                    parts = line.split(":", 1)
                    if len(parts) > 1:
                        status_text = parts[1].strip().lower()
                        if "validée" in status_text or "validé" in status_text:
                            status = "validée"
                        elif "retravailler" in status_text:
                            status = "à retravailler"
                        else:
                            status = "refusée"

            # Fallback score if not extracted
            if score is None:
                # Simple heuristic based on response length
                response_length = len(payload.responses.strip())
                if response_length < 10:
                    score = 20
                elif response_length < 50:
                    score = 40
                elif response_length < 100:
                    score = 60
                else:
                    score = 70
                
                logger.warning(f"Using fallback score: {score}")

            # Set status and color based on score
            if score >= 70:
                status = "validée"
                status_color = "success"
            elif score >= 50:
                status = "à retravailler"
                status_color = "warning"
            else:
                status = "refusée"
                status_color = "danger"

            # Ensure feedback is not empty
            if not feedback:
                if score >= 70:
                    feedback = "Réponse satisfaisante."
                elif score >= 50:
                    feedback = "Réponse correcte mais peut être améliorée."
                else:
                    feedback = "Réponse insuffisante, nécessite plus de détails."

            logger.info(f"Analysis complete - Score: {score}, Status: {status}")

            return {
                "score": score,
                "feedback": feedback,
                "status": status,
                "status_color": status_color
            }

    except httpx.TimeoutException:
        logger.error("Timeout while connecting to Ollama")
        return {
            "error": "Timeout avec Ollama",
            "details": "Le service d'analyse met trop de temps à répondre",
            "score": 0,
            "feedback": "Analyse temporairement indisponible (timeout)",
            "status": "refusée"
        }
    except httpx.ConnectError:
        logger.error("Cannot connect to Ollama")
        return {
            "error": "Connexion impossible avec Ollama",
            "details": "Le service Ollama n'est pas accessible",
            "score": 0,
            "feedback": "Service d'analyse temporairement indisponible",
            "status": "refusée"
        }
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return {
            "error": "Erreur avec Ollama",
            "details": str(e),
            "score": 0,
            "feedback": "Erreur technique lors de l'analyse",
            "status": "refusée"
        }

# Optional: Add CORS middleware if needed for cross-origin requests
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5005)