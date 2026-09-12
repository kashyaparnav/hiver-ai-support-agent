import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
AGENT_DIR = PROJECT_ROOT / "src" / "agent"

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(AGENT_DIR))

from agent import SupportAgent
from tickets import TicketManager
agent = SupportAgent()
ticket_manager = TicketManager()
# ---------------------------------------------------------
# Import existing AI Support Agent
# ---------------------------------------------------------


# ---------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------

app = FastAPI(
    title="Hiver AI Support Agent API",
    description="API for the Hiver AI customer support agent",
    version="1.0.0",
)


# ---------------------------------------------------------
# CORS
# ---------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Load Agent
# ---------------------------------------------------------

agent = SupportAgent()


# ---------------------------------------------------------
# Request Schema
# ---------------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    top_k: int = 3


# ---------------------------------------------------------
# Health Check
# ---------------------------------------------------------

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Hiver AI Support Agent API",
    }


@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "agent": "online",
    }


# ---------------------------------------------------------
# Chat Endpoint
# ---------------------------------------------------------

@app.post("/api/chat")
def chat(request: ChatRequest):

    message = request.message.strip()

    if not message:
        return {
            "success": False,
            "error": "Message cannot be empty.",
        }

    try:
        result = agent.process(
            message,
            top_k=request.top_k,
        )
        ticket = None

        if result.get("escalated"):
            ticket = ticket_manager.create_ticket(
              message=message,
              intent=result.get("intent"),
              confidence=result.get("confidence", 0),
              reason=result.get("escalation_reason") or "Human escalation required",
        )
        return {
            "success": True,
            "message": result.get("message"),
            "intent": result.get("intent"),
            "confidence": result.get("confidence"),
            "response": result.get("response"),
            "sources": result.get("sources", []),
            "escalated": result.get("escalated", False),
            "escalation_reason": result.get("escalation_reason"),
            "ticket": ticket,
        }


    except Exception as e:

        return {
            "success": False,
            "error": str(e),
        }