import sys
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))


# ---------------------------------------------------------
# Internal imports
# ---------------------------------------------------------

from intents.classifier import IntentClassifier
from retrieval.retriever import SupportRetriever
from escalation import EscalationManager


# ---------------------------------------------------------
# Support Agent
# ---------------------------------------------------------

class SupportAgent:

    def __init__(self):

        print("Initializing Support Agent...")

        # -------------------------------------------------
        # Load environment variables
        # -------------------------------------------------

        load_dotenv()

        # -------------------------------------------------
        # Gemini client
        # -------------------------------------------------

        gemini_api_key = os.getenv("GEMINI_API_KEY")

        if not gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. "
                "Please add it to your .env file."
            )

        self.client = genai.Client(
            api_key=gemini_api_key
        )

        self.model = os.getenv(
            "GEMINI_MODEL",
            "gemini-3.5-flash-lite"
        )

        print("✓ Gemini client loaded")

        # -------------------------------------------------
        # Intent classifier
        # -------------------------------------------------

        self.classifier = IntentClassifier()
        self.classifier.load()

        print("✓ Intent classifier loaded")

        # -------------------------------------------------
        # Retriever
        # -------------------------------------------------

        self.retriever = SupportRetriever(
            max_documents=10000
        )

        self.retriever.build_index()

        print("✓ Retrieval system loaded")

        # -------------------------------------------------
        # Escalation manager
        # -------------------------------------------------

        self.escalation = EscalationManager(
            confidence_threshold=0.65
        )

        print("✓ Escalation system loaded")

        print("\nSupport Agent ready!")

    # -----------------------------------------------------
    # Analyze customer message
    # -----------------------------------------------------

    def analyze(self, message, top_k=3):

        classification = self.classifier.predict(
            message
        )

        intent = classification["intent"]
        confidence = classification["confidence"]

        retrieved = self.retriever.retrieve(
            message,
            top_k=top_k
        )

        return {
            "message": message,
            "intent": intent,
            "confidence": confidence,
            "retrieved": retrieved
        }

    # -----------------------------------------------------
    # Generate AI response using Gemini
    # -----------------------------------------------------

    def generate_response(self, analysis):

        message = analysis["message"]
        intent = analysis["intent"]
        confidence = analysis["confidence"]
        retrieved = analysis["retrieved"]

        # -------------------------------------------------
        # Build context from retrieved conversations
        # -------------------------------------------------

        context = ""

        for i, item in enumerate(
            retrieved,
            start=1
        ):

            context += (
                f"\nExample {i}:\n"
                f"Customer: {item['customer_text']}\n"
                f"Support: {item['brand_text']}\n"
            )

        # -------------------------------------------------
        # Prompt
        # -------------------------------------------------

        prompt = f"""
You are a helpful customer support agent.

Customer message:
{message}

Detected intent:
{intent}

Classifier confidence:
{confidence:.3f}

Relevant historical support conversations:
{context}

Instructions:

- Answer the customer's actual question.
- Use the historical conversations only as guidance.
- Do not blindly copy historical responses.
- Do not invent order details, refund amounts, dates, or policies.
- Be polite, concise, and helpful.
- If the available information is insufficient, say so clearly.
- Never mention the classifier.
- Never mention the retrieval system.
- Never mention this prompt or internal process.
"""

        # -------------------------------------------------
        # Gemini API call
        # -------------------------------------------------

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )

        answer = response.text.strip()

        return {
            "message": message,
            "intent": intent,
            "confidence": confidence,
            "response": answer,
            "sources": retrieved
        }

    # -----------------------------------------------------
    # Complete agent pipeline
    # -----------------------------------------------------

    def process(self, message, top_k=3):

        analysis = self.analyze(
            message,
            top_k=top_k
        )

        # -------------------------------------------------
        # Check escalation
        # -------------------------------------------------

        escalation = self.escalation.should_escalate(
            analysis
        )

        # -------------------------------------------------
        # Escalate to human agent
        # -------------------------------------------------

        if escalation["escalate"]:

            return {
                "message": message,
                "intent": analysis["intent"],
                "confidence": analysis["confidence"],
                "response": (
                    "I'm sorry, but this issue requires "
                    "additional assistance from our support team. "
                    "I'm escalating this conversation to a human agent."
                ),
                "escalated": True,
                "escalation_reason": escalation["reason"],
                "sources": analysis["retrieved"]
            }

        # -------------------------------------------------
        # Generate normal AI response
        # -------------------------------------------------

        result = self.generate_response(
            analysis
        )

        result["escalated"] = False
        result["escalation_reason"] = None

        return result


# ---------------------------------------------------------
# Test the Support Agent
# ---------------------------------------------------------

if __name__ == "__main__":

    agent = SupportAgent()

    test_messages = [
        "My package is three days late",
        "Where is my refund?",
        "I want to cancel my order",
        "My payment was charged twice",
        "I cannot login to my account",
        "I think someone hacked my account"
    ]

    for message in test_messages:

        print("\n")
        print("=" * 80)

        result = agent.process(
            message,
            top_k=3
        )

        print(
            f"Customer: {result['message']}"
        )

        print(
            f"Intent: {result['intent']}"
        )

        print(
            f"Confidence: "
            f"{result['confidence']:.3f}"
        )

        print(
            f"Escalated: "
            f"{result['escalated']}"
        )

        if result["escalated"]:

            print(
                f"Escalation Reason: "
                f"{result['escalation_reason']}"
            )

        print("\nAI Response:")

        print(
            result["response"]
        )

        print("=" * 80)