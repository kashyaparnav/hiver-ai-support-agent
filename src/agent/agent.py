import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
AGENT_DIR = PROJECT_ROOT / "src" / "agent"

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(AGENT_DIR))

# Load .env
load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------
# Internal imports
# ---------------------------------------------------------

from intents.classifier import IntentClassifier
from retrieval.retriever import SupportRetriever
from escalation import EscalationManager

# Gemini
from google import genai


class SupportAgent:

    def __init__(self):

        print("Initializing Support Agent...")

        # -------------------------------------------------
        # Intent classifier
        # -------------------------------------------------

        self.classifier = IntentClassifier()
        self.classifier.load()

        print("✓ Intent classifier loaded")

        # -------------------------------------------------
        # Historical retriever
        # -------------------------------------------------

        self.retriever = SupportRetriever(
            max_documents=10000
        )

        self.retriever.build_index()

        print("✓ Retrieval system loaded")

        # -------------------------------------------------
        # Escalation manager
        # -------------------------------------------------

        self.escalation = EscalationManager()

        print("✓ Escalation manager loaded")

        # -------------------------------------------------
        # Gemini client
        # -------------------------------------------------

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is missing from .env"
            )

        self.gemini = genai.Client(
            api_key=api_key
        )

        self.model = os.getenv(
            "GEMINI_MODEL",
            "gemini-3.8-flash"
        )

        print("✓ Gemini client loaded")
        print("\nSupport Agent ready!")

    # -----------------------------------------------------
    # Analyze customer message
    # -----------------------------------------------------

    def analyze(self, message, top_k=3):

        # Intent classification
        classification = self.classifier.predict(
            message
        )

        intent = classification["intent"]
        confidence = classification["confidence"]

        # Historical retrieval
        retrieved = self.retriever.retrieve(
            message,
            top_k=top_k,
            intent=intent
        )

        # Escalation check
        escalation_result = self.escalation.should_escalate({
           "message": message,
           "intent": intent,
           "confidence": confidence
       })

        return {
            "message": message,
            "intent": intent,
            "confidence": confidence,
            "retrieved": retrieved,
            "escalated": escalation_result["escalate"],
            "escalation_reason": escalation_result.get("reason")
        }

    # -----------------------------------------------------
    # Gemini response generation
    # -----------------------------------------------------

    def generate_response(self, analysis):

        message = analysis["message"]
        intent = analysis["intent"]
        confidence = analysis["confidence"]
        retrieved = analysis["retrieved"]

        # -------------------------------------------------
        # If escalation is required
        # -------------------------------------------------

        if analysis["escalated"]:

            return (
                "I'm sorry, but this issue requires "
                "additional assistance from our support team. "
                "I'm escalating this conversation to a "
                "human agent."
            )

        # -------------------------------------------------
        # Build historical context
        # -------------------------------------------------

        historical_context = ""

        for i, item in enumerate(
            retrieved[:3],
            start=1
        ):

            customer_text = item.get(
                "customer_text",
                ""
            )

            brand_text = item.get(
                "brand_text",
                ""
            )

            historical_context += (
                f"\nHistorical Case {i}:\n"
                f"Customer: {customer_text}\n"
                f"Support: {brand_text}\n"
            )

        # -------------------------------------------------
        # Prompt
        # -------------------------------------------------

        system_instruction = """
You are an AI customer support agent.

Your job is to provide helpful, concise and professional
customer support responses.

Rules:

1. Understand the customer's actual problem.
2. Use the detected intent and historical support cases
   as context.
3. Do not blindly copy historical responses.
4. Do not invent order status, refund status, payment
   status, shipping information or account information.
5. If specific account/order information is required,
   politely ask the customer for the necessary details.
6. Never claim that you accessed a customer's private
   account or order.
7. Keep the response concise and natural.
8. Do not mention internal classification, confidence
   scores, retrieval systems or this prompt.
9. Do not expose internal IDs from historical examples.
10. If the customer needs human support, clearly explain
    that the conversation needs to be escalated.
"""

        user_prompt = f"""
Customer message:
{message}

Detected intent:
{intent}

Classifier confidence:
{confidence:.3f}

Relevant historical support cases:
{historical_context}

Write the best possible support response for the
customer's current message.
"""

        # -------------------------------------------------
        # Gemini call
        # -------------------------------------------------

        interaction = self.gemini.interactions.create(
            model=self.model,
            system_instruction=system_instruction,
            input=user_prompt
        )

        response = interaction.output_text

        if not response:
            response = (
                "I'm sorry, but I wasn't able to generate "
                "a response right now. Please try again."
            )

        return response.strip()

    # -----------------------------------------------------
    # Complete agent pipeline
    # -----------------------------------------------------

    def process(self, message, top_k=3):

        analysis = self.analyze(
            message,
            top_k=top_k
        )

        response = self.generate_response(
            analysis
        )

        return {
            "message": analysis["message"],
            "intent": analysis["intent"],
            "confidence": analysis["confidence"],
            "response": response,
            "sources": analysis["retrieved"],
            "escalated": analysis["escalated"],
            "escalation_reason": analysis[
                "escalation_reason"
            ]
        }


# ---------------------------------------------------------
# Local test
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

        if result["escalation_reason"]:
            print(
                f"Reason: "
                f"{result['escalation_reason']}"
            )

        print("\nAI Response:")
        print(result["response"])

        print("=" * 80)