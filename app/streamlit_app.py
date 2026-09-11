import sys
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

# ---------------------------------------------------------
# Internal import
# ---------------------------------------------------------

from agent.agent import SupportAgent


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Hiver AI Support Agent",
    page_icon="🤖",
    layout="wide",
)


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("🤖 Hiver AI Support Agent")
st.caption(
    "AI-powered customer support using intent classification, "
    "historical retrieval, Gemini and escalation."
)

st.divider()


# ---------------------------------------------------------
# Load agent once
# ---------------------------------------------------------

@st.cache_resource
def load_agent():
    return SupportAgent()


try:
    agent = load_agent()
except Exception as e:
    st.error("Failed to initialize the Support Agent.")
    st.exception(e)
    st.stop()


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:
    st.header("⚙️ System")

    st.success("Agent Online")

    st.markdown(
        """
        **Pipeline**

        1. Intent Classification
        2. Historical Retrieval
        3. Escalation Check
        4. Gemini Response
        """
    )

    st.divider()

    st.metric(
        "Intent Accuracy",
        "94.39%"
    )

    st.metric(
        "Macro F1",
        "94.46%"
    )


# ---------------------------------------------------------
# Customer message
# ---------------------------------------------------------

st.subheader("💬 Customer Message")

message = st.text_area(
    "Enter a customer support query:",
    placeholder="Example: My package is three days late",
    height=120,
)


# ---------------------------------------------------------
# Example queries
# ---------------------------------------------------------

st.markdown("**Try an example:**")

examples = [
    "My package is three days late",
    "Where is my refund?",
    "I want to cancel my order",
    "My payment was charged twice",
    "I cannot login to my account",
    "I think someone hacked my account",
]

cols = st.columns(3)

for i, example in enumerate(examples):
    if cols[i % 3].button(
        example,
        key=f"example_{i}",
        use_container_width=True,
    ):
        st.session_state["message"] = example


if "message" in st.session_state and not message:
    message = st.session_state["message"]


# ---------------------------------------------------------
# Process request
# ---------------------------------------------------------

if st.button(
    "🚀 Generate Support Response",
    type="primary",
    use_container_width=True,
):

    if not message.strip():
        st.warning("Please enter a customer message.")
        st.stop()

    with st.spinner("Analyzing customer message..."):

        try:
            result = agent.process(
                message.strip(),
                top_k=3,
            )

        except Exception as e:
            st.error("Something went wrong while processing the request.")
            st.exception(e)
            st.stop()


    # -----------------------------------------------------
    # Result summary
    # -----------------------------------------------------

    st.divider()

    st.subheader("📊 Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Detected Intent",
            result["intent"].replace("_", " ").title(),
        )

    with col2:
        st.metric(
            "Confidence",
            f"{result['confidence']:.1%}",
        )

    with col3:

        if result.get("escalated", False):
            st.metric(
                "Status",
                "Human Escalation",
            )
        else:
            st.metric(
                "Status",
                "Automated",
            )


    # -----------------------------------------------------
    # Escalation warning
    # -----------------------------------------------------

    if result.get("escalated", False):

        st.warning(
            f"⚠️ **Human escalation required**\n\n"
            f"Reason: {result.get('escalation_reason', 'Sensitive or low-confidence issue detected.')}"
        )

    else:

        st.success(
            "✅ This conversation can be handled automatically."
        )


    # -----------------------------------------------------
    # AI Response
    # -----------------------------------------------------

    st.subheader("🤖 AI Response")

    st.info(result["response"])


    # -----------------------------------------------------
    # Retrieved conversations
    # -----------------------------------------------------

    st.subheader("🔎 Retrieved Historical Cases")

    sources = result.get("sources", [])

    if not sources:

        st.caption(
            "No relevant historical conversations were retrieved."
        )

    else:

        for i, source in enumerate(sources, start=1):

            with st.expander(
                f"Historical Case {i}"
            ):

                similarity = source.get(
                    "similarity",
                    source.get("score", 0),
                )

                st.write(
                    f"**Similarity:** {float(similarity):.3f}"
                )

                st.markdown(
                    f"**Customer:** {source.get('customer_text', '')}"
                )

                st.markdown(
                    f"**Support:** {source.get('brand_text', '')}"
                )


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.divider()

st.caption(
    "Hiver AI Support Agent • "
    "TF-IDF + Logistic Regression + Retrieval + Gemini"
)
