import sys
from pathlib import Path
import io
import json

import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))


# ---------------------------------------------------------
# Internal imports
# ---------------------------------------------------------

from agent.agent import SupportAgent
from agent.tickets import TicketManager

# ---------------------------------------------------------
# AI Response Quality & Safety Helpers
# ---------------------------------------------------------

def validate_ai_response(result):
    """
    Lightweight safety/quality validation for the generated response.
    This is a guardrail, not a replacement for human review.
    """

    response = str(result.get("response", "")).strip()
    message = str(result.get("message", "")).strip()
    intent = str(result.get("intent", "")).strip()

    issues = []

    # Empty or obviously broken response.
    if not response:
        issues.append("Empty AI response")

    if len(response) < 10:
        issues.append("AI response is too short")

    # Avoid fabricated private/account/order facts.
    risky_patterns = [
        "your order number is",
        "your tracking number is",
        "your account balance is",
        "your refund has been processed",
        "i can see your account",
        "i checked your account",
        "i checked your order",
        "i accessed your account",
    ]

    response_lower = response.lower()

    for pattern in risky_patterns:
        if pattern in response_lower:
            issues.append(
                "Potentially fabricated private/account information"
            )
            break

    # Very lightweight intent relevance checks.
    intent_keywords = {
        "delivery_delay": [
            "delivery", "deliver", "package", "parcel",
            "late", "delay", "arrive", "shipping",
        ],
        "delivery_status": [
            "delivery", "deliver", "package", "parcel",
            "track", "tracking", "shipping", "status",
        ],
        "delivery_date": [
            "delivery", "deliver", "arrive", "date",
            "package", "shipping",
        ],
        "missing_package": [
            "package", "parcel", "delivery", "missing",
            "delivered", "not received",
        ],
        "refund_status": [
            "refund", "money", "reimburse", "credited",
        ],
        "return_issue": [
            "return", "returned", "send back",
        ],
        "order_cancellation": [
            "cancel", "cancellation", "order",
        ],
        "payment_issue": [
            "payment", "charged", "charge", "transaction",
            "card", "billing",
        ],
        "account_issue": [
            "account", "login", "log in", "password",
            "access",
        ],
        "prime_issue": [
            "prime", "membership", "subscription",
        ],
        "product_issue": [
            "product", "item", "device", "product issue",
        ],
    }

    keywords = intent_keywords.get(intent, [])

    if keywords:
        if not any(
            keyword in response_lower
            for keyword in keywords
        ):
            issues.append(
                "Response may not be aligned with detected intent"
            )

    passed = len(issues) == 0

    return {
        "passed": passed,
        "issues": issues,
    }


def safe_fallback_response(result):
    """Return a safe fallback when response validation fails."""

    intent = str(
        result.get("intent", "support_issue")
    ).replace("_", " ")

    return (
        "Thanks for reaching out. I understand this is related "
        f"to your {intent}. I don't want to guess about your "
        "specific account or order details. Please share the "
        "relevant order or case details with our support team "
        "so they can assist you accurately."
    )



# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Hiver AI Support Agent",
    page_icon="🤖",
    layout="wide",
)


# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------

if "latest_result" not in st.session_state:
    st.session_state.latest_result = None

if "latest_ticket" not in st.session_state:
    st.session_state.latest_ticket = None

if "ticket_manager" not in st.session_state:
    st.session_state.ticket_manager = TicketManager()

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "feedback" not in st.session_state:
    st.session_state.feedback = []


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


ticket_manager = st.session_state.ticket_manager


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
        5. Human Ticket Handoff
        6. Response Feedback
        7. Analytics Dashboard
        8. Response Quality & Safety
        9. Conversation Export
        10. Ticket Management
        11. Advanced Search & Filters
        12. Conversation History
        """
    )

    st.divider()

    st.metric("Intent Accuracy", "94.39%")
    st.metric("Macro F1", "94.46%")

    st.divider()

    st.metric(
        "Conversations",
        len(st.session_state.conversation_history),
    )


# ---------------------------------------------------------
# Customer message
# ---------------------------------------------------------

st.subheader("💬 Customer Message")

message = st.text_area(
    "Enter a customer support query:",
    value=st.session_state.get("message", ""),
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
        st.session_state["latest_result"] = None
        st.session_state["latest_ticket"] = None
        st.rerun()


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

    cleaned_message = message.strip()

    with st.spinner("Analyzing customer message..."):

        try:

            result = agent.process(
                cleaned_message,
                top_k=3,
            )

            # Validate the generated response before displaying it.
            quality_result = validate_ai_response(result)

            result["quality_passed"] = quality_result["passed"]
            result["quality_issues"] = quality_result["issues"]

            # Use a safe fallback if the generated response fails validation.
            if not quality_result["passed"]:
                result["response"] = safe_fallback_response(result)

            # Save current result
            st.session_state["latest_result"] = result

            # Save message
            st.session_state["message"] = cleaned_message

            # Reset previous ticket for new query
            st.session_state["latest_ticket"] = None

            # Add conversation to history
            st.session_state["conversation_history"].append(
                {
                    "customer": cleaned_message,
                    "intent": result["intent"],
                    "confidence": result["confidence"],
                    "response": result["response"],
                    "escalated": result.get("escalated", False),
                    "escalation_reason": result.get(
                        "escalation_reason"
                    ),
                    "quality_passed": result.get(
                        "quality_passed",
                        True,
                    ),
                    "quality_issues": result.get(
                        "quality_issues",
                        [],
                    ),
                }
            )

        except Exception as e:

            st.error(
                "Something went wrong while processing the request."
            )

            st.exception(e)
            st.stop()


# ---------------------------------------------------------
# Get latest result
# ---------------------------------------------------------

result = st.session_state.get("latest_result")


# ---------------------------------------------------------
# Display result
# ---------------------------------------------------------

if result is not None:

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
    # AI Response Quality & Safety
    # -----------------------------------------------------

    st.subheader("🛡️ Response Quality & Safety")

    quality_passed = result.get(
        "quality_passed",
        True,
    )

    quality_issues = result.get(
        "quality_issues",
        [],
    )

    if quality_passed:

        st.success(
            "✅ Quality Check Passed — response passed "
            "basic safety and intent-alignment checks."
        )

    else:

        st.warning(
            "⚠️ Review Recommended — the generated response "
            "did not pass all automated guardrails."
        )

        for issue in quality_issues:
            st.write(f"• {issue}")

        st.caption(
            "A safe fallback response was used instead of the "
            "potentially unsafe generated response."
        )


    # -----------------------------------------------------
    # Human Escalation
    # -----------------------------------------------------

    if result.get("escalated", False):

        st.warning(
            "⚠️ **Human escalation required**\n\n"
            f"Reason: {result.get('escalation_reason', 'Sensitive or low-confidence issue detected.')}"
        )

        st.markdown("### 🎫 Human Support Handoff")

        if st.session_state.get("latest_ticket") is None:

            if st.button(
                "🎫 Create Support Ticket",
                type="secondary",
                use_container_width=True,
            ):

                ticket = ticket_manager.create_ticket(
                    message=result["message"],
                    intent=result["intent"],
                    confidence=result["confidence"],
                    reason=result.get(
                        "escalation_reason",
                        "Sensitive or low-confidence issue detected.",
                    ),
                    priority="High",
                )

                st.session_state["latest_ticket"] = ticket
                st.rerun()


        ticket = st.session_state.get("latest_ticket")

        if ticket is not None:

            st.success(
                f"✅ Ticket created successfully: {ticket['ticket_id']}"
            )

            st.write("### 🎫 Ticket Details")

            ticket_col1, ticket_col2, ticket_col3 = st.columns(3)

            with ticket_col1:
                st.metric(
                    "Ticket ID",
                    ticket["ticket_id"],
                )

            with ticket_col2:
                st.metric(
                    "Priority",
                    ticket["priority"],
                )

            with ticket_col3:
                st.metric(
                    "Status",
                    ticket["status"],
                )

            st.write(
                f"**Assigned To:** {ticket['assigned_to']}"
            )

            st.write(
                f"**Created At:** {ticket['created_at']}"
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
    # Response Feedback
    # -----------------------------------------------------

    st.markdown("### 📝 Response Feedback")

    feedback_col1, feedback_col2 = st.columns(2)

    with feedback_col1:

        if st.button(
            "👍 Helpful",
            key="feedback_helpful",
            use_container_width=True,
        ):

            st.session_state.feedback.append(
                {
                    "message": result["message"],
                    "intent": result["intent"],
                    "feedback": "Helpful",
                }
            )

            st.success("Thanks! Your feedback has been recorded.")

    with feedback_col2:

        if st.button(
            "👎 Not Helpful",
            key="feedback_not_helpful",
            use_container_width=True,
        ):

            st.session_state.feedback.append(
                {
                    "message": result["message"],
                    "intent": result["intent"],
                    "feedback": "Not Helpful",
                }
            )

            st.info("Thanks! We'll use this feedback to improve the agent.")


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
# Analytics Dashboard
# ---------------------------------------------------------

st.divider()

st.subheader("📈 Analytics Dashboard")

history = st.session_state.get(
    "conversation_history",
    [],
)

tickets = ticket_manager.get_tickets()

total_conversations = len(history)

automated_count = sum(
    1
    for conversation in history
    if not conversation.get("escalated", False)
)

escalated_count = sum(
    1
    for conversation in history
    if conversation.get("escalated", False)
)

escalation_rate = (
    (escalated_count / total_conversations) * 100
    if total_conversations
    else 0
)

average_confidence = (
    sum(
        conversation.get("confidence", 0)
        for conversation in history
    ) / total_conversations
    if total_conversations
    else 0
)

metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.metric(
        "Total Conversations",
        total_conversations,
    )

with metric2:
    st.metric(
        "Automated Handled",
        automated_count,
    )

with metric3:
    st.metric(
        "Human Escalations",
        escalated_count,
    )

metric4, metric5, metric6 = st.columns(3)

with metric4:
    st.metric(
        "Escalation Rate",
        f"{escalation_rate:.1f}%",
    )

with metric5:
    st.metric(
        "Average Confidence",
        f"{average_confidence:.1%}",
    )

with metric6:
    st.metric(
        "Tickets Created",
        len(tickets),
    )


if history:

    st.markdown("### 🎯 Intent Distribution")

    intent_counts = {}

    for conversation in history:

        intent = conversation.get(
            "intent",
            "unknown",
        )

        intent_counts[intent] = (
            intent_counts.get(intent, 0) + 1
        )

    intent_data = sorted(
        intent_counts.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    for intent, count in intent_data:

        percentage = (
            count / total_conversations
        ) * 100

        st.write(
            f"**{intent.replace('_', ' ').title()}** — "
            f"{count} ({percentage:.1f}%)"
        )

        st.progress(
            percentage / 100
        )


    st.markdown("### 🧑‍💻 Support Handling")

    handling_col1, handling_col2 = st.columns(2)

    with handling_col1:

        st.markdown("**Automated vs Human**")

        st.write(
            f"🟢 Automated: **{automated_count}**"
        )

        st.write(
            f"🔴 Human Escalation: **{escalated_count}**"
        )

    with handling_col2:

        st.markdown("**Confidence Overview**")

        high_confidence = sum(
            1
            for conversation in history
            if conversation.get("confidence", 0) >= 0.8
        )

        medium_confidence = sum(
            1
            for conversation in history
            if 0.6 <= conversation.get("confidence", 0) < 0.8
        )

        low_confidence = sum(
            1
            for conversation in history
            if conversation.get("confidence", 0) < 0.6
        )

        st.write(
            f"🟢 High (≥80%): **{high_confidence}**"
        )

        st.write(
            f"🟡 Medium (60–79%): **{medium_confidence}**"
        )

        st.write(
            f"🔴 Low (<60%): **{low_confidence}**"
        )


if tickets:

    st.markdown("### 🎫 Recent Support Tickets")

    for ticket in reversed(tickets[-5:]):

        with st.expander(
            f"{ticket['ticket_id']} • "
            f"{ticket['priority']} • "
            f"{ticket['status']}"
        ):

            st.write(
                f"**Intent:** "
                f"{ticket['intent'].replace('_', ' ').title()}"
            )

            st.write(
                f"**Confidence:** "
                f"{ticket['confidence']:.2f}%"
            )

            st.write(
                f"**Reason:** {ticket['reason']}"
            )

            st.write(
                f"**Assigned To:** "
                f"{ticket['assigned_to']}"
            )

            st.write(
                f"**Created At:** "
                f"{ticket['created_at']}"
            )

else:

    st.caption(
        "No support tickets have been created in this session."
    )


st.markdown("### 🛡️ Quality & Safety")

quality_results = [
    conversation
    for conversation in history
    if "quality_passed" in conversation
]

quality_passed_count = sum(
    1
    for conversation in quality_results
    if conversation.get("quality_passed", False)
)

quality_review_count = (
    len(quality_results) - quality_passed_count
)

quality_pass_rate = (
    (quality_passed_count / len(quality_results)) * 100
    if quality_results
    else 0
)

quality_col1, quality_col2, quality_col3 = st.columns(3)

with quality_col1:
    st.metric(
        "Quality Checks",
        len(quality_results),
    )

with quality_col2:
    st.metric(
        "Passed",
        quality_passed_count,
    )

with quality_col3:
    st.metric(
        "Pass Rate",
        f"{quality_pass_rate:.1f}%",
    )

if quality_review_count:
    st.caption(
        f"Review recommended: {quality_review_count}"
    )
else:
    st.caption(
        "No automated quality issues detected in this session."
    )


st.markdown("### 👍 Response Feedback")

feedback_items = st.session_state.get(
    "feedback",
    [],
)

helpful_count = sum(
    1
    for item in feedback_items
    if item.get("feedback") == "Helpful"
)

not_helpful_count = sum(
    1
    for item in feedback_items
    if item.get("feedback") == "Not Helpful"
)

total_feedback = len(feedback_items)

positive_rate = (
    (helpful_count / total_feedback) * 100
    if total_feedback
    else 0
)

feedback_col1, feedback_col2, feedback_col3 = st.columns(3)

with feedback_col1:
    st.metric("Total Feedback", total_feedback)

with feedback_col2:
    st.metric("Helpful", helpful_count)

with feedback_col3:
    st.metric(
        "Positive Feedback Rate",
        f"{positive_rate:.1f}%",
    )

if total_feedback:
    st.caption(
        f"Not Helpful: {not_helpful_count}"
    )
else:
    st.caption(
        "No response feedback has been submitted in this session."
    )


st.caption(
    "Analytics are calculated from the current Streamlit session "
    "and are not persistent production analytics."
)


# ---------------------------------------------------------
# Conversation Export
# ---------------------------------------------------------

st.divider()

st.subheader("📥 Export Support Data")

history = st.session_state.get(
    "conversation_history",
    [],
)

tickets = ticket_manager.get_tickets()

feedback_items = st.session_state.get(
    "feedback",
    [],
)


# Build CSV-friendly conversation records.
conversation_export = []

for index, conversation in enumerate(history, start=1):

    conversation_export.append(
        {
            "conversation_id": index,
            "customer_message": conversation.get("customer", ""),
            "intent": conversation.get("intent", ""),
            "confidence": round(
                conversation.get("confidence", 0) * 100,
                2,
            ),
            "status": (
                "Human Escalation"
                if conversation.get("escalated", False)
                else "Automated"
            ),
            "escalation_reason": conversation.get(
                "escalation_reason",
                "",
            ),
            "ai_response": conversation.get(
                "response",
                "",
            ),
        }
    )


conversation_df = pd.DataFrame(
    conversation_export
)

if not conversation_df.empty:

    csv_buffer = io.StringIO()

    conversation_df.to_csv(
        csv_buffer,
        index=False,
    )

    st.download_button(
        label="📄 Download Conversation History (CSV)",
        data=csv_buffer.getvalue(),
        file_name="hiver_conversation_history.csv",
        mime="text/csv",
        use_container_width=True,
    )

else:

    st.caption(
        "No conversation history available for export."
    )


# Build complete JSON report.
report = {
    "project": "Hiver AI Support Agent",
    "total_conversations": len(history),
    "conversations": history,
    "tickets": tickets,
    "feedback": feedback_items,
}

json_data = json.dumps(
    report,
    indent=2,
    ensure_ascii=False,
)

st.download_button(
    label="📋 Download Full Session Report (JSON)",
    data=json_data,
    file_name="hiver_support_session_report.json",
    mime="application/json",
    use_container_width=True,
)

if tickets:

    ticket_df = pd.DataFrame(tickets)

    ticket_csv_buffer = io.StringIO()

    ticket_df.to_csv(
        ticket_csv_buffer,
        index=False,
    )

    st.download_button(
        label="🎫 Download Support Tickets (CSV)",
        data=ticket_csv_buffer.getvalue(),
        file_name="hiver_support_tickets.csv",
        mime="text/csv",
        use_container_width=True,
    )

else:

    st.caption(
        "No support tickets available for export."
    )


st.caption(
    "Exports contain data from the current Streamlit session only."
)


# ---------------------------------------------------------
# Support Ticket Management
# ---------------------------------------------------------

st.divider()

st.subheader("🎫 Support Ticket Management")

tickets = ticket_manager.get_tickets()

if not tickets:

    st.info(
        "No support tickets have been created in this session."
    )

else:

    open_tickets = sum(
        1
        for ticket in tickets
        if ticket.get("status") == "Open"
    )

    in_progress_tickets = sum(
        1
        for ticket in tickets
        if ticket.get("status") == "In Progress"
    )

    resolved_tickets = sum(
        1
        for ticket in tickets
        if ticket.get("status") == "Resolved"
    )

    ticket_metric1, ticket_metric2, ticket_metric3, ticket_metric4 = st.columns(4)

    with ticket_metric1:
        st.metric("Total Tickets", len(tickets))

    with ticket_metric2:
        st.metric("Open", open_tickets)

    with ticket_metric3:
        st.metric("In Progress", in_progress_tickets)

    with ticket_metric4:
        st.metric("Resolved", resolved_tickets)

    st.markdown("### 📋 Ticket Queue")

    for ticket in reversed(tickets):

        ticket_id = ticket["ticket_id"]

        with st.expander(
            f"🎫 {ticket_id} • "
            f"{ticket['priority']} • "
            f"{ticket['status']}"
        ):

            detail_col1, detail_col2 = st.columns(2)

            with detail_col1:

                st.write(
                    f"**Intent:** "
                    f"{ticket['intent'].replace('_', ' ').title()}"
                )

                st.write(
                    f"**Confidence:** "
                    f"{ticket['confidence']:.2f}%"
                )

                st.write(
                    f"**Priority:** {ticket['priority']}"
                )

                st.write(
                    f"**Assigned To:** "
                    f"{ticket['assigned_to']}"
                )

            with detail_col2:

                st.write(
                    f"**Created At:** "
                    f"{ticket['created_at']}"
                )

                st.write(
                    f"**Reason:** {ticket['reason']}"
                )

                st.write(
                    f"**Customer Message:** "
                    f"{ticket['message']}"
                )

            st.markdown("**Update Ticket Status**")

            current_status = ticket.get(
                "status",
                "Open",
            )

            status_options = [
                "Open",
                "In Progress",
                "Resolved",
            ]

            selected_status = st.selectbox(
                "Status",
                status_options,
                index=status_options.index(current_status)
                if current_status in status_options
                else 0,
                key=f"status_{ticket_id}",
            )

            if st.button(
                "🔄 Update Status",
                key=f"update_{ticket_id}",
                use_container_width=True,
            ):

                updated_ticket = ticket_manager.update_status(
                    ticket_id,
                    selected_status,
                )

                if updated_ticket:

                    st.success(
                        f"Ticket {ticket_id} updated to "
                        f"**{selected_status}**."
                    )

                    st.rerun()

                else:

                    st.error(
                        "Ticket could not be updated."
                    )


# ---------------------------------------------------------
# Advanced Conversation Search & Filters
# ---------------------------------------------------------

st.divider()

st.subheader("🔍 Search & Filter Conversations")

history = st.session_state.get(
    "conversation_history",
    [],
)

if not history:

    st.caption(
        "No conversations available to search yet."
    )

else:

    search_col, intent_col, status_col = st.columns(3)

    with search_col:

        search_text = st.text_input(
            "Search message",
            placeholder="Search customer message...",
            key="conversation_search",
        )

    with intent_col:

        all_intents = sorted(
            {
                conversation.get(
                    "intent",
                    "unknown",
                )
                for conversation in history
            }
        )

        intent_options = ["All"] + all_intents

        selected_intent = st.selectbox(
            "Filter by intent",
            intent_options,
            key="conversation_intent_filter",
        )

    with status_col:

        selected_status = st.selectbox(
            "Filter by status",
            [
                "All",
                "Automated",
                "Human Escalation",
            ],
            key="conversation_status_filter",
        )

    confidence_col, feedback_col = st.columns(2)

    with confidence_col:

        confidence_filter = st.selectbox(
            "Filter by confidence",
            [
                "All",
                "High (≥80%)",
                "Medium (60–79%)",
                "Low (<60%)",
            ],
            key="conversation_confidence_filter",
        )

    with feedback_col:

        feedback_filter = st.selectbox(
            "Filter by feedback",
            [
                "All",
                "Helpful",
                "Not Helpful",
                "No Feedback",
            ],
            key="conversation_feedback_filter",
        )


    # Build feedback lookup by customer message.
    feedback_lookup = {}

    for item in st.session_state.get("feedback", []):

        feedback_lookup[item.get("message", "")] = item.get(
            "feedback"
        )


    filtered_history = []

    for conversation in history:

        customer_message = conversation.get(
            "customer",
            "",
        )

        intent = conversation.get(
            "intent",
            "unknown",
        )

        confidence = conversation.get(
            "confidence",
            0,
        )

        escalated = conversation.get(
            "escalated",
            False,
        )

        feedback = feedback_lookup.get(
            customer_message
        )

        # Search filter
        if search_text.strip():

            query = search_text.strip().lower()

            if query not in customer_message.lower():
                continue

        # Intent filter
        if (
            selected_intent != "All"
            and intent != selected_intent
        ):
            continue

        # Status filter
        if selected_status == "Automated" and escalated:
            continue

        if (
            selected_status == "Human Escalation"
            and not escalated
        ):
            continue

        # Confidence filter
        if confidence_filter == "High (≥80%)":
            if confidence < 0.80:
                continue

        elif confidence_filter == "Medium (60–79%)":

            if not (
                0.60 <= confidence < 0.80
            ):
                continue

        elif confidence_filter == "Low (<60%)":

            if confidence >= 0.60:
                continue

        # Feedback filter
        if (
            feedback_filter != "All"
            and feedback_filter != "No Feedback"
            and feedback != feedback_filter
        ):
            continue

        if (
            feedback_filter == "No Feedback"
            and feedback is not None
        ):
            continue

        filtered_history.append(
            conversation
        )


    st.markdown(
        f"**Showing {len(filtered_history)} of "
        f"{len(history)} conversations**"
    )


    if not filtered_history:

        st.warning(
            "No conversations match the selected filters."
        )

    else:

        for index, conversation in enumerate(
            reversed(filtered_history),
            start=1,
        ):

            intent_label = (
                conversation.get(
                    "intent",
                    "unknown",
                )
                .replace("_", " ")
                .title()
            )

            status_label = (
                "🔴 Human Escalation"
                if conversation.get(
                    "escalated",
                    False,
                )
                else "🟢 Automated"
            )

            with st.expander(
                f"{intent_label} • "
                f"{conversation.get('confidence', 0):.1%} • "
                f"{status_label}"
            ):

                st.markdown(
                    f"**👤 Customer:** "
                    f"{conversation.get('customer', '')}"
                )

                st.markdown(
                    f"**🎯 Intent:** {intent_label}"
                )

                st.markdown(
                    f"**📊 Confidence:** "
                    f"{conversation.get('confidence', 0):.1%}"
                )

                st.markdown(
                    f"**Status:** {status_label}"
                )

                if conversation.get("escalated", False):

                    st.markdown(
                        f"**⚠️ Escalation Reason:** "
                        f"{conversation.get('escalation_reason', 'N/A')}"
                    )

                current_feedback = feedback_lookup.get(
                    conversation.get("customer", "")
                )

                st.markdown(
                    f"**👍 Feedback:** "
                    f"{current_feedback or 'No Feedback'}"
                )

                st.markdown(
                    f"**🤖 AI Response:** "
                    f"{conversation.get('response', '')}"
                )


# ---------------------------------------------------------
# Conversation History
# ---------------------------------------------------------

st.divider()

st.subheader("🕘 Conversation History")

history = st.session_state.get(
    "conversation_history",
    [],
)

if not history:

    st.caption(
        "No previous conversations in this session."
    )

else:

    for index, conversation in enumerate(
        reversed(history),
        start=1,
    ):

        conversation_number = len(history) - index + 1

        intent_label = (
            conversation["intent"]
            .replace("_", " ")
            .title()
        )

        status_label = (
            "🔴 Human Escalation"
            if conversation["escalated"]
            else "🟢 Automated"
        )

        with st.expander(
            f"Conversation {conversation_number} • {intent_label}"
        ):

            st.markdown(
                f"**👤 Customer:** {conversation['customer']}"
            )

            st.markdown(
                f"**🎯 Intent:** {intent_label}"
            )

            st.markdown(
                f"**📊 Confidence:** "
                f"{conversation['confidence']:.1%}"
            )

            st.markdown(
                f"**Status:** {status_label}"
            )

            quality_label = (
                "✅ Passed"
                if conversation.get("quality_passed", True)
                else "⚠️ Review Recommended"
            )

            st.markdown(
                f"**🛡️ Response Quality:** {quality_label}"
            )

            if not conversation.get("quality_passed", True):
                for issue in conversation.get("quality_issues", []):
                    st.caption(f"• {issue}")

            if conversation["escalated"]:

                st.markdown(
                    f"**⚠️ Escalation Reason:** "
                    f"{conversation.get('escalation_reason', 'N/A')}"
                )

            st.markdown(
                f"**🤖 AI Response:** {conversation['response']}"
            )


    if st.button(
        "🗑️ Clear Conversation History",
        use_container_width=True,
    ):

        st.session_state.conversation_history = []
        st.session_state.feedback = []
        st.session_state.latest_result = None
        st.session_state.latest_ticket = None
        st.rerun()


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.divider()

st.caption(
    "Hiver AI Support Agent • "
    "TF-IDF + Logistic Regression + Retrieval + Gemini"
)
