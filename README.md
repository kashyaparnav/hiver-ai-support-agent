# 🤖 Hiver AI Support Agent

An AI-powered customer support agent that combines **intent classification, historical conversation retrieval, multilingual support, Gemini-based response generation, and human escalation** to automate common customer-support queries.

The project is designed as an end-to-end support automation system that can classify incoming customer requests, retrieve relevant historical cases, generate grounded responses, and hand off sensitive or uncertain issues to human support.

---

## 🚀 Overview

This project demonstrates a production-oriented AI customer-support pipeline built using the **Customer Support on Twitter** dataset.

The system can:

- Understand customer intent
- Support English, Hindi, Hinglish, Bengali, Tamil and other languages
- Normalize multilingual customer messages for intent classification
- Retrieve similar historical support conversations
- Generate context-aware AI responses using Gemini
- Detect sensitive and risky support issues
- Escalate low-confidence or sensitive conversations
- Automatically create human-support tickets
- Provide an interactive Streamlit dashboard
- Provide a standalone embeddable customer-support widget
- Expose the support agent through a FastAPI backend
- Collect response feedback and conversation analytics

---

## 🧠 System Architecture

```text
                         Customer Message
                                │
                                ▼
                    ┌─────────────────────┐
                    │ Multilingual        │
                    │ Message Handling    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Intent Classification│
                    │ TF-IDF + Logistic    │
                    │ Regression           │
                    └──────────┬──────────┘
                               │
                               ▼
                       Intent + Confidence
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Historical Retrieval│
                    │ TF-IDF + Cosine     │
                    │ Similarity          │
                    └──────────┬──────────┘
                               │
                               ▼
                       Escalation Check
                         │           │
              Sensitive/Low          │ Safe & Confident
              Confidence             │
                   │                 │
                   ▼                 ▼
          ┌────────────────┐   ┌───────────────┐
          │ Human Support  │   │ Gemini LLM    │
          │ Escalation     │   │ Response      │
          └───────┬────────┘   └───────┬───────┘
                  │                    │
                  ▼                    ▼
           Ticket Creation       Final Response
                  │                    │
                  └──────────┬─────────┘
                             ▼
                       Customer / Widget
🏗️ Key Components
1. Intent Classification

The intent classifier uses:

TF-IDF
Logistic Regression
Confidence scoring

The current taxonomy contains 11 support intents:

account_issue
delivery_date
delivery_delay
delivery_status
missing_package
order_cancellation
payment_issue
prime_issue
product_issue
refund_status
return_issue

The classifier is trained on prepared AmazonHelp customer-support examples.

2. Multilingual Support

The agent supports multilingual customer messages.

Tested languages include:

English
Hindi
Hinglish
Bengali
Tamil

The system can process messages such as:

Mera refund abhi tak nahi aaya.
আমার রিফান্ড এখনও আসেনি, দয়া করে সাহায্য করুন।
எனது பணத்தைத் திரும்பப் பெறவில்லை, தயவுசெய்து உதவுங்கள்.

The agent can understand the customer's intent and generate a response in the customer's language when supported by the generative model.

Multilingual Pipeline
Customer Message
       ↓
Language Normalization
       ↓
Intent Classification
       ↓
Historical Retrieval
       ↓
Escalation Check
       ↓
Gemini Response
       ↓
Response in Customer Language
3. Historical Retrieval

Historical customer-support conversations are retrieved using:

TF-IDF
Unigram + bigram features
Cosine similarity
Intent-aware retrieval

The current development retrieval index contains 10,000 conversations.

Retrieved historical examples are provided as context to help the response generator produce relevant and grounded answers.

The system is explicitly instructed not to blindly copy historical responses.

4. Gemini Response Generation

Gemini is used for final response generation.

The response-generation stage receives:

Customer message
Detected intent
Classifier confidence
Retrieved historical support examples

The model is instructed to:

Stay grounded in available support context
Avoid inventing account/order information
Avoid claiming access to private customer systems
Avoid blindly copying historical responses
Ask for relevant information when required
Escalate when the issue requires human assistance
5. Escalation & Human Handoff

The agent can escalate conversations when:

Intent classification confidence is below the configured threshold
Sensitive keywords are detected
Potential account-security issues are detected
The request requires human assistance

Example:

Customer:
I think someone hacked my account and someone made an unauthorized payment.

Intent:
account_issue

Status:
Human Escalation

Reason:
Sensitive issue detected
6. Automatic Ticket Creation

When a conversation is escalated, the system automatically creates a support ticket.

Example:

Ticket ID: HVR-5FA5BB
Priority: High
Status: Open
Assigned To: Human Support Team

This creates a simple human-in-the-loop handoff mechanism.

📊 Evaluation
Intent Classification

The current classifier was evaluated on 660 test examples.

Metric	Score
Accuracy	94.39%
Macro F1	94.46%
Weighted F1	94.46%
Baseline Comparison
Approach	Accuracy	Macro F1
Majority Class	9.09%	1.52%
Simple TF-IDF	93.64%	93.71%
Trained Classifier	94.39%	94.46%
Retrieval

Average top-1 retrieval similarity on the evaluation sample:

0.4043
Important Evaluation Caveat

The current intent dataset contains weak/silver labels generated during dataset preparation rather than a fully human-annotated ground-truth benchmark.

Therefore, the reported 94.39% accuracy and 94.46% Macro F1 should not be interpreted as validated production performance or human-agreement performance.

A human-reviewed golden set is planned as the next evaluation improvement.

🔬 Failure Analysis

Several failure modes were identified during development.

1. Multilingual Classification

The original classifier was trained primarily on English support examples.

Without multilingual normalization, messages in languages such as Bengali or Tamil could receive low classification confidence.

The multilingual normalization layer improves this behavior by converting customer messages into a form that can be processed by the existing classifier.

2. Similar Intent Boundaries

Some intents have overlapping language.

For example:

delivery_date
delivery_delay
delivery_status

can contain similar vocabulary.

This can cause confusion when the customer message does not clearly specify whether they are asking for an expected date, reporting a delay, or requesting current delivery status.

3. Historical Retrieval

TF-IDF retrieval is primarily lexical.

Semantically similar messages using different vocabulary may receive lower similarity scores.

This is one reason semantic embeddings or hybrid retrieval are planned for future versions.

4. Real-Time Account Information

The agent does not directly access:

Customer accounts
Orders
Payment systems
Shipping systems
Refund systems

Therefore, it cannot independently verify real-time order or refund information.

Instead, it asks customers for relevant information or escalates when human assistance is required.

🖥️ Streamlit Dashboard

The project includes an interactive Streamlit application.

The dashboard provides:

Customer message input
Intent classification
Confidence score
Automation/escalation status
AI-generated response
Historical retrieved cases
Response feedback
Conversation history
Analytics
Response quality and safety information
Conversation export
Ticket management
Advanced search and filters

Run:

streamlit run app/streamlit_app.py
💬 Embeddable AI Support Widget

The project also includes a standalone customer-support chatbot widget.

The widget communicates with the FastAPI backend and can be embedded into another website using a single script.

<script src="https://hiver-ai-support-widget.onrender.com/widget.js"></script>

This allows a company website to add the AI support interface without rebuilding the chatbot UI.

Widget Flow
Website
   │
   ▼
AI Support Widget
   │
   ▼
FastAPI Backend
   │
   ▼
Support Agent
   │
   ├── Intent Classification
   ├── Retrieval
   ├── Escalation
   └── Gemini
   │
   ▼
AI Response / Human Ticket
🔌 FastAPI Backend

The support agent is exposed through a FastAPI API.

Health Check
GET /api/health
Chat Endpoint
POST /api/chat

Example request:

{
  "message": "My package is three days late",
  "top_k": 3
}

Example response structure:

{
  "success": true,
  "message": "My package is three days late",
  "intent": "delivery_delay",
  "confidence": 0.94,
  "response": "...",
  "sources": [],
  "escalated": false,
  "escalation_reason": null,
  "ticket": null
}

For escalated conversations, the response can include a generated ticket.

🌐 Live Demo
Embeddable Support Widget

https://hiver-ai-support-widget.onrender.com

FastAPI Backend

https://hiver-ai-support-agent-r2d7.onrender.com

Note: The backend is hosted on a free Render instance, so the first request after inactivity may take longer because the service can spin down.

💬 Example Queries
Delivery Issue
My package is three days late

Expected:

Intent: delivery_delay
Status: Automated
Refund
Where is my refund?

Expected:

Intent: refund_status
Status: Automated
Hindi / Hinglish
Mera refund abhi tak nahi aaya, please help karo.

Expected:

Intent: refund_status
Status: Automated
Response: Hindi/Hinglish
Bengali
আমার রিফান্ড এখনও আসেনি, দয়া করে সাহায্য করুন।

Expected:

Intent: refund_status
Status: Automated
Response: Bengali
Tamil
எனது பணத்தைத் திரும்பப் பெறவில்லை, தயவுசெய்து உதவுங்கள்.

Expected:

Intent: refund_status
Status: Automated
Response: Tamil
Cancellation
I want to cancel my order

Expected:

Intent: order_cancellation
Status: Automated
Payment
My payment was charged twice

Expected:

Intent: payment_issue
Sensitive Issue
I think someone hacked my account

Expected:

Intent: account_issue
Status: Human Escalation
📁 Project Structure
hiver-ai-support-agent/
│
├── api/
│   └── main.py
│
├── app/
│   └── streamlit_app.py
│
├── widget/
│   ├── index.html
│   ├── widget.js
│   └── widget.css
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── golden/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_intent_discovery.ipynb
│   └── 03_evaluation.ipynb
│
├── reports/
│   └── report.md
│
├── src/
│   ├── agent/
│   │   ├── agent.py
│   │   ├── escalation.py
│   │   ├── pipeline.py
│   │   ├── prompts.py
│   │   └── tickets.py
│   │
│   ├── data/
│   │   ├── loader.py
│   │   ├── cleaner.py
│   │   └── conversations.py
│   │
│   ├── evaluation/
│   │   ├── baselines.py
│   │   ├── evaluate.py
│   │   ├── judge.py
│   │   └── metrics.py
│   │
│   ├── intents/
│   │   ├── taxonomy.py
│   │   ├── classifier.py
│   │   ├── prompts.py
│   │   ├── build_golden.py
│   │   ├── review_golden.py
│   │   ├── prepare_dataset.py
│   │   └── train_classifier.py
│   │
│   └── retrieval/
│       └── retriever.py
│
├── DECISIONS.md
├── LICENSE
├── README.md
├── requirements.txt
└── .env.example
⚙️ Installation
1. Clone the repository
git clone https://github.com/kashyaparnav/hiver-ai-support-agent.git
cd hiver-ai-support-agent
2. Create a virtual environment
Windows
python -m venv .venv
.venv\Scripts\activate
macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
3. Install dependencies
pip install -r requirements.txt
4. Configure environment variables

Create a .env file based on .env.example.

GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-3.5-flash-lite

Never commit .env or expose API keys publicly.

▶️ Running the Project
Run Streamlit Dashboard
streamlit run app/streamlit_app.py
Run FastAPI Backend
uvicorn api.main:app --reload

API documentation:

http://127.0.0.1:8000/docs
Run the Support Agent Directly
python src/agent/agent.py
Run Intent Evaluation
python src/evaluation/metrics.py
Run Baseline Comparison
python src/evaluation/baselines.py
Run Evaluation Pipeline
python src/evaluation/evaluate.py
🛠️ Tech Stack
Category	Technology
Language	Python
Machine Learning	Scikit-learn
Classification	TF-IDF + Logistic Regression
Retrieval	TF-IDF + Cosine Similarity
Generative AI	Google Gemini
API	FastAPI
UI	Streamlit
Chat Widget	HTML + CSS + JavaScript
Data Processing	Pandas, NumPy
Environment	python-dotenv
Version Control	Git + GitHub
Backend Deployment	Render
Widget Deployment	Render Static Site
⚠️ Limitations

The current implementation has several limitations:

Intent labels are weak/silver labels rather than fully human-annotated labels.
The classifier benchmark is not a production-grade human-reviewed evaluation.
TF-IDF retrieval depends primarily on lexical similarity.
The local retrieval index is limited to 10,000 conversations.
Historical support responses may contain outdated information.
The agent does not directly access real order, payment, shipping, refund, or account systems.
Real-time customer-specific actions are therefore not supported.
The current response-quality judge is an offline heuristic evaluation rather than a validated LLM-as-judge benchmark with human agreement.
Gemini API availability and rate limits can affect response latency.
The free Render deployment can introduce cold-start latency after inactivity.
🔮 Future Improvements
Evaluation
Build a fully human-reviewed 150–250 example golden set
Measure human agreement
Add a validated LLM-as-judge evaluation
Evaluate response quality, groundedness, safety and helpfulness
Add confidence calibration
Perform systematic failure analysis
Retrieval
Semantic embedding-based retrieval
Hybrid lexical + semantic search
Vector database integration
Metadata-aware retrieval
Better intent-conditioned retrieval
Agent
Better multilingual classification
More robust language detection
Conversation memory
Tool calling
Real-time order/shipping/refund integrations
Human feedback loop
Production
Authentication
Rate limiting
Observability and monitoring
Persistent ticket storage
Database-backed conversation history
Production-grade deployment
Company-specific knowledge bases
Per-company support policies and retrieval indexes
📌 One-Week Next Steps

If continuing development for another week, the highest-priority improvements would be:

Day 1–2

Create and manually review a 150–250 example golden evaluation set.

Day 3

Add LLM-as-judge evaluation with explicit criteria for:

Correctness
Groundedness
Helpfulness
Safety
Day 4

Measure agreement between human reviewers and the automated judge.

Day 5

Replace TF-IDF retrieval with semantic embeddings and compare retrieval quality.

Day 6

Add better monitoring, structured logs and persistent ticket storage.

Day 7

Run the complete evaluation again and document failure cases and improvements.

📚 Dataset

The project uses the Customer Support on Twitter dataset.

The development pipeline reconstructs customer/company conversation pairs and focuses on AmazonHelp conversations for the support-agent use case.

The full raw dataset is not required for deployment; processed subsets and derived artifacts are used for experimentation and evaluation.

🔐 Safety & Privacy

The agent is designed not to invent private customer information.

It does not claim to have access to:

Account credentials
Payment details
Order databases
Shipping systems
Private customer records

Sensitive issues such as suspected account compromise can be escalated to human support.

API keys and other secrets should always be stored in environment variables or deployment secrets and must never be committed to GitHub.

📌 Conclusion

This project demonstrates an end-to-end AI customer-support architecture combining:

Classical machine learning
Intent classification
Information retrieval
Multilingual message handling
Generative AI
Human escalation
Automatic ticket creation
FastAPI
Streamlit
An embeddable web widget

The current classifier achieves:

94.39% Accuracy
94.46% Macro F1
94.46% Weighted F1

on the current weakly supervised evaluation set.

These numbers provide a useful development benchmark, but should not be interpreted as validated production performance until a human-reviewed golden set and human-agreement evaluation are completed.

The system is currently deployed with a live FastAPI backend and an embeddable customer-support widget, demonstrating the complete path from customer message to automated response or human handoff.