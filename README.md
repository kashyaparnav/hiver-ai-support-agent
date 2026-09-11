README.md ka pura existing content delete karke ye paste karo:

# Hiver AI Support Agent

An AI-powered customer support agent that classifies customer queries, retrieves relevant historical support conversations, generates grounded responses using Gemini, and escalates sensitive or low-confidence cases to human support.

---

## 🚀 Overview

The system combines:

- Machine Learning based intent classification
- TF-IDF based information retrieval
- Gemini-powered response generation
- Confidence-based escalation
- Sensitive issue detection
- Offline evaluation and benchmarking

### Architecture

```text
Customer Message
       │
       ▼
┌─────────────────────┐
│ Intent Classifier   │
│ TF-IDF + Logistic   │
│ Regression          │
└──────────┬──────────┘
           │
           ▼
   Intent + Confidence
           │
           ▼
┌─────────────────────┐
│ Historical Retriever│
│ TF-IDF + Cosine     │
│ Similarity          │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Escalation Manager  │
└──────────┬──────────┘
           │
      ┌────┴─────┐
      │          │
      ▼          ▼
   Normal     Sensitive /
   Request    Low Confidence
      │          │
      ▼          ▼
   Gemini     Human Agent
   Response   Escalation
📊 Dataset

This project uses the Customer Support on Twitter dataset.

The raw dataset contains:

2,811,774 tweets

After processing and reconstructing AmazonHelp customer-support interactions:

168,814 conversation pairs

Each conversation contains:

customer_text
brand_text
🧠 Intent Taxonomy

The system currently supports 11 customer-support intents:

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
🏗️ Project Structure
hiver-ai-support-agent/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   │   └── twcs.csv
│   ├── processed/
│   │   ├── amazonhelp_conversations.csv
│   │   ├── models/
│   │   ├── evaluation_results.json
│   │   └── response_quality_results.json
│   └── golden/
│       ├── amazonhelp_golden_candidates.csv
│       ├── train.csv
│       ├── validation.csv
│       └── test.csv
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_intent_discovery.ipynb
│   └── 03_evaluation.ipynb
│
├── src/
│   ├── agent/
│   │   ├── agent.py
│   │   ├── escalation.py
│   │   ├── pipeline.py
│   │   └── prompts.py
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
├── reports/
│   └── report.md
│
├── README.md
├── DECISIONS.md
├── requirements.txt
└── .env.example
⚙️ Tech Stack
Machine Learning
Python
Pandas
NumPy
Scikit-learn
TF-IDF
Logistic Regression
Cosine Similarity
Generative AI
Google Gemini API
google-genai
Data & Evaluation
Customer Support on Twitter dataset
Weak supervision
Classification metrics
Baseline benchmarking
Retrieval evaluation
Rule-based response quality evaluation
Development
VS Code
Python virtual environment
Jupyter Notebooks
Git
🔄 Data Pipeline
Raw Twitter Dataset
        ↓
Data Loading
        ↓
Conversation Reconstruction
        ↓
AmazonHelp Conversations
        ↓
Intent Discovery
        ↓
Weakly-Supervised Labels
        ↓
Train / Validation / Test
        ↓
Intent Classifier
🎯 Intent Classification

The classifier uses:

Customer Text
     ↓
TF-IDF Vectorization
     ↓
Logistic Regression
     ↓
Intent + Confidence

The trained artifacts are stored in:

data/processed/models/

Files:

intent_vectorizer.joblib
intent_classifier.joblib
🔎 Retrieval

The retrieval system searches historical support conversations using TF-IDF and cosine similarity.

For local development, the retriever indexes:

10,000 conversations

Example:

Customer:
My package is three days late

        ↓

Retriever

        ↓

Similar historical customer-support conversations

These retrieved conversations are provided as context to the response-generation model.

🤖 AI Response Generation

Gemini generates the final response using:

Customer message
Detected intent
Classifier confidence
Retrieved historical conversations

The model is instructed to:

Answer the actual customer question.
Use historical conversations as guidance.
Avoid blindly copying previous responses.
Avoid inventing order details.
Avoid inventing refund amounts.
Avoid inventing dates or policies.
Be concise and helpful.
Clearly state when information is insufficient.
🛡️ Escalation

The agent includes an escalation layer.

A conversation can be escalated when:

Classifier confidence is below 0.65
A sensitive issue is detected

Example:

Customer:
I think someone hacked my account

Intent:
account_issue

Escalated:
True

Reason:
Sensitive issue detected: hacked

This prevents potentially sensitive cases from being handled entirely through automated responses.

📈 Evaluation

The classifier was evaluated on:

660 test examples

Results
Metric	Score
Accuracy	94.39%
Macro F1	94.46%
Weighted F1	94.46%
Retrieval

Evaluation was performed on:

100 test examples

Average top-1 retrieval similarity:

0.4043

🧪 Baseline Comparison
Majority Class
Metric	Score
Accuracy	9.09%
Macro F1	0.0152
Simple TF-IDF
Metric	Score
Accuracy	93.64%
Macro F1	93.71%
Final Classifier
Metric	Score
Accuracy	94.39%
Macro F1	94.46%

The final classifier improves over the simple TF-IDF baseline by approximately 0.75 percentage points in both accuracy and Macro F1.

🧪 Response Quality

A lightweight offline quality judge checks:

Response presence
Response length
Helpful language
Exposure of internal system details
Obvious fabricated customer-specific information

The handcrafted examples achieved:

100%

on this rule-based evaluation.

This is a small deterministic evaluation and should not be interpreted as production-level response quality.

▶️ Setup
1. Clone the repository
git clone <your-repository-url>
cd hiver-ai-support-agent
2. Create virtual environment

Windows:

python -m venv .venv

Activate:

.venv\Scripts\Activate.ps1
3. Install dependencies
pip install -r requirements.txt
4. Configure environment variables

Create a .env file in the project root.

GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=your_available_gemini_model

Do not commit .env to Git.

▶️ Running the Project
Test Intent Classifier
python src/intents/train_classifier.py
Test Retrieval
python src/retrieval/retriever.py
Test Pipeline
python src/agent/pipeline.py
Run Complete AI Agent
python src/agent/agent.py
Evaluate Classifier
python src/evaluation/metrics.py
Run End-to-End Offline Evaluation
python src/evaluation/evaluate.py
Run Baseline Comparison
python src/evaluation/baselines.py
Run Response Quality Evaluation
python src/evaluation/judge.py
📁 Evaluation Outputs

Evaluation artifacts are stored in:

data/processed/

Important files:

evaluation_results.json
response_quality_results.json
⚠️ Limitations
Weakly Supervised Labels

The intent labels were created using weak supervision. Therefore, the reported 94.39% accuracy represents performance on a silver-label benchmark rather than a fully human-annotated benchmark.

Retrieval Scale

Only 10,000 conversations are indexed for local development.

A production system should use a scalable vector database and a larger retrieval corpus.

Retrieval Metric

The current retrieval evaluation uses similarity rather than human relevance judgments.

Future evaluation should include:

Recall@K
Precision@K
MRR
Human relevance evaluation
Response Evaluation

The current response judge is rule-based and operates on a small handcrafted evaluation set.

A production system should include human evaluation and/or a validated LLM-as-a-judge framework.

External API Dependency

Gemini response generation depends on an external API and is therefore subject to:

API availability
Rate limits
Latency
Usage costs
🚀 Future Improvements

Potential improvements include:

Human-verified intent labels
Better separation of delivery-related intents
Semantic embedding-based retrieval
Vector database integration
Retrieval reranking
Multi-turn conversation memory
Order and refund lookup tools
Stronger hallucination detection
Human response-quality evaluation
Production monitoring
Automated regression testing
Production Streamlit UI
📌 Key Takeaways

This project demonstrates an end-to-end AI customer-support architecture combining:

Machine Learning
      +
Information Retrieval
      +
Generative AI
      +
Safety / Escalation
      +
Evaluation

The current system achieves:

94.39% intent classification accuracy

and provides a modular foundation for building a production-grade AI customer-support assistant.

📄 Documentation

Detailed technical analysis is available in:

reports/report.md

Design decisions are documented in:

DECISIONS.md
👨‍💻 Project Status

Core implementation:

 Dataset loading
 Data processing
 Conversation reconstruction
 Intent discovery
 Intent taxonomy
 Golden/silver dataset creation
 Intent classifier
 Classifier evaluation
 Historical retrieval
 AI response generation
 Escalation system
 Baseline comparison
 Response quality evaluation
 Technical report
 README documentation

Next potential step:

 Production Streamlit UI
 Semantic retrieval
 Human-verified evaluation set
 Tool-based customer/order actions
