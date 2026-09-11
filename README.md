# 🤖 Hiver AI Support Agent

An AI-powered customer support agent that combines **intent classification, historical conversation retrieval, Gemini-based response generation, and human escalation** to automate common customer-support queries.

---

## 🚀 Overview

This project demonstrates an end-to-end AI customer-support pipeline built using the Customer Support on Twitter dataset.

The system can:

- Understand customer intent
- Retrieve similar historical support conversations
- Generate context-aware AI responses
- Detect sensitive support issues
- Escalate risky or uncertain conversations to human agents
- Provide an interactive Streamlit interface

---

## 🏗️ Architecture

```text
Customer Message
       │
       ▼
┌──────────────────────┐
│ Intent Classification │
│ TF-IDF + LogisticReg │
└──────────┬───────────┘
           │
           ▼
   Intent + Confidence
           │
           ▼
┌──────────────────────┐
│ Historical Retrieval │
│ TF-IDF + Cosine Sim. │
└──────────┬───────────┘
           │
           ▼
    Escalation Check
       │         │
       │         └──────────────┐
       ▼                        ▼
   Sensitive/             Safe & Confident
   Low Confidence               │
       │                        ▼
       ▼                   Gemini LLM
 Human Escalation               │
                                ▼
                         Final Response
🧠 Key Components
1. Intent Classification

The classifier uses:

TF-IDF
Logistic Regression

The current taxonomy contains 11 intents:

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
2. Historical Retrieval

Historical customer-support conversations are retrieved using:

TF-IDF
Unigram + bigram features
Cosine similarity

The development retrieval index currently contains 10,000 conversations.

3. Gemini Response Generation

Gemini generates the final response using:

Customer message
Detected intent
Classifier confidence
Retrieved historical support examples

The model is instructed not to blindly copy historical responses or invent customer/order information.

4. Escalation

Sensitive or uncertain conversations can be escalated to human support.

For example:

Customer:
I think someone hacked my account

Intent:
account_issue

Status:
Human Escalation

Reason:
Sensitive issue detected: hacked
📊 Evaluation
Intent Classification

The classifier was evaluated on 660 test examples.

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

Note: The intent dataset uses weak/silver labels generated during dataset preparation. Therefore, the reported classifier metrics should not be interpreted as performance on a fully human-annotated production benchmark.

🖥️ Demo

The project includes a Streamlit interface where users can enter customer-support queries and view:

Detected intent
Confidence score
Automation/escalation status
AI-generated response
Retrieved historical cases

Run the application with:

streamlit run app/streamlit_app.py
💬 Example Queries
Delivery Issue
My package is three days late

Expected behavior:

Intent: delivery_delay
Status: Automated
Refund
Where is my refund?

Expected behavior:

Intent: refund_status
Status: Automated
Cancellation
I want to cancel my order

Expected behavior:

Intent: order_cancellation
Status: Automated
Payment
My payment was charged twice

Expected behavior:

Intent: payment_issue
Sensitive Issue
I think someone hacked my account

Expected behavior:

Intent: account_issue
Status: Human Escalation
📁 Project Structure
hiver-ai-support-agent/
│
├── app/
│   └── streamlit_app.py
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

Windows:

python -m venv .venv
.venv\Scripts\activate
3. Install dependencies
pip install -r requirements.txt
4. Configure environment variables

Create a .env file based on .env.example.

Add your Gemini API configuration:

GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-2.5-flash

Never commit the .env file or expose API keys publicly.

▶️ Running the Project
Run the Streamlit application
streamlit run app/streamlit_app.py
Run intent evaluation
python src/evaluation/metrics.py
Run baseline comparison
python src/evaluation/baselines.py
Run retrieval/evaluation pipeline
python src/evaluation/evaluate.py
Run the support agent directly
python src/agent/agent.py
🛠️ Tech Stack
Category	Technology
Language	Python
ML	Scikit-learn
Classification	TF-IDF + Logistic Regression
Retrieval	TF-IDF + Cosine Similarity
Generative AI	Gemini
UI	Streamlit
Data Processing	Pandas, NumPy
Environment	python-dotenv
Version Control	Git + GitHub
⚠️ Limitations

The current implementation has several limitations:

Intent labels are weak/silver labels rather than fully human-annotated labels.
TF-IDF retrieval depends primarily on lexical similarity.
The local retrieval index is limited to 10,000 conversations.
Historical support responses may contain outdated information or links.
The agent does not directly access real order, payment, shipping, or customer-account systems.
Therefore, it cannot independently verify real-time order or refund information.
🔮 Future Improvements

Possible next improvements include:

Semantic embedding-based retrieval
Hybrid keyword + semantic search
Vector database integration
Human-reviewed intent dataset
Confidence calibration
Better response-quality evaluation
Real order/shipping/payment integrations
Conversation memory
Production monitoring
Human feedback loop
📌 Conclusion

This project demonstrates a practical customer-support architecture combining classical machine learning, information retrieval, generative AI, and human escalation.

The system achieves 94.39% intent classification accuracy and 94.46% Macro F1 on the current weakly supervised evaluation set while providing an end-to-end interactive support experience through Streamlit.