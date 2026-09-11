Bilkul bhai. 🔥 reports/report.md ko pura replace karke ye content paste karo:

# Hiver AI Support Agent
## Technical Report

### 1. Project Overview

This project implements an AI-powered customer support agent designed to understand customer queries, identify their intent, retrieve relevant historical support conversations, generate helpful responses, and escalate sensitive or uncertain cases to a human support agent.

The system follows a modular pipeline:

```text
Customer Message
       ↓
Intent Classification
       ↓
Confidence Score
       ↓
Historical Conversation Retrieval
       ↓
Escalation Check
       ↓
Gemini AI Response
       ↓
Final Response / Human Escalation
2. Problem Statement

Customer support teams receive a large number of repetitive queries related to orders, deliveries, payments, refunds, returns, accounts, and other common issues.

The objective of this project is to build a support agent that can:

Understand incoming customer messages.
Automatically classify the customer's intent.
Retrieve similar historical support conversations.
Generate a concise and helpful response.
Avoid inventing customer-specific information.
Escalate sensitive or low-confidence cases to human support.
Provide measurable evaluation results.
3. Dataset

The project uses the Customer Support on Twitter dataset.

The raw dataset contains customer-support interactions from Twitter/X and includes fields such as:

tweet_id
author_id
inbound
created_at
text
response_tweet_id
in_response_to_tweet_id

The raw dataset used in this project contains:

2,811,774 tweets

The data was processed to reconstruct customer-support conversation pairs.

After reconstruction:

168,814 AmazonHelp conversation pairs

were generated.

Each reconstructed conversation contains:

customer_text
brand_text

where:

customer_text represents the customer's message.
brand_text represents the corresponding support response.
4. Data Processing

The data processing pipeline performs the following steps:

Load the raw Twitter customer-support dataset.
Identify relevant AmazonHelp conversations.
Match customer messages with corresponding support responses.
Construct customer-support conversation pairs.
Store the processed conversations in:
data/processed/amazonhelp_conversations.csv

The processed conversations are then used for retrieval and intent discovery.

5. Intent Discovery

Candidate intents were discovered using text analysis and clustering techniques.

The exploratory analysis identified common support topics including:

Delivery
Prime
Product
Payment
Account
Refund
Return
Cancellation

These topics were further refined into a structured intent taxonomy.

6. Intent Taxonomy

The final system contains 11 intents:

Intent	Description
account_issue	Problems related to account access or account functionality
delivery_date	Questions about expected delivery dates
delivery_delay	Complaints about delayed deliveries
delivery_status	Questions about current delivery/order status
missing_package	Reports of missing or undelivered packages
order_cancellation	Requests to cancel an order
payment_issue	Payment, duplicate charge, or payment-related problems
prime_issue	Problems related to Prime services
product_issue	Problems or questions related to products
refund_status	Questions about pending or missing refunds
return_issue	Questions or problems related to product returns
7. Golden Dataset Construction

A candidate golden dataset was created using weak supervision and intent-specific selection rules.

The resulting candidate dataset contains:

3,300 examples

with:

300 examples per intent

The dataset was divided into:

Split	Samples
Training	2,112
Validation	528
Test	660

The test set contains 60 examples for each intent.

Because these labels were generated through weak supervision rather than complete human annotation, the evaluation should be considered a weakly supervised / silver-label benchmark, rather than a fully human-annotated ground-truth benchmark.

8. Intent Classification

The intent classifier uses a classical machine-learning approach:

Customer Text
      ↓
TF-IDF Vectorization
      ↓
Logistic Regression
      ↓
Intent + Confidence

The classifier uses TF-IDF features to represent customer messages and Logistic Regression for multiclass classification.

The trained model is stored at:

data/processed/models/intent_classifier.joblib

The TF-IDF vectorizer is stored at:

data/processed/models/intent_vectorizer.joblib
9. Classifier Evaluation

The classifier was evaluated on 660 test examples.

Overall Results
Metric	Result
Accuracy	94.39%
Macro F1	94.46%
Weighted F1	94.46%

The classifier performs strongly across most intents.

Class-level Results
Intent	Precision	Recall	F1
account_issue	1.00	0.97	0.98
delivery_date	0.78	0.95	0.86
delivery_delay	0.97	0.97	0.97
delivery_status	0.93	0.85	0.89
missing_package	0.90	0.95	0.93
order_cancellation	0.95	1.00	0.98
payment_issue	0.96	0.87	0.91
prime_issue	0.97	0.97	0.97
product_issue	0.97	0.95	0.96
refund_status	1.00	0.98	0.99
return_issue	1.00	0.93	0.97

The main areas for improvement are the closely related delivery intents, particularly delivery_date and delivery_status.

10. Retrieval System

The project implements a TF-IDF based retrieval system over historical customer-support conversations.

The retrieval pipeline is:

Customer Query
      ↓
TF-IDF Vectorization
      ↓
Cosine Similarity
      ↓
Top-K Historical Conversations

For local development, the retriever indexes:

10,000 conversations

The retriever uses:

Lowercase text normalization
English stop-word removal
Unigrams and bigrams
TF-IDF representation
Cosine similarity

The retriever provides historical customer-support examples to the response-generation stage.

11. Retrieval Evaluation

Retrieval was evaluated on a sample of:

100 test examples

The average top-1 retrieval similarity was:

0.4043

This metric measures the similarity between the customer query and the highest-ranked retrieved historical conversation.

The score should be interpreted as a retrieval-similarity indicator rather than a direct measure of answer correctness.

12. AI Response Generation

After classification and retrieval, the system uses Google's Gemini API to generate the final support response.

The response-generation process is:

Customer Message
      +
Detected Intent
      +
Classifier Confidence
      +
Retrieved Historical Conversations
      ↓
Gemini
      ↓
Grounded Support Response

The model is instructed to:

Answer the customer's actual question.
Use historical conversations as guidance.
Avoid blindly copying historical responses.
Avoid inventing order details.
Avoid inventing refund amounts.
Avoid inventing dates or policies.
Remain polite and concise.
Clearly state when available information is insufficient.
Never expose internal system details.
13. Escalation System

The system contains an escalation layer that determines whether a customer query should be handled automatically or sent to human support.

Escalation can occur when:

The classifier confidence is below the configured threshold.
A sensitive issue is detected.

The current confidence threshold is:

0.65

Sensitive cases such as hacked-account scenarios are escalated.

Example:

Customer:
I think someone hacked my account

Escalated:
True

Reason:
Sensitive issue detected: hacked

For escalated cases, the system does not generate a normal automated support response. Instead, it returns a human-escalation message.

14. End-to-End Agent

The complete agent is implemented in:

src/agent/agent.py

The pipeline combines:

Intent classifier
Historical retriever
Gemini response generation
Escalation manager

Example normal flow:

Customer:
I cannot login to my account

Intent:
account_issue

Confidence:
0.997

Escalated:
False

AI Response:
Generated by Gemini using the detected intent
and retrieved support examples.

Example escalation flow:

Customer:
I think someone hacked my account

Intent:
account_issue

Confidence:
0.902

Escalated:
True

Reason:
Sensitive issue detected: hacked
15. Baseline Comparison

Two baselines were evaluated against the trained classifier.

Majority Class Baseline
Metric	Result
Accuracy	9.09%
Macro F1	0.0152
Simple TF-IDF Baseline
Metric	Result
Accuracy	93.64%
Macro F1	93.71%
Our Trained Classifier
Metric	Result
Accuracy	94.39%
Macro F1	94.46%

The trained classifier improves over the simple TF-IDF baseline:

Accuracy improvement:
94.39% - 93.64% = 0.75 percentage points

Macro F1 improvement:
94.46% - 93.71% = 0.75 percentage points

The improvement is modest because both systems use TF-IDF-based representations, but the final classifier configuration provides the strongest measured performance in this benchmark.

16. Response Quality Evaluation

A lightweight offline response-quality judge was implemented to check basic properties of generated responses.

The evaluator checks:

Whether a response is present.
Whether response length is reasonable.
Whether supportive/helpful language is used.
Whether internal system information is exposed.
Whether obvious customer-specific information is fabricated.

The handcrafted example responses achieved:

100%

on this rule-based quality check.

However, this should not be interpreted as a production-level response-quality score. It was evaluated on a small set of manually constructed examples using deterministic rules.

17. Project Structure
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
18. Key Results

The completed system achieved the following measured results:

Component	Result
Raw tweets processed	2,811,774
Conversation pairs	168,814
Candidate labeled examples	3,300
Intent classes	11
Test examples	660
Intent accuracy	94.39%
Macro F1	94.46%
Weighted F1	94.46%
Retrieval test samples	100
Average top-1 similarity	0.4043
Majority baseline accuracy	9.09%
Simple TF-IDF baseline accuracy	93.64%
19. Limitations

There are several limitations in the current implementation.

19.1 Weakly Supervised Labels

The intent dataset was generated using weak supervision rather than being completely human annotated.

Therefore, the reported 94.39% accuracy should not be interpreted as production accuracy.

19.2 Limited Retrieval Index

The current retriever indexes 10,000 conversations for local development rather than the complete processed conversation dataset.

A production system should use a scalable vector database or larger retrieval index.

19.3 Retrieval Metric

The current retrieval evaluation uses average top-1 similarity.

Similarity alone does not guarantee that the retrieved conversation is actually useful for answering the query.

Future evaluation should include metrics such as:

Recall@K
Precision@K
MRR
Human relevance judgments
19.4 Response Evaluation

The current response-quality judge is rule-based and evaluated on a small set of examples.

A stronger evaluation would use:

Human evaluation
LLM-as-a-judge with a validated rubric
Factuality checks
Helpfulness scoring
Safety evaluation
19.5 External Model Dependency

The final response-generation stage depends on an external Gemini API.

This introduces:

API availability dependency
Rate limits
Cost considerations
Potential latency
20. Future Improvements

Potential improvements include:

Create a larger human-verified intent dataset.
Improve separation between delivery-related intents.
Use semantic embeddings instead of only TF-IDF retrieval.
Introduce a vector database for scalable retrieval.
Add reranking after initial retrieval.
Add conversation memory for multi-turn support.
Add structured tools for order/refund lookup.
Add stronger hallucination detection.
Add human evaluation for response quality.
Add monitoring for confidence and escalation rates.
Add a production-ready Streamlit interface.
Add automated regression tests for the complete agent pipeline.
21. Conclusion

This project demonstrates a complete AI customer-support pipeline combining classical machine learning, information retrieval, generative AI, and human escalation.

The system can:

Understand customer queries.
Classify support intent.
Retrieve relevant historical conversations.
Generate grounded responses using Gemini.
Detect sensitive or uncertain cases.
Escalate appropriate conversations to human agents.
Evaluate classification and retrieval performance.

The intent classifier achieved 94.39% accuracy and 94.46% Macro F1 on the weakly supervised test benchmark, while the end-to-end evaluation produced an average top-1 retrieval similarity of 0.4043.