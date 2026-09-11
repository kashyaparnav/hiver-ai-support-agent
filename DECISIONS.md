# Architecture & Design Decisions

This document records the major technical decisions made while building the Hiver AI Support Agent.

---

## 1. Why Customer Support on Twitter Dataset?

The Customer Support on Twitter dataset was selected because it contains real-world customer-support interactions between customers and brands.

It provides:

- Customer messages
- Brand/support responses
- Conversation relationships
- Timestamps
- Tweet relationships

This makes it suitable for building both an intent-classification dataset and a retrieval corpus.

---

## 2. Why AmazonHelp Conversations?

The project focuses on AmazonHelp conversations because they contain a large variety of customer-support scenarios related to:

- Orders
- Deliveries
- Payments
- Refunds
- Returns
- Accounts
- Prime
- Products

This provides enough diversity to build a useful support-intent taxonomy.

---

## 3. Why Conversation Reconstruction?

The raw dataset consists of individual tweets rather than clean question-answer pairs.

Therefore, conversation reconstruction was implemented to create:

```text
Customer Message
      ↓
Support Response

pairs.

The reconstructed conversations are stored in:

data/processed/amazonhelp_conversations.csv

This format is easier to use for retrieval and response-generation experiments.

4. Why Weak Supervision?

Creating thousands of human-labeled intent examples would require significant manual annotation effort.

Therefore, weak supervision was used to create an initial candidate dataset.

The process:

Historical Conversations
        ↓
Intent-specific rules
        ↓
Candidate labels
        ↓
Silver/weakly-labeled dataset

This allowed rapid experimentation while keeping the project reproducible.

Trade-off

The main disadvantage is label noise.

Therefore, classifier performance is explicitly described as performance on a weakly supervised/silver-label benchmark rather than a fully human-annotated benchmark.

5. Why 11 Intents?

The taxonomy was designed around common and operationally meaningful customer-support scenarios.

The final taxonomy contains:

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

The taxonomy attempts to balance:

Coverage
Interpretability
Classification difficulty
Practical support usefulness
6. Why TF-IDF for Intent Classification?

TF-IDF was selected as the initial text representation because:

It is simple.
It is fast.
It is interpretable.
It works well for short customer messages.
It provides a strong classical ML baseline.
It does not require GPU infrastructure.

For a take-home project, this provides a good balance between implementation complexity and measurable performance.

7. Why Logistic Regression?

Logistic Regression was selected for multiclass intent classification because it is:

Efficient
Strong for sparse TF-IDF features
Easy to train
Easy to evaluate
Capable of producing class probabilities/confidence scores

The confidence score is also useful for the escalation layer.

8. Why Not Use an LLM for Intent Classification?

The final response generation uses Gemini, but intent classification is handled separately using a trained classifier.

This separation provides several benefits:

Intent Classification
        ↓
Deterministic ML model

Response Generation
        ↓
Generative AI

Advantages:

Lower inference cost for classification
Faster classification
More predictable intent labels
Explicit confidence scores
Easier evaluation
Easier escalation decisions
9. Why TF-IDF Retrieval?

TF-IDF retrieval was selected as the initial retrieval approach because it is:

Simple
Fast
Transparent
Easy to reproduce
Suitable for a prototype

The retriever uses cosine similarity to identify historical conversations that are lexically similar to the incoming customer query.

10. Why Limit the Retrieval Index to 10,000 Conversations?

The full processed dataset contains a large number of conversations.

For local development, the retriever indexes 10,000 conversations.

This keeps:

Startup time manageable
Memory usage reasonable
Development iteration fast

A production system would use a scalable retrieval infrastructure.

11. Why Use Retrieval Before Generation?

The response generator should not rely only on its pretrained knowledge.

Historical support conversations provide domain-specific examples.

Therefore:

Customer Query
      ↓
Retrieve Similar Support Cases
      ↓
Provide Cases as Context
      ↓
Generate Response

This helps the model produce responses aligned with historical support behavior.

12. Why Gemini for Response Generation?

Gemini was selected for the generative-response layer because the project requires natural-language support responses.

The model receives:

Customer message
Detected intent
Classifier confidence
Retrieved historical examples

The model is instructed to use historical responses as guidance rather than blindly copying them.

13. Why Separate Classification, Retrieval and Generation?

The system follows a modular architecture:

Classifier
    ↓
Retriever
    ↓
Escalation
    ↓
Generator

This makes individual components independently testable.

For example:

Classifier can be evaluated with accuracy/F1.
Retriever can be evaluated with similarity/relevance metrics.
Generator can be evaluated for response quality.
Escalation can be tested independently.

This is preferable to implementing the entire system as one opaque LLM prompt.

14. Why Confidence-Based Escalation?

Automated systems should not confidently answer uncertain queries.

The classifier produces a confidence score.

The current threshold is:

0.65

If confidence is below this threshold, the conversation can be escalated.

Conceptually:

Confidence >= 0.65
        ↓
Automated handling

Confidence < 0.65
        ↓
Human escalation

The threshold can be tuned using validation data.

15. Why Sensitive Issue Detection?

Some customer issues should receive additional human attention even when the classifier is highly confident.

For example:

"I think someone hacked my account"

should not simply receive a generic automated response.

Therefore, the escalation layer also checks for sensitive issue indicators.

This creates a safety layer independent of classifier confidence.

16. Why Not Automatically Resolve Sensitive Cases?

The system avoids taking potentially risky actions based only on a text classification.

Instead, sensitive cases are escalated:

Sensitive Issue
      ↓
Human Support

This reduces the risk of inappropriate automated handling.

17. Why Offline Evaluation?

The project separates evaluation from live Gemini generation.

Classifier and retrieval evaluation can run without consuming generative-AI API credits.

This provides:

Reproducibility
Faster experimentation
Lower cost
Deterministic evaluation

Gemini is used only where generative response quality is actually required.

18. Why Baseline Comparison?

A model's performance is more meaningful when compared against simpler approaches.

Two baselines were implemented:

Majority Class

Predicts the most frequent class for every example.

Result:

Accuracy: 9.09%
Macro F1: 0.0152
Simple TF-IDF

Uses a basic TF-IDF + Logistic Regression configuration.

Result:

Accuracy: 93.64%
Macro F1: 93.71%

The final classifier achieved:

Accuracy: 94.39%
Macro F1: 94.46%

This demonstrates that the final configuration performs better than the tested baseline.

19. Why Evaluate Response Quality Separately?

Classification correctness does not guarantee a good customer-support response.

A system can correctly identify:

payment_issue

but still generate an unhelpful answer.

Therefore response quality is evaluated separately.

The current offline judge checks:

Response presence
Reasonable response length
Helpful language
Internal information leakage
Obvious fabricated customer-specific information
20. Why Not Claim 100% Response Quality?

The response-quality evaluator was intentionally kept lightweight.

The current 100% result comes from a small handcrafted evaluation set and deterministic rules.

It should therefore not be interpreted as production-level response quality.

A stronger system would use:

Human reviewers
Larger evaluation datasets
Structured rubrics
LLM-based judging with validation
Factuality and safety checks
21. Current Architecture

The final architecture is:

                         ┌──────────────────┐
                         │ Customer Message │
                         └────────┬─────────┘
                                  │
                                  ▼
                       ┌──────────────────────┐
                       │ Intent Classifier    │
                       │ TF-IDF + Logistic    │
                       │ Regression           │
                       └──────────┬───────────┘
                                  │
                           Intent + Confidence
                                  │
                                  ▼
                       ┌──────────────────────┐
                       │ Historical Retriever │
                       │ TF-IDF + Cosine      │
                       │ Similarity           │
                       └──────────┬───────────┘
                                  │
                                  ▼
                       ┌──────────────────────┐
                       │ Escalation Manager   │
                       └──────────┬───────────┘
                                  │
                         ┌────────┴────────┐
                         │                 │
                         ▼                 ▼
                    Normal Case       Sensitive /
                         │             Low Confidence
                         ▼                 │
                    ┌─────────┐           ▼
                    │ Gemini  │      Human Agent
                    └────┬────┘
                         │
                         ▼
                   Final Response
22. Current Measured Performance
Component	Metric	Result
Intent Classifier	Accuracy	94.39%
Intent Classifier	Macro F1	94.46%
Intent Classifier	Weighted F1	94.46%
Retrieval	Average Top-1 Similarity	0.4043
Majority Baseline	Accuracy	9.09%
Simple TF-IDF Baseline	Accuracy	93.64%
23. Future Architecture

A production version could evolve into:

Customer
   ↓
API / Chat Interface
   ↓
Intent + Safety Router
   ↓
Semantic Retrieval
   ↓
Reranker
   ↓
Tool Calling
   ├── Order Lookup
   ├── Refund Lookup
   ├── Delivery Status
   └── Account Support
   ↓
LLM Response
   ↓
Policy / Safety Validation
   ↓
Human Escalation if required
   ↓
Customer
24. Final Design Philosophy

The project intentionally uses a hybrid architecture rather than relying entirely on a generative model.

The core principle is:

Use ML where deterministic classification is useful.
Use retrieval where historical context is useful.
Use an LLM where natural-language generation is useful.
Use human escalation where automation should not be trusted.

This provides a more controllable and measurable foundation for an AI customer-support system.