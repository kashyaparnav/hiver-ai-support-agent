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

`
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
