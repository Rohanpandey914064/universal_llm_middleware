#  Universal LLM Middleware (`universal_llm_middleware`)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Pydantic v2](https://img.shields.io/badge/Pydantic-v2.7-red.svg)](https://docs.pydantic.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests: 99 Passed](https://img.shields.io/badge/Tests-99%20Passed-brightgreen.svg)]()

> A production-ready, modular AI Middleware that intercepts LLM requests and responses to enforce **zero-coupling security, context state management, and semantic memory compression**.

---

##  Key Architectural Features

`universal_llm_middleware` sits transparently between your applications and upstream LLM providers (Groq, OpenAI, Anthropic, Ollama). It enforces strict separation of concerns across three independent engines:

```
[ Application / Client ]
           │
           ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      UNIVERSAL LLM MIDDLEWARE                          │
│                                                                        │
│  ┌──────────────────┐  ┌─────────────────────┐  ┌───────────────────┐  │
│  │  Security Engine│   │  History Manager    │  │  Compressor       │  │
│  │ • Injection Guard│  │ • Zone Isolation    │  │ • Sliding Window  │  │
│  │ • PII Anonymizer │  │ • Session Store     │  │ • Drift Validator │  │
│  │ • Canary Token   │  │ • TTL Eviction      │  │ • Token Budgeting │  │
│  └──────────────────┘  └─────────────────────┘  └───────────────────┘  │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │  (Sanitised & Compressed Payload)
                                   ▼
                   [ Upstream LLM (Groq / OpenAI) ]
```

### 1.  Security Engine
- **Prompt Injection Guard**: Dual-layer heuristic inspection (19 rule patterns) with optional ONNX/DeBERTa model fallback to detect jailbreaks, role overrides, and instructions extraction attacks before hitting the LLM.
- **PII Anonymization & Unmasking**: Automatic masking of sensitive data (Emails, Phone Numbers, Credit Cards, API Keys) via Regex & spaCy NER prior to upstream dispatch, with zero-leakage round-trip unmasking.
- **Canary Token Audit**: Injects unique UUID4 canary tokens into system directives to verify that the LLM response does not leak internal prompt instructions.

### 2.  Conversational History Engine
- **Immutable Zone Isolation**: Enforces isolation between immutable system directives (`system`, `developer`) and mutable chat history (`user`, `assistant`). System directives are strictly protected from modification or token trimming.
- **Thread-Safe Session Store**: TTL-based `InMemorySessionStore` with automatic background eviction daemon and per-session state isolation.

### 3.  Memory Compression Engine
- **Pluggable Architecture**: Implements `BaseCompressor` slot for custom context reduction algorithms.
- **Recency-Weighted Sliding Window**: Keeps recent turns untouched while compressing older turns within a configurable token budget.
- **Semantic Drift Validation**: Uses TF-IDF cosine similarity to verify that compressed context preserves original intent (similarity threshold ≥ 0.90).

---

##  Universal Integration Modes

The middleware provides **two universal integration modes**:

### Mode A: FastAPI Reverse Proxy Gateway (Drop-in HTTP Proxy)
Run as a standalone microservice that exposes an OpenAI-compatible `/v1/chat/completions` endpoint. Any project in any language (Python, Node.js, React, Go) can use it simply by changing the `base_url`:

```bash
python main.py
# Server runs on http://localhost:8080
```

### Mode B: Python SDK Client Wrapper
Wrap any existing OpenAI/Groq Python SDK client with `UniversalAIWrapper`:

```python
from openai import OpenAI
from interfaces.sdk_wrapper import UniversalAIWrapper

# Wrap native OpenAI / Groq client
raw_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key="gsk_your_groq_key"
)

client = UniversalAIWrapper(
    native_client=raw_client,
    session_id="user-session-101"
)

# Use exactly like the standard OpenAI SDK
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": "Hello!"}]
)

print(response.choices[0].message.content)
```

---

##  Quick Start Guide

### 1. Prerequisites
- Python 3.10+
- Groq API Key (Free at [console.groq.com](https://console.groq.com/keys)) or OpenAI API Key

### 2. Installation
```bash
git clone https://github.com/Rohanpandey914064/universal_llm_middleware.git
cd universal_llm_middleware

# Install dependencies
pip install -r requirements.txt

# Download spaCy NER model for PII masking
python -m spacy download en_core_web_sm
```

### 3. Environment Configuration
Copy `.env.example` to `.env` and set your upstream API key:

```env
UPSTREAM_LLM_URL=https://api.groq.com/openai/v1
UPSTREAM_API_KEY=gsk_your_real_groq_api_key
DEFAULT_MODEL=llama-3.3-70b-versatile

HOST=0.0.0.0
PORT=8080
INJECTION_THRESHOLD=0.58
SESSION_TTL_SECONDS=3600
MAX_HISTORY_TOKENS=3000
```

### 4. Start the Middleware Gateway
```bash
python main.py
```

### 5. Send a Test Request (PowerShell / cURL)
```powershell
Invoke-RestMethod -Method POST -Uri "http://localhost:8080/v1/chat/completions" `
  -Headers @{"Content-Type"="application/json"; "X-Session-ID"="demo-session-1"} `
  -Body '{"model":"llama-3.3-70b-versatile","messages":[{"role":"user","content":"Hello!"}]}'
```

---

##  Demonstration Chatbot Application (`rohanAI`)

The repository includes a dedicated video-ready ChatGPT-style web application in `demo_chatbot/` to demonstrate the security before-and-after:

```bash
# Terminal 1: Middleware
python main.py

# Terminal 2: Demo Chatbot
cd demo_chatbot
python server.py
```
Open **`http://localhost:3000`** in your browser to interact with **rohanAI**.

- ** Direct Mode**: Bypasses middleware; prompt attacks successfully steal the hidden system passcode.
- ** Protected Mode**: Routes through middleware; attacks are instantly blocked with a ` THREAT BLOCKED` alert banner, PII is masked, and sessions are securely persisted.
- ** Middleware Inspector**: Live top-right dashboard showing session memory turns, compression ratio, and security metrics.

---

##  Testing

Run the full pytest suite (99 test cases covering security, history, compression, pipeline, and FastAPI proxy):

```bash
pytest tests/ -v
```

---

##  Repository Structure

```
universal_llm_middleware/
├── main.py                        # Uvicorn entrypoint for reverse proxy
├── config/
│   └── settings.py                # Pydantic v2 Settings (env-driven)
├── core/
│   ├── schemas.py                 # Pydantic v2 schemas & exceptions
│   └── pipeline.py                # UniversalPipeline (10-stage orchestrator)
├── modules/
│   ├── security/                  # InjectionGuard, PIIAnonymizer, CanaryGuard
│   ├── history/                   # ZoneSplitter, InMemorySessionStore
│   └── compression/               # SlidingWindowCompressor, DriftValidator
├── interfaces/
│   ├── reverse_proxy.py           # FastAPI OpenAI-compatible HTTP Gateway
│   └── sdk_wrapper.py             # Python SDK client interceptor
├── demo_chatbot/                  # rohanAI Web Demo App
└── tests/                         # Complete test suite (99 tests)
```

---

##  License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
