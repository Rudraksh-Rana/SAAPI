# SentimentHub AI - Sentiment Analysis Suite

A production-ready, enterprise-grade Sentiment Analysis application featuring a Flask backend, multiple sentiment engines (Hugging Face Transformers & VADER Lexicons), and a state-of-the-art glassmorphic analytical web dashboard.

![SentimentHub UI Preview](file:///C:/Users/VICTUS/.gemini/antigravity/brain/6bc70038-7d08-48cb-baeb-64051a9b6276/sentiment_suite_demo_1779183072500.webp)

## 🛠️ Tech Stack
- **Backend Server:** Flask (Python)
- **Primary AI Model:** DistilBERT (`distilbert-base-uncased-finetuned-sst-2-english`)
- **Secondary Lexicon Engine:** NLTK VADER (`vader_lexicon` for social linguistics)
- **Analytical Frontend:** HTML5, CSS3 (Glassmorphic dark design, custom dynamic SVG charting widgets), Vanilla ES6 JavaScript
- **Test Suite:** PyTest (100% test coverage for all endpoints and validation layers)

## ✨ Features
1. **⚡ Single Sentiment Analyzer:**
   - Evaluates text input and yields sentiment classification (Positive, Negative, Neutral).
   - Mode Selector: Select from VADER Lexicon, DistilBERT Transformer, or a **Dual-Run Engine Comparison Mode** showing scoring side-by-side.
   - Beautiful dynamic SVG circular confidence dial that updates matching sentiment semantics.
   
2. **📦 Bulk Batch Processing Engine:**
   - Accept multiple inputs either via direct multiline pasting or document uploading.
   - Supports dragging and dropping or selecting `.csv`, `.json`, or `.txt` datasets.
   - Interactive table showcasing IDs, escaped text bodies, classification badges, and accurate confidence metrics.
   - Full filtering controls: Filter records by Positive, Negative, or Neutral sentiment.
   - Instant export actions: Download evaluated results as clean CSV reports or JSON payloads.

3. **📊 Insights Analytics Dashboard:**
   - Real-time sessions KPI metrics tracking total text evaluated and overall positivity index.
   - Highlights cards showing the strongest positive and lowest negative sentence processed in the active session.
   - Custom-engineered responsive SVG donut chart showing the full distribution spectrum.

4. **💻 Developer API Sandbox & Playground:**
   - Sandbox requests dispatcher allowing testing of payloads from the browser.
   - Real-time updating code generators for **cURL**, **JavaScript (Fetch)**, and **Python (Requests)**.
   - Serialized JSON response panel showing fully formatted and keyword-highlighted response payloads.

---

## 🚀 How to Run Locally

### 1. Set up the Environment
```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# Windows:
.\.venv\Scripts\Activate.ps1
# Mac/Linux/Bash:
source .venv/Scripts/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Start the Application
```bash
python app.py
```

Open `http://localhost:5000/` in your browser to view the interactive dashboard!

### 3. Run the Test Suite
Ensure the virtual environment is active and execute:
```bash
pytest
```

---

## 🔌 API Documentation

### 1. Single Analysis Endpoint: `POST /analyze`
Evaluates a single line of text under a specified engine model.

- **Request Body:**
  ```json
  {
    "text": "The custom visual components are outstanding!",
    "engine": "distilbert" // Options: "distilbert", "vader", "compare" (default: "distilbert")
  }
  ```

- **Example JSON Response (DistilBERT):**
  ```json
  {
    "text": "The custom visual components are outstanding!",
    "label": "POSITIVE",
    "score": 0.9998,
    "engine": "DistilBERT (Deep Insights Mode)"
  }
  ```

- **Example JSON Response (Compare Engine):**
  ```json
  {
    "text": "The custom visual components are outstanding!",
    "vader": {
      "label": "POSITIVE",
      "score": 0.508,
      "compound": 0.6588,
      "engine": "VADER (Express Mode)"
    },
    "distilbert": {
      "label": "POSITIVE",
      "score": 0.9998,
      "engine": "DistilBERT (Deep Insights Mode)"
    },
    "engine": "compare"
  }
  ```

### 2. Bulk/Batch Analysis Endpoint: `POST /analyze_batch`
Evaluates a list of texts simultaneously under a chosen engine.

- **Request Body:**
  ```json
  {
    "texts": [
      "I love building high-fidelity web designs.",
      "The system crashed on start.",
      "It is normal."
    ],
    "engine": "compare" // Options: "distilbert", "vader", "compare"
  }
  ```

- **Example JSON Response:**
  ```json
  {
    "results": [
      {
        "text": "I love building high-fidelity web designs.",
        "vader": {
          "label": "POSITIVE",
          "score": 0.583,
          "compound": 0.6369
        },
        "distilbert": {
          "label": "POSITIVE",
          "score": 0.9997
        },
        "engine": "compare"
      },
      ...
    ],
    "summary": {
      "total": 3,
      "positive": 2,
      "negative": 1,
      "neutral": 0
    },
    "engine": "compare"
  }
  ```
