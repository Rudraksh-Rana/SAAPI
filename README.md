# Sentiment Analysis API

A production-ready Sentiment Analysis application featuring a Flask backend, Hugging Face transformer models for AI inferencing, and a beautiful modern web frontend.

## Tech Stack
- **Backend Framework:** Flask (Python)
- **AI Model:** DistilBERT (`distilbert-base-uncased-finetuned-sst-2-english`)
- **Frontend:** HTML5, Vanilla JavaScript, CSS (Glassmorphism design)
- **Testing:** PyTest

## Features
- **RESTful API Endpoint:** `POST /analyze` evaluates text and returns continuous POSITIVE or NEGATIVE sentiment confidence scores.
- **Dynamic Web UI:** A sleek, fully featured dark-mode interface built to comfortably test the API from your browser.
- **Robust Error Handling:** Safely handles empty payloads, missing JSON requests, and invalid data types gracefully without crashing the server.

## How to Run Locally

### 1. Set up the Environment
```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# Windows:
.\venv\Scripts\Activate.ps1
# Mac/Linux/Bash:
source venv/Scripts/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Start the Application
```bash
python app.py
```

Open `http://localhost:5000/` in your browser to view the interface!

## Manual API Usage
You can bypass the UI and use the API directly via `curl`:
```bash
curl -X POST http://localhost:5000/analyze \
     -H "Content-Type: application/json" \
     -d '{"text": "Flask with Transformers makes building APIs so incredibly simple!"}'
```
