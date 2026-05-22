from flask import Flask, request, jsonify, render_template
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import os
import requests

app = Flask(__name__)

# Configure NLTK data directory for serverless (read-only filesystem workaround)
nltk_data_dir = os.path.join('/tmp', 'nltk_data') if os.environ.get('VERCEL') else None
if nltk_data_dir:
    if nltk_data_dir not in nltk.data.path:
        nltk.data.path.append(nltk_data_dir)

# Download VADER lexicon (needed for VADER sentiment analysis)
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    if nltk_data_dir:
        os.makedirs(nltk_data_dir, exist_ok=True)
        nltk.download('vader_lexicon', download_dir=nltk_data_dir)
    else:
        nltk.download('vader_lexicon')

# Initialize VADER analyzer
vader_analyzer = SentimentIntensityAnalyzer()

distilbert_model = None

# Initialize the sentiment analysis pipeline using DistilBERT.
# Try loading locally first; if not installed or fails, we fall back to Hugging Face Inference API.
try:
    from transformers import pipeline
    print("Loading local model 'distilbert-base-uncased-finetuned-sst-2-english'...")
    distilbert_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    print("DistilBERT model loaded successfully.")
    print("VADER sentiment analyzer ready.")
except Exception as e:
    print(f"Local DistilBERT model could not be loaded: {e}")
    print("System will seamlessly use the cloud Hugging Face Inference API for DistilBERT mode.")

def get_distilbert_sentiment(text):
    global distilbert_model
    
    # 1. Try local pipeline if it was loaded successfully
    if distilbert_model is not None:
        try:
            result = distilbert_model(text)[0]
            return {
                "label": result['label'],
                "score": float(result['score']),
                "engine": "DistilBERT (Deep Insights Mode)"
            }
        except Exception as local_err:
            print(f"Local DistilBERT inference failed: {local_err}. Trying API fallback...")
            
    # 2. Try Hugging Face Inference API (Serverless Cloud Fallback)
    HF_API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
    # Use HF_API_KEY from environment variables if present (highly recommended for production rate-limiting)
    HF_API_KEY = os.environ.get("HF_API_KEY")
    headers = {"Authorization": f"Bearer {HF_API_KEY}"} if HF_API_KEY else {}
    
    try:
        response = requests.post(HF_API_URL, headers=headers, json={"inputs": text}, timeout=10)
        
        # Hugging Face API can return a 503 while the model is loading. We can handle it gracefully.
        if response.status_code == 503:
            est_time = response.json().get("estimated_time", 20)
            raise Exception(f"Hugging Face Model is waking up. Please retry in {int(est_time)} seconds.")
            
        if response.status_code != 200:
            err_details = response.json().get("error", "Unknown API error")
            raise Exception(f"Hugging Face API returned error: {err_details}")
            
        res_json = response.json()
        
        # Hugging Face text classification models return: [[{"label": "...", "score": ...}, ...]]
        if isinstance(res_json, list) and len(res_json) > 0 and isinstance(res_json[0], list):
            predictions = res_json[0]
            best_prediction = max(predictions, key=lambda x: x['score'])
            return {
                "label": best_prediction['label'],
                "score": float(best_prediction['score']),
                "engine": "DistilBERT (Cloud Insights Mode)"
            }
        else:
            raise Exception("Hugging Face API response structure is unexpected.")
            
    except Exception as api_err:
        raise Exception(f"DistilBERT Sentiment analysis failed: {str(api_err)}")

@app.route('/', methods=['GET'])
def index():
    """Serves the frontend Web UI."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    # Validate JSON input
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "Empty JSON request"}), 400

    if 'text' not in data:
        return jsonify({"error": "Missing 'text' field in the JSON payload"}), 400
    
    text = data['text']
    engine = data.get('engine', 'distilbert').lower()  # Default to DistilBERT

    # Validate the text field is not empty or pure whitespace and is a string
    if not isinstance(text, str):
        return jsonify({"error": "'text' field must be a string"}), 400
        
    text = text.strip()
    if not text:
        return jsonify({"error": "'text' field cannot be empty"}), 400

    # Validate engine choice
    if engine not in ['vader', 'distilbert', 'compare']:
        return jsonify({"error": "Invalid engine. Choose 'vader', 'distilbert', or 'compare'"}), 400

    try:
        if engine == 'vader':
            # VADER (Valence Aware Dictionary and sEntiment Reasoner) - Fast & Good for social media
            scores = vader_analyzer.polarity_scores(text)
            compound = scores['compound']
            
            # Map compound score to POSITIVE/NEGATIVE/NEUTRAL
            if compound >= 0.05:
                label = 'POSITIVE'
                score = scores['pos']
            elif compound <= -0.05:
                label = 'NEGATIVE'
                score = scores['neg']
            else:
                label = 'NEUTRAL'
                score = scores['neu']
            
            return jsonify({
                "text": text,
                "label": label,
                "score": float(score),
                "engine": "VADER (Express Mode)",
                "compound": float(compound)
            }), 200
        
        elif engine == 'distilbert':
            # Perform inference with DistilBERT (Dynamic local pipeline or Cloud API)
            try:
                res = get_distilbert_sentiment(text)
                return jsonify({
                    "text": text,
                    "label": res['label'],
                    "score": res['score'],
                    "engine": res['engine']
                }), 200
            except Exception as err:
                return jsonify({"error": str(err)}), 500

        else:  # compare mode
            # VADER
            v_scores = vader_analyzer.polarity_scores(text)
            v_compound = v_scores['compound']
            if v_compound >= 0.05:
                v_label = 'POSITIVE'
                v_score = v_scores['pos']
            elif v_compound <= -0.05:
                v_label = 'NEGATIVE'
                v_score = v_scores['neg']
            else:
                v_label = 'NEUTRAL'
                v_score = v_scores['neu']

            # DistilBERT (Dynamic local pipeline or Cloud API)
            try:
                d_res = get_distilbert_sentiment(text)
                d_label = d_res['label']
                d_score = d_res['score']
            except Exception as err:
                return jsonify({"error": f"DistilBERT failure in compare mode: {str(err)}"}), 500

            return jsonify({
                "text": text,
                "vader": {
                    "label": v_label,
                    "score": float(v_score),
                    "compound": float(v_compound),
                    "engine": "VADER (Express Mode)"
                },
                "distilbert": {
                    "label": d_label,
                    "score": d_score,
                    "engine": d_res['engine']
                },
                "engine": "compare"
            }), 200
        
    except Exception as e:
        # Catch unexpected errors during inference
        return jsonify({"error": f"An error occurred during analysis: {str(e)}"}), 500


@app.route('/analyze_batch', methods=['POST'])
def analyze_sentiment_batch():
    # Validate JSON input
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    
    if data is None:
        return jsonify({"error": "Empty JSON request"}), 400

    if 'texts' not in data:
        return jsonify({"error": "Missing 'texts' field in the JSON payload"}), 400
    
    texts = data['texts']
    engine = data.get('engine', 'distilbert').lower()  # Default to DistilBERT

    if not isinstance(texts, list):
        return jsonify({"error": "'texts' field must be a list of strings"}), 400

    if not texts:
        return jsonify({"error": "'texts' list cannot be empty"}), 400

    if len(texts) > 100:
        return jsonify({"error": "Batch size exceeds limit of 100 items"}), 400

    # Validate engine choice
    if engine not in ['vader', 'distilbert', 'compare']:
        return jsonify({"error": "Invalid engine. Choose 'vader', 'distilbert', or 'compare'"}), 400

    results = []
    summary = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}

    for idx, raw_text in enumerate(texts):
        if not isinstance(raw_text, str):
            return jsonify({"error": f"Item at index {idx} must be a string"}), 400
        
        text = raw_text.strip()
        if not text:
            return jsonify({"error": f"Item at index {idx} cannot be empty or whitespace"}), 400

        try:
            if engine == 'vader':
                scores = vader_analyzer.polarity_scores(text)
                compound = scores['compound']
                if compound >= 0.05:
                    label = 'POSITIVE'
                    score = scores['pos']
                elif compound <= -0.05:
                    label = 'NEGATIVE'
                    score = scores['neg']
                else:
                    label = 'NEUTRAL'
                    score = scores['neu']
                
                results.append({
                    "text": text,
                    "label": label,
                    "score": float(score),
                    "compound": float(compound),
                    "engine": "VADER (Express Mode)"
                })
                summary[label] += 1

            elif engine == 'distilbert':
                try:
                    res = get_distilbert_sentiment(text)
                    results.append({
                        "text": text,
                        "label": res['label'],
                        "score": res['score'],
                        "engine": res['engine']
                    })
                    summary[res['label']] += 1
                except Exception as err:
                    return jsonify({"error": f"DistilBERT failure at index {idx}: {str(err)}"}), 500

            else:  # compare mode
                # Run VADER
                v_scores = vader_analyzer.polarity_scores(text)
                v_compound = v_scores['compound']
                if v_compound >= 0.05:
                    v_label = 'POSITIVE'
                    v_score = v_scores['pos']
                elif v_compound <= -0.05:
                    v_label = 'NEGATIVE'
                    v_score = v_scores['neg']
                else:
                    v_label = 'NEUTRAL'
                    v_score = v_scores['neu']

                # Run DistilBERT
                try:
                    d_res = get_distilbert_sentiment(text)
                    d_label = d_res['label']
                    d_score = d_res['score']
                except Exception as err:
                    return jsonify({"error": f"DistilBERT failure at index {idx}: {str(err)}"}), 500

                results.append({
                    "text": text,
                    "vader": {
                        "label": v_label,
                        "score": float(v_score),
                        "compound": float(v_compound)
                    },
                    "distilbert": {
                        "label": d_label,
                        "score": d_score
                    },
                    "engine": "compare"
                })
                # Count primary DistilBERT label for high-level visualization summary
                summary[d_label] += 1

        except Exception as e:
            return jsonify({"error": f"An error occurred during analysis at index {idx}: {str(e)}"}), 500

    return jsonify({
        "results": results,
        "summary": {
            "total": len(texts),
            "positive": summary["POSITIVE"],
            "negative": summary["NEGATIVE"],
            "neutral": summary["NEUTRAL"]
        },
        "engine": engine
    }), 200


if __name__ == '__main__':
    # Run server
    app.run(host='0.0.0.0', port=5000)
