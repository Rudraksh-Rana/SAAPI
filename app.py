from flask import Flask, request, jsonify, render_template
from transformers import pipeline
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

app = Flask(__name__)

# Download VADER lexicon (needed for VADER sentiment analysis)
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# Initialize VADER analyzer
vader_analyzer = SentimentIntensityAnalyzer()

# Initialize the sentiment analysis pipeline using DistilBERT.
# This model represents a great balance between accuracy and size (ideal for CPUs).
try:
    print("Loading model 'distilbert-base-uncased-finetuned-sst-2-english'...")
    distilbert_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    print("DistilBERT model loaded successfully.")
    print("VADER sentiment analyzer ready.")
except Exception as e:
    distilbert_model = None
    print(f"Error loading DistilBERT model: {e}")

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
            if distilbert_model is None:
                return jsonify({"error": "DistilBERT model not loaded properly. Please check server logs."}), 500
            
            # Perform inference with DistilBERT - More accurate for nuanced text
            result = distilbert_model(text)[0]
            
            return jsonify({
                "text": text,
                "label": result['label'],
                "score": float(result['score']),
                "engine": "DistilBERT (Deep Insights Mode)"
            }), 200

        else:  # compare mode
            if distilbert_model is None:
                return jsonify({"error": "DistilBERT model not loaded properly. Please check server logs."}), 500
            
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

            # DistilBERT
            d_res = distilbert_model(text)[0]
            d_label = d_res['label']
            d_score = d_res['score']

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
                    "score": float(d_score),
                    "engine": "DistilBERT (Deep Insights Mode)"
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
                if distilbert_model is None:
                    return jsonify({"error": "DistilBERT model not loaded properly. Please check server logs."}), 500
                
                res = distilbert_model(text)[0]
                label = res['label']
                results.append({
                    "text": text,
                    "label": label,
                    "score": float(res['score']),
                    "engine": "DistilBERT (Deep Insights Mode)"
                })
                summary[label] += 1

            else:  # compare mode
                if distilbert_model is None:
                    return jsonify({"error": "DistilBERT model not loaded properly. Please check server logs."}), 500
                
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
                d_res = distilbert_model(text)[0]
                d_label = d_res['label']
                d_score = d_res['score']

                results.append({
                    "text": text,
                    "vader": {
                        "label": v_label,
                        "score": float(v_score),
                        "compound": float(v_compound)
                    },
                    "distilbert": {
                        "label": d_label,
                        "score": float(d_score)
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
