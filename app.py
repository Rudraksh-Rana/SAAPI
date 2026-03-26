from flask import Flask, request, jsonify, render_template
from transformers import pipeline

app = Flask(__name__)

# Initialize the sentiment analysis pipeline using DistilBERT.
# This model represents a great balance between accuracy and size (ideal for CPUs).
try:
    print("Loading model 'distilbert-base-uncased-finetuned-sst-2-english'...")
    sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    print("Model loaded successfully.")
except Exception as e:
    sentiment_model = None
    print(f"Error loading model: {e}")

@app.route('/', methods=['GET'])
def index():
    """Serves the frontend Web UI."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    if sentiment_model is None:
        return jsonify({"error": "Model not loaded properly. Please check server logs."}), 500

    # Validate JSON input
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "Empty JSON request"}), 400

    if 'text' not in data:
        return jsonify({"error": "Missing 'text' field in the JSON payload"}), 400
    
    text = data['text']

    # Validate the text field is not empty or pure whitespace and is a string
    if not isinstance(text, str):
        return jsonify({"error": "'text' field must be a string"}), 400
        
    text = text.strip()
    if not text:
        return jsonify({"error": "'text' field cannot be empty"}), 400

    try:
        # Perform inference
        # The pipeline returns a list of dictionaries, e.g., [{'label': 'POSITIVE', 'score': 0.99}]
        result = sentiment_model(text)[0]
        
        return jsonify({
            "text": text,
            "label": result['label'],
            "score": float(result['score']) # Convert numpy float to standard Python float for JSON compatibility
        }), 200
        
    except Exception as e:
        # Catch unexpected errors during inference
        return jsonify({"error": f"An error occurred during analysis: {str(e)}"}), 500

if __name__ == '__main__':
    # Run server
    app.run(host='0.0.0.0', port=5000)
