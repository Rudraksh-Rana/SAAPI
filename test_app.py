import pytest
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_analyze_valid_input_positive(client):
    response = client.post('/analyze', json={"text": "I absolutely love this new feature!"})
    assert response.status_code == 200
    data = response.get_json()
    assert "label" in data
    assert "score" in data
    assert data["label"] == "POSITIVE"

def test_analyze_valid_input_negative(client):
    response = client.post('/analyze', json={"text": "This product is terrible and broken."})
    assert response.status_code == 200
    data = response.get_json()
    assert "label" in data
    assert "score" in data
    assert data["label"] == "NEGATIVE"

def test_analyze_missing_json(client):
    response = client.post('/analyze', data="not a json payload")
    assert response.status_code == 400
    assert "Request must be JSON" in response.get_json()["error"]

def test_analyze_empty_json(client):
    response = client.post('/analyze', json={})
    assert response.status_code == 400
    assert "Empty JSON request" in response.get_json()["error"]

def test_analyze_missing_text_field(client):
    response = client.post('/analyze', json={"data": "some value"})
    assert response.status_code == 400
    assert "Missing 'text' field" in response.get_json()["error"]

def test_analyze_empty_text(client):
    response = client.post('/analyze', json={"text": "   "})
    assert response.status_code == 400
    assert "'text' field cannot be empty" in response.get_json()["error"]

def test_analyze_invalid_type(client):
    response = client.post('/analyze', json={"text": 12345})
    assert response.status_code == 400
    assert "'text' field must be a string" in response.get_json()["error"]

def test_analyze_compare_engine(client):
    response = client.post('/analyze', json={"text": "This application works extremely well!", "engine": "compare"})
    assert response.status_code == 200
    data = response.get_json()
    assert "vader" in data
    assert "distilbert" in data
    assert data["engine"] == "compare"
    assert data["vader"]["label"] == "POSITIVE"
    assert data["distilbert"]["label"] == "POSITIVE"

def test_analyze_batch_vader(client):
    response = client.post('/analyze_batch', json={
        "texts": [
            "I love python!",
            "I hate bugs.",
            "Today is Tuesday."
        ],
        "engine": "vader"
    })
    assert response.status_code == 200
    data = response.get_json()
    assert "results" in data
    assert "summary" in data
    assert len(data["results"]) == 3
    assert data["summary"]["total"] == 3
    assert data["results"][0]["label"] == "POSITIVE"
    assert data["results"][1]["label"] == "NEGATIVE"
    assert data["results"][2]["label"] == "NEUTRAL"

def test_analyze_batch_compare(client):
    response = client.post('/analyze_batch', json={
        "texts": [
            "I love python!",
            "I hate bugs."
        ],
        "engine": "compare"
    })
    assert response.status_code == 200
    data = response.get_json()
    assert "results" in data
    assert len(data["results"]) == 2
    assert "vader" in data["results"][0]
    assert "distilbert" in data["results"][0]

def test_analyze_batch_missing_texts(client):
    response = client.post('/analyze_batch', json={"engine": "vader"})
    assert response.status_code == 400
    assert "Missing 'texts' field" in response.get_json()["error"]

def test_analyze_batch_invalid_texts_type(client):
    response = client.post('/analyze_batch', json={"texts": "not a list", "engine": "vader"})
    assert response.status_code == 400
    assert "'texts' field must be a list" in response.get_json()["error"]

