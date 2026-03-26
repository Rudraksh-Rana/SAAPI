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
