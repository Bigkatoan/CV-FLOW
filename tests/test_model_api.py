import pytest
from fastapi.testclient import TestClient
import sys
import os
sys.path.insert(0, os.path.abspath("backend"))
from app.main import app

client = TestClient(app)

def test_list_models():
    response = client.get("/api/models")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_model_not_found():
    response = client.get("/api/models/invalid-id")
    assert response.status_code == 404
