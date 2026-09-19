def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_predict_happy_path(client):
    payload = {"PU_DO": "132_264", "trip_distance": 5.0}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()


def test_predict_invalid_payload(client):
    payload = {"PU_DO": "132_264", "trip_distance": -5.0}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
