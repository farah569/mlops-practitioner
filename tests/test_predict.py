def test_predict_returns_valid_float(trained_model):
    features = {"PU_DO": "132_264", "trip_distance": 5.0}
    pred1 = trained_model.predict_one(features)
    pred2 = trained_model.predict_one(features)

    assert isinstance(pred1, float)
    assert pred1 > 0
    assert pred1 == pred2
