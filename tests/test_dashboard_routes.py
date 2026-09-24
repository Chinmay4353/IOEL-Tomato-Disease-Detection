from backend.app import app


def test_dashboard_index_served():
    client = app.test_client()
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert b"TOMATO PLANT DETECTION DASHBOARD" in response.data


def test_dashboard_static_asset_served():
    client = app.test_client()
    response = client.get("/dashboard/style.css")
    assert response.status_code == 200
    assert b"background" in response.data
