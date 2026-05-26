from fastapi.testclient import TestClient

from app.main import app


def test_healthcheck_reports_ready_backends(monkeypatch):
    monkeypatch.setattr("app.main.check_storage_connection", lambda: (True, "postgres reachable"))
    monkeypatch.setattr("app.main.check_task_queue_connection", lambda: (True, "redis reachable"))

    client = TestClient(app)
    response = client.get("/api/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["storage_ready"] is True
    assert payload["task_queue_ready"] is True


def test_healthcheck_reports_degraded_backends(monkeypatch):
    monkeypatch.setattr("app.main.check_storage_connection", lambda: (False, "db down"))
    monkeypatch.setattr("app.main.check_task_queue_connection", lambda: (True, "redis reachable"))

    client = TestClient(app)
    response = client.get("/api/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "degraded"
    assert payload["storage_ready"] is False
    assert payload["storage_message"] == "db down"
