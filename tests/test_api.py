from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200


def test_prompts_endpoint():
    response = client.get("/prompts")
    assert response.status_code == 200
    assert isinstance(response.json(), (list, dict))


def test_prompt_version_endpoint():
    response = client.get("/prompts/v1")
    assert response.status_code == 200
    assert response.json() is not None


def test_datasets_endpoint():
    response = client.get("/datasets")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_create_dataset():
    payload = {
        "name": "Automated API Test Dataset",
        "prompts": [
            {
                "question": "What is machine learning?",
                "context": "Machine learning enables systems to learn patterns from data.",
                "expected_output": "Machine learning enables systems to learn patterns from data.",
                "category": "AI Fundamentals",
                "difficulty": "Easy",
            }
        ],
    }

    response = client.post("/datasets", json=payload)

    assert response.status_code in (200, 201)
    assert response.json() is not None


def test_create_dataset_rejects_missing_name():
    payload = {
        "prompts": [
            {
                "question": "What is Python?"
            }
        ]
    }

    response = client.post("/datasets", json=payload)

    assert response.status_code == 422


def test_create_dataset_rejects_missing_prompts():
    payload = {
        "name": "Invalid Dataset"
    }

    response = client.post("/datasets", json=payload)

    assert response.status_code == 422


def test_nonexistent_run_returns_not_found():
    response = client.get("/runs/999999999")

    assert response.status_code == 404


def test_nonexistent_run_results_returns_not_found():
    response = client.get("/runs/999999999/results")

    assert response.status_code == 404


def test_nonexistent_run_judge_results_returns_not_found():
    response = client.get("/runs/999999999/judge-results")

    assert response.status_code == 404


def test_compare_requires_run_ids():
    response = client.get("/compare")

    assert response.status_code == 422
