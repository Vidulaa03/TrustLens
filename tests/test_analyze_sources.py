import base64
import io

from app import app, create_user, get_user_by_id
from werkzeug.security import generate_password_hash


def _logged_in_client():
    user = get_user_by_id(1)
    if not user:
        create_user("Demo User", "demo", "demo@example.com", generate_password_hash("demo123"))

    client = app.test_client()
    with client.session_transaction() as session:
        session["user_id"] = 1
        session["user_name"] = "Demo User"
        session["user_username"] = "demo"
    return client


def test_file_upload_source_processes_uploaded_text():
    client = _logged_in_client()
    payload = " ".join([
        "This article explains a major public policy debate with clear warnings about corruption and institutional failures.",
        "It includes evidence, analysis, and a measured tone designed to help readers understand the issue without sensational exaggeration.",
        "The report balances claims with accountability and aims to provide context for a complex civic topic.",
    ])

    response = client.post(
        "/analyze",
        data={
            "source_type": "file",
            "source_file": (io.BytesIO(payload.encode("utf-8")), "article.txt"),
        },
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert b"Combined content assessment" in response.data


def test_url_source_data_uri_processes_text():
    client = _logged_in_client()
    text = " ".join([
        "This is a detailed report about public governance and the electoral process. It discusses policy disputes, institutional accountability, and the importance of transparency.",
        "The article examines how public funding and local leadership decisions affect communities and raises concerns about accountability across multiple agencies.",
        "It remains factual, balanced, and evidence-oriented without relying on dramatic claims or clickbait phrasing.",
    ])
    encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")

    response = client.post(
        "/analyze",
        data={
            "source_type": "url",
            "url": f"data:text/plain;base64,{encoded}",
        },
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert b"Combined content assessment" in response.data
