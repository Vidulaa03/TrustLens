import base64
import io

from app import app


def _client():
    return app.test_client()


def test_file_upload_source_processes_uploaded_text():
    client = _client()
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
    client = _client()
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
