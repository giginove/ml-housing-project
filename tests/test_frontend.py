import unittest.mock as mock

from streamlit.testing.v1 import AppTest


def test_streamlit_app_renders():
    """Vérifie que l'application Streamlit se charge sans erreur."""
    at = AppTest.from_file("frontend/streamlit_app.py")
    at.run()
    assert not at.exception
    assert at.title[0].value == "Prediction du prix immobilier"


@mock.patch("requests.post")
def test_streamlit_prediction_flow(mock_post):
    """Vérifie que le clic sur le bouton déclenche l'affichage d'une prédiction."""
    # Simulation d'une réponse réussie du backend
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {"prediction": 2.45}

    at = AppTest.from_file("frontend/streamlit_app.py")
    at.run()
    at.button[0].click().run()
    assert "Prix estime : 2.45" in at.success[0].value
