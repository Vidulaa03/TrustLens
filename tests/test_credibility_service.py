import unittest
from unittest.mock import patch

from services import credibility_service


class CredibilityServiceFallbackTests(unittest.TestCase):
    def setUp(self):
        credibility_service.vectorizer = None
        credibility_service.model = None

    @patch("services.credibility_service.calculate_ai_probability", return_value=0.23)
    @patch("services.credibility_service.classify_topic", return_value={"primary_topic": "Politics", "confidence": 0.7})
    @patch("services.credibility_service._sentiment_scores", return_value={"pos": 0.05, "neg": 0.05})
    def test_analyze_news_rebuilds_on_pickle_error(self, *_):
        with patch("services.credibility_service.pickle.load", side_effect=ModuleNotFoundError("No module named 'numpy._core.numeric'")):
            with patch("services.credibility_service._train_model_assets") as train_mock:
                train_mock.return_value = None
                result = credibility_service.analyze_news(
                    "This article explains city government policy and provides factual context with careful reporting and balanced analysis for residents and researchers during a public review process. "
                    * 3
                )

        self.assertIn("trust_score", result)
        self.assertIn("label", result)
        self.assertIn("risk_level", result)


if __name__ == "__main__":
    unittest.main()
