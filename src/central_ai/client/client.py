"""Client SDK for interacting with the Central AI Handwriting Analysis API."""

from __future__ import annotations

from typing import Any, Dict, Optional
import httpx


class CentralAIClient:
    """HTTP client for consumer applications integrating with Central AI."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        api_key: str = "dev-key",
        project_id: str = "hpi",
        timeout: float = 600.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "X-Api-Key": api_key,
            "X-Project-Id": project_id,
        }
        self.timeout = timeout

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def health(self) -> Dict[str, Any]:
        """Check API service health status."""
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(self._url("/api/v1/health"))
            response.raise_for_status()
            return response.json()

    def analyze_handwriting(
        self,
        file_bytes: bytes,
        filename: str = "note.jpg",
        content_type: str = "image/jpeg",
        engine: str = "paddle",
        verified_scores: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Synchronously perform full handwriting analysis and HPI assessment.
        
        :param engine: 'paddle' (recommended: fast, exact CTC) or 'trocr' (deep transformer).
        :param verified_scores: Optional dictionary of verified trait scores (e.g. {'Execution': 85})
                                to train the machine learning model side-by-side with this upload.
        """
        import json

        data_form = {}
        if verified_scores:
            data_form["verified_scores"] = json.dumps(verified_scores)

        with httpx.Client(timeout=self.timeout) as client:
            files = {"file": (filename, file_bytes, content_type)}
            response = client.post(
                self._url("/api/v1/handwriting/analyze"),
                headers=self.headers,
                params={"engine": engine},
                data=data_form,
                files=files,
            )
            response.raise_for_status()
            return response.json()

    def train_handwriting(
        self,
        file_bytes: bytes,
        scores: Dict[str, float],
        filename: str = "sample.jpg",
        content_type: str = "image/jpeg",
    ) -> Dict[str, Any]:
        """Submit a handwriting sample with verified HPI scores to immediately train the model.
        
        :param file_bytes: Image bytes of the handwriting sample.
        :param scores: Dictionary of verified trait scores (e.g. {'Execution': 88, 'Focus': 82}).
        """
        import json

        with httpx.Client(timeout=self.timeout) as client:
            files = {"file": (filename, file_bytes, content_type)}
            data = {"scores": json.dumps(scores)}
            response = client.post(
                self._url("/api/v1/training/train"),
                headers=self.headers,
                data=data,
                files=files,
            )
            response.raise_for_status()
            return response.json()

    def get_model_status(self) -> Dict[str, Any]:
        """Fetch the active learning model status, trained sample count, and active version."""
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(
                self._url("/api/v1/training/model"),
                headers=self.headers,
            )
            response.raise_for_status()
            return response.json()

    def assess_hpi(
        self,
        analysis_result_or_metrics: Dict[str, Any],
        ocr_confidence: Optional[float] = None,
        quality_confidence: Optional[float] = None,
        sample_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Assess HPI methodology from previously extracted handwriting parameters.

        Accepts either:
        - The full perception response dictionary from client.analyze_handwriting(...)
        - Or a dictionary of physical biometric metrics {'slant_angle': 2.5, 'pressure_density': 0.08, ...}
        """
        payload: Dict[str, Any] = {}
        if "handwriting_features" in analysis_result_or_metrics or "quality_gate" in analysis_result_or_metrics:
            payload = {
                "result_id": analysis_result_or_metrics.get("result_id"),
                "handwriting_features": analysis_result_or_metrics.get("handwriting_features"),
                "quality_gate": analysis_result_or_metrics.get("quality_gate"),
                "ocr": analysis_result_or_metrics.get("ocr"),
            }
        elif "metrics" in analysis_result_or_metrics:
            payload = analysis_result_or_metrics
        else:
            payload = {"metrics": analysis_result_or_metrics}

        if ocr_confidence is not None:
            payload["ocr_confidence"] = ocr_confidence
        if quality_confidence is not None:
            payload["quality_confidence"] = quality_confidence
        if sample_id is not None:
            payload["sample_id"] = sample_id

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                self._url("/api/v1/hpi/assess"),
                headers=self.headers,
                json=payload,
            )
            response.raise_for_status()
            return response.json()

    def get_hpi_rules(self) -> Dict[str, Any]:
        """Inspect active client-approved HPI methodology rules registry."""
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(
                self._url("/api/v1/hpi/rules"),
                headers=self.headers,
            )
            response.raise_for_status()
            return response.json()

