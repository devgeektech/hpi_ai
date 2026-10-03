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
                self._url("/api/v1/handwriting/train"),
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
                self._url("/api/v1/handwriting/model"),
                headers=self.headers,
            )
            response.raise_for_status()
            return response.json()

    def validate_image(
        self,
        file_bytes: bytes,
        filename: str = "note.jpg",
        content_type: str = "image/jpeg",
    ) -> Dict[str, Any]:
        """Validate image quality, blur, and handwriting presence."""
        with httpx.Client(timeout=self.timeout) as client:
            files = {"file": (filename, file_bytes, content_type)}
            response = client.post(
                self._url("/api/v1/images/validate"),
                headers=self.headers,
                files=files,
            )
            response.raise_for_status()
            return response.json()
