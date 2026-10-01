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
        org_id: str = "org-default",
        timeout: float = 600.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "X-Api-Key": api_key,
            "X-Project-Id": project_id,
            "X-Org-Id": org_id,
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
    ) -> Dict[str, Any]:
        """Synchronously perform full handwriting analysis and HPI assessment."""
        with httpx.Client(timeout=self.timeout) as client:
            files = {"file": (filename, file_bytes, content_type)}
            response = client.post(
                self._url("/api/v1/handwriting/analyze"),
                headers=self.headers,
                files=files,
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
