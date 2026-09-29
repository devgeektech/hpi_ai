from typing import Any, Dict, Optional
import httpx


class CentralAIClient:
    """HTTP client used by HMB, Boardroom, and HPI apps."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        api_key: str = "dev-key",
        project_id: str = "hpi",
        org_id: str = "org-default",
        timeout: float = 600.0,
    ):
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
        with httpx.Client(timeout=self.timeout) as client:
            r = client.get(self._url("/api/v1/health"))
            r.raise_for_status()
            return r.json()

    def analyze_handwriting(
        self,
        file_bytes: bytes,
        filename: str = "note.jpg",
        content_type: str = "image/jpeg",
        device_id: str = "web-client",
    ) -> Dict[str, Any]:
        with httpx.Client(timeout=self.timeout) as client:
            files = {"file": (filename, file_bytes, content_type)}
            headers = {**self.headers, "X-Device-ID": device_id}
            r = client.post(
                self._url("/api/v1/handwriting/analyze"),
                headers=headers,
                files=files,
            )
            r.raise_for_status()
            return r.json()

    def get_job(self, job_id: str) -> Dict[str, Any]:
        with httpx.Client(timeout=self.timeout) as client:
            r = client.get(self._url(f"/api/v1/jobs/{job_id}"), headers=self.headers)
            r.raise_for_status()
            return r.json()

    def get_job_result(self, job_id: str) -> Dict[str, Any]:
        with httpx.Client(timeout=self.timeout) as client:
            r = client.get(
                self._url(f"/api/v1/jobs/{job_id}/result"),
                headers=self.headers,
            )
            r.raise_for_status()
            return r.json()

    def validate_image(self, file_bytes: bytes, filename: str = "note.jpg") -> Dict[str, Any]:
        with httpx.Client(timeout=self.timeout) as client:
            files = {"file": (filename, file_bytes, "image/jpeg")}
            r = client.post(
                self._url("/api/v1/images/validate"),
                headers=self.headers,
                files=files,
            )
            r.raise_for_status()
            return r.json()

    def handwriting_features(self, file_bytes: bytes, filename: str = "note.jpg") -> Dict[str, Any]:
        with httpx.Client(timeout=self.timeout) as client:
            files = {"file": (filename, file_bytes, "image/jpeg")}
            r = client.post(
                self._url("/api/v1/handwriting/features"),
                headers=self.headers,
                files=files,
            )
            r.raise_for_status()
            return r.json()

    def nlp_explain(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        with httpx.Client(timeout=self.timeout) as client:
            r = client.post(
                self._url("/api/v1/nlp/explain"),
                headers=self.headers,
                json=payload,
            )
            r.raise_for_status()
            return r.json()
