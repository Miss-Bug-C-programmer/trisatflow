from __future__ import annotations

import json
from typing import Any, Dict, Optional

import requests


class SatEdgeSimClientError(RuntimeError):
    """Raised when the SatEdgeSim REST server returns an invalid response."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        error_type: str = "request_error",
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type


from dataclasses import dataclass


@dataclass
class SatEdgeSimClient:
    """Small REST client for the long-running SatEdgeSim server.

    Expected server endpoints:
        POST /reset
        GET  /get_state
        POST /step
        GET  /get_metrics
        POST /close
        GET  /health
    """

    base_url: str = "http://127.0.0.1:8088"
    timeout: float = 60.0

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")

    def _request(self, method: str, path: str, **kwargs: Any) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        payload = kwargs.get("json")
        try:
            response = requests.request(method, url, timeout=self.timeout, **kwargs)
            response.raise_for_status()
        except requests.RequestException as exc:
            response = getattr(exc, "response", None)
            status_code = getattr(response, "status_code", None)
            error_type = "request_error"
            if isinstance(exc, requests.Timeout):
                error_type = "http_timeout"
            elif isinstance(exc, requests.ConnectionError):
                error_type = "http_connection_error"
            response_text = None
            if response is not None:
                try:
                    response_text = response.text
                except Exception:  # noqa: BLE001 - best effort diagnostics
                    response_text = "<failed to read response text>"
            payload_text = json.dumps(payload, ensure_ascii=False, sort_keys=True) if payload is not None else None
            details = [f"SatEdgeSim request failed: {method} {url}: {exc}"]
            if status_code is not None:
                details.append(f"status_code={status_code}")
            if payload_text is not None:
                details.append(f"request_payload={payload_text}")
            if response_text:
                details.append(f"response_text={response_text}")
            raise SatEdgeSimClientError(" | ".join(details), status_code=status_code, error_type=error_type) from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise SatEdgeSimClientError(f"SatEdgeSim returned non-JSON response from {method} {url}", status_code=response.status_code) from exc

        if isinstance(payload, dict) and payload.get("status") == "ERROR":
            raise SatEdgeSimClientError(
                f"SatEdgeSim error from {method} {url}: {payload.get('message', payload)}",
                status_code=response.status_code,
                error_type="server_error",
            )
        if not isinstance(payload, dict):
            raise SatEdgeSimClientError(
                f"SatEdgeSim returned unexpected payload type from {method} {url}: {type(payload)}",
                status_code=response.status_code,
            )
        payload.setdefault("_httpStatusCode", response.status_code)
        return payload

    def health(self) -> Dict[str, Any]:
        return self._request("GET", "/health")

    def version(self) -> Dict[str, Any]:
        return self._request("GET", "/version")

    def ensure_healthy(self) -> Dict[str, Any]:
        try:
            payload = self.health()
        except SatEdgeSimClientError as exc:
            raise SatEdgeSimClientError(
                f"SatEdgeSim health check failed at {self.base_url}/health. "
                f"Start the REST server first. Details: {exc}"
            ) from exc
        if payload.get("status") != "OK" and not bool(payload.get("ok", False)):
            raise SatEdgeSimClientError(f"SatEdgeSim health endpoint returned unexpected payload: {payload}")
        return payload

    def reset(
        self,
        *,
        devices_count: int = 20,
        algorithm_index: int = 0,
        architecture_index: int = 0,
        seed: int = 0,
        clean_output_folder: bool = False,
        wait_for_first_decision: bool = True,
        wait_timeout_ms: int = 30000,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "devicesCount": int(devices_count),
            "algorithmIndex": int(algorithm_index),
            "architectureIndex": int(architecture_index),
            "seed": int(seed),
            "cleanOutputFolder": bool(clean_output_folder),
            "waitForFirstDecision": bool(wait_for_first_decision),
            "waitTimeoutMs": int(wait_timeout_ms),
        }
        if extra:
            payload.update(extra)
        return self._request("POST", "/reset", json=payload)

    def get_state(self) -> Dict[str, Any]:
        return self._request("GET", "/get_state")

    def step(self, action: Dict[str, Any], *, wait_timeout_ms: int = 30000) -> Dict[str, Any]:
        payload = {
            "action": action,
            "waitTimeoutMs": int(wait_timeout_ms),
        }
        return self._request("POST", "/step", json=payload)

    def apply_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "action": action,
        }
        return self._request("POST", "/apply_action", json=payload)

    def get_metrics(self) -> Dict[str, Any]:
        return self._request("GET", "/get_metrics")

    def close(self) -> Dict[str, Any]:
        url = f"{self.base_url}/close"
        try:
            response = requests.request("POST", url, timeout=min(self.timeout, 2.0))
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            raise SatEdgeSimClientError(f"SatEdgeSim close request failed: POST {url}: {exc}", error_type="http_connection_error") from exc
        except ValueError as exc:
            raise SatEdgeSimClientError(f"SatEdgeSim returned non-JSON response from POST {url}") from exc
        if not isinstance(payload, dict):
            raise SatEdgeSimClientError(f"SatEdgeSim returned unexpected payload type from POST {url}: {type(payload)}")
        payload.setdefault("_httpStatusCode", response.status_code)
        return payload
