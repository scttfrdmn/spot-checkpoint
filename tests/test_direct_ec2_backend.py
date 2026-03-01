"""Tests for DirectEC2Backend IMDSv2 token flow."""

import json
import time
from unittest.mock import MagicMock, call, patch

import pytest

from spot_checkpoint.lifecycle import DirectEC2Backend, InterruptEvent, InterruptReason


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _MockResponse:
    """Minimal urllib response stand-in."""

    def __init__(self, status: int, body: str | bytes) -> None:
        self.status = status
        self._body = body.encode() if isinstance(body, str) else body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_MockResponse":
        return self

    def __exit__(self, *args: object) -> None:
        pass


_INTERRUPT_BODY = json.dumps({
    "action": "terminate",
    "time": "2026-06-01T12:02:00Z",
})

_TOKEN = "fake-imdsv2-token"


# ---------------------------------------------------------------------------
# Token acquisition
# ---------------------------------------------------------------------------

class TestGetImdsToken:
    def test_returns_token_on_success(self):
        backend = DirectEC2Backend()
        resp = _MockResponse(200, _TOKEN)
        with patch("urllib.request.urlopen", return_value=resp):
            token = backend._get_imds_token()
        assert token == _TOKEN

    def test_uses_correct_ttl_header(self):
        backend = DirectEC2Backend()
        resp = _MockResponse(200, _TOKEN)
        with patch("urllib.request.urlopen", return_value=resp) as mock_open:
            backend._get_imds_token()
        req = mock_open.call_args[0][0]
        assert req.get_header("X-aws-ec2-metadata-token-ttl-seconds") == str(
            DirectEC2Backend._TOKEN_TTL_SECONDS
        )

    def test_returns_none_on_exception(self):
        backend = DirectEC2Backend()
        with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
            token = backend._get_imds_token()
        assert token is None


# ---------------------------------------------------------------------------
# Token refresh logic
# ---------------------------------------------------------------------------

class TestMaybeRefreshToken:
    def test_fetches_token_when_none(self):
        backend = DirectEC2Backend()
        assert backend._imds_token is None
        resp = _MockResponse(200, _TOKEN)
        with patch("urllib.request.urlopen", return_value=resp):
            backend._maybe_refresh_token()
        assert backend._imds_token == _TOKEN
        assert backend._token_expiry > time.time()

    def test_no_refresh_when_token_still_valid(self):
        backend = DirectEC2Backend()
        backend._imds_token = _TOKEN
        backend._token_expiry = time.time() + 3600  # valid for an hour
        with patch("urllib.request.urlopen") as mock_open:
            backend._maybe_refresh_token()
        mock_open.assert_not_called()
        assert backend._imds_token == _TOKEN

    def test_refreshes_when_near_expiry(self):
        backend = DirectEC2Backend()
        backend._imds_token = "old-token"
        # Within the refresh margin
        backend._token_expiry = time.time() + backend._TOKEN_REFRESH_MARGIN - 1
        new_token = "refreshed-token"
        resp = _MockResponse(200, new_token)
        with patch("urllib.request.urlopen", return_value=resp):
            backend._maybe_refresh_token()
        assert backend._imds_token == new_token

    def test_clears_token_when_refresh_fails(self):
        backend = DirectEC2Backend()
        backend._imds_token = "old-token"
        backend._token_expiry = 0.0  # expired
        with patch("urllib.request.urlopen", side_effect=OSError("timeout")):
            backend._maybe_refresh_token()
        assert backend._imds_token is None
        assert backend._token_expiry == 0.0


# ---------------------------------------------------------------------------
# Poll loop: interrupt detection
# ---------------------------------------------------------------------------

class TestPollLoopInterruptDetection:
    def test_interrupt_detected_with_imdsv2_token(self):
        """Full flow: token acquired, interrupt endpoint returns 200, callback fires."""
        backend = DirectEC2Backend(poll_interval=0.05, interrupt_headroom=0)

        token_resp = _MockResponse(200, _TOKEN)
        interrupt_resp = _MockResponse(200, _INTERRUPT_BODY)
        no_interrupt_resp = _MockResponse(404, "")

        events: list[InterruptEvent] = []

        # urlopen call sequence: PUT (token), GET (no interrupt), GET (interrupt)
        with patch("urllib.request.urlopen", side_effect=[
            token_resp, no_interrupt_resp, interrupt_resp,
        ]):
            backend.start(on_interrupt=events.append)
            time.sleep(0.3)
            backend.stop()

        assert len(events) == 1
        assert events[0].reason == InterruptReason.SPOT_RECLAIM

    def test_token_sent_as_header_on_metadata_request(self):
        """Verify the token header is attached to the metadata GET."""
        backend = DirectEC2Backend(poll_interval=0.05, interrupt_headroom=0)

        token_resp = _MockResponse(200, _TOKEN)
        interrupt_resp = _MockResponse(200, _INTERRUPT_BODY)

        captured_requests: list[object] = []

        def _fake_urlopen(req: object, timeout: float = 2) -> _MockResponse:
            captured_requests.append(req)
            import urllib.request
            if isinstance(req, urllib.request.Request) and req.get_method() == "PUT":
                return token_resp
            return interrupt_resp

        events: list[InterruptEvent] = []
        with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            backend.start(on_interrupt=events.append)
            time.sleep(0.3)
            backend.stop()

        # Find the first GET request (not the PUT for the token)
        import urllib.request as urlreq
        get_requests = [
            r for r in captured_requests
            if isinstance(r, urlreq.Request) and r.get_method() == "GET"
        ]
        assert get_requests, "No GET requests were made"
        first_get = get_requests[0]
        assert isinstance(first_get, urlreq.Request)
        assert first_get.get_header("X-aws-ec2-metadata-token") == _TOKEN

    def test_fallback_to_imdsv1_when_token_unavailable(self):
        """When token PUT fails, metadata GET should have no token header."""
        backend = DirectEC2Backend(poll_interval=0.05, interrupt_headroom=0)

        interrupt_resp = _MockResponse(200, _INTERRUPT_BODY)

        captured_requests: list[object] = []

        def _fake_urlopen(req: object, timeout: float = 2) -> _MockResponse:
            captured_requests.append(req)
            import urllib.request
            if isinstance(req, urllib.request.Request) and req.get_method() == "PUT":
                raise OSError("IMDS token endpoint not available")
            return interrupt_resp

        events: list[InterruptEvent] = []
        with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            backend.start(on_interrupt=events.append)
            time.sleep(0.3)
            backend.stop()

        # Interrupt should still be detected (IMDSv1 fallback)
        assert len(events) == 1
        assert events[0].reason == InterruptReason.SPOT_RECLAIM

        # GET requests should have no token header
        import urllib.request as urlreq
        get_requests = [
            r for r in captured_requests
            if isinstance(r, urlreq.Request) and r.get_method() == "GET"
        ]
        assert get_requests
        first_get = get_requests[0]
        assert isinstance(first_get, urlreq.Request)
        assert first_get.get_header("X-aws-ec2-metadata-token") is None

    def test_no_interrupt_when_404(self):
        """404 from the metadata endpoint means no interruption pending."""
        backend = DirectEC2Backend(poll_interval=0.05)

        token_resp = _MockResponse(200, _TOKEN)
        no_interrupt_resp = _MockResponse(404, "")

        events: list[InterruptEvent] = []
        with patch("urllib.request.urlopen", side_effect=[token_resp] + [no_interrupt_resp] * 20):
            backend.start(on_interrupt=events.append)
            time.sleep(0.3)
            backend.stop()

        assert len(events) == 0
