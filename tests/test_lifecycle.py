"""Tests for lifecycle backends and SpotLifecycleManager."""

import json
import os
import time
from pathlib import Path
from unittest.mock import patch

from spot_checkpoint.lifecycle import (
    DirectEC2Backend,
    InterruptEvent,
    InterruptReason,
    SlurmLifecycleBackend,
    SporeLifecycleBackend,
    detect_backend,
)


class TestSporeLifecycleBackend:
    def test_detects_signal_file(self, tmp_path: Path):
        """Backend fires callback when spored signal file appears."""
        backend = SporeLifecycleBackend(poll_interval=0.1, interrupt_headroom=10)
        backend.SPOT_SIGNAL_FILE = str(tmp_path / "spot-signal.json")

        events: list[InterruptEvent] = []
        backend.start(on_interrupt=events.append)

        # Write signal file
        signal_data = {
            "event": "spot-interruption",
            "action": "terminate",
            "time": "2026-01-01T12:02:00Z",
            "detected_at": "2026-01-01T12:00:00Z",
        }
        Path(backend.SPOT_SIGNAL_FILE).write_text(json.dumps(signal_data))

        time.sleep(0.3)
        backend.stop()

        assert len(events) == 1
        assert events[0].reason == InterruptReason.SPOT_RECLAIM

    def test_no_false_positive(self, tmp_path: Path):
        """Backend does not fire when no signal file exists."""
        backend = SporeLifecycleBackend(poll_interval=0.1)
        backend.SPOT_SIGNAL_FILE = str(tmp_path / "nonexistent.json")

        events: list[InterruptEvent] = []
        backend.start(on_interrupt=events.append)

        time.sleep(0.3)
        backend.stop()

        assert len(events) == 0


class TestSlurmLifecycleBackend:
    def test_detect_slurm_env(self):
        """detect_backend picks Slurm when SLURM_JOB_ID is set."""
        with patch.dict(os.environ, {"SLURM_JOB_ID": "12345"}):
            backend = detect_backend()
            assert isinstance(backend, SlurmLifecycleBackend)


class TestDetectBackend:
    def test_fallback_to_direct_ec2(self):
        """Without Slurm or spored, falls back to DirectEC2Backend."""
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("spot_checkpoint.lifecycle._spored_is_running", return_value=False),
        ):
            backend = detect_backend()
            assert isinstance(backend, DirectEC2Backend)

    def test_spored_detected(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("spot_checkpoint.lifecycle._spored_is_running", return_value=True),
        ):
            backend = detect_backend()
            assert isinstance(backend, SporeLifecycleBackend)
