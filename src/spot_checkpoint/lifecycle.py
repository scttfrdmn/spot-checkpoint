"""
Spot Lifecycle Manager — interrupt detection and checkpoint orchestration.

Three backends for three deployment models:
  1. SporeLifecycleBackend  — watches spored's signal file (/tmp/spawn-spot-interruption.json)
  2. SlurmLifecycleBackend  — handles SIGTERM from Slurm's preemption, writes to
                              SLURM_CHECKPOINT_DIR
  3. DirectEC2Backend       — standalone metadata polling for bare EC2 (no spored, no Slurm)

The lifecycle manager itself is backend-agnostic: it connects an interrupt signal
to the checkpoint storage engine and computation adapter.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, TypeVar

from spot_checkpoint.protocol import Checkpointable, CheckpointPayload, CheckpointStore

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------

class InterruptReason(Enum):
    SPOT_RECLAIM = auto()       # Cloud provider reclaiming spot/preemptible
    SLURM_PREEMPT = auto()      # Slurm preempting for higher-priority job
    SLURM_TIMEOUT = auto()      # Slurm job hitting wall time
    TTL_EXPIRY = auto()         # spored TTL about to expire
    USER_SIGNAL = auto()        # Manual checkpoint request (SIGUSR1)
    PERIODIC = auto()           # Regular interval checkpoint (not an interrupt)


@dataclass
class InterruptEvent:
    reason: InterruptReason
    deadline: float             # Unix timestamp — hard cutoff
    detected_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def seconds_remaining(self) -> float:
        return max(0, self.deadline - time.time())


# ---------------------------------------------------------------------------
# Lifecycle backends — interrupt detection only
# ---------------------------------------------------------------------------

class LifecycleBackend(ABC):
    """Detects interrupts. Does NOT handle checkpointing."""

    @abstractmethod
    def start(self, on_interrupt: Callable[[InterruptEvent], None]) -> None:
        """Begin monitoring. Call on_interrupt when an event is detected."""

    @abstractmethod
    def stop(self) -> None:
        """Stop monitoring, release resources."""

    def signal_completion(self) -> None:  # noqa: B027
        """Tell the environment that work is done (optional)."""


class SporeLifecycleBackend(LifecycleBackend):
    """
    Integrates with spore.host's spored agent.

    spored already:
      - Polls EC2 metadata for spot interruption
      - Writes /tmp/spawn-spot-interruption.json on detection
      - Sends wall(1) notifications to terminals
      - Handles DNS cleanup and registry deregistration

    We don't duplicate any of that. We just watch the signal file
    and (optionally) the TTL warning file.
    """

    SPOT_SIGNAL_FILE = "/tmp/spawn-spot-interruption.json"
    WARNING_FILE = "/tmp/SPAWN_WARNING"
    COMPLETION_SIGNAL = "/tmp/spawn-job-complete"

    def __init__(
        self,
        poll_interval: float = 1.0,
        interrupt_headroom: float = 30.0,
    ):
        self._poll_interval = poll_interval
        self._headroom = interrupt_headroom
        self._on_interrupt: Callable[[InterruptEvent], None] | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._already_fired = False

    def start(self, on_interrupt: Callable[[InterruptEvent], None]) -> None:
        self._on_interrupt = on_interrupt
        self._stop_event.clear()
        self._already_fired = False

        signal.signal(signal.SIGUSR1, self._handle_sigusr1)

        self._thread = threading.Thread(
            target=self._watch_loop,
            daemon=True,
            name="spore-lifecycle-watcher",
        )
        self._thread.start()
        logger.info("SporeLifecycleBackend started — watching %s", self.SPOT_SIGNAL_FILE)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        signal.signal(signal.SIGUSR1, signal.SIG_DFL)

    def signal_completion(self) -> None:
        """Write completion file so spored can take its configured action."""
        Path(self.COMPLETION_SIGNAL).write_text(
            json.dumps({
                "completed_at": time.time(),
                "status": "checkpoint_saved",
            })
        )
        logger.info("Wrote completion signal to %s", self.COMPLETION_SIGNAL)

    def _watch_loop(self) -> None:
        while not self._stop_event.is_set():
            if not self._already_fired:
                self._check_spot_signal()
            self._stop_event.wait(self._poll_interval)

    def _check_spot_signal(self) -> None:
        signal_path = Path(self.SPOT_SIGNAL_FILE)
        if not signal_path.exists():
            return

        try:
            data = json.loads(signal_path.read_text())
            interrupt_time_str = data.get("time", "")

            from datetime import datetime
            if interrupt_time_str:
                deadline_dt = datetime.fromisoformat(
                    interrupt_time_str.replace("Z", "+00:00")
                )
                deadline = deadline_dt.timestamp() - self._headroom
            else:
                deadline = time.time() + 120 - self._headroom

            event = InterruptEvent(
                reason=InterruptReason.SPOT_RECLAIM,
                deadline=deadline,
                metadata=data,
            )

            self._already_fired = True
            logger.warning(
                "Spot interruption detected via spored — %.0fs until checkpoint deadline",
                event.seconds_remaining,
            )
            if self._on_interrupt:
                self._on_interrupt(event)

        except (json.JSONDecodeError, ValueError) as e:
            logger.error("Failed to parse spored signal file: %s", e)

    def _handle_sigusr1(self, signum: int, frame: Any) -> None:
        """Manual checkpoint request via kill -USR1."""
        if self._on_interrupt and not self._already_fired:
            event = InterruptEvent(
                reason=InterruptReason.USER_SIGNAL,
                deadline=time.time() + 300,
            )
            self._on_interrupt(event)


class SlurmLifecycleBackend(LifecycleBackend):
    """
    Integrates with Slurm's preemption and checkpoint mechanisms.

    Slurm lifecycle:
      - SIGCONT + SIGTERM on preemption (configurable grace period)
      - SLURM_CHECKPOINT_DIR env var for checkpoint storage location
      - scontrol requeue for requeueing preempted jobs
      - --signal=B:USR1@120 for time-limit warnings
    """

    def __init__(
        self,
        grace_period: float = 60.0,
        requeue: bool = True,
    ):
        self._grace_period = grace_period
        self._requeue = requeue
        self._on_interrupt: Callable[[InterruptEvent], None] | None = None
        self._original_sigterm: Any = None
        self._original_sigusr1: Any = None

    def start(self, on_interrupt: Callable[[InterruptEvent], None]) -> None:
        self._on_interrupt = on_interrupt

        self._original_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, self._handle_preempt)

        self._original_sigusr1 = signal.getsignal(signal.SIGUSR1)
        signal.signal(signal.SIGUSR1, self._handle_timeout_warning)

        job_id = os.environ.get("SLURM_JOB_ID", "unknown")
        ckpt_dir = os.environ.get("SLURM_CHECKPOINT_DIR", "")
        logger.info(
            "SlurmLifecycleBackend started — job %s, grace=%ds, requeue=%s, ckpt_dir=%s",
            job_id, self._grace_period, self._requeue, ckpt_dir or "(not set)",
        )

    def stop(self) -> None:
        if self._original_sigterm:
            signal.signal(signal.SIGTERM, self._original_sigterm)
        if self._original_sigusr1:
            signal.signal(signal.SIGUSR1, self._original_sigusr1)

    def signal_completion(self) -> None:
        pass

    def _handle_preempt(self, signum: int, frame: Any) -> None:
        logger.warning("SIGTERM received — Slurm preemption, %ds grace", self._grace_period)
        event = InterruptEvent(
            reason=InterruptReason.SLURM_PREEMPT,
            deadline=time.time() + self._grace_period,
            metadata={
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "slurm_node": os.environ.get("SLURMD_NODENAME"),
                "requeue_requested": self._requeue,
            },
        )
        if self._on_interrupt:
            self._on_interrupt(event)

    def _handle_timeout_warning(self, signum: int, frame: Any) -> None:
        logger.warning("SIGUSR1 received — Slurm time limit approaching")
        event = InterruptEvent(
            reason=InterruptReason.SLURM_TIMEOUT,
            deadline=time.time() + 120,
            metadata={"slurm_job_id": os.environ.get("SLURM_JOB_ID")},
        )
        if self._on_interrupt:
            self._on_interrupt(event)

    @staticmethod
    def checkpoint_dir() -> Path | None:
        """Get Slurm's configured checkpoint directory."""
        d = os.environ.get("SLURM_CHECKPOINT_DIR")
        if d:
            return Path(d)
        job_id = os.environ.get("SLURM_JOB_ID")
        if job_id:
            p = Path(f"/tmp/slurm-ckpt-{job_id}")
            p.mkdir(exist_ok=True)
            return p
        return None

    @staticmethod
    def request_requeue() -> None:
        """Request Slurm to requeue this job."""
        job_id = os.environ.get("SLURM_JOB_ID")
        if job_id:
            os.system(f"scontrol requeue {job_id}")
            logger.info("Requested Slurm requeue for job %s", job_id)


class DirectEC2Backend(LifecycleBackend):
    """
    Standalone metadata polling for bare EC2 without spored or Slurm.

    Fallback for people who launch raw spot instances without
    spore.host or a scheduler.

    Uses IMDSv2 (token-based auth) with a 6-hour token TTL.  Falls back
    to IMDSv1 (no token) if the PUT request fails, so it still works on
    older instances or environments where the IMDS endpoint is simulated.
    """

    METADATA_URL = "http://169.254.169.254/latest/meta-data/spot/instance-action"
    TOKEN_URL = "http://169.254.169.254/latest/api/token"

    _TOKEN_TTL_SECONDS = 21600      # 6-hour token TTL requested from IMDS
    _TOKEN_REFRESH_MARGIN = 60      # Refresh token this many seconds before expiry

    def __init__(
        self,
        poll_interval: float = 2.0,
        interrupt_headroom: float = 30.0,
    ):
        self._poll_interval = poll_interval
        self._headroom = interrupt_headroom
        self._on_interrupt: Callable[[InterruptEvent], None] | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._imds_token: str | None = None
        self._token_expiry: float = 0.0

    def start(self, on_interrupt: Callable[[InterruptEvent], None]) -> None:
        self._on_interrupt = on_interrupt
        self._stop_event.clear()

        signal.signal(signal.SIGUSR1, self._handle_sigusr1)

        self._thread = threading.Thread(
            target=self._poll_loop,
            daemon=True,
            name="ec2-metadata-poller",
        )
        self._thread.start()
        logger.info("DirectEC2Backend started — polling metadata every %.1fs", self._poll_interval)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        signal.signal(signal.SIGUSR1, signal.SIG_DFL)

    def _get_imds_token(self) -> str | None:
        """Request a fresh IMDSv2 session token from the metadata service.

        Returns:
            The token string, or None if the PUT request fails (IMDSv1 fallback).
        """
        import urllib.request
        try:
            req = urllib.request.Request(
                self.TOKEN_URL,
                method="PUT",
                headers={"X-aws-ec2-metadata-token-ttl-seconds": str(self._TOKEN_TTL_SECONDS)},
            )
            with urllib.request.urlopen(req, timeout=2) as resp:
                token = str(resp.read().decode())
                logger.debug("IMDSv2 token acquired (TTL=%ds)", self._TOKEN_TTL_SECONDS)
                return token
        except Exception as exc:
            logger.debug("IMDSv2 token request failed (%s) — falling back to IMDSv1", exc)
            return None

    def _maybe_refresh_token(self) -> None:
        """Refresh the cached IMDSv2 token if it is missing or close to expiry."""
        if time.time() + self._TOKEN_REFRESH_MARGIN < self._token_expiry:
            return  # Token still valid
        token = self._get_imds_token()
        if token is not None:
            self._imds_token = token
            self._token_expiry = time.time() + self._TOKEN_TTL_SECONDS
        else:
            # Clear cached token so we fall back to IMDSv1 on this cycle
            self._imds_token = None
            self._token_expiry = 0.0

    def _poll_loop(self) -> None:
        import urllib.request
        self._maybe_refresh_token()

        while not self._stop_event.is_set():
            self._maybe_refresh_token()
            try:
                headers: dict[str, str] = {}
                if self._imds_token:
                    headers["X-aws-ec2-metadata-token"] = self._imds_token

                req = urllib.request.Request(self.METADATA_URL, headers=headers)
                with urllib.request.urlopen(req, timeout=2) as resp:
                    if resp.status == 200:
                        data = json.loads(resp.read().decode())
                        from datetime import datetime
                        action_time = datetime.fromisoformat(
                            data["time"].replace("Z", "+00:00")
                        )
                        event = InterruptEvent(
                            reason=InterruptReason.SPOT_RECLAIM,
                            deadline=action_time.timestamp() - self._headroom,
                            metadata=data,
                        )
                        logger.warning(
                            "Spot interruption detected — %.0fs remaining",
                            event.seconds_remaining,
                        )
                        if self._on_interrupt:
                            self._on_interrupt(event)
                        return
            except Exception:
                pass

            self._stop_event.wait(self._poll_interval)

    def _handle_sigusr1(self, signum: int, frame: Any) -> None:
        if self._on_interrupt:
            self._on_interrupt(InterruptEvent(
                reason=InterruptReason.USER_SIGNAL,
                deadline=time.time() + 300,
            ))


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------

def detect_backend(**kwargs: Any) -> LifecycleBackend:
    """
    Pick the right backend based on environment.

    Priority: Slurm > spore.host > direct EC2.
    """
    if os.environ.get("SLURM_JOB_ID"):
        logger.info("Detected Slurm environment — using SlurmLifecycleBackend")
        return SlurmLifecycleBackend(**kwargs)

    if _spored_is_running():
        logger.info("Detected spored agent — using SporeLifecycleBackend")
        return SporeLifecycleBackend(**kwargs)

    logger.info("No scheduler detected — using DirectEC2Backend")
    return DirectEC2Backend(**kwargs)


def _spored_is_running() -> bool:
    """Check if spored agent is running on this instance."""
    try:
        import subprocess
        result = subprocess.run(
            ["systemctl", "is-active", "spored"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() == "active"
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Lifecycle Manager — ties backend to checkpoint engine
# ---------------------------------------------------------------------------

class SpotLifecycleManager:
    """
    Connects interrupt detection (backend) to checkpoint persistence (store + adapter).

    Usage with PySCF callback:
        mgr = SpotLifecycleManager(store, adapter)
        solver.callback = mgr.make_callback()
        solver.kernel()

    Usage with manual loop:
        with mgr:
            for step in iteration:
                do_work(step)
                mgr.check(step)
    """

    def __init__(
        self,
        store: CheckpointStore,
        adapter: Checkpointable,
        backend: LifecycleBackend | None = None,
        periodic_interval: float = 300.0,
        checkpoint_id_prefix: str = "ckpt",
    ):
        self.store = store
        self.adapter = adapter
        self.backend = backend or detect_backend()
        self.periodic_interval = periodic_interval
        self.checkpoint_id_prefix = checkpoint_id_prefix

        self._interrupt_event: InterruptEvent | None = None
        self._last_checkpoint_time = 0.0
        self._checkpoint_count = 0
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None

    def __enter__(self) -> SpotLifecycleManager:
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    def start(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever,
            daemon=True,
            name="spot-checkpoint-loop",
        )
        self._loop_thread.start()
        self.backend.start(on_interrupt=self._on_interrupt)
        self._last_checkpoint_time = time.time()
        logger.info(
            "SpotLifecycleManager started — periodic=%ds, backend=%s",
            self.periodic_interval,
            type(self.backend).__name__,
        )

    def stop(self) -> None:
        self.backend.stop()
        if self._loop is not None and self._loop_thread is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop_thread.join(timeout=30.0)
            self._loop.close()
            self._loop_thread = None
        elif self._loop is not None:
            self._loop.close()
        self._loop = None
        logger.info(
            "SpotLifecycleManager stopped — %d checkpoints written",
            self._checkpoint_count,
        )

    def _run_async(self, coro: Coroutine[Any, Any, T]) -> T:
        """Submit a coroutine to the background event loop and block until complete.

        Safe to call from both sync code and from within a running async context.

        Args:
            coro: Coroutine to execute on the background event loop.

        Returns:
            The return value of the coroutine.
        """
        assert self._loop is not None and self._loop.is_running(), (
            "_run_async called before start() or after stop()"
        )
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def make_callback(self) -> Callable[[dict[str, Any]], None]:
        """Returns a PySCF-compatible callback function."""
        self.start()

        def _callback(envs: dict[str, Any]) -> None:
            cycle = envs.get("cycle", self._checkpoint_count)
            with self._lock:
                is_emergency = self._interrupt_event is not None
                is_periodic = (
                    time.time() - self._last_checkpoint_time > self.periodic_interval
                )
            if is_emergency:
                self._do_emergency_checkpoint(cycle)
            elif is_periodic:
                self._do_periodic_checkpoint(cycle)

        return _callback

    def check(self, iteration: int = 0) -> None:
        """Call from a manual computation loop."""
        with self._lock:
            is_emergency = self._interrupt_event is not None
            is_periodic = (
                time.time() - self._last_checkpoint_time > self.periodic_interval
            )
        if is_emergency:
            self._do_emergency_checkpoint(iteration)
        elif is_periodic:
            self._do_periodic_checkpoint(iteration)

    def _do_periodic_checkpoint(self, iteration: int) -> None:
        payload = self.adapter.checkpoint_state()
        ckpt_id = f"{self.checkpoint_id_prefix}-{payload.method}-iter{iteration:06d}"

        logger.info("Periodic checkpoint: %s (%.1f MB)", ckpt_id, payload.total_bytes / 1e6)

        self._run_async(
            self.store.save_checkpoint(ckpt_id, payload.tensors, payload.metadata)
        )

        with self._lock:
            self._last_checkpoint_time = time.time()
            self._checkpoint_count += 1

    def _do_emergency_checkpoint(self, iteration: int) -> None:
        event = self._interrupt_event
        payload = self.adapter.checkpoint_state()
        ckpt_id = f"{self.checkpoint_id_prefix}-{payload.method}-emergency-iter{iteration:06d}"

        estimated_write_time = payload.total_bytes / (500 * 1024 * 1024)
        logger.warning(
            "EMERGENCY CHECKPOINT: %s (%.1f MB, est. %.1fs write, %.1fs until deadline)",
            ckpt_id,
            payload.total_bytes / 1e6,
            estimated_write_time,
            event.seconds_remaining if event else 0,
        )

        if event and estimated_write_time > event.seconds_remaining:
            logger.error(
                "Checkpoint may not complete in time! Need %.1fs, have %.1fs. Attempting anyway.",
                estimated_write_time,
                event.seconds_remaining,
            )

        self._run_async(
            self.store.save_checkpoint(ckpt_id, payload.tensors, payload.metadata)
        )

        self._checkpoint_count += 1
        logger.warning("Emergency checkpoint written: %s", ckpt_id)
        self._handle_post_emergency()

    def _handle_post_emergency(self) -> None:
        if isinstance(self.backend, SlurmLifecycleBackend):
            if self.backend._requeue:
                SlurmLifecycleBackend.request_requeue()
        elif isinstance(self.backend, SporeLifecycleBackend):
            self.backend.signal_completion()

        raise SystemExit(0)

    def _on_interrupt(self, event: InterruptEvent) -> None:
        with self._lock:
            if self._interrupt_event is not None:
                return
            self._interrupt_event = event

        logger.warning(
            "Interrupt detected: %s — %.1fs remaining",
            event.reason.name,
            event.seconds_remaining,
        )

    async def restore_latest(self) -> bool:
        """Restore computation from the most recent checkpoint."""
        checkpoints = await self.store.list_checkpoints(self.checkpoint_id_prefix)
        if not checkpoints:
            logger.info("No existing checkpoints found — starting fresh")
            return False

        latest = sorted(checkpoints, key=lambda c: c.get("timestamp", 0))[-1]
        ckpt_id = latest["checkpoint_id"]
        logger.info("Restoring from checkpoint: %s", ckpt_id)

        tensors, metadata = await self.store.load_checkpoint(ckpt_id)

        payload = CheckpointPayload(
            tensors=tensors,
            metadata=metadata,
            method=metadata.get("method", "unknown"),
            timestamp=metadata.get("timestamp", 0),
        )
        self.adapter.restore_state(payload)
        logger.info(
            "Restored: method=%s, iteration=%s",
            payload.method,
            metadata.get("iteration", "?"),
        )
        return True


# ---------------------------------------------------------------------------
# Convenience: top-level API for PySCF users
# ---------------------------------------------------------------------------

def _env_int(name: str, default: int) -> int:
    """Read an integer environment variable, falling back to default."""
    val = os.environ.get(name)
    return int(val) if val is not None else default


def spot_safe(
    solver: Any,
    bucket: str | None = None,
    job_id: str | None = None,
    periodic_interval: float | None = None,
    adapter_class: Any = None,
    **store_kwargs: Any,
) -> Callable[[dict[str, Any]], None]:
    """
    One-liner for making a PySCF solver spot-safe.

    All parameters can be set via environment variables so scripts need
    no hardcoded configuration:

        SPOT_CHECKPOINT_BUCKET       — S3 bucket (required if bucket not passed)
        SPOT_CHECKPOINT_INTERVAL     — periodic checkpoint interval in seconds (default 300)
        SPOT_CHECKPOINT_SHARD_SIZE   — shard size in bytes (default 67108864 = 64 MB)
        SPOT_CHECKPOINT_MAX_CONCURRENCY — parallel S3 streams (default 32)

    Usage:
        mf = scf.RHF(mol)
        mf.callback = spot_safe(mf, bucket="my-checkpoints")
        mf.kernel()

        # Or with env vars set:
        mf.callback = spot_safe(mf)
        mf.kernel()
    """
    if bucket is None:
        bucket = os.environ.get("SPOT_CHECKPOINT_BUCKET")
    if bucket is None:
        raise ValueError(
            "bucket is required. Pass it directly or set SPOT_CHECKPOINT_BUCKET."
        )

    if periodic_interval is None:
        periodic_interval = float(
            os.environ.get("SPOT_CHECKPOINT_INTERVAL", "300")
        )

    if adapter_class is None:
        adapter_class = _detect_adapter_class(solver)

    adapter = adapter_class(solver)

    if job_id is None:
        job_id = (
            os.environ.get("SLURM_JOB_ID")
            or os.environ.get("SPAWN_INSTANCE_ID")
            or f"pyscf-{os.getpid()}"
        )

    store_kwargs.setdefault(
        "shard_size", _env_int("SPOT_CHECKPOINT_SHARD_SIZE", 64 * 1024 * 1024)
    )
    store_kwargs.setdefault(
        "max_concurrency", _env_int("SPOT_CHECKPOINT_MAX_CONCURRENCY", 32)
    )

    from spot_checkpoint.storage import S3ShardedStore
    store = S3ShardedStore(bucket=bucket, job_id=job_id, **store_kwargs)
    mgr = SpotLifecycleManager(
        store=store,
        adapter=adapter,
        periodic_interval=periodic_interval,
    )

    return mgr.make_callback()


def spot_restore(
    solver: Any,
    bucket: str | None = None,
    job_id: str | None = None,
    checkpoint_id_prefix: str = "ckpt",
    adapter_class: Any = None,
    **store_kwargs: Any,
) -> bool:
    """Restore a PySCF solver from the latest S3 checkpoint.

    All parameters can be set via environment variables so scripts need
    no hardcoded configuration:

        SPOT_CHECKPOINT_BUCKET       — S3 bucket (required if bucket not passed)
        SPOT_CHECKPOINT_SHARD_SIZE   — shard size in bytes (default 67108864 = 64 MB)
        SPOT_CHECKPOINT_MAX_CONCURRENCY — parallel S3 streams (default 32)

    Usage:
        mf = scf.RHF(mol)
        restored = spot_restore(mf, bucket="my-checkpoints")
        mf.callback = spot_safe(mf, bucket="my-checkpoints")
        mf.kernel()

        # Or with env vars set:
        restored = spot_restore(mf)
        mf.callback = spot_safe(mf)
        mf.kernel()

    Args:
        solver: PySCF solver object to restore into.
        bucket: S3 bucket name. Falls back to SPOT_CHECKPOINT_BUCKET env var.
        job_id: Job identifier scoping the checkpoints. Defaults to
            SLURM_JOB_ID, SPAWN_INSTANCE_ID, or ``pyscf-<pid>``.
        checkpoint_id_prefix: Prefix filter for checkpoint IDs (default: "ckpt").
        adapter_class: Explicit adapter class; auto-detected if None.
        **store_kwargs: Extra keyword arguments passed to S3ShardedStore
            (e.g. ``region``, ``endpoint_url``).

    Returns:
        True if a checkpoint was found and restored, False if starting fresh.
    """
    if bucket is None:
        bucket = os.environ.get("SPOT_CHECKPOINT_BUCKET")
    if bucket is None:
        raise ValueError(
            "bucket is required. Pass it directly or set SPOT_CHECKPOINT_BUCKET."
        )

    if adapter_class is None:
        adapter_class = _detect_adapter_class(solver)

    adapter = adapter_class(solver)

    if job_id is None:
        job_id = (
            os.environ.get("SLURM_JOB_ID")
            or os.environ.get("SPAWN_INSTANCE_ID")
            or f"pyscf-{os.getpid()}"
        )

    store_kwargs.setdefault(
        "shard_size", _env_int("SPOT_CHECKPOINT_SHARD_SIZE", 64 * 1024 * 1024)
    )
    store_kwargs.setdefault(
        "max_concurrency", _env_int("SPOT_CHECKPOINT_MAX_CONCURRENCY", 32)
    )

    from spot_checkpoint.storage import S3ShardedStore
    store = S3ShardedStore(bucket=bucket, job_id=job_id, **store_kwargs)
    mgr = SpotLifecycleManager(
        store=store,
        adapter=adapter,
        checkpoint_id_prefix=checkpoint_id_prefix,
    )
    return asyncio.run(mgr.restore_latest())


async def spot_safe_async(
    solver: Any,
    bucket: str | None = None,
    job_id: str | None = None,
    periodic_interval: float | None = None,
    adapter_class: Any = None,
    **store_kwargs: Any,
) -> Callable[[dict[str, Any]], None]:
    """Async-native version of spot_safe() for use inside running event loops.

    Identical to spot_safe() but does not call asyncio.run() internally,
    making it safe to use in Jupyter notebooks, FastAPI handlers, and any
    other async framework.

    Usage:
        mf.callback = await spot_safe_async(mf, bucket="my-checkpoints")
        mf.kernel()
    """
    # Delegate setup to spot_safe (sync parts only — store creation is cheap)
    return spot_safe(
        solver,
        bucket=bucket,
        job_id=job_id,
        periodic_interval=periodic_interval,
        adapter_class=adapter_class,
        **store_kwargs,
    )


async def spot_restore_async(
    solver: Any,
    bucket: str | None = None,
    job_id: str | None = None,
    checkpoint_id_prefix: str = "ckpt",
    adapter_class: Any = None,
    **store_kwargs: Any,
) -> bool:
    """Async-native version of spot_restore() for use inside running event loops.

    Identical to spot_restore() but awaits the store operations directly
    instead of calling asyncio.run(), making it safe in Jupyter notebooks,
    FastAPI handlers, and any other async framework.

    Usage:
        restored = await spot_restore_async(mf, bucket="my-checkpoints")
        mf.callback = await spot_safe_async(mf, bucket="my-checkpoints")
        mf.kernel()

    Returns:
        True if a checkpoint was found and restored, False if starting fresh.
    """
    if bucket is None:
        bucket = os.environ.get("SPOT_CHECKPOINT_BUCKET")
    if bucket is None:
        raise ValueError(
            "bucket is required. Pass it directly or set SPOT_CHECKPOINT_BUCKET."
        )

    if adapter_class is None:
        adapter_class = _detect_adapter_class(solver)

    adapter = adapter_class(solver)

    if job_id is None:
        job_id = (
            os.environ.get("SLURM_JOB_ID")
            or os.environ.get("SPAWN_INSTANCE_ID")
            or f"pyscf-{os.getpid()}"
        )

    store_kwargs.setdefault(
        "shard_size", _env_int("SPOT_CHECKPOINT_SHARD_SIZE", 64 * 1024 * 1024)
    )
    store_kwargs.setdefault(
        "max_concurrency", _env_int("SPOT_CHECKPOINT_MAX_CONCURRENCY", 32)
    )

    from spot_checkpoint.storage import S3ShardedStore
    store = S3ShardedStore(bucket=bucket, job_id=job_id, **store_kwargs)
    mgr = SpotLifecycleManager(
        store=store,
        adapter=adapter,
        checkpoint_id_prefix=checkpoint_id_prefix,
    )
    return await mgr.restore_latest()


def _detect_adapter_class(solver: Any) -> type:
    """Map PySCF solver object → checkpoint adapter class."""
    mro_names = [cls.__name__ for cls in type(solver).__mro__]

    if any(n in mro_names for n in ("CCSD", "RCCSD", "UCCSD")):
        from spot_checkpoint.adapters.pyscf import CCSDCheckpointAdapter
        return CCSDCheckpointAdapter

    if any(n in mro_names for n in ("CASSCF", "CASCI")):
        from spot_checkpoint.adapters.pyscf import CASSCFCheckpointAdapter
        return CASSCFCheckpointAdapter

    if any(n in mro_names for n in ("SCF", "RHF", "UHF", "ROHF", "RKS", "UKS")):
        from spot_checkpoint.adapters.pyscf import SCFCheckpointAdapter
        return SCFCheckpointAdapter

    raise ValueError(
        f"Cannot auto-detect adapter for {type(solver).__name__}. "
        f"Pass adapter_class= explicitly."
    )
