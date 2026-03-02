"""
PyTorch training checkpoint adapter.

Checkpoints model weights (all parameters and buffers) and optimizer state
as numpy arrays, enabling resume after preemption without restarting from
epoch zero.

PyTorch is an optional dependency — this module imports it lazily at
checkpoint/restore time and fails with a clear message if not installed.

Device handling:
    Arrays are always detached to CPU before checkpointing.  On restore,
    each tensor is moved to the device inferred from ``next(model.parameters())``.

Large model support:
    Each ``"model/{name}"`` is a separate tensor key, so the S3 sharding
    layer can write all parameter shards in parallel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from spot_checkpoint.protocol import AdapterError, CheckpointPayload

if TYPE_CHECKING:
    import torch
    import torch.nn
    import torch.optim


class PyTorchTrainingAdapter:
    """Checkpoint adapter for a PyTorch training loop.

    Saves model weights (parameters + buffers) and optimizer state as numpy
    arrays.  On restore, weights are moved to the device currently used by
    the model.

    Args:
        model: The ``torch.nn.Module`` being trained.
        optimizer: The ``torch.optim.Optimizer`` instance.
        epoch: Current epoch index (updated by caller between checkpoints).
        step: Current global step (updated by caller between checkpoints).
        loss: Most recent training loss (updated by caller; may be ``None``).

    Raises:
        AdapterError: At checkpoint time if the model has no parameters.

    Example:
        >>> adapter = PyTorchTrainingAdapter(model, optimizer)
        >>> for epoch in range(start_epoch, num_epochs):
        ...     adapter.epoch = epoch
        ...     for step, batch in enumerate(loader):
        ...         loss = train_step(batch)
        ...         adapter.step += 1
        ...         adapter.loss = loss.item()
        ...     # lifecycle manager checkpoints adapter when needed
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int = 0,
        step: int = 0,
        loss: float | None = None,
    ) -> None:
        self._model = model
        self._optimizer = optimizer
        self.epoch = epoch
        self.step = step
        self.loss = loss

    def checkpoint_state(self) -> CheckpointPayload:
        """Capture model weights and optimizer state.

        Returns:
            Payload with tensors keyed as ``"model/{name}"`` for weights
            and ``"opt/state/{idx}/{key}"`` for optimizer tensors.

        Raises:
            AdapterError: If the model has no parameters/buffers.
        """
        import torch  # noqa: F401 — lazy import; raise clear error if absent

        if sum(1 for _ in self._model.parameters()) == 0:
            raise AdapterError(
                "Model has no parameters — nothing to checkpoint. "
                "Is the model correctly initialised?"
            )

        tensors: dict[str, np.ndarray] = {}

        # --- Model weights (parameters + buffers) ---
        for name, tensor in self._model.state_dict().items():
            tensors[f"model/{name}"] = tensor.detach().cpu().numpy()

        # --- Optimizer state ---
        opt_sd = self._optimizer.state_dict()
        non_tensor_state: dict[str, Any] = {}

        for param_idx, param_state in opt_sd["state"].items():
            for key, val in param_state.items():
                if hasattr(val, "detach"):  # torch.Tensor
                    tensors[f"opt/state/{param_idx}/{key}"] = (
                        val.detach().cpu().numpy()
                    )
                else:
                    non_tensor_state.setdefault(str(param_idx), {})[key] = val

        import torch as _torch

        return CheckpointPayload(
            tensors=tensors,
            metadata={
                "epoch": self.epoch,
                "step": self.step,
                "loss": self.loss,
                "opt_param_groups": opt_sd["param_groups"],
                "opt_non_tensor_state": non_tensor_state,
                "torch_version": _torch.__version__,
            },
            method="pytorch-training",
        )

    def restore_state(self, payload: CheckpointPayload) -> None:
        """Restore model weights and optimizer state from a checkpoint.

        Args:
            payload: Previously saved checkpoint payload.
        """
        import torch

        # Determine target device from the current model parameters
        try:
            device = next(self._model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        # --- Restore model ---
        model_sd = {
            name.removeprefix("model/"): torch.from_numpy(arr.copy()).to(device)
            for name, arr in payload.tensors.items()
            if name.startswith("model/")
        }
        self._model.load_state_dict(model_sd)

        # --- Reconstruct optimizer state ---
        opt_state: dict[int, dict[str, Any]] = {}

        for name, arr in payload.tensors.items():
            if name.startswith("opt/state/"):
                parts = name.split("/")  # ["opt", "state", idx, key]
                idx = int(parts[2])
                key = parts[3]
                opt_state.setdefault(idx, {})[key] = (
                    torch.from_numpy(arr.copy()).to(device)
                )

        non_tensor: dict[str, Any] = payload.metadata.get("opt_non_tensor_state", {})
        for idx_str, vals in non_tensor.items():
            opt_state.setdefault(int(idx_str), {}).update(vals)

        self._optimizer.load_state_dict(
            {
                "state": opt_state,
                "param_groups": payload.metadata["opt_param_groups"],
            }
        )

        self.epoch = int(payload.metadata.get("epoch", 0))
        self.step = int(payload.metadata.get("step", 0))
        loss_val = payload.metadata.get("loss")
        self.loss = float(loss_val) if loss_val is not None else None

    @property
    def checkpoint_size_estimate(self) -> int:
        """Estimated bytes: sum of all model parameter/buffer nbytes."""
        return int(
            sum(int(p.numel()) * int(p.element_size()) for p in self._model.parameters())
            + sum(int(b.numel()) * int(b.element_size()) for b in self._model.buffers())
        )


__all__: list[str] = ["PyTorchTrainingAdapter"]
