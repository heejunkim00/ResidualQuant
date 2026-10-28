"""Pinned prefill KV storage with bounded, overlapped device transfers."""

import torch

_MIN_PREFETCH_LAYERS = 2
_MAX_PREFETCH_LAYERS = 8
_PREFETCH_TARGET_BYTES = 800 * 1024**2


class PrefillCPUOffload:
    def __init__(self, chunk_tokens, pending_records, heads, head_dim):
        self.chunk_tokens = chunk_tokens
        self.copy_stream = torch.cuda.Stream()
        self.ready = torch.cuda.Event()
        self.copied = torch.cuda.Event()
        self.staging = None
        self.inflight = False
        self.lookahead_enabled = True
        self.full_stream = torch.cuda.Stream()
        self.full_slots = _MAX_PREFETCH_LAYERS
        self.full_workspace = None
        self.prefetched = {}
        self.full_ready = [torch.cuda.Event() for _ in range(_MAX_PREFETCH_LAYERS)]
        self.full_consumed = [torch.cuda.Event() for _ in range(_MAX_PREFETCH_LAYERS)]
        self.full_used = [False] * _MAX_PREFETCH_LAYERS
        # Allocate transfer buffers before vLLM profiles KV cache capacity.
        self.workspace = torch.empty(
            (2, pending_records, 2, chunk_tokens, heads, head_dim),
            dtype=torch.bfloat16,
            device="cuda",
        )

    def reserve(self, tokens):
        """Include transfer buffers in vLLM's pre-KV-pool memory profile."""
        pending_records = self.workspace.size(1)
        heads, head_dim = self.workspace.shape[-2:]
        self.staging = torch.empty(
            (2, tokens, heads, head_dim), dtype=torch.bfloat16, device="cuda"
        )
        self.staging.record_stream(self.copy_stream)
        if self.lookahead_enabled:
            self._reserve_full(tokens, pending_records, heads, head_dim)

    def _reserve_full(self, tokens, count, heads, head_dim):
        layer_bytes = count * 2 * tokens * heads * head_dim * 2
        self.full_slots = max(
            _MIN_PREFETCH_LAYERS,
            min(_MAX_PREFETCH_LAYERS, _PREFETCH_TARGET_BYTES // layer_bytes),
        )
        self.full_workspace = torch.empty(
            (self.full_slots, count, 2, tokens, heads, head_dim),
            dtype=torch.bfloat16,
            device="cuda",
        )
        self.full_workspace.record_stream(self.full_stream)

    def store(self, key, value, buffer):
        tokens, heads, head_dim = key.shape
        if buffer is None or buffer.size(1) < tokens:
            buffer = torch.empty(
                (2, tokens, heads, head_dim),
                dtype=key.dtype,
                device="cpu",
                pin_memory=True,
            )
        current = torch.cuda.current_stream()
        if self.inflight:
            current.wait_event(self.copied)
        if self.staging is None or self.staging.size(1) < tokens:
            self.staging = torch.empty(
                (2, tokens, heads, head_dim), dtype=key.dtype, device=key.device
            )
            self.staging.record_stream(self.copy_stream)
        # Stage on the producing stream before asynchronous D2H transfer.
        self.staging[0, :tokens].copy_(key)
        self.staging[1, :tokens].copy_(value)
        self.ready.record(current)
        with torch.cuda.stream(self.copy_stream):
            # This also orders pinned-buffer writes after previous H2D reads.
            self.copy_stream.wait_event(self.ready)
            buffer[0, :tokens].copy_(self.staging[0, :tokens], non_blocking=True)
            buffer[1, :tokens].copy_(self.staging[1, :tokens], non_blocking=True)
            self.copied.record(self.copy_stream)
        self.inflight = True
        return buffer

    def _ensure_full(self, group, tokens):
        count = len(group.pending)
        heads, head_dim = group.pending[0][0].shape[1:]
        if (
            self.full_workspace is None
            or self.full_workspace.size(3) < tokens
            or self.full_workspace.size(1) < count
        ):
            self._reserve_full(tokens, count, heads, head_dim)

    def _prefetch(self, group, tokens):
        if group in self.prefetched:
            return
        slot = group.offload_index % self.full_slots
        current = torch.cuda.current_stream()
        self.full_stream.wait_stream(current)
        if self.inflight:
            self.full_stream.wait_event(self.copied)
        with torch.cuda.stream(self.full_stream):
            if self.full_used[slot]:
                self.full_stream.wait_event(self.full_consumed[slot])
            for i, (key, value) in enumerate(group.pending):
                self.full_workspace[slot, i, 0, :tokens].copy_(
                    key[:tokens], non_blocking=True
                )
                self.full_workspace[slot, i, 1, :tokens].copy_(
                    value[:tokens], non_blocking=True
                )
            self.full_ready[slot].record(self.full_stream)
        self.prefetched[group] = slot

    def preload(self, group, tokens):
        """Start the first ring of H2D reads during the preceding loop."""
        if group.offload_index == 0:
            self.prefetched.clear()
        self._ensure_full(group, tokens)
        if group.offload_index < self.full_slots:
            self._prefetch(group, tokens)

    def prepare(self, group, tokens):
        """Refill consumed ring slots ahead of the current completing layer."""
        if group.offload_index == 0 and group not in self.prefetched:
            self.prefetched.clear()
        self._ensure_full(group, tokens)
        chain = group.offload_chain
        index = group.offload_index
        count = len(group.pending)
        for future in chain[index : index + self.full_slots]:
            if len(future.pending) == count:
                self._prefetch(future, tokens)

    def chunks(self, records, slot_mapping, group=None):
        if group is not None and group in self.prefetched:
            slot = self.prefetched.pop(group)
            current = torch.cuda.current_stream()
            current.wait_event(self.full_ready[slot])
            tokens = slot_mapping.numel()
            staged = [
                (
                    self.full_workspace[slot, i, 0, :tokens],
                    self.full_workspace[slot, i, 1, :tokens],
                )
                for i in range(len(records) - 1)
            ]
            staged.append(records[-1])
            yield staged, slot_mapping
            self.full_consumed[slot].record(current)
            self.full_used[slot] = True
            return
        current = torch.cuda.current_stream()
        # Reuse workspace only after the preceding group has consumed it.
        self.copy_stream.wait_stream(current)
        ready = [torch.cuda.Event(), torch.cuda.Event()]
        consumed = [torch.cuda.Event(), torch.cuda.Event()]
        used = [False, False]
        total = slot_mapping.numel()

        def prefetch(begin, slot):
            end = min(begin + self.chunk_tokens, total)
            count = end - begin
            with torch.cuda.stream(self.copy_stream):
                if used[slot]:
                    self.copy_stream.wait_event(consumed[slot])
                for index, (key, value) in enumerate(records[:-1]):
                    self.workspace[slot, index, 0, :count].copy_(
                        key[begin:end], non_blocking=True
                    )
                    self.workspace[slot, index, 1, :count].copy_(
                        value[begin:end], non_blocking=True
                    )
                ready[slot].record(self.copy_stream)

        if total:
            prefetch(0, 0)
        for step, begin in enumerate(range(0, total, self.chunk_tokens)):
            slot = step % 2
            end = min(begin + self.chunk_tokens, total)
            count = end - begin
            current.wait_event(ready[slot])
            if end < total:
                prefetch(end, 1 - slot)
            staged = [
                (
                    self.workspace[slot, i, 0, :count],
                    self.workspace[slot, i, 1, :count],
                )
                for i in range(len(records) - 1)
            ]
            key, value = records[-1]
            staged.append((key[begin:end], value[begin:end]))
            yield staged, slot_mapping[begin:end]
            # The caller enqueues quantization before resuming this generator.
            consumed[slot].record(current)
            used[slot] = True
