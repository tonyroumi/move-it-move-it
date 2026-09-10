import torch


class TensorCircularBuffer:
    def __init__(
        self,
        capacity: int,
        sample_shape: tuple[int, ...],
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ):
        self.capacity = capacity

        self._buffer = torch.zeros(
            (capacity, *sample_shape),
            device=device,
            dtype=dtype,
        )

        self._write_idx = 0
        self._size = 0

        # Monotonic counters (never wrapped) used to compute how many
        # new samples have arrived since the last windowed retrieval.
        self._total_appended = 0
        self._last_window_total = 0

    def append(self, samples: torch.Tensor) -> None:
        if samples.ndim == self._buffer.ndim - 1:
            samples = samples.unsqueeze(0)

        num_samples = samples.shape[0]

        if num_samples == 0:
            return

        if num_samples >= self.capacity:
            samples = samples[-self.capacity:]
            num_samples = self.capacity

            self._buffer.copy_(samples)
            self._write_idx = 0
            self._size = self.capacity
            self._total_appended += num_samples
            return

        first_count = min(
            num_samples,
            self.capacity - self._write_idx,
        )

        self._buffer[
            self._write_idx:self._write_idx + first_count
        ].copy_(samples[:first_count])

        remaining = num_samples - first_count

        if remaining > 0:
            self._buffer[:remaining].copy_(samples[first_count:])

        self._write_idx = (
            self._write_idx + num_samples
        ) % self.capacity

        self._size = min(
            self._size + num_samples,
            self.capacity,
        )

        self._total_appended += num_samples

    def sample(self, batch_size: int) -> torch.Tensor:
        if self._size == 0:
            raise RuntimeError("Cannot sample from an empty buffer.")

        indices = torch.randint(
            0,
            self._size,
            (batch_size,),
            device=self._buffer.device,
        )

        return self._buffer[indices]

    def _slice_circular(self, start: int, count: int) -> torch.Tensor:
        """
        Return `count` samples starting at physical index `start`,
        wrapping around the end of storage if needed.
        """
        if count == 0:
            return self._buffer[:0]

        end = start + count
        if end <= self.capacity:
            return self._buffer[start:end]

        first_part = self._buffer[start:]
        second_part = self._buffer[:end - self.capacity]
        return torch.cat((first_part, second_part), dim=0)

    def get_since_last(self) -> torch.Tensor:
        """
        Return samples appended since the last call to `get_since_last`,
        in chronological order (oldest -> newest), then advance the
        internal marker so the next call only returns newer samples.

        Intended for the rollout -> update pattern: append every env
        step, then call this once after rollout collection to pull out
        exactly that rollout's observations. If more samples were
        appended than `capacity` since the last call, older ones have
        already been overwritten, so this is clamped to `capacity`
        (equivalent to a full `get()`).
        """
        num_new = min(
            self._total_appended - self._last_window_total,
            self._size,
        )

        self._last_window_total = self._total_appended

        if num_new == 0:
            return self._buffer[:0]

        # The window ends at the sample most recently written, i.e.
        # physical index (write_idx - 1), and spans `num_new` samples
        # backward from there.
        start = (self._write_idx - num_new) % self.capacity
        return self._slice_circular(start, num_new)

    def clear(self) -> None:
        self._write_idx = 0
        self._size = 0
        self._total_appended = 0
        self._last_window_total = 0

    def __len__(self) -> int:
        return self._size