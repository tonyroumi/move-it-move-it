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

    @property
    def size(self) -> int:
        return self._size

    @property
    def full(self) -> bool:
        return self._size == self.capacity

    @property
    def storage(self) -> torch.Tensor:
        """Underlying storage. Not necessarily chronological."""
        return self._buffer

    def append(self, samples: torch.Tensor) -> None:
        """
        Append one or more samples.

        Expected shape:
            [num_samples, *sample_shape]

        If more samples than capacity are provided, only the most recent
        `capacity` samples are retained.
        """
        if samples.ndim == self._buffer.ndim - 1:
            samples = samples.unsqueeze(0)

        num_samples = samples.shape[0]

        if num_samples == 0:
            return

        # If the incoming batch itself exceeds capacity,
        # only the newest samples can possibly survive.
        if num_samples >= self.capacity:
            samples = samples[-self.capacity:]
            num_samples = self.capacity

            self._buffer.copy_(samples)
            self._write_idx = 0
            self._size = self.capacity
            return

        # Number of samples that fit before reaching end of storage.
        first_count = min(
            num_samples,
            self.capacity - self._write_idx,
        )

        self._buffer[
            self._write_idx:self._write_idx + first_count
        ].copy_(samples[:first_count])

        remaining = num_samples - first_count

        # Wrap around to beginning.
        if remaining > 0:
            self._buffer[:remaining].copy_(samples[first_count:])

        self._write_idx = (
            self._write_idx + num_samples
        ) % self.capacity

        self._size = min(
            self._size + num_samples,
            self.capacity,
        )

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

    def get(self) -> torch.Tensor:
        """
        Return valid samples in chronological order:
        oldest -> newest.
        """
        if self._size == 0:
            return self._buffer[:0]

        if self._size < self.capacity:
            return self._buffer[:self._size]

        # Once full, write_idx points to the oldest element.
        if self._write_idx == 0:
            return self._buffer

        return torch.cat(
            (
                self._buffer[self._write_idx:],
                self._buffer[:self._write_idx],
            ),
            dim=0,
        )

    def clear(self) -> None:
        self._write_idx = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size