import torch


class CircularBuffer:
    """Fixed-capacity FIFO circular buffer."""

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

    def append(self, samples: torch.Tensor) -> None:
        """Append one sample or a batch, keeping only the newest `capacity` samples."""
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

    def sample(self, batch_size: int) -> torch.Tensor:
        """Uniformly sample `batch_size` stored samples, with replacement."""
        if self._size == 0:
            raise RuntimeError("Cannot sample from an empty buffer.")

        indices = torch.randint(
            0,
            self._size,
            (batch_size,),
            device=self._buffer.device,
        )

        return self._buffer[indices]

    def clear(self) -> None:
        self._write_idx = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size
