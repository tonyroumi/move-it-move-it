"""Training-log counts of how often each manifest motion is sampled and each behavior completes.

Pure torch, so it is tested without a simulator.
"""

import numpy as np
import torch

from .behavior import BehaviorManager
from .motion_set import MotionSet
from .utils.logger import Logger

SAMPLE_HISTOGRAM_TAG = "Motion Sample Histogram"
"""Histogram of the motion ids drawn since the previous write. Bucket i is the manifest's i-th motion."""
SAMPLE_COUNT_PREFIX = "Motion Samples"
"""`<prefix>/<motion>`: how many times that motion has been drawn since training started."""
COMPLETION_COUNT_PREFIX = "Behavior Completions"
"""`<prefix>/<behavior>`: how many times that behavior has completed since training started."""


class MotionMetrics:
    """Logs `MotionSet.sample_counts` and `BehaviorManager.completions` each time the logger writes.

    A draw is any `sample_ids` or `sample_locomotion_ids` call: env resets, discriminator reference
    batches and the locomotion motions behavior resets return to. Every manifest motion has a
    histogram bucket and a count, including one never drawn.
    """

    def __init__(self, motions: MotionSet, behaviors: BehaviorManager):
        self._motions = motions
        self._behaviors = behaviors
        self._bins = np.arange(motions.num_motions + 1, dtype=np.float64) - 0.5
        self._behavior_names = [motions.names[motion_id] for motion_id in motions.behavior_motion_ids.tolist()]
        self._written_sample_counts = torch.zeros(motions.num_motions, dtype=torch.long)

    def write(self, logger: Logger) -> None:
        """Log the histogram of draws since the previous write, and each motion's and behavior's total."""
        sample_counts = self._motions.sample_counts.cpu().clone()
        new_samples = sample_counts - self._written_sample_counts
        self._written_sample_counts = sample_counts
        sampled_ids = torch.repeat_interleave(torch.arange(self._motions.num_motions), new_samples)
        logger.log_histogram(SAMPLE_HISTOGRAM_TAG, sampled_ids.numpy(), logger.timestep, bins=self._bins)

        scalars = {
            f"{SAMPLE_COUNT_PREFIX}/{name}": count for name, count in zip(self._motions.names, sample_counts.tolist())
        }
        completions = self._behaviors.completions.tolist()
        scalars.update(
            {f"{COMPLETION_COUNT_PREFIX}/{name}": count for name, count in zip(self._behavior_names, completions)}
        )
        logger.log_scalars(scalars, logger.timestep)
