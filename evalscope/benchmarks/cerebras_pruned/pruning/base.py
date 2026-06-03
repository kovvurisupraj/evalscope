# Copyright (c) Cerebras Systems. All rights reserved.
from abc import ABC, abstractmethod
from typing import List
import numpy as np


class BasePruner(ABC):
    """
    Abstract base class for benchmark pruners.

    A pruner takes a precomputed score matrix (n_questions × n_models)
    and returns the indices of questions to keep, given a target prune ratio.

    This design is model-agnostic: the score matrix is computed offline from
    existing model runs, and used only to characterise question properties —
    not to select questions that favour specific models.
    """

    @abstractmethod
    def select_indices(
        self,
        score_matrix: np.ndarray,
        prune_ratio: float,
        **kwargs,
    ) -> List[int]:
        """
        Select question indices to keep.

        Args:
            score_matrix: float array of shape (n_questions, n_models).
                          Each cell is the score (0.0 or 1.0) that a model
                          received on a question.
            prune_ratio:  fraction of questions to KEEP (e.g. 0.2 = keep 20%).
            **kwargs:     strategy-specific keyword arguments.

        Returns:
            Sorted list of integer question indices to keep.
        """
        ...

    @abstractmethod
    def name(self) -> str:
        """Short identifier for this strategy (used in dataset-args)."""
        ...
