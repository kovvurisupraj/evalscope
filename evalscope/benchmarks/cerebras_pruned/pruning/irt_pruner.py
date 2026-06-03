# Copyright (c) Cerebras Systems. All rights reserved.
"""
IRT-inspired discrimination pruner.

Selects questions by their ability to separate strong models from weak ones.

Discrimination for question i is defined as the absolute Pearson correlation
between:
  - the per-model pass/fail vector for question i
  - the per-model overall ability score (mean across all questions)

This is a simplified Item Response Theory (IRT) discrimination parameter.
It is model-agnostic: the existing models are used only as a measuring
instrument to characterise question difficulty and discrimination. The
selected subset should generalise to unseen models because the questions
themselves are inherently discriminating — not because they happened to
trip up a specific model.

Questions where all models score the same (std = 0) receive discrimination
= 0 and are never selected, regardless of prune_ratio.
"""

from typing import List

import numpy as np

from .base import BasePruner


class IRTDiscriminationPruner(BasePruner):
    """Select questions by IRT discrimination score."""

    def name(self) -> str:
        return 'irt_discrimination'

    def select_indices(
        self,
        score_matrix: np.ndarray,
        prune_ratio: float,
        noise_weight: float = 1.0,
        **kwargs,
    ) -> List[int]:
        """
        Args:
            score_matrix:  (n_questions, n_models) float array.
            prune_ratio:   fraction of questions to keep (0 < prune_ratio <= 1).
            noise_weight:  multiplier applied to all discrimination scores
                           before ranking (default 1.0). Set < 1.0 for
                           benchmarks with noisy judges (e.g. AA-LCR) to
                           slightly deflate scores and reduce over-reliance
                           on marginal signal.

        Returns:
            Sorted list of question indices to keep.
        """
        if not 0 < prune_ratio <= 1.0:
            raise ValueError(f'prune_ratio must be in (0, 1], got {prune_ratio}')

        n_questions, n_models = score_matrix.shape
        n_keep = max(1, int(n_questions * prune_ratio))

        # Step 1: overall ability per model = mean score across all questions
        model_ability = score_matrix.mean(axis=0)  # shape: (n_models,)

        # Step 2: discrimination per question
        discrimination = np.zeros(n_questions)
        for i in range(n_questions):
            q_scores = score_matrix[i]
            if q_scores.std() == 0:
                # All models scored the same — zero discriminative signal
                discrimination[i] = 0.0
            else:
                corr = np.corrcoef(q_scores, model_ability)[0, 1]
                discrimination[i] = abs(corr) * noise_weight

        # Step 3: keep top-N by discrimination
        ranked = np.argsort(discrimination)[::-1]
        kept = sorted(ranked[:n_keep].tolist())
        return kept
