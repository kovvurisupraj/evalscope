# Copyright (c) Cerebras Systems. All rights reserved.
"""
AA-LCR (Pruned) — Cerebras benchmark compression extension.

Wraps the existing aa_lcr benchmark and filters questions to a
high-discrimination subset selected offline via IRT analysis.

Note on noise: AA-LCR is graded by an LLM judge which is non-deterministic.
A noise_weight of 0.9 is applied by default to slightly deflate discrimination
scores and reduce over-reliance on marginal signal from noisy judgements.

Usage:
    evalscope eval \
        --model <model_id> \
        --datasets aa_lcr_pruned \
        --dataset-args '{"pruning_strategy": "irt_discrimination", "prune_ratio": 0.3}'

Developed against evalscope commit c14dbaf.
"""

import os
from typing import Any, Dict

import numpy as np

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

from .pruning.irt_pruner import IRTDiscriminationPruner

logger = get_logger()

_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
_DEFAULT_MATRIX = os.path.join(_DATA_DIR, 'aa_lcr_score_matrix.npy')

_PRUNERS = {
    'irt_discrimination': IRTDiscriminationPruner(),
}


@register_benchmark(
    BenchmarkMeta(
        name='aa_lcr_pruned',
        pretty_name='AA-LCR (Pruned — Cerebras)',
        tags=[Tags.KNOWLEDGE, Tags.REASONING, Tags.LONG_CONTEXT],
        description="""
## Overview

A pruned version of AA-LCR that retains only the highest-discrimination
questions for efficient long-context reasoning assessment.

## Note on Noise

AA-LCR uses an LLM judge for scoring, which introduces non-determinism.
A noise_weight parameter (default 0.9) slightly deflates discrimination
scores to reduce over-reliance on marginal signal from noisy judgements.

## Default Configuration

- Strategy: irt_discrimination
- Prune ratio: 0.30 (keeps 30 of 100 questions)
- Noise weight: 0.9
- Calibrated on: gpt-oss-120b, kimi-k2.5, minimax-m2.5
- Evalscope commit: c14dbaf
""",
        dataset_id='evalscope/AA-LCR',
        metric_list=['acc'],
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template=(
            'BEGIN INPUT DOCUMENTS\n\n'
            '{documents_text}\n\n'
            'END INPUT DOCUMENTS\n\n'
            'Answer the following question using the input documents provided above.\n\n'
            'START QUESTION\n\n{question}\n\nEND QUESTION\n'
        ),
        extra_params={
            'pruning_strategy': {
                'type': 'str',
                'description': 'Pruning strategy. Supported: irt_discrimination',
                'value': 'irt_discrimination',
            },
            'prune_ratio': {
                'type': 'float',
                'description': (
                    'Fraction of questions to keep. '
                    'Validated minimum: 0.30 preserves full model ranking.'
                ),
                'value': 0.30,
            },
            'noise_weight': {
                'type': 'float',
                'description': (
                    'Multiplier applied to discrimination scores before ranking. '
                    'Values < 1.0 reduce reliance on noisy LLM judge signal. '
                    'Default: 0.9'
                ),
                'value': 0.9,
            },
            'score_matrix_path': {
                'type': 'str | null',
                'description': (
                    'Path to a custom score matrix (.npy). '
                    'If null, uses the bundled matrix.'
                ),
                'value': None,
            },
            'text_dir': {
                'type': 'str | null',
                'description': (
                    'Local directory containing extracted AA-LCR text files. '
                    'If null, auto-downloads.'
                ),
                'value': None,
            },
        },
    )
)
class AALCRPrunedAdapter(DefaultDataAdapter):
    """Pruned AA-LCR adapter using IRT discrimination selection."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pruning_strategy = self.extra_params.get(
            'pruning_strategy', 'irt_discrimination'
        )
        self.prune_ratio = float(self.extra_params.get('prune_ratio', 0.30))
        self.noise_weight = float(self.extra_params.get('noise_weight', 0.9))
        self.score_matrix_path = self.extra_params.get('score_matrix_path')
        self.text_dir = self.extra_params.get('text_dir')
        self._kept_indices: set = set()

    def load(self):
        dataset = super().load()
        self._kept_indices = set(self._compute_kept_indices())
        logger.info(
            f'[aa_lcr_pruned] Keeping {len(self._kept_indices)} questions '
            f'(prune_ratio={self.prune_ratio}, '
            f'strategy={self.pruning_strategy}, '
            f'noise_weight={self.noise_weight})'
        )
        return dataset

    def _compute_kept_indices(self):
        matrix_path = self.score_matrix_path or _DEFAULT_MATRIX

        if not os.path.exists(matrix_path):
            raise FileNotFoundError(
                f'Score matrix not found: {matrix_path}. '
                f'Re-run scripts/build_score_matrices.py to regenerate.'
            )

        pruner = _PRUNERS.get(self.pruning_strategy)
        if pruner is None:
            raise ValueError(
                f'Unknown pruning strategy: {self.pruning_strategy}. '
                f'Supported: {list(_PRUNERS.keys())}'
            )

        matrix = np.load(matrix_path)
        return pruner.select_indices(
            matrix,
            self.prune_ratio,
            noise_weight=self.noise_weight,
        )

    def sample_filter(self, sample: Sample) -> bool:
        return sample.index in self._kept_indices

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        from evalscope.benchmarks.aa_lcr.aa_lcr_adapter import AALCRAdapter
        # reuse context loading from existing adapter
        adapter = AALCRAdapter.__new__(AALCRAdapter)
        adapter.text_dir = self.text_dir
        context = adapter._get_context(record)
        prompt = self.prompt_template.format(
            documents_text=context,
            question=record['question'],
        )
        return Sample(
            input=[ChatMessageUser(content=prompt)],
            target=record['answer'],
            metadata={
                'question': record['question'],
                'data_source_urls': record['data_source_urls'],
                'input_tokens': record.get('input_tokens', 0),
            },
        )

    def llm_match_score(
        self,
        original_prediction: str,
        filtered_prediction: str,
        reference: str,
        task_state: TaskState,
    ) -> Score:
        import re
        from evalscope.benchmarks.aa_lcr.aa_lcr_adapter import JUDGE_PROMPT

        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )

        judge_prompt = JUDGE_PROMPT.format(
            question=task_state.metadata['question'],
            correct_answer=reference,
            response=filtered_prediction,
        )

        judge_response = self.llm_judge.judge(prompt=judge_prompt)
        is_correct = bool(
            re.search(r'\bCORRECT\b', judge_response, re.IGNORECASE)
        )
        score.value = {'acc': 1.0 if is_correct else 0.0}
        score.explanation = f'LLM judge: {judge_response}'
        score.metadata = {
            'source': 'llm_judge',
            'judge_strategy': self.judge_strategy,
            'model': self.llm_judge.model_id,
        }
        score.main_score_name = 'acc'
        return score
