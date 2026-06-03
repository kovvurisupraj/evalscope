# Copyright (c) Cerebras Systems. All rights reserved.
"""
LiveCodeBench (Pruned) — Cerebras benchmark compression extension.

Wraps the existing live_code_bench benchmark and filters questions
to a high-discrimination subset selected offline via IRT analysis.

Usage:
    evalscope eval \
        --model <model_id> \
        --datasets live_code_bench_pruned \
        --dataset-args '{"pruning_strategy": "irt_discrimination", "prune_ratio": 0.2}'

Developed against evalscope commit c14dbaf.
"""

import os
from typing import Any, Dict

import numpy as np

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages.chat_message import ChatMessageUser
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

from .pruning.irt_pruner import IRTDiscriminationPruner

logger = get_logger()

# Path to bundled precomputed score matrix (315 questions × 3 models)
_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
_DEFAULT_MATRIX = os.path.join(_DATA_DIR, 'lcb_score_matrix.npy')
_DEFAULT_INDICES = os.path.join(_DATA_DIR, 'lcb_kept_indices.npy')

_PRUNERS = {
    'irt_discrimination': IRTDiscriminationPruner(),
}


@register_benchmark(
    BenchmarkMeta(
        name='live_code_bench_pruned',
        pretty_name='Live-Code-Bench (Pruned — Cerebras)',
        tags=[Tags.CODING],
        description="""
## Overview

A pruned version of LiveCodeBench v5 that retains only the highest-discrimination
questions for efficient model quality assessment.

## Pruning Method

Questions are ranked by IRT discrimination score — the absolute correlation between
per-model pass/fail on this question and per-model overall benchmark score. Questions
where all models score identically (zero variance) are discarded. The top N questions
by discrimination score are retained.

## Default Configuration

- Strategy: irt_discrimination
- Prune ratio: 0.20 (keeps 63 of 315 questions)
- Calibrated on: gpt-oss-120b, kimi-k2.5, minimax-m2.5
- Evalscope commit: c14dbaf

## Parameters

- pruning_strategy: which pruner to use (currently: irt_discrimination)
- prune_ratio: fraction of questions to keep (default 0.20)
- score_matrix_path: path to a custom .npy score matrix (optional)
""",
        dataset_id='evalscope/livecodebench_code_generation_lite_parquet',
        subset_list=['release_v5'],
        metric_list=['acc'],
        aggregation='mean_and_pass_at_k',
        eval_split='test',
        prompt_template=(
            '### Question:\n{question_content}\n\n'
            '{format_prompt} ### Answer: '
            '(use the provided format with backticks)\n\n'
        ),
        review_timeout=6,
        extra_params={
            'pruning_strategy': {
                'type': 'str',
                'description': (
                    'Pruning strategy to use. '
                    'Supported: irt_discrimination'
                ),
                'value': 'irt_discrimination',
            },
            'prune_ratio': {
                'type': 'float',
                'description': (
                    'Fraction of questions to keep. '
                    'e.g. 0.2 = keep top 20% by discrimination score. '
                    'Validated minimum: 0.20 preserves full model ranking.'
                ),
                'value': 0.20,
            },
            'score_matrix_path': {
                'type': 'str | null',
                'description': (
                    'Path to a custom precomputed score matrix (.npy). '
                    'Shape must be (n_questions, n_models). '
                    'If null, uses the bundled matrix calibrated on '
                    'gpt-oss-120b, kimi-k2.5, minimax-m2.5.'
                ),
                'value': None,
            },
        },
    )
)
class LiveCodeBenchPrunedAdapter(DefaultDataAdapter):
    """Pruned LiveCodeBench adapter using IRT discrimination selection."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pruning_strategy = self.extra_params.get(
            'pruning_strategy', 'irt_discrimination'
        )
        self.prune_ratio = float(self.extra_params.get('prune_ratio', 0.20))
        self.score_matrix_path = self.extra_params.get('score_matrix_path')
        self._kept_indices: set = set()

    def load(self):
        dataset = super().load()
        self._kept_indices = set(self._compute_kept_indices())
        logger.info(
            f'[live_code_bench_pruned] Keeping {len(self._kept_indices)} '
            f'questions (prune_ratio={self.prune_ratio}, '
            f'strategy={self.pruning_strategy})'
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
        return pruner.select_indices(matrix, self.prune_ratio)

    def sample_filter(self, sample: Sample) -> bool:
        return sample.index in self._kept_indices

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        from evalscope.benchmarks.live_code_bench.load_utils import transform
        record = transform(record)
        question_content = record['question_content']
        format_prompt = record['format_prompt']
        full_prompt = self.prompt_template.format(
            question_content=question_content,
            format_prompt=format_prompt,
        )
        return Sample(
            input=[ChatMessageUser(content=full_prompt)],
            target='',
            metadata={
                'evaluation_sample': record['evaluation_sample'],
                'contest_date': record['contest_date'],
            },
        )

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        from evalscope.benchmarks.live_code_bench.extract_utils import (
            extract_code_generation,
        )
        return extract_code_generation(prediction)

    def match_score(
        self,
        original_prediction: str,
        filtered_prediction: str,
        reference: str,
        task_state: TaskState,
    ) -> Score:
        from evalscope.benchmarks.live_code_bench.evaluate_utils import (
            codegen_metrics,
        )
        from evalscope.utils.io_utils import convert_normal_types

        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )

        references = [{'input_output': task_state.metadata['evaluation_sample']}]
        predictions = [[filtered_prediction]]

        try:
            metrics, eval_results, final_metadata = codegen_metrics(
                references,
                predictions,
                k_list=[1],
                num_process_evaluate=1,
                timeout=self.review_timeout,
            )
            pass_rate = metrics['pass@1'] / 100
            score.value = {'acc': float(pass_rate > 0)}
            score.explanation = f"Pass@1: {metrics['pass@1']}%"
            score.metadata = {
                'pass_rate': float(pass_rate),
                'eval_results': convert_normal_types(eval_results),
                'final_metadata': convert_normal_types(final_metadata),
            }
        except Exception as e:
            score.value = {'acc': False}
            score.explanation = f'Evaluation failed: {str(e)}'
            score.metadata = {'error': str(e)}

        score.main_score_name = 'acc'
        return score
