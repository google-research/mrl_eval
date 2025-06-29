# coding=utf-8
# Copyright 2025 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Metrics for evaluating the different tasks."""

from collections.abc import Callable, Sequence
from typing import Any
from mrl_eval.evaluation import metrics_utils
from rouge_score import rouge_scorer
from rouge_score import scoring

MetricsFn = Callable[[list[Any], list[Any]], dict[str, float]]


def tlnls(
    targets,
    predictions,
    null_answer_text = None,
):
  """Returns the tlnls metric.

  Args:
    targets: A sequence of sequences of targets for a single example.
    predictions: Each string is a prediction for a single example.
    null_answer_text: The text of the null answer. If the prediction or any of
      the targets is equal to this text, the metric will be calculated as EM for
      this example.

  Returns:
    A dictionary with the tlnls metric.
  """

  if isinstance(targets[0], str):
    targets = [targets]

  return {
      "tlnls": metrics_utils.tlnls_calc(targets, predictions, null_answer_text)
  }


def f1(
    targets, predictions
):
  """Returns the f1 metric.

  Args:
    targets: A sequence of sequences of targets for a single example.
    predictions: Each string is a prediction for a single example.

  Returns:
    A dictionary with the f1 metric.
  """
  targets = [[metrics_utils.normalize_squad(t) for t in u] for u in targets]
  predictions = [metrics_utils.normalize_squad(p) for p in predictions]
  return {"f1": metrics_utils.f1_multi_targets(targets, predictions)}


def em(
    targets, predictions
):
  """Returns the em metric.

  Args:
    targets: A sequence of sequences of targets for a single example.
    predictions: Each string is a prediction for a single example.

  Returns:
    A dictionary with the em metric.
  """
  targets = [[metrics_utils.normalize_squad(t) for t in u] for u in targets]
  predictions = [metrics_utils.normalize_squad(p) for p in predictions]
  return {"em": metrics_utils.em_multi_targets(targets, predictions)}


def rouge(
    targets,
    predictions,
    score_keys = ("rouge1", "rouge2", "rougeL", "rougeLsum"),
    **kwargs,
):
  """Computes rouge score nondeterministically using the bootstrap.

  Args:
    targets: sequence of strings.
    predictions: sequence of strings.
    score_keys: sequence of strings with the keys to compute.
    **kwargs: additional keyword arguments for RougeScorer.

  Returns:
    dict with score_key: rouge score across all targets and predictions
  """

  scorer = rouge_scorer.RougeScorer(
      rouge_types=score_keys,
      tokenizer=metrics_utils.WhiteSpaceTokenizer(),
      **kwargs,
  )
  aggregator = scoring.BootstrapAggregator()

  for prediction, target in zip(predictions, targets):
    target = metrics_utils.prepare_summary_rouge(target)
    prediction = metrics_utils.prepare_summary_rouge(prediction)
    aggregator.add_scores(scorer.score(target=target, prediction=prediction))
  result = aggregator.aggregate()
  return {key: result[key].mid.fmeasure * 100 for key in score_keys}


def get_macro_f1_fn(expected_classes):
  """Returns a macro f1 metric function.

  Args:
    expected_classes: A sequence of expected classes.

  Returns:
    A macro f1 metric function.
  """

  def _macro_f1(
      targets, predictions
  ):
    per_class_f1 = metrics_utils.per_class_f1(
        targets, predictions, expected_classes
    )
    return {
        "macro_f1": sum(per_class_f1.values()) / len(expected_classes),
        **per_class_f1,
    }

  return _macro_f1


def accuracy(
    targets, predictions
):
  """Computes the accuracy score."""
  return {"accuracy": metrics_utils.accuracy(targets, predictions)}


def token_level_span_f1(
    targets, predictions
):
  """Computes the accuracy score."""
  return {
      "token_level_span_f1": metrics_utils.token_level_span_f1(
          targets, predictions
      )
  }


def get_em_cluster_matching_f1_fn(
    seq_to_cluster_parsing_fn,
):
  """Returns a cluster matching f1 metric function.

  Args:
    seq_to_cluster_parsing_fn: A function that parses a string into a list of
      clusters.

  Returns:
    A cluster matching f1 metric function.
  """

  def cluster_matching_f1(
      targets,
      predictions,
  ):
    """Computes the cluster matching f1 score."""
    matched_clusters = []
    # pair the gt clusters with the pred clusters in each example to caclulate
    # the f1 score stats:
    for target, prediction in zip(targets, predictions):
      gold_clusters = seq_to_cluster_parsing_fn(target)
      predicted_clusters = seq_to_cluster_parsing_fn(prediction)
      matched_clusters += metrics_utils.average_score_match_clusters(
          gold_clusters, predicted_clusters
      )
    macro_f1 = metrics_utils.macro_f1_for_matching_clusters(
        matched_clusters,
        comparing_fn=metrics_utils.exactly_comparing_clusters,
    )
    return {
        "macro_f1": macro_f1,
    }

  return cluster_matching_f1
