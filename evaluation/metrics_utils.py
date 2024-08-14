# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Utility functions for calculating metrics for text generation."""

import collections
import re
import string
from typing import Callable, List, Sequence, Tuple, Union
import Levenshtein
import numpy as np

Tag = str
Entity = str
TaggedEntities = List[Tuple[Tag, Entity]]

WordLabels = List[str]
SentenceLabels = List[WordLabels]
DatasetLabels = List[SentenceLabels]


def _normalize_answer(text, punc_chars, punc_repl):
  """Lower text and remove punctuation, articles and extra whitespace."""

  def remove_articles(s):
    return re.sub(r"\b(a|an|the)\b", " ", s)

  def replace_punctuation(s):
    to_replace = set(punc_chars)
    return "".join(punc_repl if ch in to_replace else ch for ch in s)

  def white_space_fix(s):
    return " ".join(s.split())

  text = text.lower()
  text = replace_punctuation(text)
  text = remove_articles(text)
  text = white_space_fix(text)

  return text


def normalize_squad(answer):
  """Normalization used in official SQuAD evaluation script."""
  return _normalize_answer(answer, punc_chars=string.punctuation, punc_repl="")


def _split_to_squad_tokens(text):
  return normalize_squad(text).split()


def _metric_max_over_ground_truths(
    targets,
    prediction,
    metric_fn,
):
  """Computes the maximum of the metric over all ground truths."""
  return max(metric_fn(ground_truth, prediction) for ground_truth in targets)


def _average_max_over_ground_truths(
    targets,
    predictions,
    metric_fn,
):
  """Computes the maximum of the metric over all ground truths."""
  return (
      np.mean([
          _metric_max_over_ground_truths(t, p, metric_fn)
          for p, t in zip(predictions, targets)
      ])
      * 100
  )


def _exact_match_score(target, prediction):
  return target == prediction


def _calc_f1(precision, recall):
  return 2.0 * ((precision * recall) / (precision + recall + 1e-13))


def _f1_score(target, prediction):
  """Computes token f1 score for a single target and prediction."""
  prediction_tokens = _split_to_squad_tokens(prediction)
  target_tokens = _split_to_squad_tokens(target)
  common = collections.Counter(prediction_tokens) & collections.Counter(
      target_tokens
  )
  num_same = sum(common.values())
  if num_same == 0:
    return 0
  precision = 1.0 * num_same / len(prediction_tokens)
  recall = 1.0 * num_same / len(target_tokens)
  return _calc_f1(precision, recall)


def f1_multi_targets(
    targets, predictions
):
  return _average_max_over_ground_truths(targets, predictions, _f1_score)


def em_multi_targets(
    targets, predictions
):
  return _average_max_over_ground_truths(
      targets, predictions, _exact_match_score
  )


def _levenshtein_norm(text1, text2):
  """Calculates the normalized Levenshtein distance between two strings."""
  distance = Levenshtein.distance(text1, text2)
  return float(distance) / max(len(text1), len(text2))


def _tlnls_single_prediction(target, prediction):
  """Computes the token-level normalized Levenshtein similarity."""
  target_tokens = _split_to_squad_tokens(target)
  prediction_tokens = _split_to_squad_tokens(prediction)
  if target_tokens and not prediction_tokens:
    return 0

  score = 0
  for t_tok in target_tokens:
    score += max(
        [1 - _levenshtein_norm(p_tok, t_tok) for p_tok in prediction_tokens]
    )

  score /= max(len(target_tokens), len(prediction_tokens))
  return score


def tlnls_calc(
    targets, predictions
):
  """Computes the token-level normalized Levenshtein similarity."""

  def _is_half_or_more_digits(text):
    digit_count = sum(char.isdigit() for char in text)
    return digit_count >= len(text) / 2

  score = 0
  for target_options, prediction in zip(targets, predictions):
    if _is_half_or_more_digits(prediction):
      single_prediction_metric_fn = _f1_score
    else:
      single_prediction_metric_fn = _tlnls_single_prediction

    score += _metric_max_over_ground_truths(
        target_options, prediction, single_prediction_metric_fn
    )
  tlnls_score = score * 100 / len(targets)
  return tlnls_score


def prepare_summary_rouge(summary):
  """Add newlines between sentences so that rougeLsum is computed correctly."""
  summary = summary.replace(" . ", " .\n")
  return summary


class WhiteSpaceTokenizer:

  def tokenize(self, text):
    return text.split()


def per_class_f1(
    targets,
    predictions,
    expected_classes,
):
  """Calculates the per-class F1 score."""
  tp_fp_fn_per_label = {
      label: {"tp": 0, "fp": 0, "fn": 0} for label in expected_classes
  }

  for target, prediction in zip(targets, predictions):
    if target == prediction:
      tp_fp_fn_per_label[target]["tp"] += 1
    else:
      tp_fp_fn_per_label[target]["fn"] += 1

      if prediction in expected_classes:
        tp_fp_fn_per_label[prediction]["fp"] += 1

  per_class_f1_scores = {}
  for label in tp_fp_fn_per_label:  # Iterate over labels and counts
    tp_fp_fn = tp_fp_fn_per_label[label]
    tp, fp, fn = tp_fp_fn["tp"], tp_fp_fn["fp"], tp_fp_fn["fn"]
    if tp == 0:
      class_f1 = 0
    else:
      precision = tp / (tp + fp)
      recall = tp / (tp + fn)
      class_f1 = _calc_f1(precision, recall)
    per_class_f1_scores[f"f1_{label}"] = class_f1 * 100
  return per_class_f1_scores


def accuracy(targets, predictions):
  return np.mean([t == p for t, p in zip(targets, predictions)]) * 100


def token_level_span_f1(
    targets,
    predictions,
):
  """Computes Span based F1 score for token level NER.

  Args:
    targets: sequence of strings or sequence of sequence of strings if multiple
      references are present.
    predictions: sequence of strings

  Returns:
    span f1 across all targets and predictions
  """
  true_positives = collections.defaultdict(int)
  false_positives = collections.defaultdict(int)
  false_negatives = collections.defaultdict(int)

  def compute_f1_metrics(true_positives, false_positives, false_negatives):
    precision = float(true_positives) / float(
        true_positives + false_positives + 1e-56
    )
    recall = float(true_positives) / float(
        true_positives + false_negatives + 1e-56
    )
    return precision, recall, _calc_f1(precision, recall)

  for target, pred in zip(targets, predictions):
    gold_spans = entity_markers_to_spans(target)
    predicted_spans = entity_markers_to_spans(pred)

    for span in predicted_spans:
      if span in gold_spans:
        true_positives[span[0]] += 1
        gold_spans.remove(span)
      else:
        false_positives[span[0]] += 1
    # These spans weren't predicted.
    for span in gold_spans:
      false_negatives[span[0]] += 1

  _, _, f1_measure = compute_f1_metrics(
      sum(true_positives.values()),
      sum(false_positives.values()),
      sum(false_negatives.values()),
  )

  return f1_measure * 100


def entity_markers_to_spans(tag_sequence):
  """Extracts spans from entity markers sequences with BIO or BIOSE tags.

  This function receives a string with entity markers with BIO/BIOSE tags and
  returns a list of tuples of tags and entities.
  For example: entity_markers_to_spans("[PER Barack Obama] was born in [GPE
  Honolulu]") will return [("PER", "Barack Obama"), ("GPE", "Honolulu")].

  Our methodology from extracting entities is taking all strings between closest
  brackets where the first token is the tag and the rest is the entity name.

  Args:
    tag_sequence: string of alternativing words and tags.

  Returns:
    List of tags and entities tuples.
  """

  entity_markers = _text_between_brackets(tag_sequence)

  ret = []
  for entity_marker in entity_markers:
    marker_tokens = entity_marker.split()
    if len(marker_tokens) < 2:
      continue

    tag = marker_tokens[0]
    entity = " ".join(marker_tokens[1:])

    ret.append((tag, entity))
  return ret


def _text_between_brackets(tag_sequence):
  pattern = r"\[(.*?)\]"
  return re.findall(pattern, tag_sequence)
