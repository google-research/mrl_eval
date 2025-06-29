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

"""Library for evaluating predictions."""

from typing import Any

from mrl_eval.datasets import dataset_lib


def align_predictions_with_targets(
    predictions,
    gold_targets,
):
  """Aligns predictions with gold targets."""
  targets = []
  for prediction in predictions:
    id_ = prediction["input"]["id"]
    targets.append(gold_targets[id_])
  return targets


def get_scores(
    metrics,
    targets,
    predictions,
):
  """Calculates scores for the predictions."""
  scores = {}
  for metric_fn in metrics:
    scores.update(metric_fn(targets, predictions))
  return scores
