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

"""Evaluates downstream tasks.

This script receives name of task and path to prediction file and returns the
score of the task metric.
"""

from collections.abc import Sequence

from absl import app
from absl import flags

from mrl_eval.datasets import constants
from mrl_eval.datasets import dataset_factory
from mrl_eval.evaluation import evaluate_lib
from mrl_eval.utils import io_utils

_DATASET = flags.DEFINE_enum(
    "dataset",
    None,
    constants.DATASETS,
    "The dataset you'd like to evaluate.",
)

_PREDICTIONS_PATH = flags.DEFINE_string(
    "predictions_path", None, "Path to the predictions file."
)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  dataset_instance = dataset_factory.dataset_factory(_DATASET.value)
  gold_targets = dataset_instance.read_test_targets()
  predictions = io_utils.read_jsonl(_PREDICTIONS_PATH.value)
  aligned_targets = evaluate_lib.align_predictions_with_targets(
      predictions, gold_targets
  )
  scores = evaluate_lib.get_scores(
      dataset_instance.metrics,
      aligned_targets,
      [p["prediction"] for p in predictions],
  )

  print(scores)


if __name__ == "__main__":
  app.run(main)
