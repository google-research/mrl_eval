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

"""Preprocess dataset into jsonl and tfrecords."""

from collections.abc import Sequence

from absl import app
from absl import flags

from mrl_eval.datasets import constants
from mrl_eval.datasets import dataset_factory


_DATASET = flags.DEFINE_enum(
    "dataset",
    None,
    [
        constants.NEMO,
        constants.HEQ,
        constants.HEBNLI,
        constants.HESENTIMENT,
        constants.HESUM,
    ],
    "The dataset you'd like to ingest.",
)

_SAVE_TFRECORD = flags.DEFINE_bool(
    "save_tfrecord",
    False,
    "If true, saves the data in tfrecords format in addition to jsonl.",
)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  dataset = dataset_factory.dataset_factory(_DATASET.value)
  dataset.preprocess_dataset(_SAVE_TFRECORD.value)


if __name__ == "__main__":
  app.run(main)
