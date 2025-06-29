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

"""Arabic NLI dataset.

This dataset is a subset of the XNLI dataset available in the HuggingFace Hub
(https://huggingface.co/datasets/facebook/xnli/viewer/ar).
"""

from collections.abc import Mapping, Sequence
import copy
import pathlib
from typing import Any, Union

import immutabledict
import tensorflow as tf

from mrl_eval.datasets import constants
from mrl_eval.datasets import dataset_lib
from mrl_eval.evaluation import metrics
from mrl_eval.utils import io_utils

TextField = str
EntailmentLabelField = str

RawExample = dict[str, Union[int, str]]
FeatureMap = dict[str, tf.train.Feature]
RawDataset = dataset_lib.RawDataset

ENTAILMENT = 0
NEUTRAL = 1
CONTRADICTION = 2
LABEL_KEY = "label"
PREMISE_KEY = "premise"
HYPOTHESIS_KEY = "hypothesis"

LABEL_TRANSLATIONS = immutabledict.immutabledict({
    ENTAILMENT: "استتباع",
    CONTRADICTION: "تناقض",
    NEUTRAL: "محايد",
})


class ArabicNLI(dataset_lib.Dataset):
  """Arabic NLI dataset.

  This dataset is a subset of the XNLI dataset available in the HuggingFace Hub
  (https://huggingface.co/datasets/facebook/xnli/viewer/ar).
  """

  @property
  def dataset_name(self):
    return constants.ARABIC_NLI

  @property
  def raw_files(self):
    files = {"train": "train-00000-of-00001.parquet",
             "val": "validation-00000-of-00001.parquet",
             "test": "test-00000-of-00001.parquet"}
    return files

  @property
  def metrics(self):
    """Returns the metrics to be calculated for this dataset."""
    return [
        metrics.accuracy,
        metrics.get_macro_f1_fn(list(LABEL_TRANSLATIONS.values())),
    ]

  def read_raw_data_file(self, file_path):
    dataset = io_utils.read_parquet(file_path)
    return dataset.to_dict(orient="records")

  def process_raw_examples(
      self, dataset
  ):
    processed_dataset = []
    for idx, example in enumerate(dataset):
      processed_example = copy.deepcopy(example)
      processed_example[LABEL_KEY] = _translate_label(
          processed_example[LABEL_KEY]
      )
      processed_example["id"] = f"ArNLI_{idx}"
      processed_dataset.append(processed_example)
    return processed_dataset

  def name_to_features(self):
    return immutabledict.immutabledict({
        "id": tf.io.FixedLenFeature([], tf.string),
        PREMISE_KEY: tf.io.FixedLenFeature([], tf.string),
        HYPOTHESIS_KEY: tf.io.FixedLenFeature([], tf.string),
        LABEL_KEY: tf.io.FixedLenFeature([], tf.string),
    })

  def map_to_feature(self, ex):
    # Disabling pytype here as it can't infer correct type for the dict values.
    # pytype: disable=attribute-error
    feature = {
        "id": self.bytes_feature([ex["id"].encode()]),
        PREMISE_KEY: self.bytes_feature(
            [ex[PREMISE_KEY].encode()]
        ),
        HYPOTHESIS_KEY: self.bytes_feature(
            [ex[HYPOTHESIS_KEY].encode()]
        ),
        LABEL_KEY: self.bytes_feature(
            [ex[LABEL_KEY].encode()]
        ),
    }
    # pytype: enable=attribute-error

    return feature

  def get_inputs(self, example):
    return (
        "مقدمة: "
        + example[PREMISE_KEY]
        + "\n"
        + "فرضية: "
        + example[HYPOTHESIS_KEY]
    )

  def get_outputs(self, example):
    return example[LABEL_KEY]

  def get_example_id(self, example):
    return example["id"]


def _translate_label(label):
  """Translate the NLI labels to Arabic."""

  if label not in LABEL_TRANSLATIONS.keys():
    raise ValueError(f"Invalid label: {label}")
  return LABEL_TRANSLATIONS[label]
