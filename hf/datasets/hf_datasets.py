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

"""Defining HF datasets."""

from collections.abc import Sequence
import pathlib
import re

from mrl_eval.datasets import constants
from mrl_eval.evaluation import metrics
from mrl_eval.hf.datasets import hf_datasets_lib


class HfHeSentiment(hf_datasets_lib.HfDataset):
  """Hebrew sentiment classification dataset."""

  def _preprocess_example(self, sample):
    return {
        "inputs": sample["text"],
        "targets": sample["sentiment"],
        "id": sample["id"],
    }

  @property
  def dataset_name(self):
    return constants.HESENTIMENT

  def metrics(self):
    return [
        metrics.accuracy,
        metrics.get_macro_f1_fn(["חיובי", "שלילי", "ניטרלי"]),
    ]


class HfHeQ(hf_datasets_lib.HfDataset):
  """Hebrew question answering dataset."""

  def _preprocess_example(self, sample):
    answers = sample["answers"]["text"]
    question = sample["question"]
    context = sample["context"]
    inputs = _string_join(["שאלה:", question, "הקשר:", context])
    return {
        "inputs": inputs,
        "targets": answers[0],
        "id": sample["id"],
        "context": context,
        "question": question,
        "answers": answers,
    }

  @property
  def dataset_name(self):
    return constants.HEQ

  def metrics(self):
    return [metrics.em, metrics.f1, metrics.tlnls]

  def _postprocess_val_targets(
      self, targets
  ):
    return [[t] for t in targets]


class HfHeQQuestionGen(hf_datasets_lib.HfDataset):
  """Hebrew question generation dataset."""

  def _preprocess_example(self, sample):
    answers = sample["answers"]["text"]
    question = sample["question"]
    context = sample["context"]
    inputs = _string_join(["תשובה:", answers[0], "הקשר:", context])
    return {
        "inputs": inputs,
        "targets": question,
        "id": sample["id"],
        "context": context,
        "question": question,
        "answers": answers,
    }

  @property
  def dataset_name(self):
    return constants.HEQ_QUESTION_GEN

  def metrics(self):
    return [metrics.rouge]

  def get_data_file_path(self, split):
    return (
        pathlib.Path(constants.BASE_PATH)
        / constants.HEQ
        / "jsonl"
        / f"{split}.jsonl"
    )


class HfHeSum(hf_datasets_lib.HfDataset):
  """Hebrew summarization dataset."""

  def _preprocess_example(self, sample):
    return {
        "id": sample["id"],
        "inputs": sample["article"],
        "targets": sample["summary"],
    }

  @property
  def dataset_name(self):
    return constants.HESUM

  def metrics(self):
    return [metrics.rouge]


class HfNemo(hf_datasets_lib.HfDataset):
  """Hebrew entity linking dataset."""

  _level = None

  def _preprocess_example(self, sample):
    return {
        "id": sample["id"],
        "inputs": sample["inputs"],
        "targets": sample[f"targets_as_entity_markers_{self._level}_level"],
    }

  def get_data_file_path(self, split):
    return (
        pathlib.Path(constants.BASE_PATH) / "nemo" / "jsonl" / f"{split}.jsonl"
    )

  @property
  def dataset_name(self):
    return constants.HESUM

  def metrics(self):
    return [metrics.token_level_span_f1]


class HfNemoToken(HfNemo):
  _level = "token"

  @property
  def dataset_name(self):
    return constants.NEMO_TOKEN


class HfNemoMorph(HfNemo):
  _level = "morph"

  @property
  def dataset_name(self):
    return constants.NEMO_MORPH


class HfHebNLI(hf_datasets_lib.HfDataset):
  """Hebrew NLI dataset."""

  def _preprocess_example(self, sample):
    sentence1 = sample["translation1"]
    sentence2 = sample["translation2"]
    inputs = _string_join(["משפט 1:", sentence1, "משפט 2:", sentence2])
    return {
        "id": sample["id"],
        "inputs": inputs,
        "targets": sample["label_in_hebrew"],
    }

  @property
  def dataset_name(self):
    return constants.HEBNLI

  def metrics(self):
    return [
        metrics.accuracy,
        metrics.get_macro_f1_fn(["היסק", "סתירה", "ניטרלי"]),
    ]


def _string_join(lst):
  """Joins elements on space, collapsing consecutive spaces."""
  return re.sub(r"\s+", " ", " ".join(lst))
