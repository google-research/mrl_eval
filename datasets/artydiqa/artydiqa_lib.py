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

"""Preprocessing the TyDiQA dataset and writing it to storage."""

from collections.abc import Callable, Mapping, Sequence
import functools
import pathlib
from typing import Any

import immutabledict
import tensorflow as tf

from mrl_eval.datasets import constants
from mrl_eval.datasets import dataset_lib
from mrl_eval.evaluation import metrics
from mrl_eval.evaluation import metrics_utils
from mrl_eval.utils import io_utils


# TyDiQA format
FeatureMap = Mapping[str, tf.train.Feature | list[tf.train.Feature]]
RawDataset = dataset_lib.RawDataset
RawExample = Mapping[str, Any]

_ARTYDIQA_CONTEXT_PROMPT = "النص: "
_ARTYDIQA_ANSWER_PROMPT = "الجواب: "
_ARTYDIQA_QUESTION_PROMPT = "السؤال: "
ARABIC_YES_LABEL = "نعم"
ARABIC_NO_LABEL = "لا"
ARABIC_NULL_LABEL = "لا يمكن الإجابة على السؤال من النص"

# Arabic labels.
_ARABIC_LABELS = immutabledict.immutabledict({
    "yes": ARABIC_YES_LABEL,
    "no": ARABIC_NO_LABEL,
    "null": ARABIC_NULL_LABEL,
})


def _max_suffix_to_prefix_overlap(s1, s2):
  """Finds the longest suffix of s1 that is a prefix to s2."""
  suffix_start_idx = max(0, len(s1) - len(s2))
  idx = suffix_start_idx
  while idx < len(s1):
    prefix_end_idx = idx - suffix_start_idx
    if s1[idx] != s2[prefix_end_idx]:
      suffix_start_idx = idx + 1
    idx += 1

  return s1[suffix_start_idx:]


def _get_maximum_span_overlap(target, pred):
  """Returns the span that is common between the target and pred."""
  if target in pred:
    in_both = target
  elif pred in target:
    in_both = pred
  else:
    max_pred_suffix_and_target_prefix = _max_suffix_to_prefix_overlap(
        pred, target
    )
    max_target_suffix_and_pred_prefix = _max_suffix_to_prefix_overlap(
        target, pred
    )
    in_both = max(
        max_pred_suffix_and_target_prefix,
        max_target_suffix_and_pred_prefix,
        key=len,
    )
  return in_both


def _tydiqa_text_span_f1_score(target, prediction):
  """Returns the F1 score based on TyDiQA's answer span credit system.

  TyDiQA's credit system assumes the answers are provided as text spans (start
  and end byte indices) and defines the F1 score as follows:

  * precision = in_both / (in_both + only_in_pred)
  * recall = in_both / (in_both + only_in_target)
  * f1 = 2 * precision * recall / (precision + recall)

  Where
  - in_both: intersection of the two spans.
  - only_in_pred: the part of the prediction span that is not in the target
  span.
  - only_in_target: the part of the target span that is not in the prediction
    span.
  If the spans are not overlapping, no credit is assigned.
  Since our Seq-2-Seq format does not provide the start and end byte indices,
  we will approximate the credit system above by working on the sequence level.
  Where a span overlap can mean any of the following cases:
  - The prediction is a substring of the target.
  - The target is a substring of the prediction.
  - The longest of
    - The longest prediction suffix that is a prefix of the target.
    - The longest target suffix that is a prefix of the prediction.

  Args:
    target: Gold answer.
    prediction: Model prediction.

  Returns:
    The F1 score based on TyDiQA's answer span credit system.
  """
  in_both = _get_maximum_span_overlap(target, prediction)
  only_in_pred = prediction.replace(in_both, "", 1)
  only_in_target = target.replace(in_both, "", 1)
  # Convert to bytes length.
  in_both_byte_length = len(in_both.encode("utf-8"))
  only_in_pred_byte_length = len(only_in_pred.encode("utf-8"))
  only_in_target_byte_length = len(only_in_target.encode("utf-8"))
  precision = in_both_byte_length / (
      in_both_byte_length + only_in_pred_byte_length + 1e-15
  )
  recall = in_both_byte_length / (
      in_both_byte_length + only_in_target_byte_length + 1e-15
  )
  return 2 * precision * recall / (precision + recall + 1e-15)


def _tydiqa_credit_system(
    target, prediction, func
):
  """Returns the score based on TyDiQA's credit system, using the given function (F1 or TNLNS)."""
  if target in _ARABIC_LABELS.values():
    return 1 if target == prediction else 0
  return func(target, prediction)


def tydiqa_f1_score(
    targets,
    predictions,
):
  """Computes the TyDiQA F1 score based on TyDiQA's credit system."""
  targets = [[metrics_utils.normalize_squad(t) for t in u] for u in targets]
  predictions = [metrics_utils.normalize_squad(p) for p in predictions]
  return {
      "tydiqa_f1": metrics_utils.average_max_over_ground_truths(
          targets,
          predictions,
          functools.partial(
              _tydiqa_credit_system, func=_tydiqa_text_span_f1_score
          ),
      ),
      "tydiqa_tlnls": metrics_utils.average_max_over_ground_truths(
          targets,
          predictions,
          functools.partial(
              _tydiqa_credit_system, func=metrics_utils.tlnls_single_prediction
          ),
      ),
  }


class ArTyDiQA(dataset_lib.Dataset):
  """Implementation of dataset_lib.Dataset for the Arabic subset of TyDiQA."""

  @property
  def dataset_name(self):
    return constants.ARTYDIQA

  @property
  def raw_files(self):
    return {
        "test": "test.jsonl",
        "train": "train.jsonl",
        "val": "val.jsonl",
    }

  @property
  def metrics(self):
    """Returns the metrics to be calculated for this dataset."""
    return [metrics.em, metrics.f1, tydiqa_f1_score]

  def map_to_feature(self, ex):
    """Maps RawExample into a FeatureMap as expected for TFRecord."""
    feature = {
        "id": self.bytes_feature([ex["id"].encode()]),
        "title": self.bytes_feature([ex["title"].encode()]),
        "context": self.bytes_feature([ex["context"].encode()]),
        "question": self.bytes_feature([ex["question"].encode()]),
        "answers/text": self.bytes_feature(
            [text.encode() for text in ex["answers"]["text"]]
        ),
        "answers/answer_start": self.int64_feature(
            ex["answers"]["answer_start"]
        ),
    }
    return feature

  def name_to_features(self):
    return immutabledict.immutabledict({
        "id": tf.io.FixedLenFeature([], tf.string),
        "title": tf.io.FixedLenFeature([], tf.string),
        "question": tf.io.FixedLenFeature([], tf.string),
        "context": tf.io.FixedLenFeature([], tf.string),
        "answers/text": tf.io.FixedLenSequenceFeature(
            [], tf.string, allow_missing=True
        ),
        "answers/answer_start": tf.io.FixedLenSequenceFeature(
            [], tf.int64, allow_missing=True
        ),
    })

  def read_raw_data_file(self, file_path):
    return io_utils.read_jsonl(file_path)

  def process_raw_examples(self, dataset):
    """Loads ArTyDiQA raw examples.

    Since the ArTyDiQA dataset is already in the correct format, this function
    is a no-op.

    Args:
      dataset: The raw dataset to process.

    Returns:
      The processed dataset.
    """
    return dataset

  def get_inputs(self, example):
    return (
        _ARTYDIQA_CONTEXT_PROMPT
        + example["context"]
        + "\n"
        + _ARTYDIQA_QUESTION_PROMPT
        + example["question"]
    )

  def get_outputs(self, example):
    return example["answers"]["text"]

  def get_example_id(self, example):
    return example["id"]


class ArTyDiQAQuestionGen(ArTyDiQA):
  """Implementation of dataset_lib.Dataset for ArTyDiQA for the question generation task."""

  @property
  def dataset_name(self):
    return constants.ARTYDIQA_QUESTION_GEN

  @property
  def metrics(self):
    return [metrics.rouge, metrics.tlnls]

  @property
  def dataset_dir(self):
    """Returns the directory of the raw files.

    The ArTyDiQA question generation task uses the ArTyDiQA QA task.
    """
    return pathlib.Path(constants.BASE_PATH) / constants.ARTYDIQA_QUESTION_GEN

  def get_inputs(self, example):
    return (
        _ARTYDIQA_ANSWER_PROMPT
        + "\n"
        + _ARTYDIQA_CONTEXT_PROMPT
        + example["context"]
    )

  def get_outputs(self, example):
    return str(example["question"])
