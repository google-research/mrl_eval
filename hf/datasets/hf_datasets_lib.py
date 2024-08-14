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

"""Base class for HF datasets."""

import abc
import pathlib
from typing import Any, Sequence

import numpy as np
import tqdm

from mrl_eval.datasets import constants
from mrl_eval.evaluation import metrics
from mrl_eval.utils import io_utils


class HfDataset(abc.ABC):
  """Base class for HF datasets."""

  def __init__(self, data_args, tokenizer):
    self.data_args = data_args
    self.tokenizer = tokenizer

    if data_args.load_train:
      self._train_set = self.init_split(self.get_data_file_path("train"))

    if data_args.load_validation:
      self._validation_set = self.init_split(self.get_data_file_path("val"))

    if data_args.load_test:
      self._test_set = self.init_split(self.get_data_file_path("test"))

  @abc.abstractmethod
  def _preprocess_example(self, sample):
    raise NotImplementedError()

  @property
  @abc.abstractmethod
  def dataset_name(self):
    raise NotImplementedError()

  def get_data_file_path(self, split):
    return (
        pathlib.Path(constants.BASE_PATH)
        / self.dataset_name
        / "jsonl"
        / f"{split}.jsonl"
    )

  def init_split(self, path):
    data = io_utils.read_jsonl(path)
    print(f"Read {len(data)} samples from {path}")
    data = self.preprocess_input_target_fields(data)
    data = self.tokenize_samples(data)
    return data

  def _get_target(self, example):
    return example["targets"]

  def _postprocess_val_targets(
      self, targets
  ):
    return list(targets)

  def compute_metrics(self, eval_preds):
    """Compute metrics for a batch of predictions."""
    pred_ids, target_ids = eval_preds.predictions, eval_preds.label_ids
    detok_targets = self.tokenizer.batch_decode(
        np.where(target_ids < 0, 0, target_ids), skip_special_tokens=True
    )
    detok_targets = self._postprocess_val_targets(detok_targets)

    detok_preds = self.tokenizer.batch_decode(
        np.where(pred_ids < 0, 0, pred_ids), skip_special_tokens=True
    )
    detok_preds = [p.replace("<extra_id_0>", "") for p in detok_preds]

    scores = {}
    for metric in self.metrics():
      scores.update(metric(detok_targets, detok_preds))
    return scores

  def metrics(self):
    raise NotImplementedError()

  def train_set(self):
    return self._train_set

  def validation_set(self):
    return self._validation_set

  def test_set(self):
    return self._test_set

  def tokenize_samples(
      self, samples
  ):
    """Tokenize samples into input and target ids."""
    for sample in tqdm.tqdm(samples, "Tokenizing samples"):
      model_inputs = self.tokenizer(
          sample["inputs"],
          max_length=self.data_args.max_inputs_length,
          truncation=True,
          return_tensors="pt",
      )
      sample["input_ids"] = model_inputs["input_ids"].squeeze()
      sample["attention_mask"] = model_inputs["attention_mask"].squeeze()

      sample["labels"] = self.tokenizer(
          text_target=sample["targets"],
          max_length=self.data_args.max_targets_length,
          truncation=True,
          return_tensors="pt",
      )["input_ids"].squeeze()
    return samples

  def preprocess_input_target_fields(
      self, data
  ):
    return [
        self._preprocess_example(sample)
        for sample in tqdm.tqdm(data, desc="Preprocessing samples")
    ]
