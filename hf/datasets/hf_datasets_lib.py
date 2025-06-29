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

"""Base class for HF datasets."""

import abc
from collections.abc import MutableMapping, Sequence
import pathlib
from typing import Any, TypeAlias

import numpy as np
from rich import progress
import torch
from torch.utils.data import Dataset

from mrl_eval.datasets import constants
from mrl_eval.evaluation import metrics
from mrl_eval.utils import io_utils


Sample: TypeAlias = MutableMapping[str, Any]


_DECODER_ANSWER_SEP = " ### Answer:"


class DatasetSplit(Dataset):

  def __init__(self, data):
    self.data = data

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    return self.data[idx]


class HfDataset(abc.ABC):
  """Base class for HF datasets."""

  def __init__(
      self,
      data_args,
      tokenizer,
      for_decoder_only=False,
      decoder_answer_sep=_DECODER_ANSWER_SEP,
  ):
    self.data_args = data_args
    self.tokenizer = tokenizer
    self.for_decoder_only = for_decoder_only
    self.decoder_answer_sep = decoder_answer_sep

    def _load_split(
        split_name,
        add_targets_to_input,
        mask_prompt_labels = False,
    ):
      return self.init_split(
          self.get_data_file_path(split_name),
          add_targets_to_input=add_targets_to_input,
          mask_prompt_labels=mask_prompt_labels,
      )

    if data_args.load_train:
      if self.for_decoder_only:
        self._train_set = _load_split(
            "train",
            add_targets_to_input=True,
            mask_prompt_labels=True,
        )
      else:
        self._train_set = _load_split(
            "train", add_targets_to_input=False
        )

    if data_args.load_validation:
      self._validation_set = _load_split(
          "val", add_targets_to_input=False
      )

    if data_args.load_test:
      self._test_set = _load_split(
          "test", add_targets_to_input=False
      )

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

  def init_split(
      self,
      path,
      add_targets_to_input = False,
      mask_prompt_labels = False,
  ):
    """Initializes a split of the dataset."""
    data = io_utils.read_jsonl(path)
    print(f"Read {len(data)} samples from {path}")
    data = self.preprocess_input_target_fields(data)
    data = self.tokenize_samples(
        data,
        add_targets_to_input=add_targets_to_input,
        mask_prompt_labels=mask_prompt_labels,
    )
    return DatasetSplit(data)

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

    # This postprocessing is needed for mT5, and should not affect other
    # models.
    detok_preds = [p.replace("<extra_id_0>", "").strip() for p in detok_preds]

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
      self,
      samples,
      add_targets_to_input = False,
      mask_prompt_labels = False,
  ):
    """Tokenize samples into input and target ids."""

    max_input_length = self.data_args.max_inputs_length
    max_target_length = self.data_args.max_targets_length

    if self.for_decoder_only:
      if mask_prompt_labels and not add_targets_to_input:
        raise ValueError(
            "mask_prompt_labels=True requires add_targets_to_input=True"
        )

      separator_tokens = self.tokenizer(
          text_target=" " + self.decoder_answer_sep.strip(),
          max_length=None,
          truncation=False,
          return_tensors="pt",
          add_special_tokens=False,
      )["input_ids"].squeeze()

    else:
      if mask_prompt_labels:
        raise ValueError(
            "mask_prompt_labels is not supported for encoder-decoder models"
        )
      if add_targets_to_input:
        raise ValueError(
            "add_targets_to_input is not supported for encoder-decoder models"
        )

      separator_tokens = None

    for sample in progress.track(samples, "Tokenizing samples"):

      model_inputs = self.tokenizer(
          sample["inputs"],
          max_length=max_input_length,
          truncation=True,
          return_tensors="pt",
      )

      sample["input_ids"] = model_inputs["input_ids"].squeeze()

      if self.for_decoder_only:
        # Add separator tokens to the input ids
        sample["input_ids"] = torch.cat(
            [sample["input_ids"], separator_tokens], dim=0
        )

        # Add a space before the targets
        if add_targets_to_input:
          sample["targets"] = " " + sample["targets"]

        target_ids = self.tokenizer(
            text_target=sample["targets"],
            max_length=max_target_length,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"][0]

        if add_targets_to_input:
          sample["input_ids"] = torch.cat(
              [
                  sample["input_ids"],
                  target_ids,
                  torch.tensor([self.tokenizer.eos_token_id]),
              ],
              dim=0,
          )

        sample["labels"] = sample["input_ids"].clone()  # pytype: disable=attribute-error

        if mask_prompt_labels:
          sample["labels"][:-len(target_ids)-1] = -100  # pytype: disable=unsupported-operands

      else:
        target_ids = self.tokenizer(
            text_target=sample["targets"],
            max_length=max_target_length,
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze()
        sample["labels"] = target_ids

      sample["attention_mask"] = torch.ones_like(sample["input_ids"])

    return samples

  def preprocess_input_target_fields(
      self, data
  ):
    return [
        self._preprocess_example(sample)
        for sample in progress.track(data, "Preprocessing samples")
    ]
