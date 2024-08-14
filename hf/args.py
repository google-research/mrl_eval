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

"""Arguments for fine-tuning."""

import dataclasses
from typing import Optional

from mrl_eval.datasets import constants


@dataclasses.dataclass
class ModelArguments:
  """Arguments pertaining to which model/config/tokenizer we are going to fine-tune from."""

  model_name_or_path: str = dataclasses.field(
      metadata={
          "help": (
              "Path to pretrained model or model identifier from"
              " huggingface.co/models"
          )
      }
  )
  cache_dir: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "help": (
              "Where do you want to store the pretrained models downloaded from"
              " huggingface.co"
          )
      },
  )


@dataclasses.dataclass
class DataArguments:
  """Arguments pertaining to what data we are going to input our model for training and eval."""

  task_name: Optional[str] = dataclasses.field(
      default=None,
      metadata={"help": "The task to train on."},
  )

  max_inputs_length: int = dataclasses.field(
      default=2048,
      metadata={
          "help": (
              "The maximum total input sequence length after tokenization. If"
              " set, sequences longer than this will be truncated, sequences"
              " shorter will be padded."
          )
      },
  )
  max_targets_length: int = dataclasses.field(
      default=256,
      metadata={
          "help": (
              "The maximum total target sequence length after tokenization."
              " Sequences longer than this will be truncated, sequences shorter"
              " will be padded."
          )
      },
  )
  max_train_samples: Optional[int] = dataclasses.field(
      default=None,
      metadata={
          "help": (
              "For debugging purposes or quicker training, truncate the number"
              " of training examples to this value if set."
          )
      },
  )

  load_train: bool = dataclasses.field(
      default=True,
      metadata={"help": "Whether to load the train set."},
  )
  load_validation: bool = dataclasses.field(
      default=True,
      metadata={"help": "Whether to load the validation set."},
  )
  load_test: bool = dataclasses.field(
      default=False,
      metadata={"help": "Whether to load the validation set."},
  )


TASKS_CONFIGS = {
    constants.HESENTIMENT: {
        "max_inputs_length": 400,
        "max_targets_length": 4,
        "generation_max_length": 4,
        "per_device_eval_batch_size": 32,
        "metric_for_best_model": "macro_f1",
    },
    constants.HEQ: {
        "max_inputs_length": 800,
        "max_targets_length": 180,
        "generation_max_length": 180,
        "per_device_eval_batch_size": 32,
        "metric_for_best_model": "tlnls",
    },
    constants.HEQ_QUESTION_GEN: {
        "max_inputs_length": 800,
        "max_targets_length": 180,
        "generation_max_length": 180,
        "per_device_eval_batch_size": 32,
        "metric_for_best_model": "rougeL",
    },
    constants.NEMO_TOKEN: {
        "max_inputs_length": 200,
        "max_targets_length": 240,
        "generation_max_length": 240,
        "per_device_eval_batch_size": 32,
        "metric_for_best_model": "token_level_span_f1",
    },
    constants.NEMO_MORPH: {
        "max_inputs_length": 200,
        "max_targets_length": 240,
        "generation_max_length": 240,
        "per_device_eval_batch_size": 32,
        "metric_for_best_model": "token_level_span_f1",
    },
    constants.HESUM: {
        "max_inputs_length": 2000,
        "max_targets_length": 150,
        "generation_max_length": 120,
        "per_device_eval_batch_size": 10,
        "metric_for_best_model": "rouge2",
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "eval_delay": 2000,
        "gradient_accumulation_steps": 2,
        "learning_rate": 5e-5,
    },
    constants.HEBNLI: {
        "max_inputs_length": 250,
        "max_targets_length": 5,
        "generation_max_length": 5,
        "per_device_eval_batch_size": 32,
        "metric_for_best_model": "macro_f1",
    },
}
