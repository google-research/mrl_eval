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

"""Factory for the datasets."""

import transformers

from mrl_eval.datasets import constants
from mrl_eval.hf import args
from mrl_eval.hf.datasets import hf_datasets
from mrl_eval.hf.datasets import hf_datasets_lib


def hf_dataset_factory(
    dataset_name: str,
    data_args: args.DataArguments,
    tokenizer: transformers.AutoTokenizer,
) -> hf_datasets_lib.HfDataset:
  """Dataset factory function from the dataset name."""
  match dataset_name:
    case constants.HESENTIMENT:
      return hf_datasets.HfHeSentiment(data_args, tokenizer)
    case constants.HEQ:
      return hf_datasets.HfHeQ(data_args, tokenizer)
    case constants.HEQ_QUESTION_GEN:
      return hf_datasets.HfHeQQuestionGen(data_args, tokenizer)
    case constants.NEMO_MORPH:
      return hf_datasets.HfNemoMorph(data_args, tokenizer)
    case constants.NEMO_TOKEN:
      return hf_datasets.HfNemoToken(data_args, tokenizer)
    case constants.HESUM:
      return hf_datasets.HfHeSum(data_args, tokenizer)
    case constants.HEBNLI:
      return hf_datasets.HfHebNLI(data_args, tokenizer)
    case _:
      raise ValueError(f"Dataset {dataset_name} is not defined.")
