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
    for_decoder_only: bool = False,
) -> hf_datasets_lib.HfDataset:
  """Dataset factory function from the dataset name."""
  match dataset_name:
    case constants.HESENTIMENT:
      return hf_datasets.HfHeSentiment(data_args, tokenizer, for_decoder_only)
    case constants.HEQ:
      return hf_datasets.HfHeQ(data_args, tokenizer, for_decoder_only)
    case constants.HEQ_QUESTION_GEN:
      return hf_datasets.HfHeQQuestionGen(
          data_args, tokenizer, for_decoder_only
      )
    case constants.NEMO_MORPH:
      return hf_datasets.HfNemoMorph(data_args, tokenizer, for_decoder_only)
    case constants.NEMO_TOKEN:
      return hf_datasets.HfNemoToken(data_args, tokenizer, for_decoder_only)
    case constants.HESUM:
      return hf_datasets.HfHeSum(data_args, tokenizer, for_decoder_only)
    case constants.HEBNLI:
      return hf_datasets.HfHebNLI(data_args, tokenizer, for_decoder_only)
    case constants.HEBCO:
      return hf_datasets.HfHebCo(data_args, tokenizer, for_decoder_only)
    case constants.ARQ_SPOKEN:
      return hf_datasets.HfArQ("spoken", data_args, tokenizer, for_decoder_only)
    case constants.ARQ_SPOKEN_QUESTION_GEN:
      return hf_datasets.HfArQQuestionGen(
          "spoken", data_args, tokenizer, for_decoder_only
      )
    case constants.ARQ_MSA:
      return hf_datasets.HfArQ("MSA", data_args, tokenizer, for_decoder_only)
    case constants.ARQ_MSA_QUESTION_GEN:
      return hf_datasets.HfArQQuestionGen(
          "MSA", data_args, tokenizer, for_decoder_only
      )
    case constants.ARSENTIMENT:
      return hf_datasets.HfArSentiment(data_args, tokenizer, for_decoder_only)
    case constants.ARTYDIQA:
      return hf_datasets.HfArTyDiQA(data_args, tokenizer, for_decoder_only)
    case constants.ARTYDIQA_QUESTION_GEN:
      return hf_datasets.HfArTyDiQAQuestionGen(
          data_args, tokenizer, for_decoder_only
      )
    case constants.ARCOREF:
      return hf_datasets.HfArCoref(data_args, tokenizer, for_decoder_only)
    case constants.IAHLT_NER:
      return hf_datasets.HfIahltNer(data_args, tokenizer, for_decoder_only)
    case constants.HEBSUMMARIES:
      return hf_datasets.HfHebSummaries(data_args, tokenizer, for_decoder_only)
    case constants.ARABIC_NLI:
      return hf_datasets.HfArabicNLI(data_args, tokenizer, for_decoder_only)
    case constants.AR_XLSUM:
      return hf_datasets.HfArXLSum(data_args, tokenizer, for_decoder_only)

    case _:
      raise ValueError(f"Dataset {dataset_name} is not defined.")
