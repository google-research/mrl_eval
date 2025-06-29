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

"""Defining HF datasets."""

from collections.abc import Sequence
import functools
import pathlib
import re
from typing import TypeAlias

from mrl_eval.datasets import constants
from mrl_eval.datasets.ar_xlsum import ar_xlsum_lib
from mrl_eval.datasets.arabic_nli import arabic_nli_lib
from mrl_eval.datasets.arcoref import arcoref_lib
from mrl_eval.datasets.artydiqa import artydiqa_lib
from mrl_eval.datasets.hebco import hebco_lib
from mrl_eval.datasets.iahlt_ner import iahlt_ner_lib
from mrl_eval.evaluation import metrics
from mrl_eval.hf.datasets import hf_datasets_lib

_AR_ANSWER_PROMPT = "الجواب:"
_AR_QUESTION_PROMPT = "السؤال:"
_AR_CONTEXT_PROMPT = "النص:"

_HE_NULL_ANSWER_TEXT = "לא ניתן לענות על השאלה על סמך ההקשר."
_AR_NULL_ANSWER_TEXT = "لا يمكن الإجابة على السؤال من النص."
_AR_NLI_PREMISE_PROMPT = "مقدمة:"
_AR_NLI_HYPOTHESIS_PROMPT = "فرضية:"
_AR_TYDIQA_QUESTION_PROMPT = "السؤال:"
_AR_TYDIQA_CONTEXT_PROMPT = "النص:"
_AR_TYDIQA_ANSWER_PROMPT = "الجواب:"

Sample: TypeAlias = hf_datasets_lib.Sample


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
    return [
        metrics.em,
        metrics.f1,
        functools.partial(metrics.tlnls, null_answer_text=_HE_NULL_ANSWER_TEXT),
    ]

  def _postprocess_val_targets(self, targets):
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
        "id": sample["id"],  # pylint: disable=redefined-builtin
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
        / constants.HEQ_QUESTION_GEN
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


class HfHebSummaries(hf_datasets_lib.HfDataset):
  """Hebrew summarization dataset."""

  def _preprocess_example(self, sample):
    return {
        "id": sample["id"],
        "inputs": sample["text_raw"],
        "targets": sample["summary"],
    }

  @property
  def dataset_name(self):
    return constants.HEBSUMMARIES

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


class HfHebCo(hf_datasets_lib.HfDataset):
  """Hebrew coreference resolution dataset."""

  TEXT_FIELD = hebco_lib.Hebco.TEXT_FIELD
  TARGET_FIELD = hebco_lib.Hebco.TARGET_FIELD
  ID_FIELD = hebco_lib.Hebco.ID_FIELD

  INNER_SEP = hebco_lib.Hebco.INNER_SEP
  OUTER_SEP = hebco_lib.Hebco.OUTER_SEP
  WORD_SEP = hebco_lib.Hebco.WORD_SEP

  @property
  def dataset_name(self):
    return constants.HEBCO

  def _preprocess_example(self, sample):
    return {
        "id": sample[self.ID_FIELD],
        "inputs": sample[self.TEXT_FIELD],
        "targets": sample[self.TARGET_FIELD],
    }

  def metrics(self):
    return [
        metrics.get_em_cluster_matching_f1_fn(
            hebco_lib.get_parse_string_representation(
                inner_sep=self.INNER_SEP, outer_sep=self.OUTER_SEP
            )
        )
    ]


class HfArCoref(hf_datasets_lib.HfDataset):
  """Arabic coreference resolution dataset."""

  TEXT_FIELD = arcoref_lib.ArCoref.TEXT_FIELD
  TARGET_FIELD = arcoref_lib.ArCoref.TARGET_FIELD
  ID_FIELD = arcoref_lib.ArCoref.ID_FIELD

  INNER_SEP = arcoref_lib.ArCoref.INNER_SEP
  OUTER_SEP = arcoref_lib.ArCoref.OUTER_SEP
  WORD_SEP = arcoref_lib.ArCoref.WORD_SEP

  @property
  def dataset_name(self):
    return constants.ARCOREF

  def _preprocess_example(self, sample):
    return {
        "id": sample[self.ID_FIELD],
        "inputs": sample[self.TEXT_FIELD],
        "targets": sample[self.TARGET_FIELD],
    }

  def metrics(self):
    return [
        metrics.get_em_cluster_matching_f1_fn(
            hebco_lib.get_parse_string_representation(
                inner_sep=self.INNER_SEP, outer_sep=self.OUTER_SEP
            )
        )
    ]


class HfArQ(hf_datasets_lib.HfDataset):
  """Arabic Question Answering dataset."""

  def __init__(self, variant = "spoken", *args, **kwargs):
    if variant not in ["spoken", "MSA"]:
      raise ValueError(
          f"Unsupported ArQ variant: {variant}. Supported variants are 'spoken'"
          " and 'MSA'."
      )
    self._variant = variant
    super().__init__(*args, **kwargs)

  def _preprocess_example(self, sample):
    answers = sample["answers"]["text"]
    question = sample["question"]
    context = sample["context"]
    inputs = _string_join(
        [_AR_QUESTION_PROMPT, question, _AR_CONTEXT_PROMPT, context]
    )
    samp_id = sample["id"]
    return {
        "inputs": inputs,
        "targets": answers[0],
        "id": samp_id,
        "context": context,
        "question": question,
        "answers": answers,
    }

  @property
  def dataset_name(self):
    return (
        constants.ARQ_SPOKEN if self._variant == "spoken" else constants.ARQ_MSA
    )

  def metrics(self):
    return [
        metrics.em,
        metrics.f1,
        functools.partial(metrics.tlnls, null_answer_text=_AR_NULL_ANSWER_TEXT),
    ]

  def _postprocess_val_targets(self, targets):
    return [[t] for t in targets]


class HfArQQuestionGen(hf_datasets_lib.HfDataset):
  """Arabic Question Generation dataset."""

  def __init__(self, variant = "spoken", *args, **kwargs):
    if variant not in ["spoken", "MSA"]:
      raise ValueError(
          f"Unsupported ArQ variant: {variant}. Supported variants are 'spoken'"
          " and 'MSA'."
      )
    self._variant = variant
    super().__init__(*args, **kwargs)

  def _preprocess_example(self, sample):
    answers = sample["answers"]["text"]
    question = sample["question"]
    context = sample["context"]
    inputs = _string_join(
        [_AR_ANSWER_PROMPT, answers[0], _AR_CONTEXT_PROMPT, context]
    )
    samp_id = sample["id"]
    return {
        "inputs": inputs,
        "targets": question,
        "id": samp_id,
        "context": context,
        "question": question,
        "answers": answers,
    }

  @property
  def dataset_name(self):
    return (
        constants.ARQ_SPOKEN_QUESTION_GEN
        if self._variant == "spoken"
        else constants.ARQ_MSA_QUESTION_GEN
    )

  def metrics(self):
    return [metrics.rouge]


class HfArSentiment(hf_datasets_lib.HfDataset):
  """Arabic sentiment classification dataset."""

  def _preprocess_example(self, sample):
    return {
        "inputs": sample["text"],
        "targets": sample["sentiment"],
        "id": sample["id"],
    }

  @property
  def dataset_name(self):
    return constants.ARSENTIMENT

  def metrics(self):
    return [
        metrics.accuracy,
        metrics.get_macro_f1_fn(["إيجابي", "سلبي", "محايد", "معقد"]),
    ]


def _string_join(lst):
  """Joins elements on space, collapsing consecutive spaces."""
  return re.sub(r"\s+", " ", " ".join(lst))


class HfArTyDiQA(hf_datasets_lib.HfDataset):
  """Arabic question answering dataset based on TyDiQA."""

  def _preprocess_example(self, sample):
    answers = sample["answers"]["text"]
    question = sample["question"]
    context = sample["context"]
    inputs = _string_join([
        _AR_TYDIQA_QUESTION_PROMPT,
        question,
        _AR_TYDIQA_CONTEXT_PROMPT,
        context,
    ])

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
    return constants.ARTYDIQA

  def metrics(self):
    return [metrics.em, metrics.f1, artydiqa_lib.tydiqa_f1_score]

  def _postprocess_val_targets(self, targets):
    return [[t] for t in targets]


class HfArTyDiQAQuestionGen(hf_datasets_lib.HfDataset):
  """Arabic question generation dataset based on TyDiQA."""

  def _preprocess_example(self, sample):
    answers = sample["answers"]["text"]
    question = sample["question"]
    context = sample["context"]
    inputs = _string_join([
        _AR_TYDIQA_ANSWER_PROMPT,
        answers[0],
        _AR_TYDIQA_CONTEXT_PROMPT,
        context,
    ])
    samp_id = sample["id"]

    return {
        "inputs": inputs,
        "targets": question,
        "id": samp_id,
        "context": context,
        "question": question,
        "answers": answers,
    }

  @property
  def dataset_name(self):
    return constants.ARTYDIQA_QUESTION_GEN

  def metrics(self):
    return [metrics.rouge]


class HfIahltNer(hf_datasets_lib.HfDataset):
  """Arabic MSA entity linking dataset."""

  TEXT_FIELD = iahlt_ner_lib.IahltNer.TEXT_KEY
  TARGET_FIELD = iahlt_ner_lib.IahltNer.LABEL_KEY
  ID_FIELD = iahlt_ner_lib.IahltNer.ID_FIELD

  def _preprocess_example(self, sample):
    return {
        "id": sample[self.ID_FIELD],
        "inputs": sample[self.TEXT_FIELD],
        "targets": sample[self.TARGET_FIELD],
    }

  def get_data_file_path(self, split):
    return (
        pathlib.Path(constants.BASE_PATH)
        / self.dataset_name
        / "jsonl"
        / f"{split}.jsonl"
    )

  @property
  def dataset_name(self):
    return constants.IAHLT_NER

  def metrics(self):
    return [metrics.token_level_span_f1]


class HfArabicNLI(hf_datasets_lib.HfDataset):
  """Arabic NLI dataset."""

  def _preprocess_example(self, sample):
    premise = sample[arabic_nli_lib.PREMISE_KEY]
    hypothesis = sample[arabic_nli_lib.HYPOTHESIS_KEY]
    inputs = _string_join(
        [_AR_NLI_PREMISE_PROMPT, premise, _AR_NLI_HYPOTHESIS_PROMPT, hypothesis]
    )
    return {
        "id": sample["id"],
        "inputs": inputs,
        "targets": sample[arabic_nli_lib.LABEL_KEY],
    }

  @property
  def dataset_name(self):
    return constants.ARABIC_NLI

  def metrics(self):
    return [
        metrics.accuracy,
        metrics.get_macro_f1_fn(
            list(arabic_nli_lib.LABEL_TRANSLATIONS.values())
        ),
    ]


class HfArXLSum(hf_datasets_lib.HfDataset):
  """Arabic summarization dataset."""

  def _preprocess_example(self, sample):
    return {
        "id": sample[ar_xlsum_lib.ArXLSum.ID],
        "inputs": sample[ar_xlsum_lib.ArXLSum.article],
        "targets": sample[ar_xlsum_lib.ArXLSum.summary],
    }

  @property
  def dataset_name(self):
    return constants.AR_XLSUM

  def metrics(self):
    return [metrics.rouge]
