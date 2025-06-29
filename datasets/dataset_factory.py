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

"""Dataset factory."""

from mrl_eval.datasets import constants
from mrl_eval.datasets import dataset_lib
from mrl_eval.datasets.ar_xlsum import ar_xlsum_lib
from mrl_eval.datasets.arabic_nli import arabic_nli_lib
from mrl_eval.datasets.arcoref import arcoref_lib
from mrl_eval.datasets.arq import arq_lib
from mrl_eval.datasets.arsentiment import arsentiment_lib
from mrl_eval.datasets.artydiqa import artydiqa_lib
from mrl_eval.datasets.hebco import hebco_lib
from mrl_eval.datasets.hebnli import hebnli_lib
from mrl_eval.datasets.hebsummaries import hebsummaries_lib
from mrl_eval.datasets.heq import heq_lib
from mrl_eval.datasets.hesentiment import hesentiment_lib
from mrl_eval.datasets.hesum import hesum_lib
from mrl_eval.datasets.iahlt_ner import iahlt_ner_lib
from mrl_eval.datasets.nemo import nemo_lib


def dataset_factory(dataset_name: str) -> dataset_lib.Dataset:
  """Dataset factory function from the dataset name."""
  match dataset_name:
    case constants.HEQ:
      return heq_lib.HeQ()
    case constants.HEQ_QUESTION_GEN:
      return heq_lib.HeQQuestionGen()
    case constants.HEBNLI:
      return hebnli_lib.HebNLI()
    case constants.NEMO:
      return nemo_lib.Nemo()
    case constants.NEMO_TOKEN:
      return nemo_lib.NemoToken()
    case constants.NEMO_MORPH:
      return nemo_lib.NemoMorph()
    case constants.HESENTIMENT:
      return hesentiment_lib.HeSentiment()
    case constants.HESUM:
      return hesum_lib.HeSum()
    case constants.HEBCO:
      return hebco_lib.Hebco()
    case constants.ARTYDIQA:
      return artydiqa_lib.ArTyDiQA()
    case constants.ARTYDIQA_QUESTION_GEN:
      return artydiqa_lib.ArTyDiQAQuestionGen()
    case constants.ARSENTIMENT:
      return arsentiment_lib.ArSentiment()
    case constants.ARQ_SPOKEN:
      return arq_lib.ArQ(variant="spoken")
    case constants.ARQ_MSA:
      return arq_lib.ArQ(variant="MSA")
    case constants.ARQ_SPOKEN_QUESTION_GEN:
      return arq_lib.ArQQuestionGen(variant="spoken")
    case constants.ARQ_MSA_QUESTION_GEN:
      return arq_lib.ArQQuestionGen(variant="MSA")
    case constants.ARCOREF:
      return arcoref_lib.ArCoref()
    case constants.IAHLT_NER:
      return iahlt_ner_lib.IahltNer()
    case constants.HEBSUMMARIES:
      return hebsummaries_lib.HebSummaries()
    case constants.ARABIC_NLI:
      return arabic_nli_lib.ArabicNLI()
    case constants.AR_XLSUM:
      return ar_xlsum_lib.ArXLSum()

    case _:
      raise ValueError(f"Dataset {dataset_name} is not defined.")
