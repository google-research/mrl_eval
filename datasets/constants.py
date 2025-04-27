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

"""Datasets constants."""

ARTYDIQA = "artydiqa"
ARTYDIQA_QUESTION_GEN = "artydiqa_question_gen"
ARSENTIMENT = "arsentiment"
ARCOREF = "arcoref"
IAHLT_NER = "iahlt_ner"
ARQ_SPOKEN = "arq_spoken"
ARQ_MSA = "arq_MSA"
ARQ_SPOKEN_QUESTION_GEN = "arq_spoken_question_gen"
ARQ_MSA_QUESTION_GEN = "arq_MSA_question_gen"
HEQ = "heq"
HEQ_QUESTION_GEN = "heq_question_gen"
NEMO = "nemo"
NEMO_TOKEN = "nemo_token"
NEMO_MORPH = "nemo_morph"
HEBNLI = "hebnli"
HESENTIMENT = "hesentiment"
HESUM = "hesum"
HEBCO = "hebco"
HEBSUMMARIES = "hebsummaries"


DATASETS = (
    ARTYDIQA,
    ARTYDIQA_QUESTION_GEN,
    ARSENTIMENT,
    ARQ_SPOKEN,
    ARQ_MSA,
    ARQ_SPOKEN_QUESTION_GEN,
    ARQ_MSA_QUESTION_GEN,
    HEQ,
    HEQ_QUESTION_GEN,
    NEMO,
    NEMO_TOKEN,
    NEMO_MORPH,
    HEBNLI,
    HESENTIMENT,
    HESUM,
    HEBCO,
    ARCOREF,
    IAHLT_NER,
    HEBSUMMARIES,

)

BASE_PATH = "mrl_eval_data"
