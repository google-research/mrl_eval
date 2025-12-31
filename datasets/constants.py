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

"""Datasets constants."""

ARTYDIQA = "artydiqa"
ARTYDIQA_QUESTION_GEN = "artydiqa_question_gen"
ARSENTIMENT = "arsentiment"
ARCOREF = "arcoref"
ARCOREF_GOLD_MENTIONS = "arcoref_gold_mentions"
IAHLT_NER = "iahlt_ner"
ARQ_SPOKEN = "arq_spoken"
ARQ_MSA = "arq_MSA"
ARQ_SPOKEN_QUESTION_GEN = "arq_spoken_question_gen"
ARQ_MSA_QUESTION_GEN = "arq_MSA_question_gen"
AR_XLSUM = "ar_xlsum"
ARABIC_NLI = "arabic_nli"
HEQ = "heq"
HEQ_QUESTION_GEN = "heq_question_gen"
NEMO = "nemo"
NEMO_TOKEN = "nemo_token"
NEMO_MORPH = "nemo_morph"
WOJOOD_SPOKEN = "wojood_spoken"
WOJOOD_MSA = "wojood_msa"
WOJOOD_FULL = "wojood_full"
HEBNLI = "hebnli"
HESENTIMENT = "hesentiment"
HESUM = "hesum"
HEBCO = "hebco"
HEBCO_GOLD_MENTIONS = "hebco_gold_mentions"
HEBSUMMARIES = "hebsummaries"
MSA_SENTIMENT = "msa_sentiment"
ONTONOTES = "ontonotes"
ONTONOTES_GOLD_MENTIONS = "ontonotes_gold_mentions"
SHAMNER = "shamner"
ASAS = "asas"

DATASETS = (
    ARTYDIQA,
    ARTYDIQA_QUESTION_GEN,
    ARSENTIMENT,
    ARQ_SPOKEN,
    ARQ_MSA,
    ARQ_SPOKEN_QUESTION_GEN,
    ARQ_MSA_QUESTION_GEN,
    ARABIC_NLI,
    AR_XLSUM,
    HEQ,
    HEQ_QUESTION_GEN,
    NEMO,
    NEMO_TOKEN,
    NEMO_MORPH,
    HEBNLI,
    HESENTIMENT,
    HESUM,
    HEBCO,
    HEBCO_GOLD_MENTIONS,
    ARCOREF,
    ARCOREF_GOLD_MENTIONS,
    IAHLT_NER,
    HEBSUMMARIES,
    WOJOOD_SPOKEN,
    WOJOOD_MSA,
    WOJOOD_FULL,
    MSA_SENTIMENT,
    ONTONOTES,
    ONTONOTES_GOLD_MENTIONS,
    SHAMNER,
    ASAS,
)

BASE_PATH = "mrl_eval_data"
